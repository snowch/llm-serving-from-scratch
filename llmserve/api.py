"""Chapter 26: the surface a caller actually touches.

Everything before this chapter is the engine. This is the layer between it and an HTTP client, and
it is where a surprising share of production incidents live — not because it is hard, but because
it is the part nobody measures. A chat template with a stray space, a usage count that double-bills,
a stream that never terminates: each is invisible in the engine's own metrics and each is visible to
every user.

There is no web framework here. The parts that matter are the request translation, the chat
template, the event framing and the accounting, and all four are ordinary data transformations.
Bolting them to an ASGI server is the easy half and teaches nothing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from llmserve.request import Request
from llmserve.sampling import SamplingParams
from llmserve.tokenizer import ByteTokenizer, IncrementalDetokenizer


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatTemplate:
    """How a list of messages becomes one string of tokens.

    This is the single most under-tested component in a serving stack. It decides the exact bytes
    the model sees, so a change to it changes the model's behaviour *and* — because prefix caching
    is content-addressed (chapter 9) — silently changes how much of the prompt can be reused. The
    ordering rule below is the one people get wrong.
    """

    #: opens a turn, per role
    role_open: dict[str, str] = field(
        default_factory=lambda: {
            "system": "<|system|>\n",
            "user": "<|user|>\n",
            "assistant": "<|assistant|>\n",
        }
    )
    role_close: str = "<|end|>\n"
    #: the tokens that invite the model to speak
    generation_prompt: str = "<|assistant|>\n"
    bos: str = ""

    def render(self, messages: list[ChatMessage]) -> str:
        parts = [self.bos]
        for message in messages:
            parts.append(self.role_open.get(message.role, self.role_open["user"]))
            parts.append(message.content)
            parts.append(self.role_close)
        parts.append(self.generation_prompt)
        return "".join(parts)


DEFAULT_TEMPLATE = ChatTemplate()


@dataclass
class CompletionRequest:
    """The subset of ``/v1/chat/completions`` that changes what the engine does.

    Deliberately a subset. Half the fields in the real schema are either ignored by most servers or
    are aliases for each other, and pretending to support them is worse than not.
    """

    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    stop: tuple[str, ...] = ()
    model: str = "llmserve-tiny"
    #: which tenant this belongs to (chapter 21), from the API key rather than the body
    tenant: str | None = None

    def prompt(self, template: ChatTemplate = DEFAULT_TEMPLATE) -> str:
        return template.render(self.messages)

    def to_engine_request(
        self,
        tokenizer: ByteTokenizer,
        template: ChatTemplate = DEFAULT_TEMPLATE,
        *,
        seed: int | None = None,
    ) -> Request:
        return Request(
            prompt_token_ids=tokenizer.encode(self.prompt(template)),
            params=SamplingParams(
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=seed,
            ),
            tenant=self.tenant,
        )


@dataclass
class Usage:
    """What the caller is billed for.

    Counted from the token ids the engine actually processed, never from the text. Re-tokenizing
    the rendered prompt to count it is the classic way to bill for a different number than you
    served, because the count then depends on a code path the engine never ran.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def chunk_event(request_id: int, model: str, delta: str, finish_reason: str | None = None) -> str:
    """One server-sent event, in the shape every OpenAI-compatible client expects."""
    payload = {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": delta} if delta else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    # ensure_ascii=False because the stream is UTF-8 and escaping every non-ASCII character
    # inflates exactly the payloads that are already largest.
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return f"data: {body}\n\n"


def done_event() -> str:
    """The sentinel that ends the stream.

    Not optional, and not implied by the connection closing. A client that never sees it either
    hangs until its own timeout or reports a truncated response — and on the server side the
    request looks completed, so nothing alerts.
    """
    return "data: [DONE]\n\n"


class StreamingResponse:
    """Turns an engine's token ids into the event stream a client reads.

    Owns the incremental detokenizer, because that is the only place that can be right: a
    multi-byte character split across two tokens must not be emitted as two replacement characters,
    and a stateless per-token decode cannot see that.
    """

    def __init__(self, request: Request, model: str = "llmserve-tiny") -> None:
        self.request = request
        self.model = model
        self.usage = Usage(prompt_tokens=request.prompt_len)
        self._detokenizer = IncrementalDetokenizer()
        self._finished = False

    def on_tokens(self, token_ids: list[int]) -> str:
        text = "".join(self._detokenizer.append(token_id) for token_id in token_ids)
        self.usage.completion_tokens += len(token_ids)
        return chunk_event(self.request.request_id, self.model, text) if text else ""

    def on_finish(self, reason: str | None = "stop") -> str:
        tail = self._detokenizer.finalize()
        self._finished = True
        events = []
        if tail:
            events.append(chunk_event(self.request.request_id, self.model, tail))
        events.append(chunk_event(self.request.request_id, self.model, "", finish_reason=reason))
        events.append(done_event())
        return "".join(events)

    @property
    def finished(self) -> bool:
        return self._finished
