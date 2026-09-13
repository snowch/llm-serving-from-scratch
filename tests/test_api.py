"""Chapter 24: the layer between a client and the engine."""

import json

import pytest

from llmserve.api import (
    DEFAULT_TEMPLATE,
    ChatMessage,
    ChatTemplate,
    CompletionRequest,
    StreamingResponse,
    Usage,
    chunk_event,
    done_event,
)
from llmserve.request import Request
from llmserve.sampling import SamplingParams
from llmserve.tokenizer import ByteTokenizer


@pytest.fixture
def tokenizer():
    return ByteTokenizer()


def _messages(system="You are helpful.", user="Hello"):
    return [ChatMessage("system", system), ChatMessage("user", user)]


# -- templates --------------------------------------------------------------------------


def test_a_template_ends_by_inviting_the_model_to_speak():
    """Forgetting the generation prompt is the most common template bug and looks like a bad model."""
    rendered = DEFAULT_TEMPLATE.render(_messages())
    assert rendered.endswith(DEFAULT_TEMPLATE.generation_prompt)


def test_the_same_messages_render_to_the_same_bytes():
    """Prefix caching is content-addressed, so a template must be a pure function of its input."""
    assert DEFAULT_TEMPLATE.render(_messages()) == DEFAULT_TEMPLATE.render(_messages())


def test_a_changing_system_prompt_changes_the_prefix():
    """The silent failure chapter 24 measures: a per-request value destroys the shared prefix."""
    stable = DEFAULT_TEMPLATE.render(_messages())
    stamped = DEFAULT_TEMPLATE.render(_messages(system="12:01 You are helpful."))
    shared = 0
    for a, b in zip(stable, stamped, strict=False):
        if a != b:
            break
        shared += 1
    assert shared < len(DEFAULT_TEMPLATE.role_open["system"]) + 4


def test_an_unknown_role_does_not_crash_the_template():
    """A client can send anything; a server that raises on an unknown role returns a 500."""
    rendered = ChatTemplate().render([ChatMessage("tool", "result")])
    assert "result" in rendered


# -- request translation -----------------------------------------------------------------


def test_a_completion_request_becomes_an_engine_request(tokenizer):
    completion = CompletionRequest(messages=_messages(), max_tokens=12, temperature=0.7)
    request = completion.to_engine_request(tokenizer)
    assert isinstance(request, Request)
    assert request.params.max_tokens == 12
    assert request.params.temperature == pytest.approx(0.7)
    assert request.prompt_len == len(tokenizer.encode(completion.prompt()))


def test_the_tenant_comes_from_the_key_not_the_body(tokenizer):
    """Chapter 19 needs a tenant; letting the request body assert one is how you get impersonation."""
    completion = CompletionRequest(messages=_messages(), tenant="acme")
    assert completion.to_engine_request(tokenizer).tenant == "acme"


def test_an_invalid_max_tokens_is_rejected_at_the_edge(tokenizer):
    with pytest.raises(ValueError):
        CompletionRequest(messages=_messages(), max_tokens=0).to_engine_request(tokenizer)


# -- streaming ---------------------------------------------------------------------------


def test_a_chunk_is_valid_json_in_the_expected_shape():
    event = chunk_event(7, "m", "hi")
    assert event.startswith("data: ") and event.endswith("\n\n")
    payload = json.loads(event[len("data: ") :])
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"][0]["delta"]["content"] == "hi"


def test_the_stream_ends_with_the_sentinel():
    """A client that never sees [DONE] hangs until its own timeout; the server sees success."""
    request = Request(prompt_token_ids=[1, 2, 3], params=SamplingParams(max_tokens=4))
    stream = StreamingResponse(request)
    tail = stream.on_finish("stop")
    assert tail.endswith(done_event())
    assert '"finish_reason":"stop"' in tail.replace(" ", "")
    assert stream.finished


def test_a_multibyte_character_is_not_split_across_chunks():
    """Per-token decoding turns one multi-byte character into two replacement characters."""
    request = Request(prompt_token_ids=[1], params=SamplingParams(max_tokens=4))
    stream = StreamingResponse(request)
    deltas = []
    for byte in "€".encode():
        event = stream.on_tokens([byte])
        if event:
            deltas.append(json.loads(event[len("data: ") :])["choices"][0]["delta"]["content"])
    assert "".join(deltas) == "€"
    assert not any("\ufffd" in delta for delta in deltas)


def test_usage_counts_the_tokens_the_engine_processed():
    request = Request(prompt_token_ids=[1, 2, 3, 4], params=SamplingParams(max_tokens=8))
    stream = StreamingResponse(request)
    stream.on_tokens([65, 66])
    stream.on_tokens([67])
    assert stream.usage.prompt_tokens == 4
    assert stream.usage.completion_tokens == 3
    assert stream.usage.total_tokens == 7


def test_usage_serialises_the_three_fields_clients_read():
    assert set(Usage(3, 4).as_dict()) == {"prompt_tokens", "completion_tokens", "total_tokens"}
