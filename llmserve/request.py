"""What the engine schedules.

``Request`` is what a caller submits. ``RequestState`` is the engine's private bookkeeping for
it, and it grows as the book does: chapter 8 adds block tables, chapter 9 a cache-hit count,
chapter 17 an acceptance count. Keeping the two apart means the public interface stays stable
while the engine is rebuilt underneath it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import torch

from llmserve.sampling import SamplingParams

_ids = itertools.count()


@dataclass
class Request:
    """A unit of work submitted to the engine."""

    prompt_token_ids: list[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    request_id: int = field(default_factory=lambda: next(_ids))
    #: who submitted it (ch21). A single-tenant deployment leaves this None and never notices;
    #: a shared one cannot schedule fairly without it, because fairness is a property *between*
    #: callers and the engine has no other way to tell two callers apart.
    tenant: str | None = None

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)


@dataclass
class RequestState:
    """The engine's view of an in-flight request."""

    request: Request
    output_token_ids: list[int] = field(default_factory=list)
    past: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    #: physical KV blocks held by this sequence (ch08); empty for the contiguous engines
    block_table: list[int] = field(default_factory=list)
    #: how many tokens of this sequence are actually stored (ch08)
    stored: int = 0
    #: how many tokens must be cached before this sequence can decode (ch10).
    #: Not len(all_token_ids): that grows with every generated token, so comparing against it
    #: makes a decoding sequence look like it is prefilling again.
    prefill_target: int = 0
    finished: bool = False
    finish_reason: str | None = None

    @property
    def request_id(self) -> int:
        return self.request.request_id

    @property
    def all_token_ids(self) -> list[int]:
        return [*self.request.prompt_token_ids, *self.output_token_ids]

    @property
    def cache_len(self) -> int:
        """How many tokens this sequence currently has cached."""
        return 0 if self.past is None else self.past[0][0].shape[2]

    @property
    def total_len(self) -> int:
        return self.request.prompt_len + len(self.output_token_ids)

    def check_finished(self) -> bool:
        """Stop on token budget or a stop token. Returns True if this call ended the request."""
        params = self.request.params
        if self.finished:
            return False
        if self.output_token_ids and self.output_token_ids[-1] in params.stop_token_ids:
            self.finished, self.finish_reason = True, "stop_token"
            return True
        if len(self.output_token_ids) >= params.max_tokens:
            self.finished, self.finish_reason = True, "length"
            return True
        return False


@dataclass
class StepOutput:
    """What one engine step produced for one request."""

    request_id: int
    token_ids: list[int]
    finished: bool
    finish_reason: str | None = None
