"""The engine interface.

Deliberately tiny. An engine accepts requests, and advancing it by one ``step`` produces zero or
more tokens for zero or more requests. Everything the book adds later — batching, paging,
speculation, chunked prefill — changes what happens inside ``step`` without changing this shape.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from llmserve.request import Request, StepOutput


@runtime_checkable
class Engine(Protocol):
    name: str

    def add_request(self, request: Request) -> None:
        """Queue a request. Admission to the running set is the scheduler's decision, not this."""
        ...

    def step(self) -> list[StepOutput]:
        """Advance the engine by one iteration and return whatever tokens that produced."""
        ...

    def has_work(self) -> bool:
        """True while any request is queued or running."""
        ...
