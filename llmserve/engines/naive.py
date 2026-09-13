"""Chapter 1's engine: one request at a time, and no memory between tokens.

This is the strawman the rest of the book dismantles, and it is deliberately not a strawman in
the unfair sense — it is what you get by wrapping a model's forward pass in a loop, which is
what most first implementations are. Two properties define it:

* **One request at a time.** A request that arrives while another is running waits for it to
  finish completely, however long that takes.
* **No KV cache.** Every decode step re-runs the model over the entire sequence so far, so the
  cost of producing token *n* grows with *n*. Chapter 5 fixes this and measures the difference.
"""

from __future__ import annotations

import torch

from llmserve.model import TinyGPT
from llmserve.request import Request, RequestState, StepOutput
from llmserve.sampling import sample


class NaiveEngine:
    """Serves requests strictly in arrival order, recomputing the prefix on every token."""

    name = "naive"

    def __init__(self, model: TinyGPT, *, use_cache: bool = False) -> None:
        self.model = model
        self.use_cache = use_cache
        self.waiting: list[RequestState] = []
        self.current: RequestState | None = None
        self._generator = torch.Generator(device="cpu")

    def add_request(self, request: Request) -> None:
        self.waiting.append(RequestState(request=request))

    def has_work(self) -> bool:
        return self.current is not None or bool(self.waiting)

    def step(self) -> list[StepOutput]:
        """Produce exactly one token for the single request currently being served.

        Admitting a new request costs a prefill; continuing one costs a decode. Reporting them
        through the same interface is what lets the harness see time-to-first-token separately
        from inter-token latency, even for an engine this simple.
        """
        if self.current is None:
            if not self.waiting:
                return []
            self.current = self.waiting.pop(0)

        state = self.current
        params = state.request.params
        if params.seed is not None:
            self._generator.manual_seed(params.seed + len(state.output_token_ids))

        if state.past is None:
            # Prefill: run the whole prompt, keep the cache only if this engine uses one.
            ids = torch.tensor([state.request.prompt_token_ids], dtype=torch.long)
            logits, past = self.model(ids)
            state.past = past if self.use_cache else []
        elif self.use_cache:
            # Decode with a cache: one new token, attending to everything already stored.
            last = torch.tensor([[state.output_token_ids[-1]]], dtype=torch.long)
            positions = torch.tensor([state.total_len - 1], dtype=torch.long)
            logits, past = self.model(last, state.past, positions)
            state.past = past
        else:
            # Decode without a cache: re-run the entire sequence. This is the quadratic cost
            # chapter 5 removes, and it is the single largest avoidable waste in this engine.
            ids = torch.tensor([state.all_token_ids], dtype=torch.long)
            logits, _ = self.model(ids)

        prev = torch.tensor([state.all_token_ids], dtype=torch.long)
        token = sample(
            logits[:, -1, :],
            params,
            prev_ids=prev,
            generator=None if params.seed is None else self._generator,
        )
        token_id = int(token.item())
        state.output_token_ids.append(token_id)
        state.check_finished()

        out = StepOutput(
            request_id=state.request_id,
            token_ids=[token_id],
            finished=state.finished,
            finish_reason=state.finish_reason,
        )
        if state.finished:
            self.current = None
        return [out]


class CachedEngine(NaiveEngine):
    """Chapter 5's engine: still one request at a time, but with a KV cache.

    Isolating the cache as the *only* change from ``NaiveEngine`` is what makes its contribution
    measurable. Chapter 6 then adds batching on top, and chapter 7 replaces the scheduling.
    """

    name = "cached"

    def __init__(self, model: TinyGPT) -> None:
        super().__init__(model, use_cache=True)
