"""Chapter 11's engine: prefill and decode as separate pools with a KV handoff.

Every engine so far has made one pool do both jobs, and chapter 10 showed how much of scheduling
is spent refereeing between them. Disaggregation asks the obvious next question: what if they
simply stopped sharing?

A request is prefilled in one pool, its keys and values are shipped to another, and decoding
happens there. The two can then be sized, scheduled and even provisioned independently — prefill
wants compute, decode wants memory bandwidth, and nothing forces one machine to be good at both.

**What this implementation can and cannot show.** In one process the two pools take turns, so the
concurrency that makes disaggregation worthwhile is not available here. What is measurable is the
cost: every byte of KV cache crosses between pools, and that transfer is real work. Treat the
numbers as an honest lower bound on overhead, not as a verdict on the architecture.
"""

from __future__ import annotations

import time

from llmserve.engines.batched import _decode_batch, _prefill_batch, _sample_batch
from llmserve.model import TinyGPT
from llmserve.request import Request, RequestState, StepOutput


class DisaggregatedEngine:
    """Two pools, one handoff.

    ``prefill_batch_size`` and ``max_batch_size`` size the pools independently, which is the whole
    point of the architecture even though a single process cannot exploit it.
    """

    name = "disaggregated"

    def __init__(
        self,
        model: TinyGPT,
        *,
        max_batch_size: int = 8,
        prefill_batch_size: int = 2,
    ) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self.prefill_batch_size = prefill_batch_size
        self.waiting: list[RequestState] = []
        self.decoding: list[RequestState] = []
        #: bytes of KV cache moved between the pools
        self.transferred_bytes = 0
        #: seconds spent moving it — the cost the architecture must earn back
        self.transfer_seconds = 0.0
        self.transfers = 0

    def add_request(self, request: Request) -> None:
        self.waiting.append(RequestState(request=request))

    def has_work(self) -> bool:
        return bool(self.waiting or self.decoding)

    def _handoff(self, state: RequestState) -> None:
        """Move one sequence's KV cache from the prefill pool to the decode pool.

        Here that is a copy within one process. In a real deployment it is a network transfer, and
        its cost scales with the KV cache — which is why interconnect bandwidth, not compute,
        decides whether disaggregation is viable. A slow link turns every prefill into a stall at
        the other end.
        """
        assert state.past is not None
        start = time.perf_counter()
        moved = 0
        transferred = []
        for k, v in state.past:
            transferred.append((k.clone(), v.clone()))
            moved += k.numel() * k.element_size() + v.numel() * v.element_size()
        state.past = transferred
        self.transfer_seconds += time.perf_counter() - start
        self.transferred_bytes += moved
        self.transfers += 1

    def step(self) -> list[StepOutput]:
        self.decoding = [s for s in self.decoding if not s.finished]
        outputs: list[StepOutput] = []

        # -- prefill pool --------------------------------------------------------------
        room = self.max_batch_size - len(self.decoding)
        batch = self.waiting[: min(self.prefill_batch_size, max(room, 0))]
        if batch:
            self.waiting = self.waiting[len(batch) :]
            logits, caches = _prefill_batch(self.model, batch)
            for state, cache in zip(batch, caches, strict=True):
                state.past = cache
                self._handoff(state)
            outputs += self._emit(batch, _sample_batch(logits, batch))
            self.decoding += [s for s in batch if not s.finished]

        # -- decode pool ---------------------------------------------------------------
        ready = [s for s in self.decoding if s not in batch and not s.finished]
        if ready:
            outputs += self._emit(ready, _sample_batch(_decode_batch(self.model, ready), ready))

        return outputs

    def _emit(self, states: list[RequestState], tokens: list[int]) -> list[StepOutput]:
        outputs = []
        for state, token in zip(states, tokens, strict=True):
            state.output_token_ids.append(token)
            state.check_finished()
            outputs.append(
                StepOutput(
                    request_id=state.request_id,
                    token_ids=[token],
                    finished=state.finished,
                    finish_reason=state.finish_reason,
                )
            )
        return outputs

    @property
    def transfer_overhead(self) -> dict[str, float]:
        return {
            "transfers": self.transfers,
            "megabytes": round(self.transferred_bytes / 1e6, 2),
            "seconds": round(self.transfer_seconds, 4),
            "mb_per_transfer": round(self.transferred_bytes / 1e6 / max(self.transfers, 1), 3),
        }
