"""Chapter 26's engine: refusing work, on purpose.

An overloaded engine that accepts everything serves *nothing* well. Queues grow without bound, every
request misses its objective, and the only signal the caller gets is that the whole service became
slow at once. The alternative is to refuse some requests quickly, so the rest are served properly —
which feels worse and measures better, and is the single most reliable way to keep a serving system
useful under a load spike.

The hard part is not rejecting. It is deciding *when*, and the answer is not "when the CPU is busy":
that number does not predict whether the next request will meet its objective. The two signals that
do are the ones chapter 8 and chapter 25 already export.
"""

from __future__ import annotations

from llmserve.config import ModelConfig
from llmserve.engines.cancel import CancellableEngine
from llmserve.model import TinyGPT
from llmserve.request import Request, StepOutput


class SheddingEngine(CancellableEngine):
    """Serving with admission control, and a drain that does not cut live streams.

    A rejection reuses the cancellation path from chapter 22 for one reason that matters: the caller
    gets a terminal message immediately. A server that refuses work by dropping connections looks,
    from the client's side, exactly like a server that has hung — and the client's retry then makes
    the overload worse, which is how a load spike becomes an outage.
    """

    name = "shedding"

    def __init__(
        self,
        model: TinyGPT,
        config: ModelConfig,
        *,
        max_batch_size: int = 8,
        n_blocks: int = 256,
        block_size: int = 16,
        max_queue_depth: int = 8,
        max_kv_utilisation: float = 0.9,
    ) -> None:
        super().__init__(
            model, config, max_batch_size=max_batch_size, n_blocks=n_blocks, block_size=block_size
        )
        self.max_queue_depth = max_queue_depth
        self.max_kv_utilisation = max_kv_utilisation
        self.rejected = 0
        self.draining = False

    def should_admit(self) -> bool:
        """Whether the engine can take another request and still expect to serve it properly.

        Queue depth is the direct statement of "there is already more work here than can be served
        soon". KV utilisation is the one that catches the case a queue length cannot see: a few very
        long sequences can exhaust the block budget while the queue looks short, and admitting into
        that produces the preemption thrash of chapter 8 rather than an honest refusal.
        """
        if self.draining:
            return False
        if len(self.waiting) >= self.max_queue_depth:
            return False
        return self.cache.allocator.utilisation < self.max_kv_utilisation

    def add_request(self, request: Request) -> None:
        if not self.should_admit():
            self.rejected += 1
            self._abort_notices.append(
                StepOutput(
                    request_id=request.request_id,
                    token_ids=[],
                    finished=True,
                    finish_reason="rejected",
                )
            )
            return
        super().add_request(request)

    def drain(self) -> None:
        """Stop accepting new work; finish what is in flight.

        This is what a rolling upgrade needs and what a naive one skips. Terminating the process
        with sequences still decoding drops every one of those streams, and the client sees a
        truncated response rather than an error — which is worse, because nothing retries it.
        """
        self.draining = True

    @property
    def drained(self) -> bool:
        return self.draining and not self.waiting and not self.running
