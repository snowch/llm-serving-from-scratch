"""Chapter 24's engine: a request the caller no longer wants.

Every engine so far assumed a request runs to completion. Agent workloads break that assumption
constantly — a step is abandoned because a parallel branch answered first, a timeout fired, a user
hit stop, or the orchestrator gave up on the whole run. Without a way to say so, the engine keeps
decoding for a client that left, and that work is not merely wasted: it occupies a batch slot and
KV blocks that a live request is queued for.

Cancellation is therefore a *throughput* feature, not a politeness feature, and it is one of the
few places where doing less work is purely good.
"""

from __future__ import annotations

from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.request import RequestState, StepOutput


class CancellableEngine(PrefixCachedEngine):
    """Prefix-cached serving that can drop a request the caller has abandoned.

    Aborting is two different operations wearing one name. A queued request has consumed nothing
    and simply disappears. A running one holds KV blocks and a batch slot, and the engine has to
    release both — which is the same bookkeeping preemption already does, so there is very little
    new machinery here. That is the point: an engine that got chapter 8 right gets cancellation
    almost for free, and one that did not cannot bolt it on.
    """

    name = "cancellable"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        #: requests dropped before they finished
        self.aborted = 0
        #: tokens generated for requests that were later abandoned — the work cancellation saves
        self.wasted_tokens = 0
        #: terminal outputs owed to callers whose requests were dropped
        self._abort_notices: list[StepOutput] = []

    def abort(self, request_id: int) -> bool:
        """Drop a request. Returns whether there was anything to drop.

        Idempotent by design: a client disconnect and a timeout can both fire for the same request,
        and an abort that raised on the second call would turn a routine race into an error page.
        """
        for index, state in enumerate(self.waiting):
            if state.request_id == request_id:
                self.waiting.pop(index)
                self._finish_aborted(state)
                return True
        for index, state in enumerate(self.running):
            if state.request_id == request_id:
                self.running.pop(index)
                self._release(state)
                self._finish_aborted(state)
                return True
        return False

    def _finish_aborted(self, state: RequestState) -> None:
        state.finished = True
        state.finish_reason = "aborted"
        self.aborted += 1
        self.wasted_tokens += len(state.output_token_ids)
        self._abort_notices.append(
            StepOutput(
                request_id=state.request_id, token_ids=[], finished=True, finish_reason="aborted"
            )
        )

    def has_work(self) -> bool:
        return bool(self._abort_notices) or super().has_work()

    def step(self) -> list[StepOutput]:
        """Advance one step, telling whoever was streaming an aborted request that it ended.

        A cancelled request still needs a terminal output. Silence is not a termination signal: a
        caller waiting on the stream would wait forever, and the connection — along with whatever
        the gateway is holding for it — leaks. This is the single most common way cancellation is
        implemented incorrectly, because nothing in the engine's own metrics notices.
        """
        notices, self._abort_notices = self._abort_notices, []
        return notices + super().step()
