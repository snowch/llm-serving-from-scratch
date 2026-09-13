"""Chapter 21's engine: one base model, many tenants, and a queue that is fair about it.

Everything before this chapter treated the waiting queue as a single line and served it in order.
That is exactly right when every request belongs to the same caller and exactly wrong when they do
not: one tenant submitting a burst pushes every other tenant behind it, and the engine has no idea
it is doing so. Head-of-line blocking between customers is not a latency problem, it is an
isolation problem, and no amount of extra throughput fixes it.

The fix is to stop treating the queue as one line. Requests are grouped by tenant, and admission
picks the tenant that has had the least service so far, weighted by whatever share each is
entitled to. That is deficit round robin, which is thirty years old and comes from packet
scheduling, where the same problem has the same shape.
"""

from __future__ import annotations

from llmserve.config import ModelConfig
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import TinyGPT
from llmserve.request import RequestState, StepOutput

ANONYMOUS = "anonymous"


class TenantFairEngine(PrefixCachedEngine):
    """Prefix-cached serving with weighted fair admission across tenants.

    ``weights`` is each tenant's entitled share; a tenant on weight 2.0 is admitted until it has
    had twice the service of a tenant on 1.0. An absent tenant gets 1.0, so the common case needs
    no configuration and a quota is a one-line change rather than a new subsystem.
    """

    name = "tenant-fair"

    def __init__(
        self,
        model: TinyGPT,
        config: ModelConfig,
        *,
        max_batch_size: int = 8,
        n_blocks: int = 256,
        block_size: int = 16,
        weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            model, config, max_batch_size=max_batch_size, n_blocks=n_blocks, block_size=block_size
        )
        self.weights = dict(weights or {})
        #: tokens of work charged to each tenant so far
        self.served: dict[str, float] = {}
        #: requests admitted per tenant, for the fairness measurement
        self.admitted: dict[str, int] = {}

    @staticmethod
    def tenant_of(state: RequestState) -> str:
        return state.request.tenant or ANONYMOUS

    def _deficit(self, tenant: str) -> float:
        """Service received, scaled by entitlement. Lowest is admitted next."""
        return self.served.get(tenant, 0.0) / self.weights.get(tenant, 1.0)

    def _next_waiting(self) -> int:
        """Admit from whichever tenant is furthest behind its share.

        Within a tenant the order is still arrival order — fairness is between customers, not
        within one, and a tenant that reorders its own requests would be a surprising server.
        """
        if not self.waiting:
            return 0
        heads: dict[str, int] = {}
        for index, state in enumerate(self.waiting):
            heads.setdefault(self.tenant_of(state), index)
        return heads[min(heads, key=self._deficit)]

    def _prefill(self, states: list[RequestState]) -> list[StepOutput]:
        """Charge each tenant for the work it is about to cause.

        Charged on admission, not on completion, and charged for the *whole* request — prompt plus
        token budget — rather than for what has happened so far. Both choices are deliberate. A
        scheduler that charges only for work already done cannot see a burst until it has already
        served it, which is precisely when the unfairness has happened. This overcharges requests
        that stop early, and that error is in the safe direction.
        """
        for state in states:
            tenant = self.tenant_of(state)
            cost = state.request.prompt_len + state.request.params.max_tokens
            self.served[tenant] = self.served.get(tenant, 0.0) + cost
            self.admitted[tenant] = self.admitted.get(tenant, 0) + 1
        return super()._prefill(states)

    @property
    def service_share(self) -> dict[str, float]:
        """Fraction of charged work each tenant received. The fairness claim, as a number."""
        total = sum(self.served.values())
        return {t: v / total for t, v in sorted(self.served.items())} if total else {}
