"""Chapter 29: what a token costs, from quantities you can actually look up.

Nothing here is measured, and that is the point. Cost is arithmetic over four inputs — what the
hardware costs per hour, how many tokens per second it sustains, how busy it is, and what mix of
prompt and output tokens the traffic has — and the reason cost estimates are usually wrong is not
that the arithmetic is hard but that one of those four is quietly assumed.

The one that is almost always assumed is utilisation. A throughput number from a benchmark is a
*peak*; a service with a daily traffic curve runs far below it most of the time, and the cost per
token is the cost of the hardware divided by the tokens it actually produced, not the tokens it
could have produced.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Deployment:
    """Everything a cost-per-token figure depends on. State all of it or the figure is unfalsifiable.

    ``utilisation`` is the fraction of wall-clock time the hardware spends at ``tokens_per_second``.
    Setting it to 1.0 produces the number a vendor benchmark reports, and no real service achieves
    it: a serving fleet sized for the daily peak is idle at night, and a fleet sized for the mean
    misses its objective at the peak.
    """

    name: str
    dollars_per_hour: float
    #: sustained output tokens per second per instance, at the batch size the SLO permits
    tokens_per_second: float
    utilisation: float = 0.4
    instances: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.utilisation <= 1.0:
            raise ValueError("utilisation must be in (0, 1]")
        if self.tokens_per_second <= 0:
            raise ValueError("tokens_per_second must be positive")

    @property
    def tokens_per_hour(self) -> float:
        return self.tokens_per_second * 3600 * self.utilisation * self.instances

    @property
    def cost_per_hour(self) -> float:
        return self.dollars_per_hour * self.instances

    @property
    def cost_per_million_tokens(self) -> float:
        return self.cost_per_hour / self.tokens_per_hour * 1e6


def instances_for(demand_tokens_per_second: float, deployment: Deployment) -> int:
    """How many instances a demand needs, with utilisation honestly applied.

    Dividing demand by peak throughput is the standard mistake: it sizes the fleet for a world
    where every instance is always saturated, which is the world where every request also waits.
    """
    per_instance = deployment.tokens_per_second * deployment.utilisation
    return max(1, -(-demand_tokens_per_second // per_instance).__int__())


def blended_cost_per_million(
    deployment: Deployment,
    prompt_tokens: float,
    output_tokens: float,
    prefill_speedup: float = 10.0,
) -> float:
    """Cost per million *billed* tokens, when prompt tokens are cheaper to serve than output ones.

    Prefill is compute-bound and parallel across a prompt; decode is memory-bandwidth-bound and
    strictly sequential (chapter 3). A prompt token therefore costs a fraction of what an output
    token costs, and any pricing model that charges the same for both is either overcharging for
    prompts or undercharging for outputs. ``prefill_speedup`` is how many prompt tokens a
    deployment processes in the time it takes to produce one output token.
    """
    if prefill_speedup <= 0:
        raise ValueError("prefill_speedup must be positive")
    total = prompt_tokens + output_tokens
    if total <= 0:
        raise ValueError("a token mix needs some tokens in it")
    # Express the prompt in "output-token equivalents" of machine time.
    equivalent = prompt_tokens / prefill_speedup + output_tokens
    return deployment.cost_per_million_tokens * equivalent / total


def break_even_tokens_per_month(deployment: Deployment, api_dollars_per_million: float) -> float:
    """Monthly volume at which self-hosting costs the same as paying a hosted API per token.

    Note what does *not* appear: utilisation. The hardware bill is the same whether the machine is
    busy or idle, so the break-even volume depends only on that bill and the API's price. What
    utilisation changes is the cost of the tokens you *do* serve, which is the other half of the
    comparison and the reason a fleet that looks cheap on paper is not.

    Returns infinity when the fleet physically cannot produce the break-even volume in a month —
    at which point the answer is not "self-host", it is "buy more hardware and redo the sum".
    """
    if api_dollars_per_million <= 0:
        raise ValueError("an API price of zero has no break-even point")
    monthly_cost = deployment.cost_per_hour * 24 * 30
    volume = monthly_cost / api_dollars_per_million * 1e6
    ceiling = deployment.tokens_per_second * 3600 * 24 * 30 * deployment.instances
    return float("inf") if volume > ceiling else volume
