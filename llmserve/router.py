"""Routing across replicas.

Chapter 18. One engine eventually runs out of device. The usual answer is to run several and put a
load balancer in front, and the usual load balancer is round-robin — which is wrong here in a way
that is invisible until you measure it.

Round-robin assumes requests are interchangeable units of work. LLM requests are not. They differ
in cost by orders of magnitude, and — more importantly — a replica that has already seen a
request's prompt prefix can serve it far more cheaply than one that has not (chapter 9). Spreading
work evenly destroys exactly the locality the prefix cache depends on.

The router here exposes the same ``Engine`` interface as everything else, so the chapter 2 harness
measures a fleet exactly as it measures a single engine.
"""

from __future__ import annotations

from collections.abc import Callable

from llmserve.request import Request, StepOutput


class RoutingPolicy:
    """Chooses a replica for a request. The whole design space of this chapter."""

    name = "policy"

    def select(self, request: Request, replicas: list) -> int:
        raise NotImplementedError


class RoundRobin(RoutingPolicy):
    """Send each request to the next replica in turn.

    The default in every load balancer, and the baseline here precisely because it is what people
    reach for. It balances *request counts*, which is the wrong quantity twice over: requests
    differ enormously in cost, and it actively scatters requests that share a prefix.
    """

    name = "round-robin"

    def __init__(self) -> None:
        self._next = 0

    def select(self, request: Request, replicas: list) -> int:
        chosen = self._next % len(replicas)
        self._next += 1
        return chosen


class LeastOutstandingTokens(RoutingPolicy):
    """Send to whichever replica has the least *work* queued, measured in tokens.

    Least-connections is the usual refinement of round-robin and is still wrong: one connection
    can be a thousand times the work of another. Counting outstanding tokens — prompt plus
    remaining budget — is a far better proxy for how busy a replica actually is.
    """

    name = "least-tokens"

    def select(self, request: Request, replicas: list) -> int:
        return min(range(len(replicas)), key=lambda i: _outstanding_tokens(replicas[i]))


class PrefixAffinity(RoutingPolicy):
    """Send requests sharing a prompt prefix to the same replica.

    Hashing a fixed-length prefix means every request beginning with the same system prompt lands
    on the replica that already has those blocks cached. This is deliberately *not* balancing: it
    trades even load for cache hits, on the bet that a hit is worth more than a slightly shorter
    queue — which chapter 9 measured, and which chapter 18 now tests at fleet scale.

    The bet fails when one prefix dominates traffic: every request goes to one replica and the
    rest idle. Real routers therefore cap how unbalanced they will let things get.
    """

    name = "prefix-affinity"

    def __init__(self, prefix_length: int = 64, max_imbalance: float = 2.0) -> None:
        self.prefix_length = prefix_length
        self.max_imbalance = max_imbalance

    def select(self, request: Request, replicas: list) -> int:
        prefix = tuple(request.prompt_token_ids[: self.prefix_length])
        preferred = hash(prefix) % len(replicas)

        # Guard against a single hot prefix pinning all traffic to one replica.
        loads = [_outstanding_tokens(r) for r in replicas]
        mean = sum(loads) / len(loads) if loads else 0
        if mean > 0 and loads[preferred] > self.max_imbalance * mean:
            return min(range(len(replicas)), key=lambda i: loads[i])
        return preferred


def _outstanding_tokens(engine) -> int:
    """Rough work queued on a replica: prompt plus remaining budget, over everything in flight."""
    total = 0
    for state in [*getattr(engine, "waiting", []), *getattr(engine, "running", [])]:
        remaining = state.request.params.max_tokens - len(state.output_token_ids)
        total += state.request.prompt_len + max(remaining, 0)
    current = getattr(engine, "current", None)
    if current is not None:
        total += current.request.prompt_len
    return total


class Router:
    """A fleet of replicas behind one interface.

    ``step`` advances every replica once, so a fleet of N replicas does N times the work per step.
    That is the right model for independent devices and the wrong one for replicas sharing a CPU —
    which is exactly our situation, and why chapter 18 compares policies against each other rather
    than claiming a speedup over a single engine.
    """

    name = "router"

    def __init__(self, factory: Callable[[], object], n_replicas: int = 4, policy=None) -> None:
        self.replicas = [factory() for _ in range(n_replicas)]
        self.policy = policy or RoundRobin()
        self.name = f"router-{self.policy.name}"
        #: requests sent to each replica, for measuring how evenly work was spread
        self.assignments = [0] * n_replicas

    def add_request(self, request: Request) -> None:
        index = self.policy.select(request, self.replicas)
        self.assignments[index] += 1
        self.replicas[index].add_request(request)

    def has_work(self) -> bool:
        return any(replica.has_work() for replica in self.replicas)

    def step(self) -> list[StepOutput]:
        outputs: list[StepOutput] = []
        for replica in self.replicas:
            if replica.has_work():
                outputs += replica.step()
        return outputs

    @property
    def imbalance(self) -> float:
        """Ratio of the busiest replica's share to an even share. 1.0 is perfectly balanced."""
        if not self.assignments or sum(self.assignments) == 0:
            return 1.0
        mean = sum(self.assignments) / len(self.assignments)
        return max(self.assignments) / mean if mean else 1.0

    @property
    def cache_hit_rate(self) -> float:
        """Prefix-cache hit rate across the fleet, when the replicas have one."""
        hits = sum(getattr(r, "prefix", None).hits for r in self.replicas if hasattr(r, "prefix"))
        misses = sum(
            getattr(r, "prefix", None).misses for r in self.replicas if hasattr(r, "prefix")
        )
        total = hits + misses
        return hits / total if total else 0.0

    @property
    def token_reuse_rate(self) -> float:
        replicas = [r for r in self.replicas if hasattr(r, "prefix")]
        reused = sum(r.prefix.hit_tokens for r in replicas)
        total = sum(r.prefix.total_prompt_tokens for r in replicas)
        return reused / total if total else 0.0

    @property
    def inner_engines(self) -> list:
        """The engines this one runs. The benchmark harness fingerprints their source too, so a
        change to the replica engine invalidates the fleet results that were measured with it."""
        return self.replicas
