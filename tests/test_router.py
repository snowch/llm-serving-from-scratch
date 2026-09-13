"""Chapter 20: routing across replicas."""

import pytest
import torch

from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.router import (
    LeastOutstandingTokens,
    PrefixAffinity,
    RoundRobin,
    Router,
)
from llmserve.sampling import SamplingParams

TENANT_A = list(b"You are a helpful assistant. Answer concisely. " * 4)
TENANT_B = list(b"You are a code reviewer. Point out bugs clearly. " * 4)


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return build_model(REFERENCE_MODEL)


@pytest.fixture
def fleet(model):
    def factory():
        return PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=128, block_size=16)

    return factory


def _request(tokens, max_tokens=4):
    return Request(prompt_token_ids=list(tokens), params=SamplingParams(max_tokens=max_tokens))


def _drain(router):
    outputs = []
    while router.has_work():
        outputs += router.step()
    return outputs


# -- policies --------------------------------------------------------------------------


def test_round_robin_spreads_identical_prompts_across_replicas(fleet):
    """The property that makes round-robin wrong here: same prefix, different replica."""
    router = Router(fleet, n_replicas=4, policy=RoundRobin())
    for _ in range(8):
        router.add_request(_request(TENANT_A))
    assert router.assignments == [2, 2, 2, 2]


def test_prefix_affinity_keeps_one_prefix_on_one_replica(fleet):
    """The policy's core mechanism, with its imbalance guard out of the way.

    Admitting eight requests without serving any of them is the worst case for the guard: one
    replica holds everything, which is precisely the condition it exists to break up. Draining
    between arrivals — what actually happens at a moderate arrival rate — leaves every replica at
    zero load when the decision is made, so the guard never fires and the hashing shows through.
    """
    router = Router(fleet, n_replicas=4, policy=PrefixAffinity())
    for _ in range(8):
        router.add_request(_request(TENANT_A))
        _drain(router)
    assert sorted(router.assignments) == [0, 0, 0, 8]


def test_prefix_affinity_separates_distinct_prefixes(fleet):
    """Different system prompts should be able to land on different replicas.

    With 2 tenants and 4 replicas a collision is possible in principle; asserting that these two
    specific prompts do not collide would be asserting a property of ``hash``. What the policy
    actually guarantees is that each *tenant* is concentrated, which is what is checked here.
    """
    router = Router(fleet, n_replicas=4, policy=PrefixAffinity())
    for _ in range(4):
        router.add_request(_request(TENANT_A))
        router.add_request(_request(TENANT_B))
    assert sum(1 for n in router.assignments if n) <= 2
    assert all(n in (0, 4, 8) for n in router.assignments)


def test_prefix_affinity_gives_up_on_a_dominant_prefix(fleet):
    """The guard that stops one hot tenant pinning the whole fleet to one replica."""
    router = Router(fleet, n_replicas=4, policy=PrefixAffinity(max_imbalance=1.5))
    for _ in range(12):
        router.add_request(_request(TENANT_A, max_tokens=32))
    assert sum(1 for n in router.assignments if n) > 1, (
        "every request landed on one replica; the imbalance guard did not fire"
    )


def test_least_outstanding_tokens_prefers_the_emptier_replica(fleet):
    router = Router(fleet, n_replicas=2, policy=LeastOutstandingTokens())
    router.add_request(_request(TENANT_A, max_tokens=64))
    router.add_request(_request(list(b"hi"), max_tokens=1))
    assert router.assignments == [1, 1]


def test_least_outstanding_tokens_counts_work_not_requests(fleet):
    """One long request should outweigh several short ones — the whole point of the policy."""
    router = Router(fleet, n_replicas=2, policy=LeastOutstandingTokens())
    router.add_request(_request(TENANT_A * 4, max_tokens=256))
    for _ in range(3):
        router.add_request(_request(list(b"hi"), max_tokens=1))
    assert router.assignments[1] == 3, "short requests should have avoided the loaded replica"


# -- the fleet as an engine -------------------------------------------------------------


def test_router_serves_every_request(fleet):
    router = Router(fleet, n_replicas=3, policy=RoundRobin())
    requests = [_request(TENANT_A + [i]) for i in range(33, 42)]
    for request in requests:
        router.add_request(request)

    outputs = _drain(router)
    finished = {o.request_id for o in outputs if o.finished}
    assert finished == {r.request_id for r in requests}

    produced = {r.request_id: 0 for r in requests}
    for out in outputs:
        produced[out.request_id] += len(out.token_ids)
    assert set(produced.values()) == {4}


def test_affinity_reuses_more_of_the_prompt_than_round_robin(fleet):
    """The chapter's claim, as a test: concentrating a prefix is what makes the cache work.

    Round-robin makes every replica compute the shared prefix once; affinity makes exactly one
    replica compute it. That is the whole difference, and it is visible in the reuse rate.
    """
    reuse = {}
    for name, policy in (("rr", RoundRobin()), ("affinity", PrefixAffinity())):
        router = Router(fleet, n_replicas=4, policy=policy)
        for i in range(8):
            router.add_request(_request(TENANT_A + [33 + i]))
            _drain(router)
        reuse[name] = router.token_reuse_rate
    assert reuse["affinity"] > reuse["rr"]


# -- reporting --------------------------------------------------------------------------


def test_imbalance_is_one_when_perfectly_even(fleet):
    router = Router(fleet, n_replicas=4, policy=RoundRobin())
    for _ in range(8):
        router.add_request(_request(TENANT_A))
    assert router.imbalance == pytest.approx(1.0)


def test_imbalance_is_defined_before_any_request_arrives(fleet):
    """Reporting a ratio over an empty fleet must not divide by zero."""
    router = Router(fleet, n_replicas=4, policy=RoundRobin())
    assert router.imbalance == 1.0
    assert router.token_reuse_rate == 0.0
    assert router.cache_hit_rate == 0.0


def test_router_declares_its_replica_engines_for_fingerprinting(fleet):
    """A change to the replica engine has to invalidate results measured with the fleet."""
    from bench.harness import engine_sources

    router = Router(fleet, n_replicas=2, policy=RoundRobin())
    sources = engine_sources(router)
    assert "llmserve/router.py" in sources
    assert "llmserve/engines/prefix.py" in sources
