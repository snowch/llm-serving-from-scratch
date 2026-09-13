"""Chapter 19: fairness between tenants sharing one engine."""

import pytest
import torch

from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.engines.tenant import ANONYMOUS, TenantFairEngine
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return build_model(REFERENCE_MODEL)


def _engine(model, **kwargs):
    return TenantFairEngine(
        model, REFERENCE_MODEL, max_batch_size=2, n_blocks=256, block_size=16, **kwargs
    )


def _request(tenant, prompt_len=32, max_tokens=4):
    return Request(
        prompt_token_ids=[(i % 250) + 1 for i in range(prompt_len)],
        params=SamplingParams(max_tokens=max_tokens),
        tenant=tenant,
    )


def _admission_order(engine, requests):
    """Which tenant each admitted request belonged to, in the order they were admitted."""
    for request in requests:
        engine.add_request(request)
    seen: list[str] = []
    running = set()
    while engine.has_work():
        engine.step()
        for state in engine.running:
            if state.request_id not in running:
                running.add(state.request_id)
                seen.append(state.request.tenant or ANONYMOUS)
    return seen


# -- the queue -----------------------------------------------------------------------


def test_fifo_serves_a_burst_before_anyone_else(model):
    """The problem, stated as a test: one caller's burst is served before a later arrival."""
    engine = PrefixCachedEngine(model, REFERENCE_MODEL, max_batch_size=2, n_blocks=256)
    requests = [_request("hog") for _ in range(6)] + [_request("alice")]
    for request in requests:
        engine.add_request(request)

    admitted: list[str] = []
    running: set[int] = set()
    while engine.has_work() and len(admitted) < 7:
        engine.step()
        for state in engine.running:
            if state.request_id not in running:
                running.add(state.request_id)
                admitted.append(state.request.tenant)
    assert admitted.index("alice") == 6, "alice should be stuck behind the whole burst"


def test_fair_queueing_interleaves_tenants(model):
    """The fix: a later arrival from a quiet tenant does not wait for the whole burst."""
    engine = _engine(model)
    requests = [_request("hog") for _ in range(6)] + [_request("alice")]
    order = _admission_order(engine, requests)
    assert order.index("alice") < 6


def test_a_tenant_alone_is_served_in_arrival_order(model):
    """Fairness is between callers, not within one: a tenant's own requests keep their order."""
    engine = _engine(model)
    requests = [_request("solo", max_tokens=2) for _ in range(4)]
    ids = [r.request_id for r in requests]
    for request in requests:
        engine.add_request(request)

    admitted: list[int] = []
    running: set[int] = set()
    while engine.has_work():
        engine.step()
        for state in engine.running:
            if state.request_id not in running:
                running.add(state.request_id)
                admitted.append(state.request_id)
    assert admitted == ids


def test_weights_buy_a_larger_share(model):
    """A quota is a weight, not a new subsystem."""
    engine = _engine(model, weights={"big": 3.0, "small": 1.0})
    requests = [_request("big", max_tokens=2) for _ in range(6)]
    requests += [_request("small", max_tokens=2) for _ in range(6)]
    order = _admission_order(engine, requests)
    first_six = order[:6]
    assert first_six.count("big") > first_six.count("small")


def test_an_untagged_request_is_its_own_tenant(model):
    """A single-tenant deployment never sets a tenant and must still work."""
    engine = _engine(model)
    order = _admission_order(engine, [_request(None, max_tokens=2) for _ in range(3)])
    assert order == [ANONYMOUS] * 3


# -- accounting ----------------------------------------------------------------------


def test_service_share_reports_where_the_work_went(model):
    engine = _engine(model)
    _admission_order(engine, [_request("a", max_tokens=2), _request("b", max_tokens=2)])
    share = engine.service_share
    assert set(share) == {"a", "b"}
    assert sum(share.values()) == pytest.approx(1.0)


def test_service_share_is_empty_before_anything_is_served(model):
    assert _engine(model).service_share == {}


def test_charging_happens_on_admission_not_completion(model):
    """A scheduler that waits for completion cannot see a burst until it has already served it."""
    engine = _engine(model)
    engine.add_request(_request("a", prompt_len=32, max_tokens=8))
    engine.step()
    assert engine.served["a"] == 40
