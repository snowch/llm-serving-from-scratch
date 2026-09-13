"""Chapters 26 and 27: refusing work, draining, and what a token costs."""

import pytest
import torch

from llmserve.config import REFERENCE_MODEL
from llmserve.cost import Deployment, blended_cost_per_million, break_even_tokens_per_month
from llmserve.engines.shedding import SheddingEngine
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return build_model(REFERENCE_MODEL)


def _request(prompt_len=32, max_tokens=8):
    return Request(
        prompt_token_ids=[(i % 250) + 1 for i in range(prompt_len)],
        params=SamplingParams(max_tokens=max_tokens),
    )


# -- admission control ------------------------------------------------------------------


def test_a_deep_queue_stops_admitting(model):
    engine = SheddingEngine(
        model, REFERENCE_MODEL, max_batch_size=1, n_blocks=512, max_queue_depth=3
    )
    for _ in range(10):
        engine.add_request(_request())
    assert engine.rejected == 7
    assert len(engine.waiting) == 3


def test_a_rejection_reaches_the_caller(model):
    """Refusing by dropping the connection looks, to a client, exactly like hanging."""
    engine = SheddingEngine(model, REFERENCE_MODEL, n_blocks=512, max_queue_depth=0)
    request = _request()
    engine.add_request(request)
    assert engine.has_work()
    outputs = engine.step()
    assert [(o.request_id, o.finish_reason) for o in outputs] == [(request.request_id, "rejected")]


def test_kv_pressure_sheds_even_when_the_queue_is_short(model):
    """A few long sequences exhaust the block budget while the queue still looks healthy."""
    engine = SheddingEngine(
        model,
        REFERENCE_MODEL,
        max_batch_size=8,
        n_blocks=8,
        block_size=16,
        max_queue_depth=100,
        max_kv_utilisation=0.5,
    )
    for _ in range(4):
        engine.add_request(_request(prompt_len=64, max_tokens=16))
        engine.step()
    assert engine.cache.allocator.utilisation >= 0.5
    before = engine.rejected
    engine.add_request(_request())
    assert engine.rejected == before + 1


def test_draining_refuses_new_work_and_finishes_the_old(model):
    engine = SheddingEngine(model, REFERENCE_MODEL, max_batch_size=2, n_blocks=512)
    for _ in range(3):
        engine.add_request(_request(max_tokens=4))
    engine.step()
    assert engine.running

    engine.drain()
    before = engine.rejected
    engine.add_request(_request())
    assert engine.rejected == before + 1
    assert not engine.drained, "a drain that reports done with work in flight cuts live streams"

    while not engine.drained:
        engine.step()
    assert not engine.running and not engine.waiting


# -- cost -------------------------------------------------------------------------------


def test_utilisation_dominates_cost_per_token():
    """The term people leave out. Same hardware, same engine, four times the cost."""
    busy = Deployment("x", dollars_per_hour=3.0, tokens_per_second=2500, utilisation=0.8)
    idle = Deployment("x", dollars_per_hour=3.0, tokens_per_second=2500, utilisation=0.2)
    assert idle.cost_per_million_tokens == pytest.approx(4 * busy.cost_per_million_tokens)


def test_utilisation_must_be_a_fraction():
    with pytest.raises(ValueError, match="utilisation"):
        Deployment("x", dollars_per_hour=1.0, tokens_per_second=10, utilisation=0.0)


def test_a_prompt_heavy_mix_costs_less_per_billed_token():
    """Prefill is parallel and decode is not, so the mix changes the cost of a 'token'."""
    deployment = Deployment("x", dollars_per_hour=3.0, tokens_per_second=2500)
    prompt_heavy = blended_cost_per_million(deployment, 2000, 100)
    output_heavy = blended_cost_per_million(deployment, 100, 2000)
    assert prompt_heavy < output_heavy


def test_break_even_ignores_utilisation_but_respects_capacity():
    """The hardware bill is fixed; what utilisation changes is the cost of what you did serve."""
    busy = Deployment("x", dollars_per_hour=3.0, tokens_per_second=2500, utilisation=0.9)
    idle = Deployment("x", dollars_per_hour=3.0, tokens_per_second=2500, utilisation=0.1)
    assert break_even_tokens_per_month(busy, 1.0) == break_even_tokens_per_month(idle, 1.0)
    # A price so low that no achievable volume pays for the hardware.
    assert break_even_tokens_per_month(busy, 0.001) == float("inf")


def test_a_free_api_has_no_break_even_point():
    deployment = Deployment("x", dollars_per_hour=3.0, tokens_per_second=2500)
    with pytest.raises(ValueError):
        break_even_tokens_per_month(deployment, 0.0)
