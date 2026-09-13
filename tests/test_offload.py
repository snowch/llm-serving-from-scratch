"""Chapter 12: a second KV tier, and the engine that demotes into it."""

import pytest
import torch

from llmserve.arithmetic import kv_bytes_per_token
from llmserve.cache.offload import (
    OffloadTier,
    break_even_bandwidth,
    fetch_vs_recompute,
)
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.offload import OffloadEngine
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams


def _layers(value: float, n_layers: int = 2):
    return [(torch.full((2, 4, 8), value), torch.full((2, 4, 8), value)) for _ in range(n_layers)]


# -- the tier ----------------------------------------------------------------------------


def test_a_stored_block_comes_back():
    tier = OffloadTier(n_layers=2)
    tier.store(7, _layers(1.5))
    fetched = tier.fetch(7)
    assert fetched is not None
    assert torch.equal(fetched[0][0], torch.full((2, 4, 8), 1.5))
    assert tier.fetches == 1


def test_a_missing_block_is_a_miss_not_an_error():
    """A promotion that misses turns into the prefill it was avoiding — not into a crash."""
    tier = OffloadTier(n_layers=2)
    assert tier.fetch(99) is None
    assert tier.misses == 1
    assert tier.hit_rate == 0.0


def test_stored_tensors_are_copies():
    """A view into a block the allocator is about to reuse is silent cross-request corruption."""
    tier = OffloadTier(n_layers=1)
    layers = _layers(1.0, n_layers=1)
    tier.store(1, layers)
    layers[0][0].fill_(99.0)  # the allocator hands the block to somebody else
    assert torch.equal(tier.fetch(1)[0][0], torch.ones((2, 4, 8)))


def test_the_tier_evicts_its_own_oldest_when_full():
    tier = OffloadTier(n_layers=1, capacity_blocks=2)
    for key in (1, 2, 3):
        tier.store(key, _layers(float(key), n_layers=1))
    assert tier.fetch(1) is None
    assert tier.fetch(3) is not None
    assert tier.evicted == 1


def test_a_fetch_makes_a_block_recently_used():
    tier = OffloadTier(n_layers=1, capacity_blocks=2)
    tier.store(1, _layers(1.0, 1))
    tier.store(2, _layers(2.0, 1))
    tier.fetch(1)
    tier.store(3, _layers(3.0, 1))
    assert tier.fetch(1) is not None, "the block just used should not be the one evicted"


def test_a_tier_with_no_capacity_is_rejected():
    with pytest.raises(ValueError, match="not a tier"):
        OffloadTier(n_layers=1, capacity_blocks=0)


# -- the arithmetic that decides whether it pays -----------------------------------------


def test_fetching_beats_recomputing_on_any_realistic_link():
    per_token = kv_bytes_per_token(REFERENCE_MODEL)
    result = fetch_vs_recompute(
        per_token, 16, 128, link_bytes_per_second=3.2e10, prefill_tokens_per_second=10_000
    )
    assert result["speedup"] > 1


def test_break_even_bandwidth_does_not_depend_on_block_size():
    """Block size cancels: the answer is a property of the model and the hardware."""
    per_token = kv_bytes_per_token(REFERENCE_MODEL)
    bandwidth = break_even_bandwidth(per_token, 10_000)
    for block_size in (8, 16, 64):
        result = fetch_vs_recompute(
            per_token,
            block_size,
            4,
            link_bytes_per_second=bandwidth,
            prefill_tokens_per_second=10_000,
        )
        assert result["speedup"] == pytest.approx(1.0)


# -- the engine ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return build_model(REFERENCE_MODEL)


def _serve(engine, prompts, max_tokens=2):
    for prompt in prompts:
        engine.add_request(
            Request(prompt_token_ids=list(prompt), params=SamplingParams(max_tokens=max_tokens))
        )
    while engine.has_work():
        engine.step()
    return engine.prefix


def test_demoting_preserves_reuse_that_dropping_loses(model):
    """Chapter 12's claim, on a block pool deliberately too small to hold the working set."""
    prompts = [[((i * 131 + j) % 200) + 33 for j in range(96)] for i in range(4)]
    sequence = [*prompts, *prompts]  # each prefix is wanted a second time

    dropped = _serve(
        PrefixCachedEngine(model, REFERENCE_MODEL, max_batch_size=1, n_blocks=20, block_size=16),
        sequence,
    )
    demoted_engine = OffloadEngine(
        model, REFERENCE_MODEL, max_batch_size=1, n_blocks=20, block_size=16
    )
    demoted = _serve(demoted_engine, sequence)

    assert demoted.hit_tokens > dropped.hit_tokens
    assert demoted_engine.tier.fetches > 0


def test_the_engine_still_serves_every_request(model):
    engine = OffloadEngine(model, REFERENCE_MODEL, max_batch_size=2, n_blocks=24, block_size=16)
    requests = [
        Request(
            prompt_token_ids=[((i * 97 + j) % 200) + 33 for j in range(64)],
            params=SamplingParams(max_tokens=3),
        )
        for i in range(6)
    ]
    for request in requests:
        engine.add_request(request)
    finished = set()
    while engine.has_work():
        for output in engine.step():
            if output.finished:
                finished.add(output.request_id)
    assert finished == {r.request_id for r in requests}
