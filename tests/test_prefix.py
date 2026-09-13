"""Chapter 9: content-addressed reuse of prompt prefixes."""

import pytest
import torch

from llmserve.cache.prefix import PrefixCache
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.naive import CachedEngine
from llmserve.engines.paged import PagedEngine
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams

SYSTEM = list(b"You are a helpful assistant. Answer concisely and accurately. " * 2)


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return build_model(REFERENCE_MODEL)


@pytest.fixture
def specs():
    return [
        (SYSTEM + list(b"What is Rust?"), 8),
        (SYSTEM + list(b"Explain paging."), 8),
        (SYSTEM + list(b"Hello there!"), 8),
        (SYSTEM + list(b"Define a queue."), 8),
    ]


# -- the cache itself ------------------------------------------------------------------


def test_only_full_blocks_are_shareable():
    """A partial block still depends on tokens that have not arrived."""
    cache = PrefixCache(block_size=4)
    assert cache.block_keys(list(range(10))) == cache.block_keys(list(range(8)))


def test_a_published_prefix_is_reused():
    cache = PrefixCache(block_size=4)
    shared = list(range(1, 13))
    cache.lookup(shared + [100])
    cache.publish(shared + [100], [10, 11, 12])
    assert cache.lookup(shared + [200, 201]) == [10, 11, 12]


def test_a_different_prefix_never_hits():
    """Keys cover every token up to a block, so sharing later content must not collide."""
    cache = PrefixCache(block_size=4)
    shared = list(range(1, 13))
    cache.publish(shared, [10, 11, 12])
    assert cache.lookup([99, 98, 97, 96] + shared[:4]) == []


def test_lookup_stops_at_the_first_miss():
    """A hit after a gap is worthless: a prefix is only usable if everything before it is too."""
    cache = PrefixCache(block_size=4)
    cache.publish(list(range(1, 5)), [10])  # publishes only the first block
    assert cache.lookup(list(range(1, 13))) == [10]


def test_eviction_returns_the_least_recently_used():
    cache = PrefixCache(block_size=4)
    cache.publish(list(range(1, 5)), [10])
    cache.publish(list(range(100, 104)), [20])
    cache.lookup(list(range(1, 5)))  # touch the first, making the second oldest
    assert cache.evict_oldest() == 20


# -- the engine ------------------------------------------------------------------------


def _drain(engine, specs) -> list[list[int]]:
    outputs: dict[int, list[int]] = {}
    for prompt, n in specs:
        request = Request(prompt_token_ids=list(prompt), params=SamplingParams(max_tokens=n))
        outputs[request.request_id] = []
        engine.add_request(request)
    guard = 0
    while engine.has_work() and guard < 10_000:
        for out in engine.step():
            outputs[out.request_id].extend(out.token_ids)
        guard += 1
    return list(outputs.values())


def _reference(model, specs) -> list[list[int]]:
    results = []
    for prompt, n in specs:
        engine = CachedEngine(model)
        engine.add_request(
            Request(prompt_token_ids=list(prompt), params=SamplingParams(max_tokens=n))
        )
        tokens: list[int] = []
        while engine.has_work():
            for out in engine.step():
                tokens.extend(out.token_ids)
        results.append(tokens)
    return results


def test_prefix_reuse_does_not_change_output(model, specs):
    """The invariant that makes prefix caching safe: a cache hit is invisible in the output."""
    engine = PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=256, block_size=16)
    assert _drain(engine, specs) == _reference(model, specs)


def test_shared_prompts_actually_hit(model, specs):
    """Publishing on request completion would be too late — concurrent arrivals would all miss."""
    engine = PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=256, block_size=16)
    _drain(engine, specs)
    assert engine.prefix.hits >= len(specs) - 1
    assert engine.prefix.token_reuse_rate > 0.5


def test_unrelated_prompts_do_not_hit(model):
    """No shared text, no reuse. Prefix caching must not invent savings."""
    engine = PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=256, block_size=16)
    unrelated = [([(i * 37 + j) % 250 + 1 for j in range(64)], 4) for i in range(3)]
    _drain(engine, unrelated)
    assert engine.prefix.token_reuse_rate == 0.0


def test_prefix_engine_matches_paged_engine(model, specs):
    """Same answers as ch08, reached with less work."""
    paged = PagedEngine(model, REFERENCE_MODEL, n_blocks=256, block_size=16)
    prefix = PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=256, block_size=16)
    assert _drain(prefix, specs) == _drain(paged, specs)
