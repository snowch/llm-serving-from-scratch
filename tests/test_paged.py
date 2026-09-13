"""Chapter 8: paged KV storage, and the scheduling behaviour it makes possible."""

import pytest
import torch

from llmserve.cache.blocks import BlockAllocator, OutOfBlocksError, PagedKVCache
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.naive import CachedEngine
from llmserve.engines.paged import PagedEngine
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return build_model(REFERENCE_MODEL)


# -- the allocator ---------------------------------------------------------------------


def test_blocks_needed_rounds_up():
    allocator = BlockAllocator(n_blocks=8, block_size=4)
    assert allocator.blocks_needed(1) == 1
    assert allocator.blocks_needed(4) == 1
    assert allocator.blocks_needed(5) == 2


def test_allocation_round_trips_to_the_pool():
    allocator = BlockAllocator(n_blocks=8, block_size=4)
    blocks = allocator.allocate(3)
    assert allocator.n_free == 5
    allocator.free(blocks)
    assert allocator.n_free == 8


def test_exhaustion_raises_rather_than_returning_short():
    allocator = BlockAllocator(n_blocks=2, block_size=4)
    with pytest.raises(OutOfBlocksError):
        allocator.allocate(3)


def test_shared_blocks_survive_until_the_last_reference_goes():
    """Reference counting is what makes ch09's prefix sharing safe."""
    allocator = BlockAllocator(n_blocks=4, block_size=4)
    block = allocator.allocate(1)[0]
    allocator.share(block)
    allocator.free([block])
    assert allocator.n_free == 3, "a block still referenced must not return to the pool"
    allocator.free([block])
    assert allocator.n_free == 4


# -- the paged store -------------------------------------------------------------------


def test_write_then_gather_round_trips_exactly():
    cache = PagedKVCache(n_layers=1, n_kv_heads=2, head_dim=4, n_blocks=8, block_size=4)
    table = cache.allocator.allocate(3)
    k = torch.arange(2 * 10 * 4, dtype=torch.float32).reshape(2, 10, 4)
    cache.write(0, table, 0, k, -k)
    got_k, got_v = cache.gather(0, table, 10)
    assert torch.equal(got_k[0], k)
    assert torch.equal(got_v[0], -k)


def test_writes_spanning_a_block_boundary_match_a_single_write():
    """A run of tokens rarely lines up with a block, so the split path must be identical."""
    k = torch.arange(2 * 10 * 4, dtype=torch.float32).reshape(2, 10, 4)

    whole = PagedKVCache(1, 2, 4, 8, 4)
    table_a = whole.allocator.allocate(3)
    whole.write(0, table_a, 0, k, -k)

    split = PagedKVCache(1, 2, 4, 8, 4)
    table_b = split.allocator.allocate(3)
    split.write(0, table_b, 0, k[:, :3, :], -k[:, :3, :])
    split.write(0, table_b, 3, k[:, 3:, :], -k[:, 3:, :])

    assert torch.equal(whole.gather(0, table_a, 10)[0], split.gather(0, table_b, 10)[0])


# -- the engine ------------------------------------------------------------------------

SPECS = [
    (list(b"The quick brown fox jumps"), 8),
    (list(b"Hello"), 4),
    (list(b"A somewhat longer prompt that forces left padding in the batch"), 12),
]


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


def test_paging_does_not_change_output(model):
    """Where the keys and values live must not affect what the model says."""
    engine = PagedEngine(model, REFERENCE_MODEL, n_blocks=256, block_size=16)
    assert _drain(engine, SPECS) == _reference(model, SPECS)


def test_blocks_are_returned_when_requests_finish(model):
    """A leak here is invisible until the pool is empty and throughput collapses."""
    engine = PagedEngine(model, REFERENCE_MODEL, n_blocks=256, block_size=16)
    _drain(engine, SPECS)
    assert engine.cache.allocator.n_free == engine.cache.allocator.n_blocks


def test_admission_does_not_promise_blocks_it_cannot_deliver(model):
    """Regression: the admission loop must subtract what it has already promised this step.

    Checking every candidate against the same free-block count admits a batch that does not fit,
    and prefill then fails on a request the scheduler had already accepted.
    """
    engine = PagedEngine(model, REFERENCE_MODEL, max_batch_size=8, n_blocks=10, block_size=8)
    specs = [(list(b"a prompt"), 6)] * 6
    outputs = _drain(engine, specs)
    assert all(len(tokens) == 6 for tokens in outputs)


def test_preemption_makes_progress_under_memory_pressure(model):
    """Running out of blocks is a scheduling event, not a crash — and must still terminate."""
    engine = PagedEngine(model, REFERENCE_MODEL, max_batch_size=8, n_blocks=10, block_size=8)
    specs = [(list(b"a prompt"), 40)] * 6
    outputs = _drain(engine, specs)
    assert all(len(tokens) == 40 for tokens in outputs), "preempted requests must still complete"
    assert engine.preemptions > 0, "this budget should have forced preemption"
    assert engine.cache.allocator.n_free == engine.cache.allocator.n_blocks


def test_paging_beats_reserve_max_concurrency(model):
    """The whole point: on-demand blocks fit sequences a reserve-max allocator cannot.

    A contiguous allocator must reserve the maximum sequence length before knowing the real one,
    so in this budget it fits nothing at all.
    """
    block_size, n_blocks = 16, 24
    budget_tokens = block_size * n_blocks
    reserve_max_concurrency = budget_tokens // REFERENCE_MODEL.max_seq_len

    engine = PagedEngine(
        model, REFERENCE_MODEL, max_batch_size=16, n_blocks=n_blocks, block_size=block_size
    )
    _drain(engine, [(list(b"prompt " * 3), 10)] * 12)
    assert reserve_max_concurrency == 0
    assert engine.peak_running > 1
