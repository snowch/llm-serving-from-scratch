"""Chapter 13: decode attention that reads the paged cache in place."""

import pytest
import torch

from llmserve.attention import standard_attention
from llmserve.config import ModelConfig
from llmserve.kernels import (
    gather_bytes_per_step,
    in_place_bytes_per_step,
    paged_decode_attention,
)

LLAMA8B = ModelConfig(
    vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, dtype="fp16"
)


def _caches(n_blocks=16, n_kv_heads=2, block_size=8, head_dim=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (n_blocks, n_kv_heads, block_size, head_dim)
    return (
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
    )


def _reference(query, key_cache, value_cache, tables, lengths):
    """Materialise each sequence and use the chapter 12 reference. The thing we must match."""
    n_seqs, n_heads, _ = query.shape
    n_kv_heads = key_cache.shape[1]
    group = n_heads // n_kv_heads
    out = torch.empty_like(query)
    for seq in range(n_seqs):
        length = lengths[seq]
        for head in range(n_heads):
            kv_head = head // group
            keys = torch.cat([key_cache[p, kv_head] for p in tables[seq]], dim=0)[:length]
            values = torch.cat([value_cache[p, kv_head] for p in tables[seq]], dim=0)[:length]
            out[seq, head] = standard_attention(
                query[seq, head][None, None, :], keys[None], values[None]
            )[0, 0]
    return out


# -- correctness -------------------------------------------------------------------------


def test_reading_blocks_in_place_matches_materialising_them():
    """The specification a kernel has to meet: same answer, no gather."""
    key_cache, value_cache = _caches()
    tables = [[3, 7, 1], [5, 2, 9]]
    lengths = [20, 17]
    query = torch.randn(2, 4, 32, generator=torch.Generator().manual_seed(1))

    got = paged_decode_attention(query, key_cache, value_cache, tables, lengths)
    want = _reference(query, key_cache, value_cache, tables, lengths)
    assert torch.allclose(got, want, atol=1e-5)


@pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 20])
def test_a_partial_final_block_is_masked_correctly(length):
    """Off-by-one in the mask is the bug this whole chapter is most likely to introduce."""
    key_cache, value_cache = _caches(block_size=8)
    tables = [[3, 7, 1]]
    query = torch.randn(1, 2, 32, generator=torch.Generator().manual_seed(2))

    got = paged_decode_attention(query, key_cache, value_cache, tables, [length])
    want = _reference(query, key_cache, value_cache, tables, [length])
    assert torch.allclose(got, want, atol=1e-5)


def test_blocks_may_be_in_any_physical_order():
    """The block table is an indirection, so adjacency must not matter to the answer."""
    key_cache, value_cache = _caches()
    query = torch.randn(1, 2, 32, generator=torch.Generator().manual_seed(3))

    ordered = paged_decode_attention(query, key_cache, value_cache, [[0, 1, 2]], [20])
    # The same logical sequence, stored in different physical blocks.
    shuffled_keys = key_cache.clone()
    shuffled_values = value_cache.clone()
    for logical, physical in enumerate((11, 4, 6)):
        shuffled_keys[physical] = key_cache[logical]
        shuffled_values[physical] = value_cache[logical]
    scattered = paged_decode_attention(query, shuffled_keys, shuffled_values, [[11, 4, 6]], [20])
    assert torch.allclose(ordered, scattered, atol=1e-6)


def test_grouped_query_heads_share_a_kv_head_without_expanding_the_cache():
    """Expanding the cache to match the query heads would reintroduce the copy this avoids."""
    key_cache, value_cache = _caches(n_kv_heads=1)
    query = torch.randn(1, 4, 32, generator=torch.Generator().manual_seed(4))
    got = paged_decode_attention(query, key_cache, value_cache, [[2, 5]], [12])
    want = _reference(query, key_cache, value_cache, [[2, 5]], [12])
    assert torch.allclose(got, want, atol=1e-5)


def test_shapes_that_cannot_work_are_rejected():
    key_cache, value_cache = _caches(n_kv_heads=3)
    query = torch.randn(1, 4, 32)
    with pytest.raises(ValueError, match="do not divide"):
        paged_decode_attention(query, key_cache, value_cache, [[0]], [4])

    key_cache, value_cache = _caches(head_dim=16)
    with pytest.raises(ValueError, match="head_dim"):
        paged_decode_attention(torch.randn(1, 2, 32), key_cache, value_cache, [[0]], [4])

    key_cache, value_cache = _caches()
    with pytest.raises(ValueError, match="one block table"):
        paged_decode_attention(torch.randn(2, 2, 32), key_cache, value_cache, [[0]], [4])


def test_a_sequence_with_nothing_to_attend_to_is_an_error_not_a_nan():
    """Zero keys means an all-masked softmax, which is the NaN chapter 6 warns about."""
    key_cache, value_cache = _caches()
    with pytest.raises(ValueError, match="no tokens"):
        paged_decode_attention(torch.randn(1, 2, 32), key_cache, value_cache, [[0]], [0])


# -- the arithmetic that motivates it ----------------------------------------------------


def test_the_gather_moves_the_cache_more_than_once():
    gathered = gather_bytes_per_step(LLAMA8B, 2048, 32)
    in_place = in_place_bytes_per_step(LLAMA8B, 2048, 32)
    assert gathered > in_place
    assert gathered == pytest.approx(2 * in_place)


def test_traffic_scales_with_context_and_batch():
    base = in_place_bytes_per_step(LLAMA8B, 2048, 8)
    assert in_place_bytes_per_step(LLAMA8B, 2048, 16) == pytest.approx(2 * base)
    assert in_place_bytes_per_step(LLAMA8B, 4096, 8) == pytest.approx(2 * base)


def test_partial_blocks_are_charged_as_whole_blocks():
    """Memory traffic is per block, not per token: the last block is read in full either way."""
    assert in_place_bytes_per_step(LLAMA8B, 17, 1, block_size=16) == pytest.approx(
        in_place_bytes_per_step(LLAMA8B, 32, 1, block_size=16)
    )
