"""Chapter 19: splitting a model across devices, and getting the same answer."""

import pytest
import torch
from torch import nn

from llmserve.config import REFERENCE_MODEL, ModelConfig
from llmserve.parallel import (
    TensorParallelPair,
    all_reduce_bytes_per_token,
    all_reduce_time_seconds,
    max_tensor_parallel_shards,
    pipeline_bubble_fraction,
    shard_columns,
    shard_rows,
)

LLAMA8B = ModelConfig(
    vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, dtype="fp16"
)


# -- the split has to be exact -----------------------------------------------------------


@pytest.mark.parametrize("n_shards", [2, 4, 8])
def test_a_sharded_pair_computes_exactly_what_the_unsharded_one_does(n_shards):
    """The claim the whole technique rests on: splitting changes where work happens, not the answer.

    Tested at several shard counts because an off-by-one in the split shows up at some and not
    others — a two-way split is symmetric enough to hide bugs that a four-way split exposes.
    """
    torch.manual_seed(0)
    first, second = nn.Linear(64, 128, bias=False), nn.Linear(128, 64, bias=False)
    x = torch.randn(3, 5, 64)

    reference = second(first(x))
    sharded = TensorParallelPair(first, second, n_shards=n_shards)(x)
    assert torch.allclose(reference, sharded, atol=1e-5)


def test_one_collective_per_pair_not_two():
    """Column-then-row is the whole trick: cut it the other way and you pay twice."""
    torch.manual_seed(0)
    pair = TensorParallelPair(nn.Linear(64, 128, bias=False), nn.Linear(128, 64, bias=False))
    pair(torch.randn(2, 4, 64))
    assert pair.collectives == 1


def test_shapes_that_do_not_divide_fail_loudly():
    """A silent uneven split is a correctness bug; a load-time error is an inconvenience."""
    with pytest.raises(ValueError, match="divisible"):
        shard_columns(torch.zeros(7, 4), 2)
    with pytest.raises(ValueError, match="divisible"):
        shard_rows(torch.zeros(4, 7), 2)


def test_shards_partition_the_weight_without_overlap():
    weight = torch.arange(24.0).reshape(6, 4)
    assert torch.equal(torch.cat(shard_columns(weight, 3), dim=0), weight)
    assert torch.equal(torch.cat(shard_rows(weight, 2), dim=1), weight)


# -- the arithmetic that decides viability -----------------------------------------------


def test_communication_grows_with_shards_but_sublinearly():
    """A ring all-reduce moves 2(n-1)/n of the payload — approaching, never reaching, twice."""
    two = all_reduce_bytes_per_token(LLAMA8B, 2)
    eight = all_reduce_bytes_per_token(LLAMA8B, 8)
    assert two < eight < 2 * two


def test_a_single_shard_communicates_nothing():
    assert all_reduce_bytes_per_token(LLAMA8B, 1) == 0.0
    assert all_reduce_time_seconds(LLAMA8B, 1) == 0.0


def test_decode_is_latency_bound_and_prefill_is_bandwidth_bound():
    """The chapter's central point, as a test.

    Decode all-reduces a few kilobytes and pays the collective's fixed cost; the time is the same
    for two shards and eight. Prefill's payload scales with the prompt, so it does grow.
    """
    fast = {"bandwidth_bytes_per_second": 4.0e11, "latency_seconds": 5e-6}
    decode_2 = all_reduce_time_seconds(LLAMA8B, 2, tokens=1, **fast)
    decode_8 = all_reduce_time_seconds(LLAMA8B, 8, tokens=1, **fast)
    assert decode_2 == decode_8

    prefill_2 = all_reduce_time_seconds(LLAMA8B, 2, tokens=2048, **fast)
    prefill_8 = all_reduce_time_seconds(LLAMA8B, 8, tokens=2048, **fast)
    assert prefill_8 > prefill_2 > decode_2


def test_grouped_query_attention_caps_how_far_a_model_can_split():
    """A device must get whole KV heads, and GQA leaves far fewer of those than query heads."""
    assert max_tensor_parallel_shards(LLAMA8B) == 8
    assert max_tensor_parallel_shards(REFERENCE_MODEL) == REFERENCE_MODEL.n_kv_heads


def test_a_pipeline_with_one_microbatch_is_mostly_idle():
    assert pipeline_bubble_fraction(4, 1) == pytest.approx(0.75)
    assert pipeline_bubble_fraction(1, 8) == 0.0
    assert pipeline_bubble_fraction(4, 64) < 0.05


def test_a_pipeline_needs_at_least_one_of_each():
    with pytest.raises(ValueError):
        pipeline_bubble_fraction(0, 4)
    with pytest.raises(ValueError):
        pipeline_bubble_fraction(4, 0)
