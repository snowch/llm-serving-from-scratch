"""Tests for the chapter 3 arithmetic.

The first tests in the book, and they set the pattern: a hand calculation we can check by eye,
then the properties that must hold whatever the numbers are.
"""

import pytest

from llmserve.arithmetic import (
    cached_decode_flops,
    decode_ceiling_tokens_per_second,
    kv_bytes_per_token,
    max_concurrent_sequences,
    naive_decode_flops,
)
from llmserve.config import REFERENCE_MODEL, ModelConfig
from llmserve.model import build_model

# Llama-3-8B-shaped: grouped-query attention, 8 KV heads against 32 query heads.
LLAMA8B = ModelConfig(
    vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, dtype="fp16"
)


def test_kv_bytes_per_token_matches_hand_calculation():
    # 2 * 32 layers * 8 kv heads * 128 head_dim * 2 bytes = 131072 bytes/token
    assert kv_bytes_per_token(LLAMA8B) == 131072


def test_kv_quantisation_halves_the_cache():
    assert kv_bytes_per_token(LLAMA8B, kv_dtype="int8") == kv_bytes_per_token(LLAMA8B) / 2


def test_gqa_shrinks_the_cache_relative_to_mha():
    mha = ModelConfig(**{**LLAMA8B.__dict__, "n_kv_heads": 32})
    assert kv_bytes_per_token(mha) == 4 * kv_bytes_per_token(LLAMA8B)


def test_predicted_parameter_count_matches_the_built_model():
    """The arithmetic is only trustworthy if it agrees with the thing it describes."""
    model = build_model(REFERENCE_MODEL)
    assert sum(p.numel() for p in model.parameters()) == REFERENCE_MODEL.n_params


def test_decode_ceiling_falls_as_context_grows():
    bandwidth = 2e12  # ~2 TB/s, H100-class
    short = decode_ceiling_tokens_per_second(LLAMA8B, bandwidth, context_length=128)
    long = decode_ceiling_tokens_per_second(LLAMA8B, bandwidth, context_length=32768)
    assert short > long


def test_batching_raises_the_ceiling():
    """The whole reason continuous batching (ch07) works: weights are read once per step."""
    bandwidth = 2e12
    single = decode_ceiling_tokens_per_second(LLAMA8B, bandwidth, context_length=1024)
    batched = decode_ceiling_tokens_per_second(
        LLAMA8B, bandwidth, context_length=1024, batch_size=32
    )
    assert batched > 10 * single


def test_concurrency_ceiling_is_memory_over_per_sequence_cost():
    gib = 1024**3
    assert max_concurrent_sequences(LLAMA8B, 10 * gib, context_length=2048) == int(
        10 * gib // (131072 * 2048)
    )


def test_uncached_decode_is_quadratic_in_sequence_length():
    """Chapter 5's motivation, as arithmetic rather than assertion."""
    short = naive_decode_flops(REFERENCE_MODEL, prompt_len=64, n_output=16)
    long = naive_decode_flops(REFERENCE_MODEL, prompt_len=64, n_output=64)
    # Four times the output tokens costs much more than four times the work.
    assert long > 4 * short
    assert naive_decode_flops(REFERENCE_MODEL, 64, 64) > cached_decode_flops(
        REFERENCE_MODEL, 64, 64
    )


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        decode_ceiling_tokens_per_second(LLAMA8B, 2e12, batch_size=0)


def test_head_counts_must_be_divisible():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_heads=8, n_kv_heads=3)
