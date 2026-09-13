"""Tests for the ch03 arithmetic.

These are the first tests in the book and they set the pattern: a known-good hand calculation,
then the property that must hold regardless of the numbers.
"""

import pytest

from llmserve import ModelSpec, decode_ceiling_tokens_per_second, kv_bytes_per_token

# Llama-3-8B-shaped: GQA with 8 KV heads against 32 query heads.
LLAMA8B = ModelSpec(
    name="llama-3-8b",
    n_params=8_030_000_000,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    dtype="fp16",
)


def test_kv_bytes_per_token_matches_hand_calculation():
    # 2 * 32 layers * 8 kv heads * 128 head_dim * 2 bytes = 131072 bytes/token
    assert kv_bytes_per_token(LLAMA8B) == 131072


def test_kv_quantisation_halves_the_cache():
    assert kv_bytes_per_token(LLAMA8B, kv_dtype="int8") == kv_bytes_per_token(LLAMA8B) / 2


def test_gqa_shrinks_the_cache_relative_to_mha():
    mha = ModelSpec(**{**LLAMA8B.__dict__, "n_kv_heads": 32})
    assert kv_bytes_per_token(mha) == 4 * kv_bytes_per_token(LLAMA8B)


def test_decode_ceiling_falls_as_context_grows():
    bandwidth = 2e12  # ~2 TB/s, H100-class
    short = decode_ceiling_tokens_per_second(LLAMA8B, bandwidth, context_length=128)
    long = decode_ceiling_tokens_per_second(LLAMA8B, bandwidth, context_length=32768)
    assert short > long


def test_batching_raises_the_ceiling():
    bandwidth = 2e12
    single = decode_ceiling_tokens_per_second(LLAMA8B, bandwidth, context_length=1024)
    batched = decode_ceiling_tokens_per_second(
        LLAMA8B, bandwidth, context_length=1024, batch_size=32
    )
    # The whole reason continuous batching (ch07) works.
    assert batched > single


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        decode_ceiling_tokens_per_second(LLAMA8B, 2e12, batch_size=0)
