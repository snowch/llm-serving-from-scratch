"""Chapter 12: online softmax must equal the batch computation, exactly enough to rely on."""

import math

import pytest
import torch

from llmserve.attention import (
    attention_matrix_bytes,
    online_softmax_attention,
    standard_attention,
)


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)


def _qkv(seq_len: int, n_keys: int, heads: int = 4, dim: int = 32, scale: float = 1.0):
    return (
        torch.randn(2, heads, seq_len, dim) * scale,
        torch.randn(2, heads, n_keys, dim) * scale,
        torch.randn(2, heads, n_keys, dim),
    )


@pytest.mark.parametrize("block_size", [1, 16, 64, 4096])
@pytest.mark.parametrize("seq_len", [1, 16])
def test_online_softmax_matches_standard_attention(block_size, seq_len):
    """The result must not depend on the block size — that is what makes tiling safe."""
    q, k, v = _qkv(seq_len, n_keys=257)  # not a multiple of any block size above
    assert torch.allclose(
        online_softmax_attention(q, k, v, block_size=block_size),
        standard_attention(q, k, v),
        atol=1e-5,
    )


def test_survives_scores_that_would_overflow_a_naive_implementation():
    """Subtracting the running maximum is what stops exp() reaching infinity."""
    q, k, v = _qkv(4, n_keys=128, scale=50.0)
    out = online_softmax_attention(q, k, v, block_size=16)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, standard_attention(q, k, v), atol=1e-5)


def test_single_key_is_just_that_value():
    """With one key, attention has no choice, whatever the scores are."""
    q, k, v = _qkv(3, n_keys=1)
    assert torch.allclose(online_softmax_attention(q, k, v), v.expand(-1, -1, 3, -1), atol=1e-6)


def test_uniform_scores_average_the_values():
    """Identical queries and keys give a uniform distribution, so the output is the mean of V."""
    q = torch.zeros(1, 1, 1, 8)
    k = torch.zeros(1, 1, 16, 8)
    v = torch.randn(1, 1, 16, 8)
    assert torch.allclose(online_softmax_attention(q, k, v), v.mean(dim=2, keepdim=True), atol=1e-6)


def test_score_matrix_grows_quadratically():
    """The term that makes long context impossible without tiling."""
    small = attention_matrix_bytes(1024, 1024, 32)
    large = attention_matrix_bytes(4096, 4096, 32)
    assert large == 16 * small


def test_scale_defaults_to_inverse_sqrt_head_dim():
    q, k, v = _qkv(4, n_keys=32)
    explicit = standard_attention(q, k, v, scale=1.0 / math.sqrt(q.shape[-1]))
    assert torch.equal(standard_attention(q, k, v), explicit)
