"""Chapter 13: multi-head latent attention, as arithmetic and as a projection."""

import pytest
import torch
from torch import nn

from llmserve.arithmetic import kv_bytes_per_token
from llmserve.config import REFERENCE_MODEL, ModelConfig
from llmserve.latent import (
    LatentConfig,
    LatentKV,
    compression_ratio,
    latent_bytes_per_token,
)

LLAMA8B = ModelConfig(
    vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, dtype="fp16"
)


def test_expanding_the_cached_latent_is_what_the_forward_pass_produces():
    """The cached thing must reconstruct exactly what attention would have been given."""
    torch.manual_seed(0)
    mla = LatentKV(REFERENCE_MODEL, LatentConfig(d_latent=32))
    hidden = torch.randn(2, 5, REFERENCE_MODEL.d_model)

    keys, values = mla(hidden)
    from_cache = mla.expand(mla.compress(hidden))
    assert torch.equal(from_cache[0], keys)
    assert torch.equal(from_cache[1], values)


def test_reconstructed_keys_have_the_shape_attention_expects():
    mla = LatentKV(LLAMA8B, LatentConfig(d_latent=512))
    keys, values = mla(torch.randn(2, 7, LLAMA8B.d_model))
    assert keys.shape == (2, LLAMA8B.n_kv_heads, 7, LLAMA8B.head_dim)
    assert values.shape == keys.shape


def test_only_the_latent_is_cached():
    """The saving is that this is the whole cache entry — not a key and a value per KV head."""
    mla = LatentKV(LLAMA8B, LatentConfig(d_latent=512))
    latent = mla.compress(torch.randn(1, 3, LLAMA8B.d_model))
    assert latent.shape == (1, 3, 512)


def test_a_latent_of_no_dimensions_is_rejected():
    with pytest.raises(ValueError, match="caches nothing"):
        LatentConfig(d_latent=0)


# -- the arithmetic ------------------------------------------------------------------------


def test_latent_bytes_use_the_same_whole_model_convention_as_kv_bytes():
    """Regression: this was written per-layer and compared against a per-model figure, which
    overstated the compression by a factor of n_layers."""
    config = LatentConfig(d_latent=512)
    per_token = latent_bytes_per_token(LLAMA8B, config)
    assert per_token == 512 * LLAMA8B.n_layers * 2
    assert per_token > 512 * 2, "a per-layer figure would be n_layers too small"


def test_compression_is_linear_in_the_latent_dimension():
    small = compression_ratio(LLAMA8B, LatentConfig(d_latent=256))
    large = compression_ratio(LLAMA8B, LatentConfig(d_latent=512))
    assert small == pytest.approx(2 * large)


def test_a_latent_the_size_of_the_cache_it_replaces_saves_nothing():
    """The sanity check that the arithmetic is measuring what it claims to."""
    equivalent = 2 * LLAMA8B.n_kv_heads * LLAMA8B.head_dim
    ratio = compression_ratio(LLAMA8B, LatentConfig(d_latent=equivalent))
    assert ratio == pytest.approx(1.0)


def test_latent_attention_beats_grouped_query_at_realistic_settings():
    assert compression_ratio(LLAMA8B, LatentConfig(d_latent=512)) > 1
    assert latent_bytes_per_token(LLAMA8B, LatentConfig(512)) < kv_bytes_per_token(LLAMA8B)


def test_the_projections_are_the_only_parameters_added():
    """MLA trades cache for compute; the compute is these three matrices and nothing else."""
    mla = LatentKV(REFERENCE_MODEL, LatentConfig(d_latent=32))
    assert {name for name, _ in mla.named_children()} == {"down", "up_keys", "up_values"}
    assert all(isinstance(child, nn.Linear) for _, child in mla.named_children())
