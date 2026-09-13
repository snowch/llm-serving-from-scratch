"""Chapter 14: quantisation must compress, and must be honest about what it costs."""

import pytest
import torch
from torch import nn

from llmserve.quant import (
    QuantizedLinear,
    dequantize,
    dequantize_int4_grouped,
    quantization_error,
    quantize_int4_grouped,
    quantize_int8_per_tensor,
    quantize_int8_symmetric,
    quantize_kv,
    quantize_model,
)


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)


def test_int8_round_trip_stays_close():
    weight = torch.randn(64, 128)
    restored = dequantize(*quantize_int8_symmetric(weight))
    assert (weight - restored).abs().max() < weight.abs().max() / 100


def test_int4_is_lossier_than_int8():
    """Fewer levels, more error. If this ever fails, a scale is wrong somewhere."""
    weight = torch.randn(64, 128)
    int8 = (weight - dequantize(*quantize_int8_symmetric(weight))).abs().mean()
    q4, s4 = quantize_int4_grouped(weight, group_size=64)
    int4 = (weight - dequantize_int4_grouped(q4, s4, 64, 128)).abs().mean()
    assert int4 > int8


def test_per_channel_survives_an_outlier_that_ruins_per_tensor():
    """The mechanism behind per-channel scaling, as an assertion rather than advice."""
    weight = torch.randn(8, 64)
    weight[0] *= 200  # one channel dwarfing the rest

    per_channel = dequantize(*quantize_int8_symmetric(weight))
    per_tensor = dequantize(*quantize_int8_per_tensor(weight))

    others = slice(1, None)
    channel_error = (weight[others] - per_channel[others]).abs().mean()
    tensor_error = (weight[others] - per_tensor[others]).abs().mean()
    assert tensor_error > 50 * channel_error


def test_smaller_int4_groups_are_more_accurate():
    """Tighter ranges per group, at the cost of storing more scales."""
    weight = torch.randn(32, 256)

    def error(group_size: int) -> torch.Tensor:
        q, s = quantize_int4_grouped(weight, group_size=group_size)
        return (weight - dequantize_int4_grouped(q, s, 32, 256)).abs().mean()

    assert error(32) < error(256)


def test_int4_rejects_an_indivisible_shape():
    with pytest.raises(ValueError, match="divisible"):
        quantize_int4_grouped(torch.randn(8, 100), group_size=64)


@pytest.mark.parametrize("bits, expected", [(8, 4.0), (4, 8.0)])
def test_reported_compression_matches_the_bit_width(bits, expected):
    weight = torch.randn(32, 64)
    stats = quantization_error(weight, weight, bits)
    assert stats.compression == expected


def test_quantized_linear_approximates_the_original():
    linear = nn.Linear(64, 32)
    quantised = QuantizedLinear(linear, bits=8)
    x = torch.randn(4, 64)
    assert torch.allclose(linear(x), quantised(x), atol=0.05)
    assert quantised.stored_bytes() < linear.weight.numel() * 4


def test_quantize_model_replaces_linears_and_leaves_norms_alone():
    model = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8), nn.Linear(8, 4))
    quantize_model(model, bits=8)
    assert isinstance(model[0], QuantizedLinear)
    assert isinstance(model[1], nn.LayerNorm), "norms are tiny and numerically sensitive"
    assert isinstance(model[2], QuantizedLinear)


def test_kv_quantisation_preserves_shape_and_stays_close():
    cache = torch.randn(2, 4, 16, 32)
    restored = quantize_kv(cache, bits=8)
    assert restored.shape == cache.shape
    assert (cache - restored).abs().mean() < 0.02


def test_kv_int4_is_lossier_than_int8():
    cache = torch.randn(2, 4, 16, 32)
    assert (cache - quantize_kv(cache, 4)).abs().mean() > (
        cache - quantize_kv(cache, 8)
    ).abs().mean()


def test_unsupported_bit_widths_are_rejected():
    with pytest.raises(ValueError, match="unsupported"):
        quantize_kv(torch.randn(2, 2), bits=3)
    with pytest.raises(ValueError, match="unsupported"):
        QuantizedLinear(nn.Linear(4, 4), bits=3)
