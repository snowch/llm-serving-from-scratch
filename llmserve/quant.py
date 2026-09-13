"""Quantisation: storing weights and cache in fewer bits.

Chapter 15. Every optimisation so far has been free in the sense that mattered — the output was
token-identical, and only the time or memory changed. This one is different. Quantisation makes
the model's answers measurably worse, and the entire engineering question is whether the loss is
small enough to be worth what it buys.

That means a quantisation chapter without a quality measurement is not a chapter, it is a sales
brochure. Three axes, always together: **quality, speed, memory**.

Weight-only quantisation is the technique here: store weights in 8 or 4 bits, restore them to
floating point on the way into the matmul. Per chapter 3, decode is bound by bytes read from
memory rather than by arithmetic, so reading a quarter as many bytes is the win even though the
multiply still happens in floating point.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class QuantStats:
    """What a quantisation actually cost, in bytes and in error."""

    original_bytes: int
    quantised_bytes: int
    max_abs_error: float
    mean_abs_error: float

    @property
    def compression(self) -> float:
        return self.original_bytes / self.quantised_bytes


def quantize_int8_symmetric(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric INT8.

    Symmetric means zero maps to zero and one scale per channel covers ``[-max, +max]``; there is
    no zero-point to store or add. Per *channel* rather than per tensor because a single scale for
    the whole matrix is set by its largest outlier, which crushes every other channel into a
    handful of levels — the difference between INT8 working and INT8 being useless.
    """
    scale = weight.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-12)
    quantised = (weight / scale).round().clamp(-127, 127).to(torch.int8)
    return quantised, scale


def quantize_int8_per_tensor(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """One scale for the whole matrix — the naive choice, kept so it can be measured.

    Included deliberately. It is what you write first, it looks reasonable, and it is set by the
    single largest value anywhere in the tensor. Every other channel is then squeezed into
    whatever levels remain. Chapter 15 measures what that costs, because "use per-channel scales"
    is much more convincing as a number than as advice.
    """
    scale = weight.abs().amax().clamp(min=1e-12) / 127.0
    quantised = (weight / scale).round().clamp(-127, 127).to(torch.int8)
    return quantised, scale


def dequantize(quantised: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return quantised.to(scale.dtype) * scale


def quantize_int4_grouped(
    weight: torch.Tensor, group_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor]:
    """INT4 with one scale per group of ``group_size`` weights along the input dimension.

    Four bits gives sixteen levels, which is not enough to cover a whole channel's range. Smaller
    groups mean more scales — more metadata, less compression — but each group's range is tighter.
    That trade is the whole design space of INT4 formats.

    Returned values are int8-typed but confined to the INT4 range; packing two per byte is
    mechanical and omitted, so the reported size accounts for it rather than the storage doing so.
    """
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(f"in_features {in_features} is not divisible by group_size {group_size}")

    grouped = weight.reshape(out_features, in_features // group_size, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = scale.clamp(min=1e-12)
    quantised = (grouped / scale).round().clamp(-7, 7).to(torch.int8)
    return quantised, scale


def dequantize_int4_grouped(
    quantised: torch.Tensor, scale: torch.Tensor, out_features: int, in_features: int
) -> torch.Tensor:
    return (quantised.to(scale.dtype) * scale).reshape(out_features, in_features)


def quantization_error(original: torch.Tensor, restored: torch.Tensor, bits: int) -> QuantStats:
    """Compare a weight tensor with its round trip through quantisation."""
    delta = (original - restored).abs()
    element_bytes = original.element_size()
    return QuantStats(
        original_bytes=original.numel() * element_bytes,
        # Scales are stored per channel or per group; counted at full precision.
        quantised_bytes=int(original.numel() * bits / 8),
        max_abs_error=float(delta.max()),
        mean_abs_error=float(delta.mean()),
    )


class QuantizedLinear(nn.Module):
    """A ``nn.Linear`` whose weights are stored quantised and restored on the way in.

    Dequantising on every forward pass is what makes this *weight-only* quantisation. On a
    memory-bound device it still wins, because the expensive part was reading the weights. On a
    machine where they were already in cache it is pure overhead — which is exactly what this
    book's CPU measurements show, and why the chapter is careful about where the win comes from.
    """

    def __init__(
        self, linear: nn.Linear, bits: int = 8, group_size: int = 64, per_tensor: bool = False
    ) -> None:
        super().__init__()
        self.bits = bits
        self.per_tensor = per_tensor
        self.group_size = group_size
        self.out_features, self.in_features = linear.weight.shape
        self.bias = linear.bias

        if bits == 8 and per_tensor:
            quantised, scale = quantize_int8_per_tensor(linear.weight.data)
        elif bits == 8:
            quantised, scale = quantize_int8_symmetric(linear.weight.data)
        elif bits == 4:
            quantised, scale = quantize_int4_grouped(linear.weight.data, group_size)
        else:
            raise ValueError(f"unsupported bit width: {bits}")

        self.register_buffer("quantised", quantised)
        self.register_buffer("scale", scale)

    def dequantized_weight(self) -> torch.Tensor:
        if self.bits == 8:
            return dequantize(self.quantised, self.scale)
        return dequantize_int4_grouped(
            self.quantised, self.scale, self.out_features, self.in_features
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.dequantized_weight(), self.bias)

    def stored_bytes(self) -> int:
        return (
            self.quantised.numel() * self.bits // 8 + self.scale.numel() * self.scale.element_size()
        )


def quantize_model(
    model: nn.Module, bits: int = 8, group_size: int = 64, per_tensor: bool = False
) -> nn.Module:
    """Replace every ``nn.Linear`` in place. Embeddings and norms are left alone.

    Norms are tiny and numerically sensitive; the embedding and output head are large but quantise
    badly in ways that show up immediately as degraded text. Production quantisers make the same
    exclusions, for the same reasons.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(
                model,
                name,
                QuantizedLinear(child, bits=bits, group_size=group_size, per_tensor=per_tensor),
            )
        else:
            quantize_model(child, bits=bits, group_size=group_size, per_tensor=per_tensor)
    return model


def quantize_kv(tensor: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Round-trip a KV cache tensor through quantisation, per head.

    Often the bigger win than weight quantisation, and almost always the forgotten one: chapter 13
    showed KV cache is what limits concurrency at long context, and halving it doubles how many
    conversations fit.
    """
    if bits not in (8, 4):
        raise ValueError(f"unsupported bit width: {bits}")
    levels = 127.0 if bits == 8 else 7.0
    scale = tensor.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / levels
    return (tensor / scale).round().clamp(-levels, levels) * scale
