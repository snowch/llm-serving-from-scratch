"""Attention, computed without materialising the attention matrix.

Chapter 12. The standard formulation builds an ``S x N`` score matrix per head per layer, softmaxes
it, and multiplies by V. At 32k context that matrix is larger than the model's weights, and it is
written to memory and read back for no reason other than that softmax appears to need all its
inputs at once.

It does not. Softmax can be computed incrementally, one block of keys at a time, by carrying a
running maximum and a running sum and rescaling what you have so far when the maximum moves. That
single trick — *online softmax* — is what FlashAttention is built on. The real implementation adds
GPU-specific tiling and keeps everything in SRAM; the arithmetic below is the part that matters
and the part you can check.
"""

from __future__ import annotations

import math

import torch


def standard_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float | None = None
) -> torch.Tensor:
    """Attention the obvious way, materialising the full score matrix.

    The reference every other implementation is checked against, and the thing whose memory
    traffic we are trying to avoid.
    """
    scale = scale or 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-2, -1)) * scale
    return scores.softmax(dim=-1) @ v


def online_softmax_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
    scale: float | None = None,
) -> torch.Tensor:
    """The same result, one block of keys at a time.

    Carries three running quantities per query: the maximum score seen so far ``m``, the sum of
    exponentials ``l``, and the weighted value accumulator ``acc``. When a new block contains a
    larger score, everything accumulated so far is rescaled by ``exp(m_old - m_new)`` — which is
    exactly the correction that keeps the result identical to having seen all scores at once.

    Subtracting the running maximum is not an optimisation, it is what stops ``exp`` overflowing.
    Naive incremental softmax without it produces infinities on perfectly ordinary inputs.

    Peak memory for scores is ``S x block_size`` instead of ``S x N``: the saving is the whole
    point, and it grows with context length.
    """
    scale = scale or 1.0 / math.sqrt(q.shape[-1])
    *batch_dims, seq_len, head_dim = q.shape
    n_keys = k.shape[-2]

    acc = torch.zeros(*batch_dims, seq_len, head_dim, dtype=q.dtype)
    running_max = torch.full((*batch_dims, seq_len, 1), float("-inf"), dtype=q.dtype)
    running_sum = torch.zeros(*batch_dims, seq_len, 1, dtype=q.dtype)

    for start in range(0, n_keys, block_size):
        end = min(start + block_size, n_keys)
        scores = (q @ k[..., start:end, :].transpose(-2, -1)) * scale

        block_max = scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(running_max, block_max)
        # exp(-inf - -inf) is nan, so the first block needs the guard.
        correction = torch.where(
            running_max == float("-inf"),
            torch.zeros_like(running_max),
            (running_max - new_max).exp(),
        )
        weights = (scores - new_max).exp()

        acc = acc * correction + weights @ v[..., start:end, :]
        running_sum = running_sum * correction + weights.sum(dim=-1, keepdim=True)
        running_max = new_max

    return acc / running_sum


def attention_matrix_bytes(seq_len: int, n_keys: int, n_heads: int, dtype_bytes: int = 4) -> int:
    """Bytes the materialised score matrix occupies, per layer.

    Worth computing before reaching for a longer context: this is the term that makes naive
    attention impossible at scale, and it is quadratic while everything else is linear.
    """
    return seq_len * n_keys * n_heads * dtype_bytes
