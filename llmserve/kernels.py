"""Chapter 14: decode attention that reads the paged cache in place.

Chapter 8's decode path materialises each sequence's KV cache as one contiguous tensor before every
step, pads it into a batch, and hands that to the model. It is the simplest thing that works and it
moves an enormous amount of memory to produce one token per sequence.

The fix is to stop materialising. Attention over a block table does not need the blocks to be
adjacent: chapter 13's online softmax processes keys one block at a time and never needs to see them
all at once, so the block table can be walked in place and each block read exactly once, straight
into the accumulator.

This module is that algorithm, in PyTorch, checked against the reference implementation. It is
deliberately *not* a Triton kernel — see the chapter for why, and for what the translation involves.
What it gives you is the specification a kernel has to match, and the arithmetic that says how much
there is to win.
"""

from __future__ import annotations

import math

import torch

from llmserve.config import BYTES_PER_DTYPE, ModelConfig


def paged_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: list[list[int]],
    lengths: list[int],
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Attention for one decode step, reading blocks in place.

    ``query`` is ``[n_seqs, n_heads, head_dim]`` — one token per sequence, which is what decode is.
    The caches are ``[n_blocks, n_kv_heads, block_size, head_dim]``: the layout chapter 8 allocates,
    untouched. Grouped-query attention is handled by mapping each query head to its KV head rather
    than by expanding the cache, which is the whole point — expanding it would reintroduce the copy.

    No score matrix is built and no sequence is padded to a common length. Each block is read once,
    accumulated, and discarded, so the memory traffic is the cache itself rather than a copy of it.
    """
    n_seqs, n_heads, head_dim = query.shape
    n_blocks_total, n_kv_heads, block_size, cache_dim = key_cache.shape
    if cache_dim != head_dim:
        raise ValueError(f"query head_dim {head_dim} does not match cache {cache_dim}")
    if n_heads % n_kv_heads:
        raise ValueError(f"{n_heads} query heads do not divide into {n_kv_heads} KV heads")
    if len(block_tables) != n_seqs or len(lengths) != n_seqs:
        raise ValueError("one block table and one length per sequence")

    scale = scale or 1.0 / math.sqrt(head_dim)
    group = n_heads // n_kv_heads
    out = torch.empty_like(query)

    for seq in range(n_seqs):
        length = lengths[seq]
        if length < 1:
            raise ValueError(f"sequence {seq} has no tokens to attend to")
        n_used = (length + block_size - 1) // block_size
        table = block_tables[seq][:n_used]

        for head in range(n_heads):
            kv_head = head // group
            q = query[seq, head]

            # Online softmax state, exactly as chapter 13 derived it: a running maximum, a running
            # denominator, and an accumulator that is rescaled whenever the maximum moves.
            running_max = torch.tensor(float("-inf"))
            denominator = torch.zeros(())
            acc = torch.zeros(head_dim)

            for i, physical in enumerate(table):
                keys = key_cache[physical, kv_head]
                values = value_cache[physical, kv_head]

                scores = (keys @ q) * scale
                positions = i * block_size + torch.arange(block_size)
                # A partial final block must be masked, and with a finite floor rather than -inf:
                # an all-masked row would make softmax produce NaN, which chapter 6 learned the
                # hard way spreads across the whole batch.
                scores = torch.where(positions < length, scores, torch.finfo(scores.dtype).min)

                block_max = scores.max()
                new_max = torch.maximum(running_max, block_max)
                rescale = torch.exp(running_max - new_max) if running_max.isfinite() else 0.0
                weights = torch.exp(scores - new_max)

                denominator = denominator * rescale + weights.sum()
                acc = acc * rescale + weights @ values
                running_max = new_max

            out[seq, head] = acc / denominator

    return out


def gather_bytes_per_step(
    model: ModelConfig, context_length: int, batch_size: int, block_size: int = 16
) -> float:
    """Bytes chapter 8's decode path moves per step, before the model runs at all.

    The gather copies every sequence's whole cache, then pads each to the batch's longest, then the
    model concatenates the new key and value onto it — so the same bytes are written more than once.
    Counting two copies is the conservative reading and still enough to make the point.
    """
    per_sequence = _kv_bytes(model, context_length, block_size)
    return 2 * per_sequence * batch_size


def in_place_bytes_per_step(
    model: ModelConfig, context_length: int, batch_size: int, block_size: int = 16
) -> float:
    """Bytes a kernel reading blocks in place must move: the cache, once, and nothing else."""
    return _kv_bytes(model, context_length, block_size) * batch_size


def _kv_bytes(model: ModelConfig, context_length: int, block_size: int) -> float:
    """One sequence's cached keys and values, rounded up to whole blocks."""
    blocks = (context_length + block_size - 1) // block_size
    slots = blocks * block_size
    element = BYTES_PER_DTYPE[model.dtype]
    return 2 * model.n_layers * model.n_kv_heads * model.head_dim * slots * element
