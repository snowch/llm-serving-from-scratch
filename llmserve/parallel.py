"""Chapter 19: splitting one model across several devices.

A model that does not fit on one accelerator has to be split, and there are two ways to cut it.
**Tensor parallelism** cuts each layer's matrices across devices, so every device holds a slice of
every layer and they combine their partial results. **Pipeline parallelism** cuts the stack of
layers, so each device holds some layers whole and activations flow between them.

The two have completely different communication profiles, and that — not any property of the model —
decides which one is viable on your hardware. Tensor parallelism communicates *every layer*, so it
wants an interconnect measured in hundreds of GB/s. Pipeline parallelism communicates once per
stage boundary and introduces idle time instead.

This module implements the tensor-parallel split as a simulation on one device: the shards are real,
the arithmetic is real, and the all-reduce is a sum rather than a collective. That is enough to prove
the split is *correct* — which is the part people get wrong — and no substitute for measuring it,
which needs hardware the book's default tier does not have. The arithmetic that decides viability is
in :func:`all_reduce_bytes_per_token` and is exact.
"""

from __future__ import annotations

import torch
from torch import nn

from llmserve.config import BYTES_PER_DTYPE, ModelConfig


def shard_columns(weight: torch.Tensor, n_shards: int) -> list[torch.Tensor]:
    """Split an ``(out, in)`` weight along its output dimension.

    Used for the first matrix of a pair — the query/key/value projections, and the MLP's up and gate
    projections. Each device computes a slice of the output and needs no communication to do it,
    because every device has the whole input.
    """
    if weight.shape[0] % n_shards:
        raise ValueError(
            f"output dimension {weight.shape[0]} is not divisible by {n_shards} shards — "
            "the usual cause of a tensor-parallel setup failing at load time, and the reason "
            "head counts constrain how far a model can be split"
        )
    return list(torch.chunk(weight, n_shards, dim=0))


def shard_rows(weight: torch.Tensor, n_shards: int) -> list[torch.Tensor]:
    """Split an ``(out, in)`` weight along its input dimension.

    Used for the second matrix of a pair — the attention output projection and the MLP's down
    projection. Each device holds a slice of the input, so each computes a *partial* output over the
    full output dimension, and the partials must be summed. That sum is the all-reduce, and it is
    the entire communication cost of tensor parallelism.
    """
    if weight.shape[1] % n_shards:
        raise ValueError(f"input dimension {weight.shape[1]} is not divisible by {n_shards} shards")
    return list(torch.chunk(weight, n_shards, dim=1))


class TensorParallelPair(nn.Module):
    """Two chained projections, split so exactly one collective is needed between them.

    This is the pattern the whole technique rests on, and it is why the split is column-then-row
    rather than any other combination. Splitting the first matrix by column gives each device a
    slice of the intermediate, which it can feed straight into its row-slice of the second matrix
    with no communication. Only the second matrix's output needs summing.

    Cut it the other way round and you need a collective *between* the two matrices as well, which
    doubles the communication for no benefit. Every production implementation makes this choice, and
    it is the single most useful thing to recognise when reading one.
    """

    def __init__(self, first: nn.Linear, second: nn.Linear, n_shards: int = 2) -> None:
        super().__init__()
        self.n_shards = n_shards
        self.first_shards = shard_columns(first.weight.data, n_shards)
        self.second_shards = shard_rows(second.weight.data, n_shards)
        self.activation: nn.Module | None = None
        #: how many times a forward pass would call a collective
        self.collectives = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partials = []
        for first, second in zip(self.first_shards, self.second_shards, strict=True):
            hidden = x @ first.T
            if self.activation is not None:
                hidden = self.activation(hidden)
            partials.append(hidden @ second.T)
        # The all-reduce. On real hardware this is a collective over the interconnect; here it is a
        # sum, which is numerically what the collective computes.
        self.collectives += 1
        return torch.stack(partials).sum(dim=0)


def all_reduce_bytes_per_token(model: ModelConfig, n_shards: int = 2) -> float:
    """Bytes each token's forward pass all-reduces, across the whole model.

    Two collectives per layer — one after attention's output projection, one after the MLP's down
    projection — each over a vector of ``d_model`` elements. A ring all-reduce moves roughly
    ``2 × (n-1)/n`` times the payload per participant, which is the factor below.

    This is the number that decides whether tensor parallelism is viable, and it does not depend on
    the model fitting or not: it is paid on every token forever.
    """
    if n_shards < 2:
        return 0.0
    payload = model.d_model * BYTES_PER_DTYPE[model.dtype]
    collectives = 2 * model.n_layers
    return collectives * payload * 2 * (n_shards - 1) / n_shards


def all_reduce_time_seconds(
    model: ModelConfig,
    n_shards: int,
    *,
    tokens: int = 1,
    bandwidth_bytes_per_second: float = 4.0e11,
    latency_seconds: float = 5e-6,
) -> float:
    """Time the collectives cost, counting both bandwidth and per-message latency.

    Bandwidth alone gets this badly wrong, and in the direction that matters. A decode step
    all-reduces one vector per layer — a few kilobytes — and a few kilobytes over a 400 GB/s link
    takes nanoseconds. What it actually costs is the *fixed* cost of the collective, several
    microseconds, paid twice per layer. At 32 layers that is 64 collectives per token, and the
    latency term dominates by orders of magnitude.

    Prefill is the opposite: the payload scales with prompt length, so it is genuinely
    bandwidth-bound. Tensor parallelism therefore costs a different thing in each phase, which is
    the same split chapter 3 found everywhere else.
    """
    if n_shards < 2:
        return 0.0
    collectives = 2 * model.n_layers
    payload = all_reduce_bytes_per_token(model, n_shards) * tokens
    return max(payload / bandwidth_bytes_per_second, collectives * latency_seconds)


def pipeline_bubble_fraction(n_stages: int, n_microbatches: int) -> float:
    """Fraction of device time spent idle in a pipeline, from the schedule alone.

    ``(stages - 1) / (microbatches + stages - 1)``. The stages at the start and end of a pipeline
    have nothing to do while the pipeline fills and drains, and the only cure is more microbatches
    in flight — which costs memory, because every in-flight microbatch holds activations.

    For serving rather than training this is worse than it looks: a decode step produces one token
    per sequence, so the natural microbatch count is small and the bubble is large. It is the main
    reason pipeline parallelism is more common in training than in inference.
    """
    if n_stages < 1 or n_microbatches < 1:
        raise ValueError("a pipeline needs at least one stage and one microbatch")
    return (n_stages - 1) / (n_microbatches + n_stages - 1)


def max_tensor_parallel_shards(model: ModelConfig) -> int:
    """The largest split the model's shape allows.

    Attention is split by head, so a device must get whole heads — and with grouped-query attention
    (chapter 13) the binding constraint is the *key/value* head count, which is much smaller than
    the query head count. A model with 8 KV heads cannot be split 16 ways however large it is, and
    that is a surprisingly common wall to hit.
    """
    return model.n_kv_heads
