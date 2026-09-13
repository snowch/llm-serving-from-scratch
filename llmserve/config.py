"""Configuration for the reference engine.

Two objects, deliberately separate: ``ModelConfig`` describes the *model* (what it computes),
``EngineConfig`` describes the *serving* of it (how we schedule and cache). Chapter 3 shows that
almost every serving decision follows from a handful of numbers in the first, which is why they
are kept apart.
"""

from __future__ import annotations

from dataclasses import dataclass

BYTES_PER_DTYPE = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "int4": 0.5,
}


@dataclass(frozen=True)
class ModelConfig:
    """The dimensions that determine both what the model computes and what it costs to serve.

    ``n_kv_heads`` is the number of key/value heads. It equals ``n_heads`` for vanilla multi-head
    attention, is smaller under grouped-query attention, and is 1 under multi-query attention.
    It appears here rather than being derived because it is the single biggest lever on KV-cache
    size, which chapter 5 shows is the binding constraint on concurrency.
    """

    vocab_size: int = 260
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    head_dim: int = 32
    ffn_mult: int = 4
    max_seq_len: int = 2048
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    dtype: str = "fp32"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads}); "
                "each KV head serves an equal group of query heads"
            )

    @property
    def d_model(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def group_size(self) -> int:
        """Query heads per KV head. 1 means full multi-head attention."""
        return self.n_heads // self.n_kv_heads

    @property
    def bytes_per_element(self) -> float:
        return BYTES_PER_DTYPE[self.dtype]

    @property
    def n_params(self) -> int:
        """Parameter count, computed rather than measured so chapter 3 can predict before we build.

        Per layer: attention projections (q, k, v, o) plus a SwiGLU MLP (gate, up, down) plus two
        RMSNorm weight vectors. Plus embeddings and the final norm.
        """
        d = self.d_model
        kv_dim = self.n_kv_heads * self.head_dim
        hidden = self.ffn_mult * d
        attn = d * d + 2 * (d * kv_dim) + d * d
        mlp = 3 * (d * hidden)
        norms = 2 * d
        per_layer = attn + mlp + norms
        embeddings = self.vocab_size * d
        head = d * self.vocab_size
        return self.n_layers * per_layer + embeddings + head + d


# The book's reference model for the CPU tier (Tier 1 in PLAN.md §5).
#
# Small enough that a laptop serves it at interactive speed, large enough that a decode step is
# real work rather than interpreter overhead. Grouped-query attention is on (4 query heads per KV
# head) because that is what modern models do, and because it makes the KV-cache arithmetic in
# chapter 3 non-trivial.
REFERENCE_MODEL = ModelConfig(
    vocab_size=260,
    n_layers=6,
    n_heads=8,
    n_kv_heads=2,
    head_dim=32,
    ffn_mult=4,
    max_seq_len=2048,
    dtype="fp32",
    seed=0,
)


@dataclass(frozen=True)
class EngineConfig:
    """How the engine schedules and caches. Every field here is a knob some chapter turns."""

    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = "cpu"
    seed: int = 0
