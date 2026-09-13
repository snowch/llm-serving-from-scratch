"""A small decoder-only transformer, defined in code rather than downloaded.

The book needs a model whose every dimension we control and whose weights are reproducible on
any machine. Defining one here means no download, no version drift, and a model small enough
that a laptop can serve it while still exercising every mechanism the engine implements:
grouped-query attention, a KV cache, batching, and eventually paging.

The weights are random, so the text it produces is meaningless. That is fine and deliberate.
Everything this book measures — latency, throughput, cache behaviour, scheduling — depends on
the *shape* of the computation, not on the values in the weights. Where output quality genuinely
matters (chapter 15), the book says so and uses a trained model instead.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from llmserve.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


def build_rope_cache(
    max_seq_len: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute rotary cos/sin tables.

    Rotary embeddings encode position by rotating query and key vectors, which means position
    lives in the *values* we cache rather than in a separate embedding added up front. That is
    what lets a cached key stay valid no matter where the sequence goes next, and it is why
    chapter 9 can share cache blocks between requests that start with the same prefix.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate x by the angles for its positions.

    ``x`` is [batch, heads, seq, head_dim]; ``cos``/``sin`` are [batch, seq, head_dim/2]. Positions
    are per sequence rather than shared, because a padded batch has sequences sitting at different
    offsets and giving them all the same positions is a silent correctness bug (ch06).
    """
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Causal self-attention with grouped-query attention.

    The KV projections are narrower than the Q projection by a factor of ``group_size``. That
    single asymmetry is the whole of GQA, and chapter 13 shows it shrinks the KV cache — and
    therefore raises the concurrency ceiling — by the same factor.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, kv_dim, bias=False)
        self.v_proj = nn.Linear(d, kv_dim, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        cfg = self.cfg
        bsz, seq_len, _ = x.shape

        q = self.q_proj(x).view(bsz, seq_len, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        present = (k, v)

        # Expand KV heads to match query heads. A real kernel reads the shared head directly
        # instead of materialising copies; we expand for clarity, and chapter 14 fixes it.
        if cfg.group_size > 1:
            k = k.repeat_interleave(cfg.group_size, dim=1)
            v = v.repeat_interleave(cfg.group_size, dim=1)

        total_len = k.shape[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(cfg.head_dim)

        # Mask with the dtype's most negative finite value rather than -inf.
        #
        # This is not fussiness. In a left-padded batch, a query at a padding position is masked
        # at every position it may attend to. softmax over a row of all -inf is NaN, that NaN
        # reaches V at the padding positions, and although real tokens give those positions a
        # weight of zero, 0 * NaN is NaN — so one sequence's padding silently corrupts every other
        # sequence in the batch. A finite floor makes such a row a harmless uniform distribution,
        # while real rows still underflow padded weights to exactly zero.
        neg = torch.finfo(scores.dtype).min

        # Causal mask. During decode seq_len is 1 and every cached position is visible, so the
        # mask is a no-op — which is exactly why decode is cheap in FLOPs and expensive in bytes.
        if seq_len > 1:
            causal = torch.ones(seq_len, total_len, dtype=torch.bool, device=x.device).tril(
                diagonal=total_len - seq_len
            )
            scores = scores.masked_fill(~causal, neg)

        # Padding mask. Batching sequences of different lengths means some cache slots hold
        # nothing; attending to them would blend one request's padding into another's output.
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[:, None, None, :], neg)

        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(bsz, seq_len, cfg.d_model)
        return self.o_proj(out), present


class MLP(nn.Module):
    """SwiGLU feed-forward, as used by Llama-family models."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.ffn_mult * cfg.d_model
        self.gate_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, present = self.attn(self.attn_norm(x), cos, sin, past_kv, valid_mask)
        x = x + attn_out
        return x + self.mlp(self.mlp_norm(x)), present


KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class TinyGPT(nn.Module):
    """The model the book serves.

    ``forward`` takes an optional ``past`` and returns the updated cache alongside the logits,
    which is the minimum interface a serving engine needs. Chapter 5 uses it to build a real KV
    cache; chapter 8 replaces the contiguous tensors with paged blocks.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        cos, sin = build_rope_cache(
            cfg.max_seq_len, cfg.head_dim, cfg.rope_theta, torch.device("cpu")
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.eval()

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        past: KVCache | None = None,
        positions: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Serving path: no autograd, so nothing accumulates a graph while generating."""
        return self._forward(input_ids, past, positions, valid_mask)

    def _forward(
        self,
        input_ids: torch.Tensor,
        past: KVCache | None = None,
        positions: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Return logits for every input position, plus the updated cache.

        Undecorated, so training can use it (ch15 needs a model whose output can get worse, and
        that means one that has been trained). Serving goes through ``forward``, which adds
        inference mode.

        ``positions`` lets the caller say where these tokens sit in their sequence. During decode
        that is the current length, not zero, and getting it wrong is a silent correctness bug
        rather than a crash — the model simply attends as though every token were at the start.
        """
        bsz, seq_len = input_ids.shape
        if positions is None:
            start = 0 if past is None else past[0][0].shape[2]
            positions = torch.arange(start, start + seq_len, device=input_ids.device)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0).expand(bsz, -1)

        cos = self.rope_cos[positions]
        sin = self.rope_sin[positions]

        x = self.embed(input_ids)
        present: KVCache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, cos, sin, None if past is None else past[i], valid_mask)
            present.append(kv)
        return self.lm_head(self.norm(x)), present


def build_model(cfg: ModelConfig | None = None) -> TinyGPT:
    """Construct the reference model. Seeded, so two machines get identical weights."""
    return TinyGPT(cfg or ModelConfig())
