"""Multi-head latent attention: compressing the KV cache in the architecture (chapter 13).

Grouped-query attention shrinks the cache by giving several query heads one KV head. That is a
coarse dial — the sharing is all-or-nothing per group, and the floor is one KV head. MLA takes a
different route to the same goal: cache a single low-rank *latent* vector per token and reconstruct
each head's keys and values from it on the fly.

The saving is large and the reason is arithmetic rather than clever: a cache entry stops being
``2 x n_kv_heads x head_dim`` numbers and becomes ``d_latent`` of them, with ``d_latent`` chosen far
smaller than either. What it costs is a matrix multiply per layer to project back up, which is work
the memory-bound decode phase (chapter 3) has to spare.

This module is the projection and its arithmetic, checked against the attention it replaces. It is
deliberately not wired into ``TinyGPT``: MLA is a property of a model's *training*, not a serving
decision, and a randomly-initialised latent projection would tell you nothing about quality. What it
can tell you honestly is the shape of the trade, which is what chapter 13 uses it for.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from llmserve.config import BYTES_PER_DTYPE, ModelConfig


@dataclass(frozen=True)
class LatentConfig:
    """How far the cache is compressed.

    ``d_latent`` is the whole dial. DeepSeek's published configurations put it several times smaller
    than one head's worth of KV across all heads, which is where the order-of-magnitude saving comes
    from; setting it equal to ``2 * n_kv_heads * head_dim`` would reproduce today's cache exactly and
    save nothing, which is a useful sanity check rather than a configuration anyone would ship.
    """

    d_latent: int = 128

    def __post_init__(self) -> None:
        if self.d_latent < 1:
            raise ValueError("a latent of no dimensions caches nothing")


class LatentKV(nn.Module):
    """Down-project to a latent, cache that, and up-project back to keys and values.

    The cached thing is ``compress(x)``: one vector per token per layer, shared across every head.
    Everything the heads need is reconstructed from it, so the cache no longer scales with head
    count at all — which is the property that makes it a different technique from GQA rather than a
    more aggressive setting of it.
    """

    def __init__(self, model: ModelConfig, config: LatentConfig | None = None) -> None:
        super().__init__()
        self.cfg = model
        self.latent = config or LatentConfig()
        kv_dim = model.n_kv_heads * model.head_dim
        self.down = nn.Linear(model.d_model, self.latent.d_latent, bias=False)
        self.up_keys = nn.Linear(self.latent.d_latent, kv_dim, bias=False)
        self.up_values = nn.Linear(self.latent.d_latent, kv_dim, bias=False)

    def compress(self, hidden: torch.Tensor) -> torch.Tensor:
        """The only thing that gets cached: ``[batch, tokens, d_latent]``."""
        return self.down(hidden)

    def expand(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct keys and values, shaped as the attention path expects them."""
        batch, tokens, _ = latent.shape
        shape = (batch, tokens, self.cfg.n_kv_heads, self.cfg.head_dim)
        keys = self.up_keys(latent).view(shape).transpose(1, 2)
        values = self.up_values(latent).view(shape).transpose(1, 2)
        return keys, values

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.expand(self.compress(hidden))


def latent_bytes_per_token(model: ModelConfig, config: LatentConfig) -> float:
    """Bytes one token costs in the cache under MLA, across the whole model.

    Per *layer* — the same convention as :func:`~llmserve.arithmetic.kv_bytes_per_token`, and worth
    being explicit about because mixing the two conventions is an easy factor of ``n_layers``. There
    is no factor of two here: one latent is cached, not a key and a value.
    """
    return config.d_latent * model.n_layers * BYTES_PER_DTYPE[model.dtype]


def compression_ratio(model: ModelConfig, config: LatentConfig) -> float:
    """How much smaller the cache gets, against this model's grouped-query cache."""
    from llmserve.arithmetic import kv_bytes_per_token

    return kv_bytes_per_token(model) / latent_bytes_per_token(model, config)
