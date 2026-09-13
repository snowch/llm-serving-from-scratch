"""Low-rank adapters, and what it costs to serve many of them at once.

Chapter 21. A tenant wants a model fine-tuned on their data. Giving each tenant a full copy of the
weights is the obvious implementation and is immediately unaffordable: at production scale one copy
is tens of gigabytes, and the whole point of a shared serving fleet is that tenants share the
expensive part.

LoRA changes the arithmetic. A fine-tune is expressed as a low-rank update to selected projections
— two thin matrices instead of a dense one — so a tenant's weights become a rounding error next to
the base model. The base stays shared, resident once, and every tenant borrows it.

That is the memory story, and it is the easy half. The hard half is serving: a batch drawn from
several tenants wants a *different* delta for each row, and one GEMM cannot do that. Everything
interesting in multi-tenant serving follows from that single fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from llmserve.config import BYTES_PER_DTYPE, ModelConfig


@dataclass(frozen=True)
class LoRAConfig:
    """Which projections carry an adapter, and how much capacity it gets.

    ``rank`` is the whole trade: capacity against size. ``alpha`` rescales the update so that
    changing the rank does not also change its effective magnitude — without it, raising the rank
    quietly raises the learning rate too.

    Targeting the query and value projections is the convention from the LoRA paper, and it is not
    arbitrary: it is where the original ablation found most of the benefit per parameter.
    """

    rank: int = 8
    alpha: float = 16.0
    targets: tuple[str, ...] = ("q_proj", "v_proj")

    @property
    def scale(self) -> float:
        return self.alpha / self.rank


@dataclass
class LoRAAdapter:
    """One tenant's fine-tune: two thin matrices per targeted projection."""

    name: str
    config: LoRAConfig = field(default_factory=LoRAConfig)
    #: projection name -> (A of shape (rank, in_features), B of shape (out_features, rank))
    weights: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    @property
    def n_parameters(self) -> int:
        return sum(a.numel() + b.numel() for a, b in self.weights.values())

    def n_bytes(self, dtype: str = "fp16") -> float:
        return self.n_parameters * BYTES_PER_DTYPE[dtype]

    def parameters(self) -> list[torch.Tensor]:
        return [t for pair in self.weights.values() for t in pair]


def make_adapter(model: nn.Module, name: str, config: LoRAConfig | None = None, seed: int = 0):
    """Build an adapter shaped to fit ``model``'s targeted projections.

    ``B`` starts at zero, so the adapter is exactly the identity before training. That matters more
    than it looks: an adapter that perturbs the model at initialisation makes the first training
    steps fight damage they caused themselves, and makes an untrained adapter indistinguishable
    from a broken one.
    """
    config = config or LoRAConfig()
    generator = torch.Generator().manual_seed(seed)
    adapter = LoRAAdapter(name=name, config=config)
    for module_name, module in model.named_modules():
        if module_name.rsplit(".", 1)[-1] not in config.targets:
            continue
        if not isinstance(module, nn.Linear):
            continue
        a = torch.randn(config.rank, module.in_features, generator=generator) / config.rank**0.5
        b = torch.zeros(module.out_features, config.rank)
        adapter.weights[module_name] = (a, b)
    if not adapter.weights:
        raise ValueError(f"no projections named {config.targets} found in this model")
    return adapter


class LoRALinear(nn.Module):
    """A shared projection that can apply a different low-rank delta to each row of a batch.

    The single-adapter path is one extra pair of small matrix multiplies and is essentially free.
    The mixed path is the interesting one, and it is deliberately written as the naive loop so the
    cost is visible: rows are grouped by adapter and each group takes its own pass. That loop is
    what S-LoRA and Punica exist to replace with a batched kernel, and chapter 21 measures what it
    costs before reaching for one.
    """

    def __init__(self, base: nn.Linear) -> None:
        super().__init__()
        self.base = base
        self.adapters: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.scale: float = 1.0
        #: the adapter every row uses, when the batch is homogeneous
        self.active: str | None = None
        #: one adapter name (or None) per row, when it is not
        self.row_adapters: list[str | None] | None = None

    def _delta(self, x: torch.Tensor, name: str) -> torch.Tensor:
        a, b = self.adapters[name]
        return (x @ a.T) @ b.T * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.row_adapters is None:
            if self.active is not None:
                out = out + self._delta(x, self.active)
            return out

        # Mixed batch. Distinct adapters mean distinct weight matrices, so there is no single GEMM
        # that serves the batch: one pass per adapter present, over only the rows that want it.
        for name in dict.fromkeys(n for n in self.row_adapters if n is not None):
            rows = torch.tensor(
                [i for i, n in enumerate(self.row_adapters) if n == name], dtype=torch.long
            )
            out = out.index_add(0, rows, self._delta(x.index_select(0, rows), name))
        return out


def attach(model: nn.Module, adapters: list[LoRAAdapter]) -> dict[str, LoRALinear]:
    """Wrap the targeted projections so the model can serve any of ``adapters``.

    The base weights are not touched and not copied: every adapter borrows the same tensors. This
    is the whole memory argument, expressed as code — ``n`` tenants cost one model plus ``n``
    small dictionaries.
    """
    if not adapters:
        raise ValueError("attach needs at least one adapter")
    scale = adapters[0].config.scale
    if any(a.config.scale != scale for a in adapters):
        raise ValueError("adapters attached together must share alpha/rank scaling")

    wrapped: dict[str, LoRALinear] = {}
    for module_name in adapters[0].weights:
        parent_name, _, leaf = module_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        base = getattr(parent, leaf)
        layer = base if isinstance(base, LoRALinear) else LoRALinear(base)
        layer.scale = scale
        for adapter in adapters:
            if module_name in adapter.weights:
                layer.adapters[adapter.name] = adapter.weights[module_name]
        setattr(parent, leaf, layer)
        wrapped[module_name] = layer
    return wrapped


def detach(model: nn.Module) -> None:
    """Put the plain projections back, leaving the model exactly as it was."""
    for module in list(model.modules()):
        for leaf, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, leaf, child.base)


def set_active(layers: dict[str, LoRALinear], name: str | None) -> None:
    """Serve one adapter for the whole batch — the homogeneous fast path."""
    for layer in layers.values():
        layer.active = name
        layer.row_adapters = None


def set_row_adapters(layers: dict[str, LoRALinear], names: list[str | None]) -> None:
    """Serve a batch whose rows belong to different tenants."""
    for layer in layers.values():
        layer.active = None
        layer.row_adapters = list(names)


@torch.no_grad()
def merge(layers: dict[str, LoRALinear], name: str) -> None:
    """Fold one adapter into the base weights, in place.

    ``W + scale * B @ A`` is a dense matrix of exactly the base's shape, so after merging there is
    no adapter left to apply and no per-row work at all. It is the right answer for a single
    tenant and unavailable for many: the merge destroys the shared base that every other tenant
    was borrowing.
    """
    for layer in layers.values():
        a, b = layer.adapters[name]
        layer.base.weight.add_(layer.scale * (b @ a))
        layer.adapters.pop(name)
    set_active(layers, None)


def adapter_bytes(model: ModelConfig, config: LoRAConfig, n_layers: int | None = None) -> float:
    """Bytes one adapter occupies, from the shapes alone.

    Separate from :meth:`LoRAAdapter.n_bytes` because chapter 21 needs this for model sizes it
    cannot build — the argument for LoRA is strongest exactly where the base model does not fit on
    a laptop.
    """
    layers = n_layers if n_layers is not None else model.n_layers
    d = model.d_model
    kv_dim = model.n_kv_heads * model.head_dim
    per_target = {"q_proj": (d, d), "k_proj": (kv_dim, d), "v_proj": (kv_dim, d), "o_proj": (d, d)}
    total = 0
    for target in config.targets:
        out_features, in_features = per_target[target]
        total += config.rank * (in_features + out_features)
    return total * layers * BYTES_PER_DTYPE[model.dtype]
