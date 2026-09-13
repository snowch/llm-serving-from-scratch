"""Train per-tenant adapters, so chapter 21 can show that an adapter actually does something.

An untrained adapter is the identity by construction, which makes it useless as evidence: you
cannot tell a working multi-adapter server from a broken one if every adapter is a no-op. So each
tenant gets its own corpus and its own brief training run, and the chapter then checks the thing
that matters — that an adapter helps *its own* tenant's text and not the other tenant's.

Each tenant's corpus comes from a different seed of the chapter 15 word process, which gives a
genuinely different vocabulary over the same syllable inventory. That is the right shape for this
measurement: the tenants' domains are distinct but related, as two customers of one base model
would be, rather than two unrelated languages that a single adapter could never span.
"""

from __future__ import annotations

import torch
from torch import nn

from bench.train_tiny import evaluate_perplexity, make_corpus
from llmserve.lora import LoRAAdapter, LoRALinear, attach, make_adapter, set_active
from llmserve.model import TinyGPT

#: Seeds for the tenant corpora. Seed 0 is the base model's own corpus (chapter 15), so tenants
#: start at 1: an adapter trained on the data the base already saw would show nothing.
TENANT_SEEDS = {"alice": 1, "bob": 2}


def tenant_data(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One tenant's corpus, split into train and held-out halves of bytes."""
    corpus = torch.tensor(list(make_corpus(seed=seed).encode("utf-8")), dtype=torch.long)
    split = int(0.9 * len(corpus))
    return corpus[:split], corpus[split:]


def train_adapter(
    model: TinyGPT,
    layers: dict[str, LoRALinear],
    adapter: LoRAAdapter,
    data: torch.Tensor,
    *,
    steps: int = 300,
    batch_size: int = 16,
    seq_len: int = 128,
    lr: float = 5e-3,
    seed: int = 0,
    log_every: int = 100,
) -> float:
    """Train one adapter on one tenant's text, leaving the base weights alone.

    Freezing the base is not an optimisation, it is the definition: what makes an adapter cheap to
    store *and* cheap to swap is that it is the only thing that changed. A training loop that
    quietly updated the shared weights would produce better perplexity and a model that can no
    longer serve anyone else.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for tensor in adapter.parameters():
        tensor.requires_grad_(True)

    set_active(layers, adapter.name)
    optimiser = torch.optim.AdamW(adapter.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    vocab = model.cfg.vocab_size
    loss = torch.tensor(float("nan"))

    model.train()
    for step in range(steps):
        starts = torch.randint(len(data) - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([data[s : s + seq_len] for s in starts])
        y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])

        with torch.enable_grad():
            logits, _ = model._forward(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimiser.step()

        if log_every and step % log_every == 0:
            print(f"    {adapter.name} step {step:>4}  loss {loss.item():.4f}")

    model.eval()
    for tensor in adapter.parameters():
        tensor.requires_grad_(False)
    set_active(layers, None)
    return float(loss.item())


def train_tenant_adapters(
    model: TinyGPT, *, steps: int = 300, rank: int | None = None
) -> tuple[dict[str, LoRAAdapter], dict[str, LoRALinear]]:
    """Build and train one adapter per tenant, all sharing the one base model."""
    from llmserve.lora import LoRAConfig

    config = LoRAConfig(rank=rank) if rank else LoRAConfig()
    adapters = {
        name: make_adapter(model, name, config, seed=seed) for name, seed in TENANT_SEEDS.items()
    }
    layers = attach(model, list(adapters.values()))
    for name, adapter in adapters.items():
        train, _ = tenant_data(TENANT_SEEDS[name])
        train_adapter(model, layers, adapter, train, steps=steps, seed=TENANT_SEEDS[name])
    return adapters, layers


def perplexity_matrix(
    model: TinyGPT, layers: dict[str, LoRALinear], adapters: dict[str, LoRAAdapter]
) -> dict[str, dict[str, float]]:
    """Every adapter evaluated against every tenant's held-out text, plus the base model.

    The diagonal is the claim (an adapter helps its own tenant). The off-diagonal is the check that
    the claim means anything: an adapter that improved *everyone* would just be a better base model
    and would say nothing about specialisation.
    """
    scores: dict[str, dict[str, float]] = {}
    for tenant, seed in TENANT_SEEDS.items():
        _, held_out = tenant_data(seed)
        row = {}
        set_active(layers, None)
        row["base"] = evaluate_perplexity(model, held_out)
        for name in adapters:
            set_active(layers, name)
            row[name] = evaluate_perplexity(model, held_out)
        set_active(layers, None)
        scores[tenant] = row
    return scores
