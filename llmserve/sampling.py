"""Turning logits into tokens.

Every sampler here is written out rather than delegated, because chapter 4's point is that these
are the decisions ``model.generate()`` makes on your behalf — and a serving engine has to own
them to be reproducible, to stream correctly, and to stay comparable across the optimisations in
the rest of the book.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class SamplingParams:
    """What the caller asks for. Defaults are greedy, which is what the tests compare against."""

    max_tokens: int = 32
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stop_token_ids: tuple[int, ...] = field(default_factory=tuple)
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be positive")


def apply_repetition_penalty(
    logits: torch.Tensor, prev_ids: torch.Tensor, penalty: float
) -> torch.Tensor:
    """Divide the logit of every already-seen token, or multiply if it is negative.

    The asymmetry looks odd but is the standard formulation: dividing a negative logit would make
    the token *more* likely, which is the opposite of a penalty.
    """
    if penalty == 1.0 or prev_ids.numel() == 0:
        return logits
    seen = logits.gather(-1, prev_ids)
    seen = torch.where(seen > 0, seen / penalty, seen * penalty)
    return logits.scatter(-1, prev_ids, seen)


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the k highest logits, mask the rest."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    threshold = logits.topk(k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: keep the smallest set of tokens whose probability mass reaches p."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    # Shift so the token that crosses the threshold is itself kept.
    remove = cumulative - sorted_logits.softmax(dim=-1) > p
    remove_original = remove.scatter(-1, sorted_idx, remove)
    return logits.masked_fill(remove_original, float("-inf"))


def apply_min_p(logits: torch.Tensor, min_p: float) -> torch.Tensor:
    """Keep tokens at least ``min_p`` times as probable as the most likely one.

    Unlike top-p this adapts to how confident the distribution is: a peaked distribution keeps
    almost nothing, a flat one keeps a lot.
    """
    if min_p <= 0.0:
        return logits
    probs = logits.softmax(dim=-1)
    threshold = probs.max(dim=-1, keepdim=True).values * min_p
    return logits.masked_fill(probs < threshold, float("-inf"))


def sample(
    logits: torch.Tensor,
    params: SamplingParams,
    *,
    prev_ids: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Pick one token per row of ``logits`` ([batch, vocab]).

    Order matters and is the conventional one: penalties act on raw logits, temperature rescales,
    then the truncation filters narrow the field. Applying temperature after truncation would
    change which tokens survive, which is a subtle way for two implementations to disagree while
    both looking correct.
    """
    if logits.dim() != 2:
        raise ValueError(f"expected [batch, vocab] logits, got shape {tuple(logits.shape)}")

    if prev_ids is not None and params.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, prev_ids, params.repetition_penalty)

    if params.greedy:
        return logits.argmax(dim=-1)

    logits = logits / params.temperature
    logits = apply_top_k(logits, params.top_k)
    logits = apply_min_p(logits, params.min_p)
    logits = apply_top_p(logits, params.top_p)

    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
