"""Speculative decoding: breaking the one-token-at-a-time serialisation.

Chapter 15. Decode is serial by construction — token *n+1* depends on token *n* — so at low batch
the device spends most of its time waiting on a dependency rather than on arithmetic. Chapter 3
explains why that is so wasteful: the step is bound by reading the weights, and reading them to
produce one token is the worst possible ratio.

Speculation breaks the serialisation without changing the answer. Something cheap proposes *k*
tokens; the real model verifies all of them in **one** forward pass, because verification is a
prefill-shaped operation and prefill is parallel. Accepted tokens cost nothing extra.

The part that makes this respectable rather than a heuristic is that it is **exact**. With the
acceptance rule below, the tokens you get are drawn from precisely the distribution the target
model would have produced on its own. Not similar — the same. A speculative engine that is
"almost" the target model is a different model, and nobody would be able to tell which one
answered.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from llmserve.model import TinyGPT
from llmserve.sampling import SamplingParams


class Drafter:
    """Proposes candidate continuations. Cheap is the only requirement."""

    name = "drafter"

    def propose(self, token_ids: list[int], k: int) -> list[int]:
        raise NotImplementedError


class NgramDrafter(Drafter):
    """Proposes by finding where this context appeared before and copying what followed.

    No second model, no training, no memory to speak of. It works precisely when the output
    repeats its input — summarising a document, editing code, answering from a quoted context —
    which is a large slice of real traffic. Where nothing repeats it proposes nothing useful, and
    the acceptance rate simply falls to zero.
    """

    name = "ngram"

    def __init__(self, n: int = 3) -> None:
        self.n = n

    def propose(self, token_ids: list[int], k: int) -> list[int]:
        if len(token_ids) < self.n + 1:
            return []
        pattern = token_ids[-self.n :]
        # Search backwards: the most recent match is the most likely continuation.
        for start in range(len(token_ids) - self.n - 1, -1, -1):
            if token_ids[start : start + self.n] == pattern:
                return token_ids[start + self.n : start + self.n + k]
        return []


class ModelDrafter(Drafter):
    """Proposes with a smaller model of the same shape.

    The canonical form. The draft model must share the target's vocabulary; everything else about
    it is a trade between how fast it runs and how often it agrees.
    """

    name = "model"

    def __init__(self, model: TinyGPT) -> None:
        self.model = model

    def propose(self, token_ids: list[int], k: int) -> list[int]:
        ids = torch.tensor([token_ids], dtype=torch.long)
        logits, past = self.model(ids)
        proposed = []
        for _ in range(k):
            token = int(logits[:, -1, :].argmax(dim=-1).item())
            proposed.append(token)
            position = torch.tensor([[len(token_ids) + len(proposed) - 1]], dtype=torch.long)
            logits, past = self.model(torch.tensor([[token]], dtype=torch.long), past, position)
        return proposed


@dataclass
class SpeculationStats:
    """Acceptance accounting. The one number that decides whether speculation is worth it."""

    proposed: int = 0
    accepted: int = 0
    rounds: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0

    @property
    def tokens_per_round(self) -> float:
        """Mean tokens produced per target forward pass — the actual speedup factor."""
        return (self.accepted + self.rounds) / self.rounds if self.rounds else 0.0


def accept_greedy(target_logits: torch.Tensor, proposed: list[int]) -> tuple[list[int], int]:
    """Accept proposals while they match what the target would have chosen.

    With temperature at zero the target is deterministic, so "would the target have said this?"
    is just a comparison. Accept the matching prefix, stop at the first disagreement, and take the
    target's own token there.

    Returns the accepted tokens plus one correction or bonus token — so a round always produces at
    least one token, and speculation can never be slower than not speculating in token count.
    """
    accepted: list[int] = []
    for i, token in enumerate(proposed):
        target_token = int(target_logits[i].argmax(dim=-1).item())
        if target_token != token:
            return accepted, target_token
        accepted.append(token)
    # Every proposal matched: the extra position gives a free bonus token.
    return accepted, int(target_logits[len(proposed)].argmax(dim=-1).item())


def accept_sampled(
    target_logits: torch.Tensor,
    draft_probs: torch.Tensor,
    proposed: list[int],
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> tuple[list[int], int]:
    """Accept proposals by rejection sampling, preserving the target distribution exactly.

    For each proposed token ``x``, accept with probability ``min(1, p_target(x) / p_draft(x))``.
    On rejection, resample from the normalised positive part of ``p_target - p_draft``.

    That residual is the whole trick, and it is worth seeing why it works. Proposing from
    ``p_draft`` and accepting at that ratio leaves token ``x`` under-sampled by exactly
    ``max(0, p_target(x) - p_draft(x))``. Drawing the replacement from precisely that shortfall
    puts the missing mass back where it belongs, so the combined procedure samples from
    ``p_target`` — not approximately, exactly.

    Drop the residual and resample from ``p_target`` instead and the result is subtly biased
    towards tokens the draft model liked. It still produces fluent text, which is what makes the
    bug hard to notice.
    """
    accepted: list[int] = []
    temperature = max(params.temperature, 1e-6)

    for i, token in enumerate(proposed):
        target_probs = (target_logits[i] / temperature).softmax(dim=-1)
        p_target = target_probs[token]
        p_draft = draft_probs[i][token]

        ratio = (p_target / p_draft.clamp(min=1e-10)).clamp(max=1.0)
        if torch.rand(1, generator=generator).item() < ratio.item():
            accepted.append(token)
            continue

        residual = (target_probs - draft_probs[i]).clamp(min=0)
        total = residual.sum()
        if total <= 0:
            corrected = int(torch.multinomial(target_probs, 1, generator=generator).item())
        else:
            corrected = int(torch.multinomial(residual / total, 1, generator=generator).item())
        return accepted, corrected

    bonus_probs = (target_logits[len(proposed)] / temperature).softmax(dim=-1)
    return accepted, int(torch.multinomial(bonus_probs, 1, generator=generator).item())


def speculative_generate(
    target: TinyGPT,
    drafter: Drafter,
    prompt_token_ids: list[int],
    params: SamplingParams,
    k: int = 4,
    generator: torch.Generator | None = None,
    stats: SpeculationStats | None = None,
) -> list[int]:
    """Generate with speculation, verifying each round in a single target forward pass.

    Deliberately written without a KV cache for the target. Caching across speculative rounds is
    fiddly — a rejected proposal has to be rolled back out of the cache — and it would obscure the
    part of this that is worth understanding. Chapter 15's measurement therefore counts *target
    forward passes*, which is the quantity speculation reduces, rather than wall-clock.
    """
    stats = stats if stats is not None else SpeculationStats()
    tokens = list(prompt_token_ids)
    output: list[int] = []

    while len(output) < params.max_tokens:
        proposed = drafter.propose(tokens, k)

        draft_probs = None
        if not params.greedy and proposed:
            # The draft distribution is needed for the acceptance ratio. A model drafter would
            # return it directly; for drafters that only emit tokens (n-gram) we treat the
            # proposal as certain, which makes the ratio min(1, p_target) and stays exact.
            draft_probs = torch.zeros(len(proposed), target.cfg.vocab_size)
            for i, token in enumerate(proposed):
                draft_probs[i, token] = 1.0

        ids = torch.tensor([tokens + proposed], dtype=torch.long)
        logits, _ = target(ids)
        # Positions that predict the proposals, plus one extra for the bonus token.
        verify_logits = logits[0, len(tokens) - 1 :, :]

        if params.greedy:
            accepted, corrected = accept_greedy(verify_logits, proposed)
        else:
            accepted, corrected = accept_sampled(
                verify_logits, draft_probs, proposed, params, generator
            )

        stats.proposed += len(proposed)
        stats.accepted += len(accepted)
        stats.rounds += 1

        for token in [*accepted, corrected]:
            if len(output) >= params.max_tokens:
                break
            output.append(token)
            tokens.append(token)
            if token in params.stop_token_ids:
                return output

    return output[: params.max_tokens]
