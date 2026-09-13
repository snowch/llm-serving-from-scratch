"""Chapter 15: speculation must be fast *and* produce the target model's distribution."""

import pytest
import torch

from llmserve.config import REFERENCE_MODEL
from llmserve.model import build_model
from llmserve.sampling import SamplingParams
from llmserve.speculative import (
    NgramDrafter,
    SpeculationStats,
    accept_greedy,
    accept_sampled,
    speculative_generate,
)

PROMPT = list(b"the server reads memory because decode is bound. the server reads")


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return build_model(REFERENCE_MODEL)


def _greedy_reference(model, prompt, n):
    tokens, output = list(prompt), []
    for _ in range(n):
        logits, _ = model(torch.tensor([tokens]))
        token = int(logits[0, -1].argmax())
        output.append(token)
        tokens.append(token)
    return output


# -- the drafter -----------------------------------------------------------------------


def test_ngram_drafter_copies_what_followed_the_last_match():
    drafter = NgramDrafter(n=3)
    assert drafter.propose([1, 2, 3, 9, 9, 1, 2, 3], k=2) == [9, 9]


def test_ngram_drafter_proposes_nothing_without_a_match():
    assert NgramDrafter(n=3).propose([1, 2, 3, 4, 5, 6, 7], k=3) == []


def test_ngram_drafter_needs_enough_context():
    assert NgramDrafter(n=3).propose([1, 2], k=3) == []


# -- acceptance rules ------------------------------------------------------------------


def test_greedy_acceptance_stops_at_the_first_disagreement():
    # Positions 0 and 1 favour token 5; position 2 favours token 7.
    logits = torch.full((3, 8), -10.0)
    logits[0, 5] = logits[1, 5] = logits[2, 7] = 10.0
    accepted, corrected = accept_greedy(logits, [5, 3])
    assert accepted == [5]
    assert corrected == 5, "the correction is the target's own token at the mismatch"


def test_greedy_acceptance_of_everything_yields_a_bonus_token():
    logits = torch.full((3, 8), -10.0)
    logits[:, 5] = 10.0
    accepted, corrected = accept_greedy(logits, [5, 5])
    assert accepted == [5, 5]
    assert corrected == 5, "all proposals accepted, so the extra position is free"


def test_a_round_always_produces_at_least_one_token():
    """Speculation can never yield fewer tokens per pass than not speculating."""
    logits = torch.full((3, 8), -10.0)
    logits[:, 1] = 10.0
    accepted, corrected = accept_greedy(logits, [7, 7])
    assert accepted == []
    assert corrected is not None


# -- the correctness claim --------------------------------------------------------------


def test_greedy_speculation_is_token_identical(model):
    """The whole claim, at several proposal depths."""
    reference = _greedy_reference(model, PROMPT, 12)
    for k in (1, 2, 4, 8):
        assert (
            speculative_generate(model, NgramDrafter(3), PROMPT, SamplingParams(max_tokens=12), k=k)
            == reference
        )


def test_residual_rule_puts_the_missing_mass_back():
    """The acceptance rule on a distribution we can check by hand.

    Draft is a point mass on token 0. Accepting with probability p_target(0) and otherwise drawing
    from the residual must reproduce p_target exactly — which means token 0 appears with its own
    probability, not more.
    """
    torch.manual_seed(0)
    vocab = 4
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0]])
    target_probs = logits[0].softmax(dim=-1)
    draft = torch.zeros(1, vocab)
    draft[0, 0] = 1.0

    counts = torch.zeros(vocab)
    trials = 20_000
    for seed in range(trials):
        generator = torch.Generator().manual_seed(seed)
        accepted, corrected = accept_sampled(
            logits, draft, [0], SamplingParams(temperature=1.0), generator
        )
        counts[accepted[0] if accepted else corrected] += 1

    empirical = counts / trials
    assert torch.allclose(empirical, target_probs, atol=0.02), (
        f"speculative sampling must reproduce the target distribution; "
        f"got {empirical.tolist()} against {target_probs.tolist()}"
    )


def test_stats_report_tokens_per_round():
    stats = SpeculationStats(proposed=8, accepted=6, rounds=2)
    assert stats.acceptance_rate == 0.75
    assert stats.tokens_per_round == 4.0  # 6 accepted + 2 corrections over 2 rounds
