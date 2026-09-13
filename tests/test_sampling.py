"""Chapter 4's samplers, written out rather than delegated, so they need testing rather than trust."""

import pytest
import torch

from llmserve.sampling import (
    SamplingParams,
    apply_min_p,
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    sample,
)

LOGITS = torch.tensor([[3.0, 2.0, 1.0, 0.5, -1.0]])


def _kept(masked: torch.Tensor) -> int:
    return int((~masked.isinf()).sum())


def test_greedy_picks_the_argmax():
    assert sample(LOGITS, SamplingParams()).tolist() == [0]


def test_top_k_keeps_exactly_k():
    assert _kept(apply_top_k(LOGITS, 2)) == 2


def test_top_k_is_a_noop_when_k_covers_the_vocabulary():
    assert torch.equal(apply_top_k(LOGITS, 99), LOGITS)


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    """Nucleus sampling includes the token that takes cumulative mass past p, not excludes it."""
    kept = _kept(apply_top_p(LOGITS, 0.7))
    probs = LOGITS.softmax(-1).sort(descending=True).values[0]
    assert probs[: kept - 1].sum() < 0.7 <= probs[:kept].sum()


def test_min_p_adapts_to_confidence():
    """A peaked distribution keeps fewer tokens than a flat one at the same min_p."""
    flat = torch.tensor([[1.0, 0.9, 0.8, 0.7, 0.6]])
    assert _kept(apply_min_p(LOGITS, 0.5)) < _kept(apply_min_p(flat, 0.5))


def test_repetition_penalty_lowers_seen_tokens_including_negative_logits():
    """Dividing a negative logit would raise its probability, so the sign must be handled."""
    seen = torch.tensor([[0]])
    positive = apply_repetition_penalty(LOGITS.clone(), seen, 2.0)
    assert positive[0, 0] < LOGITS[0, 0]

    negative = torch.tensor([[-2.0, 1.0]])
    penalised = apply_repetition_penalty(negative.clone(), seen, 2.0)
    assert penalised[0, 0] < negative[0, 0]


def test_sampling_is_reproducible_for_a_fixed_seed():
    params = SamplingParams(temperature=1.0, seed=1234)
    draws = []
    for _ in range(2):
        generator = torch.Generator().manual_seed(1234)
        draws.append(sample(LOGITS, params, generator=generator).tolist())
    assert draws[0] == draws[1]


def test_rejects_wrongly_shaped_logits():
    with pytest.raises(ValueError, match=r"\[batch, vocab\]"):
        sample(torch.zeros(1, 2, 5), SamplingParams())


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_tokens": 0}, "max_tokens"),
        ({"top_p": 0.0}, "top_p"),
        ({"repetition_penalty": 0.0}, "repetition_penalty"),
    ],
)
def test_invalid_parameters_are_rejected_at_construction(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SamplingParams(**kwargs)
