"""Chapter 18: the grammar must make invalid output unreachable — and only claim that much."""

import pytest
import torch

from llmserve.constrain import JSONGrammar, State, is_valid_json_object


@pytest.fixture
def grammar():
    return JSONGrammar(vocab_size=260, eos_id=257)


def _walk(grammar, text: str) -> State:
    """Walk the machine, asserting every character was permitted."""
    state = State.START
    for char in text:
        assert ord(char) in grammar.allowed_bytes(state), f"{char!r} rejected in {state}"
        state = grammar.step(state, ord(char))
    return state


@pytest.mark.parametrize(
    "document",
    ['{"a": "b"}', '{"a": "b", "c": "d"}', "{}", '{"key with spaces": "value"}'],
)
def test_valid_documents_walk_to_an_accepting_state(grammar, document):
    assert _walk(grammar, document) is State.DONE
    assert is_valid_json_object(document)


def test_illegal_continuations_are_forbidden(grammar):
    after_brace = grammar.step(State.START, ord("{"))
    assert ord(",") not in grammar.allowed_bytes(after_brace)
    assert ord('"') in grammar.allowed_bytes(after_brace)


def test_a_key_must_be_followed_by_a_colon(grammar):
    state = _walk(grammar, '{"a"')
    assert state is State.EXPECT_COLON
    assert grammar.allowed_bytes(state) == {ord(":"), ord(" ")}


def test_masking_leaves_only_legal_tokens(grammar):
    logits = torch.randn(260)
    masked = grammar.apply(logits, State.START)
    legal = (masked > torch.finfo(logits.dtype).min).nonzero().flatten().tolist()
    assert legal == [ord("{")], "only an opening brace can start a JSON object"


def test_masking_never_produces_nan(grammar):
    """An all -inf row would softmax to NaN and poison the batch (the ch06 lesson)."""
    logits = torch.randn(260)
    for state in State:
        probs = grammar.apply(logits, state).softmax(dim=-1)
        assert torch.isfinite(probs).all(), f"{state} produced non-finite probabilities"


def test_masks_are_cached_per_state(grammar):
    for _ in range(10):
        grammar.mask(State.IN_VALUE)
    assert grammar.masks_built == 1


def test_uncached_masks_are_rebuilt_every_time(grammar):
    for _ in range(10):
        grammar.mask(State.IN_VALUE, cached=False)
    assert grammar.masks_built == 10


def test_a_valid_prefix_is_not_a_valid_document():
    """The chapter's central caveat, as an assertion.

    Every character below is legal, and the result does not parse. A grammar constrains shape, not
    length, so truncation defeats the guarantee people think they are buying.
    """
    truncated = '{"a": "b'
    assert not is_valid_json_object(truncated)
