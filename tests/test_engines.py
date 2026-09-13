"""Equivalence tests: the discipline that makes the rest of the book's claims safe.

Every optimisation from chapter 5 onward must leave output unchanged. Asserting that here, once
per optimisation, is what lets later chapters claim a speedup without also claiming a regression
nobody checked for.
"""

import pytest
import torch

from llmserve.config import REFERENCE_MODEL
from llmserve.engines.batched import ContinuousBatchEngine, StaticBatchEngine
from llmserve.engines.naive import CachedEngine, NaiveEngine
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams

PROMPT = list(b"The quick brown fox jumps over")


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return build_model(REFERENCE_MODEL)


def drain(engine, prompt=PROMPT, **params) -> list[int]:
    params.setdefault("max_tokens", 12)
    engine.add_request(Request(prompt_token_ids=list(prompt), params=SamplingParams(**params)))
    tokens: list[int] = []
    while engine.has_work():
        for out in engine.step():
            tokens.extend(out.token_ids)
    return tokens


def test_kv_cache_does_not_change_output(model):
    """Chapter 5's correctness claim: the cache is an optimisation, not a different model."""
    assert drain(NaiveEngine(model)) == drain(CachedEngine(model))


def test_engine_produces_exactly_max_tokens(model):
    assert len(drain(CachedEngine(model), max_tokens=7)) == 7


def test_stop_token_ends_generation_early(model):
    """Run once to learn the first token, then make it a stop token and check we stop there."""
    first = drain(CachedEngine(model), max_tokens=5)[0]
    stopped = drain(CachedEngine(model), max_tokens=5, stop_token_ids=(first,))
    assert stopped == [first]


def test_requests_are_served_in_arrival_order(model):
    engine = CachedEngine(model)
    for _ in range(3):
        engine.add_request(
            Request(prompt_token_ids=list(PROMPT), params=SamplingParams(max_tokens=3))
        )
    order: list[int] = []
    while engine.has_work():
        for out in engine.step():
            if out.finished:
                order.append(out.request_id)
    assert order == sorted(order)


def test_engine_reports_no_work_when_drained(model):
    engine = CachedEngine(model)
    assert not engine.has_work()
    drain(engine, max_tokens=2)
    assert not engine.has_work()


# --- Part II/III: batching must not change any output token ---


@pytest.fixture
def mixed_specs():
    """Deliberately unequal prompts and budgets.

    Equal lengths would batch without padding and hide every bug this file exists to catch.
    """
    return [
        (list(b"The quick brown fox jumps"), 8),
        (list(b"Hello"), 4),
        (list(b"A somewhat longer prompt that forces left padding in the batch"), 12),
    ]


def _serve_individually(model, specs) -> list[list[int]]:
    return [drain(CachedEngine(model), prompt=p, max_tokens=n) for p, n in specs]


def _serve_batched(model, engine_cls, specs) -> list[list[int]]:
    engine = engine_cls(model)
    outputs: dict[int, list[int]] = {}
    for prompt, n in specs:
        request = Request(prompt_token_ids=list(prompt), params=SamplingParams(max_tokens=n))
        outputs[request.request_id] = []
        engine.add_request(request)
    while engine.has_work():
        for out in engine.step():
            outputs[out.request_id].extend(out.token_ids)
    return list(outputs.values())


@pytest.mark.parametrize("engine_cls", [StaticBatchEngine, ContinuousBatchEngine])
def test_batching_does_not_change_output(model, mixed_specs, engine_cls):
    """ch06/ch07: batching is a scheduling change, not a different model."""
    assert _serve_batched(model, engine_cls, mixed_specs) == _serve_individually(model, mixed_specs)


def test_padding_does_not_leak_between_sequences(model):
    """Regression test for a NaN that corrupted every sequence in a padded batch.

    A query at a padding position is masked everywhere it may attend, so softmax over a row of all
    -inf produced NaN. Real tokens weight padded positions at zero, but 0 * NaN is NaN, so one
    short sequence silently destroyed the whole batch. Masking with the dtype's most negative
    finite value instead keeps such a row harmless.

    The short sequence is the canary: before the fix it generated nothing but token 0.
    """
    specs = [(list(b"a" * 24), 6), (list(b"bbb"), 6)]
    batched = _serve_batched(model, StaticBatchEngine, specs)
    assert batched == _serve_individually(model, specs)
    assert len(set(batched[1])) > 1, "short padded sequence collapsed to a constant token"


def test_static_batching_wastes_slots_on_finished_sequences(model, mixed_specs):
    """ch06's measured problem: a finished sequence holds its slot until the batch drains."""
    engine = StaticBatchEngine(model)
    _serve_batched(model, StaticBatchEngine, mixed_specs)
    engine = StaticBatchEngine(model)
    for prompt, n in mixed_specs:
        engine.add_request(
            Request(prompt_token_ids=list(prompt), params=SamplingParams(max_tokens=n))
        )
    while engine.has_work():
        engine.step()
    assert engine.wasted_slot_steps > 0


def test_continuous_batching_wastes_no_slots(model, mixed_specs):
    """ch07's fix: a finished sequence leaves immediately, so no slot-step is ever discarded."""
    engine = ContinuousBatchEngine(model)
    for prompt, n in mixed_specs:
        engine.add_request(
            Request(prompt_token_ids=list(prompt), params=SamplingParams(max_tokens=n))
        )
    while engine.has_work():
        engine.step()
    assert engine.wasted_slot_steps == 0
