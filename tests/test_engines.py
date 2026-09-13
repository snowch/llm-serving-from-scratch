"""Equivalence tests: the discipline that makes the rest of the book's claims safe.

Every optimisation from chapter 5 onward must leave output unchanged. Asserting that here, once
per optimisation, is what lets later chapters claim a speedup without also claiming a regression
nobody checked for.
"""

import pytest
import torch

from llmserve.config import REFERENCE_MODEL
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
