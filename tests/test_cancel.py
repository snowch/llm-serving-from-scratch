"""Chapter 24: dropping a request the caller no longer wants."""

import pytest
import torch

from llmserve.config import REFERENCE_MODEL
from llmserve.engines.cancel import CancellableEngine
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return build_model(REFERENCE_MODEL)


@pytest.fixture
def engine(model):
    return CancellableEngine(model, REFERENCE_MODEL, max_batch_size=2, n_blocks=256, block_size=16)


def _request(max_tokens=16, prompt_len=32):
    return Request(
        prompt_token_ids=[(i % 250) + 1 for i in range(prompt_len)],
        params=SamplingParams(max_tokens=max_tokens),
    )


def test_aborting_a_queued_request_removes_it(engine):
    kept, dropped = _request(), _request()
    engine.add_request(kept)
    engine.add_request(dropped)
    assert engine.abort(dropped.request_id) is True

    finished = set()
    while engine.has_work():
        for out in engine.step():
            if out.finished:
                finished.add(out.request_id)
    assert finished == {kept.request_id, dropped.request_id}


def test_aborting_a_running_request_releases_its_slot(engine):
    request = _request()
    engine.add_request(request)
    engine.step()
    assert engine.running, "expected the request to be admitted"
    state = engine.running[0]

    engine.abort(request.request_id)
    assert not engine.running
    assert state.block_table == []
    assert state.finish_reason == "aborted"


def test_an_aborted_prefix_is_still_worth_keeping(engine):
    """Abandoned work is not entirely wasted: the prefix it computed remains reusable.

    Chapter 9's release path publishes a sequence's full blocks to the prefix cache, and abort goes
    through the same path. That matters most for exactly this workload — an agent abandons steps
    constantly, and the next step of the same run starts with the same transcript.
    """
    prompt = [(i % 250) + 1 for i in range(64)]
    first = Request(prompt_token_ids=list(prompt), params=SamplingParams(max_tokens=8))
    engine.add_request(first)
    engine.step()
    engine.abort(first.request_id)
    engine.step()

    hits_before = engine.prefix.hit_tokens
    second = Request(prompt_token_ids=list(prompt), params=SamplingParams(max_tokens=2))
    engine.add_request(second)
    while engine.has_work():
        engine.step()
    assert engine.prefix.hit_tokens > hits_before


def test_abort_is_idempotent(engine):
    """A disconnect and a timeout can both fire; the second must not be an error."""
    request = _request()
    engine.add_request(request)
    assert engine.abort(request.request_id) is True
    assert engine.abort(request.request_id) is False
    assert engine.abort(999_999) is False


def test_an_aborted_request_gets_a_terminal_output(engine):
    """Silence is not a termination signal: a caller waiting on the stream must be told."""
    request = _request()
    engine.add_request(request)
    engine.step()
    engine.abort(request.request_id)

    outputs = engine.step()
    terminal = [o for o in outputs if o.request_id == request.request_id]
    assert terminal, "aborting produced no output for the caller"
    assert terminal[0].finished
    assert terminal[0].finish_reason == "aborted"


def test_the_engine_still_has_work_while_a_notice_is_pending(engine):
    """Aborting the only request must not strand its terminal output."""
    request = _request()
    engine.add_request(request)
    engine.abort(request.request_id)
    assert engine.has_work()
    outputs = engine.step()
    assert [o.finish_reason for o in outputs] == ["aborted"]
    assert not engine.has_work()


def test_cancellation_saves_work(model):
    """The point of the feature: fewer steps, because abandoned requests stop decoding."""
    steps = {}
    for honour in (False, True):
        engine = CancellableEngine(model, REFERENCE_MODEL, max_batch_size=4, n_blocks=512)
        requests = [_request(max_tokens=24) for _ in range(6)]
        for request in requests:
            engine.add_request(request)
        doomed = {r.request_id for r in requests[:3]}

        count = 0
        while engine.has_work():
            engine.step()
            count += 1
            if honour and count == 4:
                for request_id in doomed:
                    engine.abort(request_id)
        steps[honour] = count

    assert steps[True] < steps[False]


def test_wasted_tokens_counts_work_done_for_callers_who_left(engine):
    request = _request(max_tokens=16)
    engine.add_request(request)
    for _ in range(4):
        engine.step()
    engine.abort(request.request_id)
    assert engine.aborted == 1
    assert engine.wasted_tokens > 0
