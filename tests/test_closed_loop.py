"""Chapter 28: the load generator is part of the measurement."""

import pytest
import torch

from bench.closed_loop import run_closed_loop
from bench.harness import SLO, make_poisson_trace, run_benchmark
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import build_model


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return build_model(REFERENCE_MODEL)


def _engine(model):
    return PrefixCachedEngine(model, REFERENCE_MODEL, max_batch_size=2, n_blocks=512)


def test_a_closed_loop_serves_every_request(model):
    trace = make_poisson_trace(8, 8.0, seed=1)
    result = run_closed_loop(_engine(model), trace, n_workers=2, slo=SLO())
    assert len(result.records) == len(trace)
    assert result.summary()["completed"] == len(trace)


def test_concurrency_is_capped_at_the_worker_count(model):
    """The defining property: the generator never has more than N requests outstanding.

    Counted at the moment of each submission, excluding sequences the engine has finished but not
    yet swept out of its running list — those are done, and the worker that owned one is the worker
    submitting now.
    """
    trace = make_poisson_trace(8, 8.0, seed=1)
    engine = _engine(model)
    original = engine.add_request
    peak = 0

    def counting_add(request):
        nonlocal peak
        live = len(engine.waiting) + len([s for s in engine.running if not s.finished])
        peak = max(peak, live + 1)
        original(request)

    engine.add_request = counting_add
    run_closed_loop(engine, trace, n_workers=2, slo=SLO())
    assert peak <= 2


def test_a_closed_loop_reports_no_offered_rate(model):
    """There is no arrival process, so recording one would be the confusion itself."""
    trace = make_poisson_trace(4, 8.0, seed=1)
    result = run_closed_loop(_engine(model), trace, n_workers=2, slo=SLO())
    assert result.rate_per_second == 0.0
    assert result.meta["generator"] == "closed-loop"
    assert result.meta["workers"] == 2


def test_the_closed_loop_hides_the_queueing_the_open_loop_shows(model):
    """Coordinated omission, as a test.

    The same workload, the same engine. The open loop offers requests on a schedule the server
    cannot keep up with, so they queue and the queueing shows up in the latency. The closed loop
    submits only when a worker is free, so the same server reports a far better tail while doing
    the same amount of work.
    """
    trace = make_poisson_trace(24, 32.0, seed=2)
    slo = SLO(ttft_seconds=0.5, itl_seconds=0.05)

    open_loop = run_benchmark(_engine(model), trace, rate_per_second=32.0, slo=slo).summary()
    closed = run_closed_loop(_engine(model), trace, n_workers=2, slo=slo).summary()

    assert closed["ttft_p95"] < open_loop["ttft_p95"], (
        "the closed loop should look better on the same work — that is the whole problem"
    )
