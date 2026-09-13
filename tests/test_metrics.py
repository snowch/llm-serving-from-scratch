"""Chapter 25: the signals that explain an engine."""

import pytest
import torch

from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.metrics import LATENCY_BUCKETS, Histogram, Metrics, leading_indicator_windows
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams


def test_an_empty_histogram_has_no_percentile():
    """Reporting 0 for a metric nobody observed is how a dashboard lies about a quiet service."""
    assert Histogram(LATENCY_BUCKETS).percentile(95) is None
    assert Histogram(LATENCY_BUCKETS).mean is None


def test_a_percentile_lands_in_the_right_bucket():
    histogram = Histogram((1.0, 2.0, 4.0))
    for value in (0.5, 0.5, 0.5, 3.0):
        histogram.observe(value)
    assert histogram.percentile(50) == 1.0
    assert histogram.percentile(100) == 4.0


def test_a_value_past_the_last_bucket_overflows_rather_than_being_clamped():
    """Clamping an overflow into the top bucket hides exactly the outlier you needed to see."""
    histogram = Histogram((1.0,))
    histogram.observe(99.0)
    assert histogram.percentile(95) == float("inf")


def test_the_mean_is_kept_alongside_the_buckets():
    histogram = Histogram((1.0, 2.0))
    histogram.observe(0.5)
    histogram.observe(1.5)
    assert histogram.mean == pytest.approx(1.0)


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    return build_model(REFERENCE_MODEL)


def test_metrics_sample_the_engine_every_step(model):
    engine = PrefixCachedEngine(model, REFERENCE_MODEL, max_batch_size=2, n_blocks=256)
    metrics = Metrics()
    for _ in range(6):
        engine.add_request(
            Request(
                prompt_token_ids=[(i % 250) + 1 for i in range(48)],
                params=SamplingParams(max_tokens=4),
            )
        )
    while engine.has_work():
        metrics.observe_step(engine, engine.step())

    report = metrics.report()
    assert report["steps"] == len(metrics.snapshots) > 0
    assert report["peak_queue_depth"] > 0, "a queue formed and the metric did not see it"
    assert 0.0 < report["peak_kv_utilisation"] <= 1.0


def test_a_report_from_a_fresh_engine_is_still_well_formed():
    report = Metrics().report()
    assert report["steps"] == 0
    assert report["ttft_p95"] is None
    assert report["peak_queue_depth"] == 0


def test_windows_summarise_the_run_on_one_axis(model):
    engine = PrefixCachedEngine(model, REFERENCE_MODEL, max_batch_size=2, n_blocks=256)
    metrics = Metrics()
    for _ in range(4):
        engine.add_request(
            Request(
                prompt_token_ids=[(i % 250) + 1 for i in range(32)],
                params=SamplingParams(max_tokens=4),
            )
        )
    while engine.has_work():
        metrics.observe_step(engine, engine.step())

    windows = leading_indicator_windows(metrics.snapshots, [], window=3)
    assert windows
    assert [w["step"] for w in windows] == sorted(w["step"] for w in windows)
    assert sum(w["tokens_out"] for w in windows) == sum(s.tokens_out for s in metrics.snapshots)
