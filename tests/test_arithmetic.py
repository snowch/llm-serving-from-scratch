"""Tests for the chapter 3 arithmetic.

The first tests in the book, and they set the pattern: a hand calculation we can check by eye,
then the properties that must hold whatever the numbers are.
"""

import pytest

from bench.harness import ROOT
from llmserve.arithmetic import (
    cached_decode_flops,
    decode_ceiling_tokens_per_second,
    kv_bytes_per_token,
    max_concurrent_sequences,
    naive_decode_flops,
)
from llmserve.config import REFERENCE_MODEL, ModelConfig
from llmserve.model import build_model

# Llama-3-8B-shaped: grouped-query attention, 8 KV heads against 32 query heads.
LLAMA8B = ModelConfig(
    vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, dtype="fp16"
)


def test_kv_bytes_per_token_matches_hand_calculation():
    # 2 * 32 layers * 8 kv heads * 128 head_dim * 2 bytes = 131072 bytes/token
    assert kv_bytes_per_token(LLAMA8B) == 131072


def test_kv_quantisation_halves_the_cache():
    assert kv_bytes_per_token(LLAMA8B, kv_dtype="int8") == kv_bytes_per_token(LLAMA8B) / 2


def test_gqa_shrinks_the_cache_relative_to_mha():
    mha = ModelConfig(**{**LLAMA8B.__dict__, "n_kv_heads": 32})
    assert kv_bytes_per_token(mha) == 4 * kv_bytes_per_token(LLAMA8B)


def test_predicted_parameter_count_matches_the_built_model():
    """The arithmetic is only trustworthy if it agrees with the thing it describes."""
    model = build_model(REFERENCE_MODEL)
    assert sum(p.numel() for p in model.parameters()) == REFERENCE_MODEL.n_params


def test_decode_ceiling_falls_as_context_grows():
    bandwidth = 2e12  # ~2 TB/s, H100-class
    short = decode_ceiling_tokens_per_second(LLAMA8B, bandwidth, context_length=128)
    long = decode_ceiling_tokens_per_second(LLAMA8B, bandwidth, context_length=32768)
    assert short > long


def test_batching_raises_the_ceiling():
    """The whole reason continuous batching (ch07) works: weights are read once per step."""
    bandwidth = 2e12
    single = decode_ceiling_tokens_per_second(LLAMA8B, bandwidth, context_length=1024)
    batched = decode_ceiling_tokens_per_second(
        LLAMA8B, bandwidth, context_length=1024, batch_size=32
    )
    assert batched > 10 * single


def test_concurrency_ceiling_is_memory_over_per_sequence_cost():
    gib = 1024**3
    assert max_concurrent_sequences(LLAMA8B, 10 * gib, context_length=2048) == int(
        10 * gib // (131072 * 2048)
    )


def test_uncached_decode_is_quadratic_in_sequence_length():
    """Chapter 5's motivation, as arithmetic rather than assertion."""
    short = naive_decode_flops(REFERENCE_MODEL, prompt_len=64, n_output=16)
    long = naive_decode_flops(REFERENCE_MODEL, prompt_len=64, n_output=64)
    # Four times the output tokens costs much more than four times the work.
    assert long > 4 * short
    assert naive_decode_flops(REFERENCE_MODEL, 64, 64) > cached_decode_flops(
        REFERENCE_MODEL, 64, 64
    )


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        decode_ceiling_tokens_per_second(LLAMA8B, 2e12, batch_size=0)


def test_head_counts_must_be_divisible():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_heads=8, n_kv_heads=3)


def test_code_fingerprint_is_independent_of_where_the_repo_lives():
    """Regression: fingerprints were stored against absolute paths.

    A result generated in one checkout then failed verification in every other, because the
    recorded path did not exist there. CI failed while the same command passed locally, which is
    the most expensive shape of bug to chase.
    """
    import os

    from bench.harness import code_fingerprint, relative_to_root

    here = code_fingerprint("llmserve/engines/naive.py")
    cwd = os.getcwd()
    try:
        os.chdir("/tmp")
        assert code_fingerprint("llmserve/engines/naive.py") == here
    finally:
        os.chdir(cwd)

    assert relative_to_root("/somewhere/else/entirely/naive.py") is None
    assert not str(relative_to_root(str(ROOT / "llmserve" / "model.py"))).startswith("/")


def test_fingerprint_changes_when_a_core_source_changes(tmp_path):
    """The guard only works if it actually notices a change."""
    from bench.harness import CORE_SOURCES, code_fingerprint

    assert "llmserve/sampling.py" in CORE_SOURCES
    baseline = code_fingerprint()
    assert code_fingerprint("llmserve/engines/naive.py") != baseline


def test_fingerprint_accepts_several_sources():
    """A composed engine — a router over replicas — depends on more than one module."""
    from bench.harness import code_fingerprint

    pair = ["llmserve/engines/prefix.py", "llmserve/router.py"]
    assert code_fingerprint(pair) != code_fingerprint(pair[:1])
    assert code_fingerprint(pair) != code_fingerprint()
    # One path may be given as a bare string.
    assert code_fingerprint("llmserve/router.py") == code_fingerprint(["llmserve/router.py"])
    # The hash is over an ordered list, so discovery order would change it. That is why
    # engine_sources sorts rather than relying on the order replicas happen to be visited in.
    assert code_fingerprint(pair) != code_fingerprint(list(reversed(pair)))


def test_engine_sources_are_sorted_and_include_wrapped_engines():
    """A change to the replica engine must invalidate a fleet result measured with it."""
    from bench.harness import engine_sources

    class FakeReplica:
        pass

    class FakeFleet:
        inner_engines = [FakeReplica(), FakeReplica()]

    sources = engine_sources(FakeFleet())
    assert list(sources) == sorted(sources)
    # Both classes live in this test module, so the single path appears exactly once.
    assert len(sources) == 1


def test_every_committed_result_carries_a_real_fingerprint():
    """Regression: four results shipped with a placeholder like ``n/a-quantisation``.

    Those were the runners that do not serve a trace — arithmetic tables and microbenchmarks — and
    because nothing cited them from ``SCORECARDS`` they escaped verify-numbers entirely. A
    quantisation figure could have been measured by code that no longer existed, and the build
    would have stayed green.
    """
    import json

    results = sorted((ROOT / "bench" / "results").glob("*.json"))
    assert results, "no committed results to check"
    for path in results:
        payload = json.loads(path.read_text())
        recorded = payload.get("code_fingerprint")
        assert recorded, f"{path.name} has no code_fingerprint"
        assert not str(recorded).startswith("n/a"), (
            f"{path.name} carries a placeholder fingerprint, so nothing can detect it going stale"
        )


def test_every_derived_fragment_declares_its_results():
    """A derived fragment that reads a result must say so, or the result goes unverified."""
    import re

    from bench.scorecards import DERIVED_SOURCES

    source = (ROOT / "scripts" / "render-scorecards.py").read_text()
    derived_block = source[source.index("DERIVED = {") : source.index("def render(")]
    declared = set(re.findall(r'"([^"]+)":', derived_block))

    assert set(DERIVED_SOURCES) <= declared, (
        f"DERIVED_SOURCES names fragments the renderer does not build: "
        f"{sorted(set(DERIVED_SOURCES) - declared)}"
    )

    # Every derived renderer that calls load() must appear in DERIVED_SOURCES.
    scorecard = (ROOT / "bench" / "scorecard.py").read_text()
    for fragment in sorted(declared):
        function = _renderer_for(fragment, source)
        if function and _reads_a_result(function, scorecard):
            assert fragment in DERIVED_SOURCES, (
                f"{fragment} is rendered by {function}(), which reads a result file, but it is not "
                "declared in DERIVED_SOURCES — verify-numbers cannot check that result"
            )


def _renderer_for(fragment: str, render_source: str) -> str | None:
    """The function name the renderer maps a fragment to."""
    import re

    match = re.search(rf'"{re.escape(fragment)}":\s*(?:lambda[^,]*?)?(\w+_table)', render_source)
    return match.group(1) if match else None


def _reads_a_result(function: str, scorecard_source: str) -> bool:
    """Whether a renderer's body calls ``load()``, i.e. depends on a committed result."""
    start = scorecard_source.find(f"def {function}(")
    if start < 0:
        return False
    end = scorecard_source.find("\ndef ", start + 1)
    body = scorecard_source[start : end if end > 0 else len(scorecard_source)]
    return "load(" in body
