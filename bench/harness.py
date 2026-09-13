"""The benchmark harness. Chapter 2 builds it; every chapter after that re-runs it.

Two design decisions matter more than anything else here.

**Open loop, not closed loop.** A closed-loop harness keeps N workers that each submit a request,
wait for it, then submit the next. That sounds like load, but it is self-limiting: when the
server slows down, the harness slows down with it, so the queue never grows and overload is
invisible. This is *coordinated omission*, and it is why a closed-loop benchmark of an
overloaded server can report healthy latencies. Here, arrivals follow a schedule fixed before
the run starts, and a server that cannot keep up accumulates a queue — which is the thing we
want to see.

**Percentiles, not means.** Nobody experiences the mean. The whole book is about tails, so the
harness reports p50/p95/p99 and a goodput figure against an explicit SLO, and never a bare
average.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import platform
import subprocess
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from llmserve.engines.base import Engine
from llmserve.request import Request
from llmserve.sampling import SamplingParams

RESULTS_DIR = Path(__file__).resolve().parent / "results"
ROOT = Path(__file__).resolve().parent.parent

#: Files whose contents determine the numbers any engine produces. A result records a hash of
#: these plus the engine module that generated it, so staleness is decided by what the code *is*
#: rather than by when it was committed. Commit times cannot tell an unrelated new file apart
#: from a change to the thing being measured, and get it wrong in both directions.
CORE_SOURCES = (
    "llmserve/model.py",
    "llmserve/sampling.py",
    "llmserve/request.py",
    "llmserve/config.py",
    "bench/harness.py",
)


def relative_to_root(path: str | None) -> str | None:
    """Express a source path relative to the repository root.

    Absolute paths are machine-specific: a result generated in one checkout would not match the
    same code checked out anywhere else, so every fingerprint would differ on CI while passing
    locally. Store the relative path and resolve it against ROOT when reading.
    """
    if not path:
        return None
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return None


def code_fingerprint(sources: str | Sequence[str] | None = None) -> str:
    """Hash the sources a benchmark result depends on.

    ``sources`` are repo-relative paths hashed on top of :data:`CORE_SOURCES` — the module that
    generated the result, and anything it composes. One path may be passed as a bare string. A
    missing file raises rather than hashing nothing, because silently contributing empty bytes
    makes a broken fingerprint look like a mismatch and sends you hunting for the wrong problem.
    """
    if isinstance(sources, str):
        sources = [sources]
    digest = hashlib.sha256()
    names = [*CORE_SOURCES, *(sources or ())]
    for name in names:
        path = ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"cannot fingerprint missing source: {name}")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def engine_sources(engine: object) -> tuple[str, ...]:
    """Repo-relative modules whose contents determine what this engine measures.

    Three things count, and each was a hole when it was missing:

    * the engine's own module;
    * every module its class inherits from, because most engines in this book are a subclass that
      overrides one method and leaves the scheduling loop to its base — a change there changes the
      numbers and named only the subclass's file;
    * the engines it runs, if it runs any. A router's numbers come from its replicas, so a change
      to the replica engine has to invalidate the fleet results too. An engine that wraps others
      says so by exposing them as ``inner_engines``.

    The result is sorted, so the fingerprint does not depend on the order things were discovered.
    """
    found: set[str] = set()
    pending = [engine]
    while pending:
        current = pending.pop()
        for base in type(current).__mro__:
            # ``object`` and anything else defined in C has no source file to hash.
            try:
                source = inspect.getsourcefile(base)
            except TypeError:
                continue
            path = relative_to_root(source)
            if path:
                found.add(path)
        pending += list(getattr(current, "inner_engines", ()))
    return tuple(sorted(found))


def stamped_payload(
    *,
    engine: str,
    model: dict,
    conditions: dict,
    summary: dict,
    code_sources: Sequence[str] = (),
) -> dict:
    """Build a stamped result payload for something that does not serve a trace.

    Arithmetic tables and microbenchmarks never build a :class:`BenchResult`, and for a while they
    wrote a placeholder fingerprint instead — which meant ``verify-numbers.py`` could not tell
    whether the code behind those figures had changed since they were measured. Every writer of a
    result file now goes through either this or :meth:`BenchResult.to_json`, so the stamp cannot
    be left off.
    """
    return {
        "engine": engine,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "hardware": _hardware(),
        "model": model,
        "versions": _versions(),
        "code_fingerprint": code_fingerprint(code_sources),
        "code_sources": list(code_sources),
        "conditions": conditions,
        "summary": summary,
    }


@dataclass(frozen=True)
class RequestSpec:
    """One entry in a trace: when it arrives, how big it is, and optionally what it contains.

    ``tokens`` matters for chapter 9. Prefix caching only does anything when requests genuinely
    share text, so a trace has to be able to specify the actual prompt rather than just a length.
    When it is None the harness synthesises a prompt of ``prompt_len`` tokens.
    """

    arrival: float
    prompt_len: int
    max_tokens: int
    tokens: tuple[int, ...] | None = None
    #: which tenant submitted it (ch21); None for the single-tenant traces
    tenant: str | None = None


@dataclass(frozen=True)
class SLO:
    """The service objective goodput is measured against. State it, or the number is meaningless."""

    ttft_seconds: float = 1.0
    itl_seconds: float = 0.2


@dataclass
class RequestRecord:
    """Raw timings for one request. Everything reported is derived from these four numbers."""

    request_id: int
    arrival: float
    prompt_len: int
    first_token: float | None = None
    finish: float | None = None
    n_output: int = 0
    tenant: str | None = None

    @property
    def ttft(self) -> float | None:
        return None if self.first_token is None else self.first_token - self.arrival

    @property
    def itl(self) -> float | None:
        """Mean inter-token latency after the first token."""
        if self.finish is None or self.first_token is None or self.n_output < 2:
            return None
        return (self.finish - self.first_token) / (self.n_output - 1)

    @property
    def e2e(self) -> float | None:
        return None if self.finish is None else self.finish - self.arrival


def make_poisson_trace(
    n_requests: int,
    rate_per_second: float,
    *,
    prompt_len: tuple[int, int] = (32, 128),
    output_len: tuple[int, int] = (16, 64),
    seed: int = 0,
) -> list[RequestSpec]:
    """Poisson arrivals with uniformly drawn lengths.

    Poisson because real traffic is bursty: fixed-interval arrivals understate queueing badly,
    and any benchmark using them flatters the server. Length ranges matter just as much — chapter
    21 shows that changing the input/output ratio can reverse which engine looks faster.
    """
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / rate_per_second, size=n_requests)
    arrivals = np.cumsum(gaps)
    prompts = rng.integers(prompt_len[0], prompt_len[1] + 1, size=n_requests)
    outputs = rng.integers(output_len[0], output_len[1] + 1, size=n_requests)
    return [
        RequestSpec(float(a), int(p), int(o))
        for a, p, o in zip(arrivals, prompts, outputs, strict=True)
    ]


@dataclass
class BenchResult:
    """Everything one run produced, plus the conditions that produced it."""

    engine: str
    records: list[RequestRecord]
    wall_time: float
    rate_per_second: float
    slo: SLO
    meta: dict = field(default_factory=dict)
    code_sources: tuple[str, ...] = ()

    def summary(self) -> dict:
        ttfts = np.array([r.ttft for r in self.records if r.ttft is not None])
        itls = np.array([r.itl for r in self.records if r.itl is not None])
        completed = [r for r in self.records if r.finish is not None]
        out_tokens = sum(r.n_output for r in completed)

        met_slo = [
            r
            for r in completed
            if r.ttft is not None
            and r.ttft <= self.slo.ttft_seconds
            and (r.itl is None or r.itl <= self.slo.itl_seconds)
        ]

        def pct(a: np.ndarray, q: float) -> float | None:
            return None if a.size == 0 else round(float(np.percentile(a, q)), 4)

        summary = {
            "requests": len(self.records),
            "completed": len(completed),
            "ttft_p50": pct(ttfts, 50),
            "ttft_p95": pct(ttfts, 95),
            "ttft_p99": pct(ttfts, 99),
            "itl_p50": pct(itls, 50),
            "itl_p95": pct(itls, 95),
            "output_tokens_per_second": round(out_tokens / self.wall_time, 2),
            "requests_per_second": round(len(completed) / self.wall_time, 3),
            "goodput_per_second": round(len(met_slo) / self.wall_time, 3),
            "goodput_fraction": round(len(met_slo) / max(len(completed), 1), 3),
            "wall_time": round(self.wall_time, 3),
        }

        # A fleet-wide percentile hides exactly the thing chapter 21 is about: one tenant can be
        # served perfectly while another is starved, and the aggregate looks fine either way. The
        # key only appears when the trace actually has tenants, so single-tenant results are
        # unchanged.
        tenants = sorted({r.tenant for r in self.records if r.tenant})
        if tenants:
            summary["by_tenant"] = {t: self._tenant_summary(t) for t in tenants}
        return summary

    def _tenant_summary(self, tenant: str) -> dict:
        """The same measures as :meth:`summary`, over one tenant's requests only."""
        mine = [r for r in self.records if r.tenant == tenant]
        ttfts = np.array([r.ttft for r in mine if r.ttft is not None])
        completed = [r for r in mine if r.finish is not None]
        met_slo = [
            r
            for r in completed
            if r.ttft is not None
            and r.ttft <= self.slo.ttft_seconds
            and (r.itl is None or r.itl <= self.slo.itl_seconds)
        ]

        def pct(a: np.ndarray, q: float) -> float | None:
            return None if a.size == 0 else round(float(np.percentile(a, q)), 4)

        return {
            "requests": len(mine),
            "completed": len(completed),
            "ttft_p50": pct(ttfts, 50),
            "ttft_p95": pct(ttfts, 95),
            "output_tokens": sum(r.n_output for r in completed),
            "goodput_fraction": round(len(met_slo) / max(len(completed), 1), 3),
        }

    def to_json(self, path: Path | str) -> Path:
        """Write a stamped result file.

        The stamps are not decoration: ``scripts/verify-numbers.py`` refuses a result without
        them, because a performance number whose hardware and versions are unknown cannot be
        checked by anyone, including us six months from now.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "engine": self.engine,
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "hardware": _hardware(),
            "model": self.meta.get("model", {}),
            "versions": _versions(),
            "code_fingerprint": code_fingerprint(self.code_sources),
            "code_sources": list(self.code_sources),
            "conditions": {
                "rate_per_second": self.rate_per_second,
                "slo": asdict(self.slo),
                **{k: v for k, v in self.meta.items() if k != "model"},
            },
            "summary": self.summary(),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n")
        return path


def _hardware() -> dict:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": _cpu_count(),
        "device": "cpu",
    }


def _cpu_count() -> int:
    try:
        return int(subprocess.run(["nproc"], capture_output=True, text=True, check=True).stdout)
    except Exception:
        return 0


def _versions() -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def run_benchmark(
    engine: Engine,
    trace: list[RequestSpec],
    *,
    rate_per_second: float,
    slo: SLO | None = None,
    seed: int | None = 0,
    meta: dict | None = None,
    idle_sleep: float = 0.0005,
) -> BenchResult:
    """Drive ``engine`` through ``trace`` on an open-loop schedule and record what happened.

    The loop does three things per iteration: admit any request whose arrival time has passed,
    advance the engine one step, and stamp whatever tokens came back. When there is nothing to
    run it sleeps briefly rather than spinning, so the measurement is not competing with the
    engine for CPU.
    """
    slo = slo or SLO()
    records: dict[int, RequestRecord] = {}
    pending = sorted(trace, key=lambda s: s.arrival)
    index = 0

    start = time.perf_counter()
    while index < len(pending) or engine.has_work():
        now = time.perf_counter() - start

        while index < len(pending) and pending[index].arrival <= now:
            spec = pending[index]
            prompt = (
                list(spec.tokens)
                if spec.tokens is not None
                else [(i % 250) + 1 for i in range(spec.prompt_len)]
            )
            request = Request(
                prompt_token_ids=prompt,
                params=SamplingParams(max_tokens=spec.max_tokens, seed=seed),
                tenant=spec.tenant,
            )
            engine.add_request(request)
            records[request.request_id] = RequestRecord(
                request_id=request.request_id,
                arrival=now,
                prompt_len=spec.prompt_len,
                tenant=spec.tenant,
            )
            index += 1

        if not engine.has_work():
            time.sleep(idle_sleep)
            continue

        outputs = engine.step()
        stamp = time.perf_counter() - start
        for out in outputs:
            rec = records[out.request_id]
            if rec.first_token is None and out.token_ids:
                rec.first_token = stamp
            rec.n_output += len(out.token_ids)
            if out.finished:
                rec.finish = stamp

    wall = time.perf_counter() - start
    return BenchResult(
        engine=getattr(engine, "name", type(engine).__name__),
        code_sources=engine_sources(engine),
        records=list(records.values()),
        wall_time=wall,
        rate_per_second=rate_per_second,
        slo=slo,
        meta=meta or {},
    )
