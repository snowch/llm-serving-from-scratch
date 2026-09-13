"""Chapter 27: the signals that explain a serving engine.

Host metrics tell you the machine is busy. They cannot tell you *why* a request was slow, because
the thing that made it slow — it waited behind a prefill, it was preempted, it missed the prefix
cache, its batch was starved — happens inside the engine and is invisible from outside. An engine
that exposes only CPU, memory and request rate is an engine you cannot debug.

The signals here are chosen on one criterion: each one changes what you would do next. Queue depth
says add capacity. KV utilisation says the block budget is the constraint. Preemption rate says the
scheduler is thrashing. A batch-size histogram says whether batching is working at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class Histogram:
    """Fixed-bucket histogram, which is what a serving metric almost always wants.

    Keeping every observation is what a benchmark does; a running server keeps counts per bucket,
    because the memory is bounded and the percentiles are still good enough to alert on. The cost is
    that a percentile is only as precise as the bucket it lands in, and the honest way to present
    that is to report the bucket edge rather than pretend to a precision the data does not have.
    """

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = tuple(sorted(buckets))
        self.counts = [0] * (len(self.buckets) + 1)
        self.total = 0
        self.sum = 0.0

    def observe(self, value: float) -> None:
        self.total += 1
        self.sum += value
        for index, edge in enumerate(self.buckets):
            if value <= edge:
                self.counts[index] += 1
                return
        self.counts[-1] += 1

    def percentile(self, q: float) -> float | None:
        """The upper edge of the bucket the qth percentile falls in; ``inf`` if it overflows."""
        if not self.total:
            return None
        target = q / 100 * self.total
        seen = 0
        for index, count in enumerate(self.counts):
            seen += count
            if seen >= target:
                return self.buckets[index] if index < len(self.buckets) else float("inf")
        return float("inf")

    @property
    def mean(self) -> float | None:
        return self.sum / self.total if self.total else None


#: Latency buckets in seconds. Log-ish spacing, because the interesting range spans four orders of
#: magnitude and a linear scale wastes every bucket on the part nobody looks at.
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


@dataclass
class EngineSnapshot:
    """One step's worth of engine state — the sample a scrape would take."""

    step: int
    queue_depth: int
    running: int
    kv_utilisation: float
    preemptions: int
    tokens_out: int


@dataclass
class Metrics:
    """Everything worth exporting from one engine, and nothing that is not.

    Sampling per step rather than per scrape is deliberate. A scrape every fifteen seconds cannot
    see a queue that formed and drained in two, and those are exactly the events that produce the
    tail latency someone is complaining about.
    """

    ttft: Histogram = field(default_factory=lambda: Histogram(LATENCY_BUCKETS))
    itl: Histogram = field(default_factory=lambda: Histogram(LATENCY_BUCKETS))
    batch_size: Histogram = field(default_factory=lambda: Histogram((0, 1, 2, 4, 8, 16, 32)))
    snapshots: list[EngineSnapshot] = field(default_factory=list)
    steps: int = 0

    def observe_step(self, engine, outputs: list) -> EngineSnapshot:
        self.steps += 1
        running = len(getattr(engine, "running", []))
        self.batch_size.observe(running)
        snapshot = EngineSnapshot(
            step=self.steps,
            queue_depth=len(getattr(engine, "waiting", [])),
            running=running,
            kv_utilisation=_utilisation(engine),
            preemptions=getattr(engine, "preemptions", 0),
            tokens_out=sum(len(o.token_ids) for o in outputs),
        )
        self.snapshots.append(snapshot)
        return snapshot

    def observe_request(self, ttft: float | None, itl: float | None) -> None:
        if ttft is not None:
            self.ttft.observe(ttft)
        if itl is not None:
            self.itl.observe(itl)

    def report(self) -> dict:
        """What a scrape endpoint would return."""
        return {
            "steps": self.steps,
            "ttft_p50": self.ttft.percentile(50),
            "ttft_p95": self.ttft.percentile(95),
            "itl_p95": self.itl.percentile(95),
            "batch_size_p50": self.batch_size.percentile(50),
            "peak_queue_depth": max((s.queue_depth for s in self.snapshots), default=0),
            "peak_kv_utilisation": round(
                max((s.kv_utilisation for s in self.snapshots), default=0.0), 4
            ),
            "preemptions": max((s.preemptions for s in self.snapshots), default=0),
        }


def _utilisation(engine) -> float:
    cache = getattr(engine, "cache", None)
    allocator = getattr(cache, "allocator", None)
    return float(getattr(allocator, "utilisation", 0.0))


def leading_indicator_windows(
    snapshots: list[EngineSnapshot], arrival_times: list[float], window: int = 20
) -> list[dict]:
    """Summarise the run in windows, so a signal's *timing* can be compared against another's.

    The claim chapter 27 tests is that queue depth moves before latency does. Testing it needs both
    series on the same time axis and at a resolution finer than a scrape interval, which is why the
    engine samples itself rather than waiting to be asked.
    """
    del arrival_times  # the engine's own step index is the axis that matters here
    rows = []
    for start in range(0, len(snapshots), window):
        chunk = snapshots[start : start + window]
        if not chunk:
            continue
        rows.append(
            {
                "step": chunk[0].step,
                "mean_queue_depth": round(sum(s.queue_depth for s in chunk) / len(chunk), 2),
                "peak_queue_depth": max(s.queue_depth for s in chunk),
                "mean_running": round(sum(s.running for s in chunk) / len(chunk), 2),
                "kv_utilisation": round(max(s.kv_utilisation for s in chunk), 4),
                "tokens_out": sum(s.tokens_out for s in chunk),
            }
        )
    return rows
