---
title: "Observability for Serving Engines"
short_title: "ch27 Observability"
---

(ch27)=
# ch27 · Observability for Serving Engines

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch02](#ch02), [ch08](#ch08) |
| **Scorecard** | Queue depth moves first and recovers first. Latency lags, and keeps rising after the problem is over. |
:::

## The problem

Host metrics tell you the machine is busy. They cannot tell you *why* a request was slow, because
every reason lives inside the engine: it waited behind a prefill ({ref}`ch10`), it was preempted
({ref}`ch08`), it missed the prefix cache ({ref}`ch09`), its batch was starved ({ref}`ch07`). An
engine that exports CPU, memory and request rate is an engine you cannot debug.

Worse, the obvious signal is actively misleading. {ref}`ch03` explains why: decode is
memory-bandwidth-bound, so a replica can be completely saturated — unable to take another sequence
without hurting everyone on it — while its compute utilisation looks unremarkable. **Utilisation does
not track the constraint, so it cannot predict the failure.**

## The signals worth exporting

One criterion for inclusion: does this change what you would do next?

```{literalinclude} ../llmserve/metrics.py
:language: python
:start-at: class EngineSnapshot
:end-before:     step: int
```

- **Queue depth** says *add capacity*. It is the most direct statement of "there is more work here
  than can be served soon".
- **KV utilisation** ({ref}`ch08`) says *the block budget is the constraint*. It catches what queue
  length cannot: a few very long sequences can exhaust memory while the queue looks short.
- **Preemption rate** says *the scheduler is thrashing*, which is the mechanism by which tail latency
  falls off a cliff rather than degrading.
- **Batch size distribution** says *whether batching is working at all*. A fleet whose batch size is
  persistently one has a scheduling problem no throughput number will reveal.
- **Prefix hit rate** ({ref}`ch09`) says *whether the workload is what you think it is* — and, per
  {ref}`ch26`, whether somebody changed the chat template.
- **TTFT and ITL histograms** say *what users experienced*, which is the only thing on this list that
  is an outcome rather than a cause.

The first five explain the sixth. That is the whole design.

## Sample per step, not per scrape

```{literalinclude} ../llmserve/metrics.py
:language: python
:start-at: class Metrics:
:end-before:     ttft: Histogram
```

A scrape every fifteen seconds cannot see a queue that formed and drained in two — and those are
precisely the events that produce the tail latency someone is complaining about. The engine has a
natural sampling point, its own step, and using it costs:

```{include} _generated/ch27-overhead.md
```

A few percent, for a complete per-step record. That is a good trade on any engine you intend to
operate, and the alternative — sampling rarely enough to be free — is how a serving incident becomes
unexplainable after the fact.

Histograms are bucketed rather than exact for the same reason a running server differs from a
benchmark: bounded memory matters more than a precise percentile, and the honest way to present that
is to report the bucket edge rather than pretend to precision the data does not have.

```{literalinclude} ../llmserve/metrics.py
:language: python
:start-at: class Histogram:
:end-before:     def __init__
```

## The measurement: which signal moves first

Drive the engine into overload and watch both series on the same axis:

```{include} _generated/ch27-signals.md
```

Three things, and each one is an operational rule.

**Queue depth leads.** It is already deep in the early windows while TTFT p95 is still healthy. An
alert on queue depth fires while there is still time to do something; an alert on latency fires
after users have been affected.

**Queue depth recovers first, and latency keeps rising after it.** By the later windows the queue is
empty and TTFT p95 is still climbing — because the requests finishing then are the ones that queued
during the peak, and their clock started when they arrived. Latency is a *report about the past*.
Anyone reading a latency graph during an incident is reading history, and the instinct to keep
escalating after the cause has cleared comes directly from this lag.

**KV utilisation tells a different story from both.** It rises steadily and never approaches its
limit: on this trace memory was never the constraint. That is a useful negative — it rules out an
entire class of fix, and without the signal you would be guessing.

## What to alert on

- **Page on burn rate against the SLO**, not on a threshold. "p95 above 500 ms" fires on a blip and
  misses a slow bleed; burn rate against an error budget does neither.
- **Page on queue depth growth**, because it is the leading indicator above.
- **Alert, don't page, on preemption rate and cache hit rate.** They explain incidents and predict
  them; they are rarely themselves the emergency.
- **Never page on utilisation.** It does not track the constraint.

## Tracing

A metric says the fleet is slow; a trace says where one request's time went. The spans worth having
map exactly onto this book's chapters: queued, prefill (with a cache-hit attribute), each decode
step's batch, and preemptions. Two attributes make a trace far more useful than the default:
**tenant** ({ref}`ch21`) and **chain or session id** ({ref}`ch24`), because without the second the
agent latency problem is invisible at any sampling rate.

## The cost

- **Instrumentation costs a few percent** and the table above prices it. Sampling less often makes
  it cheaper and makes incidents unexplainable, which is the wrong saving.
- **Per-tenant dimensions multiply cardinality.** A histogram per tenant per endpoint is how a
  metrics bill exceeds a GPU bill, and the usual answer — aggregate and keep per-tenant detail only
  in traces — means the fairness problem of {ref}`ch21` is invisible in the metrics.
- **Bucketed percentiles are approximate**, and merging histograms across replicas makes them more
  so. A p99 computed from buckets is a bucket edge, and treating it as exact leads to arguments
  about numbers that were never that precise.
- **More signals means more to misread.** Six dashboards with no stated relationship between them is
  worse than two with one. The relationship here is that five of the signals are causes and one is
  the outcome; a dashboard that does not say so will be read wrongly under pressure.

## Key takeaways

- Host metrics cannot explain a serving engine, and **utilisation is actively misleading** because
  decode is memory-bound.
- Export the five causes — queue depth, KV utilisation, preemption rate, batch size, cache hit rate —
  alongside the one outcome, TTFT and ITL.
- **Sample per step.** A scrape interval cannot see the events that produce your tail.
- **Queue depth leads, latency lags.** Latency keeps rising after the cause has cleared, which is
  why incidents feel longer than they are.
- Page on SLO burn rate and on queue growth. Never page on utilisation.
- Put tenant and chain identifiers on traces, or two of this book's real problems are invisible.

## Looking ahead

{ref}`ch27`'s signals say when the engine is in trouble. {ref}`ch28` decides what it should *do* about
it — which turns out to be refusing work, on purpose, and doubling goodput by doing so.

## Further reading

The SRE literature on burn-rate alerting and on the four golden signals transfers to serving almost
unchanged, and is better written than anything specific to LLMs. For the engine-specific signals, read
what vLLM and SGLang actually export: the overlap between their metric names and the list above is
high, and the places they differ are usually places where one of them learned something.
