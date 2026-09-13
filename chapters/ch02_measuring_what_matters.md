---
title: "Measuring What Matters"
short_title: "ch02 Measuring What Matters"
---

(ch02)=
# ch02 · Measuring What Matters

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch01](#ch01) |
| **Scorecard** | Builds the harness that produces **every scorecard row in this book**. |
:::

## The problem

Chapter 1 made claims. It said the server saturates around four requests per second, that the
median user waits three seconds, that goodput falls while throughput holds. Every one of those
came from a tool that has not been described, which means none of them has been earned yet.

That is not pedantry. Benchmarking a serving system is unusually easy to get wrong, and the two
most common mistakes both produce numbers that look *better* than reality. If we are about to
spend twenty-seven chapters optimising against a measurement, the measurement has to be the first
thing we can defend.

## The idea

### What to measure

Four numbers describe a serving system. They are not interchangeable, and no two of them can be
traded without someone noticing.

**Time to first token (TTFT).** From the request arriving to the first token reaching the caller.
This is the "is it broken?" interval — the silence a user stares at. It includes queueing, and at
load that is almost all of it.

**Inter-token latency (ITL).** The gap between subsequent tokens, sometimes called time per output
token. This is the reading speed. Once text is flowing, a user tolerates a surprising amount of
total latency provided it arrives steadily; ITL spikes are far more noticeable than a uniformly
slower stream.

**Throughput.** Output tokens per second across all requests. This is what the hardware bill is
denominated in, and it is the number most benchmarks report. On its own it is close to useless,
for reasons the measurement section shows.

**Goodput.** Requests per second that met a stated service objective. This is the only number that
means anything on its own, and it is the one almost nobody reports — mostly because reporting it
requires committing to an SLO in public.

### Why means lie

Report a mean latency and you describe a user who does not exist. A server where ninety percent
of requests take 50 ms and ten percent take 5 s has a mean of 545 ms, which sounds tolerable and
describes nobody: everyone either had a fast experience or an unusable one.

This book reports p50, p95 and p99 and never a bare average. Tail latency is not an edge case in
serving — it is the product. A user whose request lands behind a long generation experiences the
tail, and at any real traffic level a lot of users do.

### Why the obvious load generator lies

Here is the load generator almost everyone writes first: start N workers; each submits a request,
waits for the response, then submits the next.

This is a **closed loop**, and it cannot measure an overloaded server. When the server slows down,
each worker's next request is delayed by exactly the amount the server is late, so the offered
load drops to match the server's capacity. The queue never grows. Latency stays flat. The
benchmark reports that everything is fine, right up until production disagrees.

The name for this is **coordinated omission**: the requests that would have shown the problem were
never sent, because the harness was politely waiting. It is the single most common way a
benchmark flatters a system, and it is invisible unless you know to look for it.

The fix is an **open loop**. Decide the arrival schedule before the run starts, and stick to it
regardless of how the server is coping. If the server cannot keep up, requests pile up — which is
precisely the thing we want to observe.

Arrivals here are Poisson rather than evenly spaced, for the same reason. Real traffic is bursty,
and a fixed-interval schedule understates queueing badly.

## The build

The harness is a loop with three jobs: admit requests whose time has come, advance the engine one
step, and stamp whatever came back.

```{literalinclude} ../bench/harness.py
:language: python
:start-at:     start = time.perf_counter()
:end-before:     wall = time.perf_counter() - start
```

Everything reported is derived from four timestamps per request, which keeps the derivation
auditable:

```{literalinclude} ../bench/harness.py
:language: python
:start-at: class RequestRecord:
:end-before:     @property
```

Note what `run_benchmark` does *not* do: it never waits for a response before sending the next
request. The schedule comes from `make_poisson_trace` and is fixed before the first request is
sent.

The SLO is a parameter rather than a constant, and it has to be stated with any goodput figure.
"Goodput" with an unstated objective is just throughput wearing a better name.

```{literalinclude} ../bench/harness.py
:language: python
:start-at: class SLO:
:end-before: @dataclass
```

## The measurement

Now chapter 1's claims can be checked. The same naive engine, past its capacity:

```{include} _generated/ch02-goodput-collapse.md
```

Read the last two columns together. Across those rows the arrival rate doubles, then doubles
again. Output tokens per second barely moves — the engine is saturated, and that flat number is
its capacity.

Goodput goes *down*.

This is the shape that justifies the whole chapter. A throughput-only benchmark of this server
would report that it handled the extra load fine: the machine stayed busy, tokens kept coming.
What actually happened is that the engine spent its fixed capacity generating tokens for requests
whose deadline had already passed. The work was done. Nobody wanted it by the time it arrived.

Two lessons follow, and they hold for every engine in this book:

1. **Throughput measures the machine. Goodput measures the service.** Optimise the first and you
   can make the second worse.
2. **Past saturation, latency is a queueing property, not a compute property.** No kernel
   optimisation fixes a three-second TTFT that is three seconds of waiting in line. That is why
   {ref}`ch07` — a scheduling change with no new mathematics in it — produces the largest single
   improvement in the book.

## The cost

The harness is now a dependency of every number this book prints, which has consequences worth
accepting deliberately:

- **Changing it invalidates every committed result.** `bench/harness.py` should be treated as
  frozen once chapters start citing it. Where it must change, all results get regenerated
  together, never piecemeal.
- **The trace is part of the measurement.** These numbers describe Poisson arrivals with prompts
  of 32–96 tokens and outputs of 16–48. {ref}`ch21` shows that changing that distribution can
  reverse which engine looks faster. A benchmark without its trace is not a result.
- **Measuring costs CPU.** The harness shares this machine with the engine. The loop sleeps rather
  than spins when idle, but on a busy system the measurement is part of the load.

## Key takeaways

- TTFT, ITL, throughput and goodput describe different things. Only goodput is meaningful alone,
  and only against a stated SLO.
- Report percentiles, never means. In serving, the tail is the product.
- A closed-loop load generator cannot observe overload, because it slows down in sympathy with the
  server. This is coordinated omission, and it makes broken systems look healthy.
- Past saturation, goodput falls while throughput holds steady. A benchmark that reports only
  throughput will tell you that collapse is fine.

## Looking ahead

We can now measure precisely, and we know the naive engine saturates at a particular number. What
we cannot yet do is say *why* that number and not one ten times larger. {ref}`ch03` derives it
from first principles, with arithmetic you can do before writing any code — and then shows where
that arithmetic is wrong, which turns out to be the more useful half.

## Further reading

Coordinated omission was named and popularised by Gil Tene; his talks on latency measurement are
the standard reference and apply well beyond serving. The goodput framing used throughout this
book comes from the DistServe paper (Appendix E), which makes the case that it, rather than
throughput, is what an inference system should be optimised for.
