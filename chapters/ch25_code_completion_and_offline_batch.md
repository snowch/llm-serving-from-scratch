---
title: "Code Completion and Offline Batch"
short_title: "ch25 Completion and Offline Batch"
---

(ch25)=
# ch25 · Code Completion and Offline Batch

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch02](#ch02), [ch07](#ch07), [ch17](#ch17) |
| **Scorecard** | Two workloads at opposite ends of one axis, reading the same table in opposite directions. |
:::

## Two workloads, one axis

Code completion and offline batch inference look similar from the outside. Similar model, similar
request sizes, no human reading a stream of tokens in either case. Almost every configuration
decision comes out opposite, and the reason is a single question: **is anyone waiting?**

For completion, someone is waiting and they are waiting *impatiently*. A suggestion that arrives
after the developer has typed the next character is worth nothing — not less, nothing. The objective
is a time to first token in the tens of milliseconds, and throughput is close to irrelevant.

For offline batch, nobody is waiting at all. There is no arrival process to model, no service
objective to miss, and exactly one number: tokens per unit of hardware time.

## Code completion

```{include} _generated/ch25-completion.md
```

The objective here is deliberately brutal, and the `Met SLO` column shows the engine mostly failing
it. That is the honest outcome and it is the useful one: **this workload is hard, and a general
serving configuration does not meet it.**

What the table does say clearly is that serving one request at a time is the worst possible choice,
which is not the intuition. A completion request is tiny; the reflex is to keep the batch small so
each one is served immediately. But at a realistic arrival rate a small batch means a queue, and
queueing is the dominant term in time to first token — so the configuration that minimises
per-request work maximises per-request latency.

What actually moves this number is not in this table, and it is worth saying so plainly:

- **{ref}`ch09`'s prefix cache**, because a developer's file barely changes between keystrokes. The
  same buffer is re-sent with one more character, which is a near-perfect cache hit. The framing
  table in {ref}`ch22` shows a high reuse rate for exactly this reason.
- **{ref}`ch17`'s speculation**, because code is the most predictable text there is. Acceptance rates
  on code are far above prose, and speculation attacks decode latency, which is what is left once
  prefill is cached away.
- **A smaller model.** The uncomfortable truth of this workload: single-digit-millisecond TTFT is
  reached by not running a large model, and the serving stack cannot fix a model that is too big for
  the objective.
- **Cancelling superseded requests** ({ref}`ch24`). Every keystroke invalidates the last request, so
  a completion service that does not cancel is doing several times the work it needs to.

## Offline batch

```{include} _generated/ch25-offline.md
```

Every request is available at time zero, so there is no arrival process — and removing it removes
the entire reason {ref}`ch02`'s harness is open-loop:

```{literalinclude} ../bench/traces.py
:language: python
:start-at: def make_offline_batch_trace
:end-before:     rng = np.random.default_rng(seed)
```

The table reads in the opposite direction to {ref}`ch22`'s. Throughput rises with batch size and
there is no latency column to trade it against, so the answer is simply *the largest batch that
fits*. The KV utilisation column is how you find that: push the batch until the allocator is close
to full, and stop before preemption starts, because a preempted sequence recomputes its prefill and
that is pure loss.

Two things this workload can do that no interactive one can:

**Sort by length.** Nothing is waiting, so the order requests are served in is free to choose.
Grouping similar lengths together dramatically reduces the padding waste of {ref}`ch06` — the reason
static batching wasted a third of its slots was length variance within a batch, and offline is the
one workload where you can simply remove it.

**Accept much worse tail latency for throughput.** Every trade-off in this book that was rejected
because it hurt the tail should be re-examined here, and most of them flip.

## The general point

These two chapters-worth of tuning come from the same engine and the same code. Nothing was
recompiled; a handful of numbers changed. That is the argument for Part VI as a whole, and the
reason the framing table in {ref}`ch22` has no "best" row.

It also means **a benchmark without a workload is not a result.** An engine tuned for offline batch
will beat one tuned for completion on a throughput benchmark, by a lot, and lose on any latency
measure — and both engines are the same software. {ref}`ch31` is about how to run comparisons that
survive this observation.

## The cost

- **Two configurations means two deployments**, or one deployment that is wrong for one of the
  workloads. Mixing completion and batch traffic on one fleet gives the batch work the completion
  fleet's low utilisation and the completion traffic the batch fleet's tail.
- **Completion's objective may not be reachable** with the model you have, and no amount of serving
  work changes that. Recognising it early is worth more than tuning.
- **Offline batch at maximum batch size runs close to the memory limit**, where {ref}`ch08`'s
  preemption waits. Preemption in a batch job is silent — no SLO to miss — and shows up only as
  throughput that is lower than it should be.
- **Length sorting changes completion order**, which matters if anything downstream assumed
  submission order. It usually did, and it usually did not say so.
- **Cancellation becomes load-bearing for completion**, with all the race conditions {ref}`ch24`
  lists, on the workload least tolerant of latency spikes.

## Key takeaways

- The question that separates these workloads is "is anyone waiting", and it decides nearly every
  other setting.
- For completion, small batches are the wrong reflex: queueing dominates time to first token, so
  the batch that minimises per-request work maximises per-request latency.
- Completion's real wins are prefix caching, speculation, cancellation and a smaller model — not
  scheduler tuning.
- Offline batch removes the arrival process entirely, which removes the reason for an open-loop
  harness and licenses every throughput-for-latency trade in this book.
- Offline batch can sort by length, which removes the padding waste {ref}`ch06` could only mitigate.
- **The same engine, tuned two ways, produces opposite answers from the same table.** A benchmark
  without a stated workload is not a result.

## Looking ahead

Part VI is done. The engine has been tuned four ways for four workloads, and every one of those
tunings assumed something that has not been built yet: that requests arrive through an API, that
somebody can see what the engine is doing, and that it degrades rather than fails when the
assumptions break. Part VII builds that, starting with the surface a caller actually touches — and
with a bug in it that silently undoes {ref}`ch09`.

## Further reading

The fill-in-the-middle literature is about training rather than serving and is worth reading anyway,
because the prompt format it dictates decides what a completion request looks like and therefore what
the cache can reuse. For the batch side, the relevant work is mostly about scheduling and bin-packing
rather than about LLMs, and the older literature transfers cleanly.
