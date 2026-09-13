---
title: "Agents and Tool Use"
short_title: "ch24 Agents and Tool Use"
---

(ch24)=
# ch24 · Agents and Tool Use

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch09](#ch09), [ch08](#ch08) |
| **Scorecard** | The best reuse in the book, and a latency result that inverts the usual advice. |
:::

## The best-case workload

An agent run is a loop: the model emits a tool call, something executes it, the result is appended,
and the model is asked again. Every step therefore replays the entire transcript so far, plus one
turn:

```{literalinclude} ../bench/traces.py
:language: python
:start-at: def make_agent_trace
:end-before:     rng = np.random.default_rng(seed)
```

Step *n* of a run is **literally step *n−1* plus one turn**. That is the strongest possible case for
{ref}`ch09`, and {ref}`ch22`'s framing table shows it: agent traffic has the longest prompts in the
table and the best time to first token, because almost none of those tokens need computing.

If you serve agents, prefix caching is not one optimisation among several. It is the difference
between viable and not, and everything else in this chapter is secondary to keeping the hit rate up.

## The result that matters is not about one request

Here is the part that changes how to read the rest of this book.

A user waiting on an agent is not waiting on a request. They are waiting on a *chain* of them, and
the chain's latency is a **sum** of per-call latencies. Sums do not behave like their terms:

```{include} _generated/ch24-chains.md
```

Read the last three columns.

**The median grows linearly.** Sixteen calls take about sixteen times as long as one. Nothing
surprising.

**The tail grows more slowly** — twelve times, not sixteen — and the `p95/p50` column says why: as
the chain lengthens, the ratio between the tail and the middle *falls*, from about 1.6 to about 1.1.
This is the central limit theorem doing what it always does. A sum of independent draws concentrates
around its mean, and the relative spread shrinks as roughly one over the square root of the number
of terms.

So for a long chain, **per-call tail latency barely matters and per-call mean latency is everything.**
One slow call in sixteen is diluted by the fifteen typical ones; a mean that is 10% worse makes the
whole chain 10% worse, every single time.

That inverts the advice this book has given since {ref}`ch02`. Tail latency is the right thing to
optimise when a human waits on one request — which is every workload in Part VI except this one. For
agents, a change that improves p99 and worsens the median is a change that makes your users' actual
experience worse, and it will look like an improvement on every dashboard you have.

The two practical consequences:

- **Optimise the median for agent traffic.** Larger batches, higher utilisation, and the throughput
  end of every trade-off in this book. The usual objection — "but the tail" — is much weaker here.
- **Report chain latency, not call latency.** A per-request p95 of a second sounds fine and is
  twenty seconds of a user waiting. Nothing in a standard serving dashboard shows this, because the
  chain does not exist as far as the engine is concerned.

## Cancellation is a throughput feature

Agents abandon work constantly: a parallel branch answers first, a timeout fires, a step is
superseded, the user stops the run. Without a way to say so, the engine keeps decoding for a client
that left — and that work is not merely wasted, it occupies a batch slot and KV blocks that a live
request is queued for.

```{literalinclude} ../llmserve/engines/cancel.py
:language: python
:start-at:     def abort
:end-before:         for index, state in enumerate(self.waiting):
```

```{include} _generated/ch24-cancellation.md
```

A third of requests abandoned recovers roughly a quarter of the engine's steps. That is a large
number for a feature usually filed under politeness, and it comes almost free: aborting a queued
request is a list removal, and aborting a running one is the same bookkeeping {ref}`ch08`'s
preemption already does. **An engine that got paged memory right gets cancellation almost free, and
one that did not cannot bolt it on.**

Two details that are easy to get wrong and expensive to debug:

**A cancelled request still needs a terminal message.** Silence is not a termination signal. A
caller waiting on the stream waits until its own timeout, the connection and whatever the gateway
holds for it leak, and on the server side the request looks completed — so nothing alerts. This is
the most common way cancellation is implemented incorrectly.

```{literalinclude} ../llmserve/engines/cancel.py
:language: python
:start-at:     def step(self) -> list[StepOutput]:
:end-before:         notices, self._abort_notices
```

**Abort must be idempotent.** A client disconnect and a timeout routinely fire for the same request;
an abort that raised on the second call would turn a normal race into an error page.

And one pleasant surprise: the work is not entirely lost. An aborted sequence's completed blocks are
published to the prefix cache on the way out, so the next step of the same run — which begins with
the same transcript — still finds them warm.

## The cost

- **The chain is invisible to the engine.** Everything in this chapter about chain latency has to be
  measured at the client, because the server sees N unrelated requests. Correlating them needs a
  trace id threaded through the API ({ref}`ch26`) and honoured by the metrics ({ref}`ch27`).
- **Optimising the median trades away the tail**, and if the same fleet also serves interactive
  traffic those two goals are in direct conflict. This is a genuine argument for separate pools, or
  at least for {ref}`ch21`'s per-tenant policy.
- **Cancellation is a new source of races.** A request can be aborted between admission and prefill,
  during decode, or after it has already finished. Every one of those paths has to be idempotent, and
  the failure mode of getting it wrong is a leaked block rather than a crash.
- **Prefix cache pressure is extreme.** Agent transcripts are long and each run's is unique, so the
  cache fills with entries only one run will ever want. LRU handles it, but the cache has to be large
  enough to hold a run's transcript for the run's duration, and sizing it is now a function of how
  long agent runs last.
- **Retries multiply everything.** An agent that retries a failed step re-submits a prompt that is
  almost identical, which is good for the cache and bad for the queue, and a retry storm from an
  agent framework is the noisy neighbour of {ref}`ch21`.

## Key takeaways

- Agent traffic replays its transcript, so step *n* contains step *n−1*. It is the best case for
  prefix caching in this book, and the hit rate is the thing to protect.
- **A chain's latency is a sum, and sums concentrate.** The median grows linearly, the tail grows
  more slowly, and the relative spread shrinks.
- Therefore: **optimise the median for agents**, not the tail — the reverse of the advice for every
  other workload here.
- Report chain latency. A per-call p95 that sounds healthy can be a twenty-second wait, and no
  server-side dashboard shows it.
- Cancellation recovers a large fraction of steps and is nearly free on a paged engine. It is a
  throughput feature.
- A cancelled request needs a terminal message, and abort must be idempotent. Both failures are
  silent.

## Looking ahead

{ref}`ch25` closes Part VI with the two extremes of the latency axis: code completion, where a
result that arrives late is worth nothing at all, and offline batch, where nothing is waiting and
the only number is tokens per unit of hardware time. The same engine, the same table, opposite
answers.

## Further reading

The serving literature has little to say about chains specifically; the useful material is in
queueing theory and in the tail-at-scale work from distributed systems, where the same arithmetic
governs a request that fans out to many servers. The direction of the effect differs — fan-out takes
the *maximum* and amplifies the tail, chains take the *sum* and dampen it — and holding both in mind
is the point.
