---
title: "Continuous Batching"
short_title: "ch07 Continuous Batching"
---

(ch07)=
# ch07 · Continuous Batching

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch06](#ch06) |
| **Scorecard** | The largest single jump in this book. |
:::

## The problem

{ref}`ch06` measured it: a third of the slot-steps in a mixed batch were spent computing tokens
for sequences that had already finished, and a request arriving just after a batch formed waited
for every member of that batch to drain before it could start.

Both come from one assumption, so deeply buried it barely looks like a decision: **the batch is
the unit of work**. Sequences enter together and leave together because that is how the loop was
written.

Nothing about the mathematics requires it.

## The idea

### Schedule per iteration, not per batch

A decode step does not care which sequences are in it. It reads the weights, applies them to
whatever queries it is given, and returns one token for each. The set of sequences can be
different on every single step and the arithmetic does not notice.

So make it different. On each step:

1. Retire any sequence that finished on the previous step. Its slot is free *now*.
2. Admit waiting requests into free slots.
3. Run one step over whatever is in the running set.

This is **iteration-level scheduling**, introduced by Orca and universally known as continuous
batching. Both names are accurate: the granularity of scheduling is one iteration, and the batch
never stops to re-form.

The two wastes from {ref}`ch06` disappear rather than shrink:

- **Ragged completion** — gone. A finished sequence leaves on the next step, so no slot-step is
  ever spent on completed work.
- **Admission delay** — gone. A request waits at most one step for a free slot, instead of waiting
  for a whole batch.

### Why this is the biggest win in the book

It is worth being precise about *why*, because the reason is not "it does less work".

Throughput improves, but only modestly — the engine was already batching, and batching was already
collecting the weight-read amortisation.

**TTFT improves enormously**, and that is a queueing effect. {ref}`ch02` showed the naive engine
collapsing because requests queued behind other requests. Static batching shortened the queue but
kept its structure: you still wait for a batch. Continuous batching removes the waiting almost
entirely, so at loads where the static engine has a visible queue, the continuous engine has
essentially none.

And because goodput counts requests that met a latency objective, an improvement concentrated in
TTFT moves goodput far more than the throughput change alone would suggest.

There is no new mathematics in this chapter. There is no new kernel. The entire gain comes from
changing *when* a sequence is allowed to join and leave.

## The build

The engine subclasses {ref}`ch06`'s and replaces `step`. That inheritance is the point: everything
about how tokens are computed is unchanged.

```{literalinclude} ../llmserve/engines/batched.py
:language: python
:start-at: class ContinuousBatchEngine
:end-before:     def _emit
```

Three details do the work.

**Retirement happens first.** The running set is filtered before admission, so a slot freed by a
sequence finishing on the previous step is reusable on this one.

**Admitted requests are prefilled separately, then join the decode batch.** A new arrival needs its
whole prompt processed; sequences already running need one token each. These are different shapes,
so this engine does them as two passes. That is honest but not ideal — the prefill pass makes the
step longer, which every currently-streaming sequence feels as an ITL spike. {ref}`ch10` addresses
exactly this.

**Nothing about sampling, caching or attention changed.** The equivalence test holds without any
tolerance:

```{literalinclude} ../tests/test_engines.py
:language: python
:start-at: def test_batching_does_not_change_output
:end-before: def test_padding_does_not_leak_between_sequences
```

And the waste counter from {ref}`ch06` now reads zero by construction:

```{literalinclude} ../tests/test_engines.py
:language: python
:start-at: def test_continuous_batching_wastes_no_slots
```

## The measurement

Against static batching, same trace, same model, same machine:

```{include} _generated/ch07-continuous-batching.md
```

The throughput column improves by a modest margin. The TTFT columns are a different story
entirely — and note what happens to them *as load rises*. Static batching's median TTFT climbs
steadily with the arrival rate. Continuous batching's barely moves. The engine is not merely
faster; it has stopped queueing.

That is the shape to remember: an engine whose TTFT is flat in the arrival rate has capacity in
hand, and one whose TTFT climbs is already queueing, whatever its throughput says.

Here is the whole journey so far, at the highest load we have measured:

```{include} _generated/ch07-running-scorecard.md
```

Four chapters of work, one saturating workload. Throughput has multiplied several times over,
median TTFT has fallen from seconds to milliseconds, and goodput — the only number that counts
requests actually served within their objective — has improved by well over an order of magnitude.

Two of those four steps (the KV cache, and this one) involved no new mathematics at all.

## The cost

Continuous batching is close to strictly better, but not entirely free:

- **The engine is now a scheduler**, with genuine state: a running set, a waiting queue, admission
  and retirement rules. Every later chapter modifies this loop, and bugs in it produce starvation
  and unfairness rather than crashes.
- **ITL gets worse, and unevenly so.** Prefilling an admitted request lengthens that step for
  everyone already streaming. A user mid-sentence experiences somebody else's arrival as a pause.
  The measurement above shows ITL rising with load for precisely this reason, and {ref}`ch10` is
  the fix.
- **Admission is now a policy decision.** First-come-first-served is one choice among several, and
  the wrong one starves somebody. {ref}`ch10` again.
- **Memory pressure becomes bursty.** Admitting several long prompts on one step can spike KV usage
  hard. With a contiguous per-sequence cache there is no graceful response available — which is
  the subject of {ref}`ch08`.

That last point deserves emphasis. This engine assembles a padded batch from per-sequence caches
on every step and takes it apart again afterwards, copying work proportional to the longest
sequence present. It is correct, and it is wasteful, and it caps how far this design scales.

## Key takeaways

- A decode step is indifferent to which sequences are in it, so the running set can change every
  step. That single observation is continuous batching.
- Retiring finished sequences immediately eliminates ragged-completion waste; admitting into freed
  slots eliminates batch-formation delay.
- The gain shows up mostly in TTFT, because it is a queueing improvement rather than a compute one
  — and goodput follows TTFT.
- Flat TTFT as load rises means capacity in hand. Rising TTFT means queueing, whatever throughput
  reports.
- The engine is now a scheduler, and its remaining problems — prefill interference, admission
  policy, bursty memory — are scheduling and memory problems, not arithmetic ones.

## Looking ahead

The scheduler wants to admit more sequences than memory allows, and its per-sequence contiguous
caches make every step copy more than it should. {ref}`ch08` replaces that memory layout with
fixed-size blocks and a block table, which raises the concurrency ceiling without buying a single
byte of extra RAM.

## Further reading

Orca (Yu et al., OSDI 2022, Appendix E) introduced iteration-level scheduling and is the primary
source for this chapter; it is short and worth reading now that you have implemented its central
idea. The vLLM paper builds on it and is better read after {ref}`ch08`.
