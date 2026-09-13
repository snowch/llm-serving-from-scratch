---
title: "Static Batching and Its Limits"
short_title: "ch06 Static Batching"
---

(ch06)=
# ch06 · Static Batching and Its Limits

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch03](#ch03), [ch05](#ch05) |
| **Scorecard** | Throughput up several-fold. Utilisation poor, and measurably so. |
:::

## The problem

{ref}`ch05`'s engine reads every weight in the model to produce a single token for a single user.
{ref}`ch03` established that it would read exactly the same weights to produce a token for thirty
users. We are paying the dominant cost of a decode step and collecting one token for it.

This is the free lunch of LLM serving, and taking it is the obvious move. What is not obvious is
how much of it the obvious implementation gives back.

## The idea

### Batching amortises the expensive part

Decode is bandwidth-bound. Per step the engine reads the weights once — a fixed cost — and each
additional sequence in the batch adds only its own KV cache to the bytes moved. Doubling the batch
therefore roughly doubles output for roughly the same weight traffic, until KV cache or compute
becomes the limit instead.

That is the whole economic argument for batching, and it is why every serious serving system does
it.

### Ragged sequences make batching awkward

Requests do not cooperate. They arrive with prompts of different lengths and generate outputs of
different lengths, and a batched matrix multiplication requires a rectangle.

Padding produces the rectangle, and then two things must be corrected for — **both of which fail
silently rather than loudly**:

**Positions.** We left-pad, so every sequence's final token lands in the same column and decode
reads one column rather than a ragged edge. But a left-padded sequence's first real token sits at
index `pad`, while its *position* is 0. Feed the index as the position and the model is told the
prompt begins somewhere it does not. No error; just worse output.

**Masking.** Padded slots hold arbitrary values, and without a mask they are attended to — one
request's padding blending into another's output.

The masking detail is worth spelling out, because getting it almost right is worse than getting it
obviously wrong:

```{literalinclude} ../llmserve/model.py
:language: python
:start-at:         # Mask with the dtype's most negative finite value
:end-before:         # Causal mask
```

Masking with `-inf` is the natural choice and it is a trap. A query sitting *at* a padding
position is masked everywhere it may attend, so its attention row is entirely `-inf`, and softmax
over that row is NaN. Those NaNs reach V at the padded positions. Real tokens weight those
positions at zero — but `0 × NaN` is NaN, so a single short sequence quietly destroys the output
of every other sequence sharing its batch.

We found this by testing, not by reasoning, which is why the regression test is explicit about the
symptom:

```{literalinclude} ../tests/test_engines.py
:language: python
:start-at: def test_padding_does_not_leak_between_sequences
:end-before: def test_static_batching_wastes_slots
```

### Where static batching gives the gain back

The batch is the unit of work: form one, run it until every member has finished, form the next.
That produces two kinds of waste.

**Ragged completion.** A sequence that wants 4 tokens sits in a batch alongside one that wants 40.
It finishes first, then keeps its slot for 36 more steps, being computed and having its output
thrown away.

**Admission delay.** A request arriving one step after a batch forms waits for *every* member of
that batch to finish before it can even start.

Both are scheduling failures, not arithmetic ones, and {ref}`ch07` fixes them without touching a
single kernel.

## The build

Prefill assembles the rectangle and takes it apart again:

```{literalinclude} ../llmserve/engines/batched.py
:language: python
:start-at: def _prefill_batch
:end-before:     lengths = [state.request.prompt_len
```

The engine itself is small, and the waste is instrumented rather than described:

```{literalinclude} ../llmserve/engines/batched.py
:language: python
:start-at:         tokens = _sample_batch(logits, self.running)
:end-before:         # The batch is released only when every member
```

That `wasted_slot_steps` counter is the chapter's real output. It counts slot-steps spent
computing tokens for sequences that had already finished — work done, paid for, discarded.

## The measurement

Against {ref}`ch05`'s one-at-a-time engine, on the same trace:

```{include} _generated/ch06-static-batching.md
```

Throughput rises sharply, and TTFT improves at every load — both as predicted. The engine is
collecting the free lunch.

Now the waste. On a small batch with deliberately mixed output budgets, the test suite records
what fraction of computed slot-steps were discarded:

```{literalinclude} ../tests/test_engines.py
:language: python
:start-at: def test_static_batching_wastes_slots_on_finished_sequences
:end-before: def test_continuous_batching_wastes_no_slots
```

A third of the slots computed in that batch produced tokens nobody received. The proportion grows
with the spread of output lengths — and real traffic has a very wide spread, far wider than this
test's. A workload mixing one-line answers with long explanations wastes most of its batch.

Notice also what happened to ITL: it got slightly *worse* than the one-at-a-time engine. Each step
now does more work, so each step takes longer. That is the trade we chose — throughput for
per-token latency — and it is the right trade here, but it is a trade.

## The cost

- **ITL rises.** A bigger batch means a longer step. Batch size is a dial between throughput and
  per-token latency, and {ref}`ch10` is about choosing where to set it.
- **TTFT now depends on batch formation.** A request can be ready to run and still wait, which is
  a latency source that did not exist before.
- **One long generation penalises everyone in its batch.** The slowest member sets the pace.
- **Padding is real work.** We compute tokens we then mask away, and the wider the length spread,
  the more of the batch is padding.
- **A whole class of silent bugs arrives with padding.** Positions and masks both fail quietly.
  The equivalence tests are not optional here.

## Key takeaways

- Decode reads the weights once per step regardless of batch size, so batching is nearly free
  throughput. This is the strongest economic fact in serving.
- Batching requires a rectangle; real requests are ragged. Padding bridges the gap and brings
  position and masking bugs that produce worse output rather than errors.
- Mask with a finite floor, not `-inf`. A fully-masked row softmaxes to NaN, and `0 × NaN`
  contaminates every other sequence in the batch.
- Static batching gives much of its gain back: finished sequences hold their slots, and arriving
  requests wait for a whole batch to drain.
- Both of those are scheduling problems. Neither needs a faster kernel to fix.

## Looking ahead

The waste is measured, and its cause is the batch boundary itself. {ref}`ch07` removes the
boundary: sequences join and leave on any step they like. It is the smallest conceptual change in
this book and it produces the largest single improvement in the scorecard.

## Further reading

Static batching is what naive implementations and most inference tutorials do, so it is rarely
written up as a named technique. Its limitations are the motivation for Orca (Appendix E), whose
introduction states the ragged-completion problem more concisely than this chapter does.
