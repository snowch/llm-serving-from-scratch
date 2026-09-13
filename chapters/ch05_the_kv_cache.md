---
title: "The KV Cache"
short_title: "ch05 The KV Cache"
---

(ch05)=
# ch05 · The KV Cache

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch03](#ch03), [ch04](#ch04) |
| **Scorecard** | First real speedup row — and the first memory constraint. |
:::

## The problem

Watch {ref}`ch01`'s engine produce its two-hundredth token. To do it, the model runs over all 199
previous tokens. Every one of those was computed on the previous step, produced exactly the same
result, and was thrown away.

The arithmetic from {ref}`ch03` says this should be expensive: generating *n* tokens without a
cache costs work proportional to *n²*, against *n* with one. For a 64-token prompt and 32 output
tokens that predicted a 27.5× penalty.

Whether it actually costs 27.5× is the interesting question, and the answer is instructive.

## The idea

### What is worth keeping

Attention computes, for each position, a query against the keys and values of every earlier
position. During decode there is exactly one new query — the token just produced — and the keys
and values of every previous token are *identical to what they were on the previous step*.

They depend only on the token at that position and its position in the sequence. Neither changes.
So compute them once and keep them. That is the entire idea, and it is the single highest-return
optimisation in inference.

Queries are not cached, because there is only ever one live query per step and it is new every
time.

This is also where rotary position embeddings earn their place. RoPE rotates keys by their
position at the moment they are computed, so a cached key already carries its position and stays
valid however long the sequence grows. A model that added a position embedding to the input
instead would need care here — and, more importantly, could not share cached keys between two
requests whose prompts start the same way. {ref}`ch09` depends on that property entirely.

### What it costs

From {ref}`ch03`:

```
kv_bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element
```

The number to hold on to is not the per-token figure but what it becomes. For a 7B model in fp16
with grouped-query attention, a single 8k-token conversation costs on the order of a gigabyte of
KV cache. Weights are a fixed cost paid once; **KV cache is a per-concurrent-request cost, and it
is what actually limits how many users a GPU can serve**.

We have traded compute for memory. Memory is now the binding constraint, and stays that way for
the rest of Part III.

## The build

The change to the engine is small, which is part of the point — the cache is an optimisation, not
a different model:

```{literalinclude} ../llmserve/engines/naive.py
:language: python
:start-at:         elif self.use_cache:
:end-before:         else:
```

Three details in those few lines matter more than their size suggests.

**Only the newest token goes in.** Not the whole sequence — just the one token produced last. The
cache supplies the rest.

**The position is passed explicitly.** During decode the new token sits at position
`total_len - 1`, not at zero. Get this wrong and there is no crash and no exception: the model
simply attends as though every token were at the start of the sequence, and the output degrades in
a way that looks like a bad model rather than a bug. This is the most common KV cache error and it
is silent.

**The cache is returned and reassigned.** The model appends the new key and value and hands back
the extended cache. {ref}`ch08` replaces this contiguous tensor with paged blocks, but the
interface stays exactly the same.

Isolating the cache as the *only* difference from {ref}`ch01`'s engine is deliberate — it is what
makes its contribution measurable rather than merely plausible:

```{literalinclude} ../llmserve/engines/naive.py
:language: python
:start-at: class CachedEngine
:end-before:     def __init__
```

## The measurement

Same trace, same model, same machine — one engine with a cache and one without:

```{include} _generated/ch05-kv-cache.md
```

Read it in two parts.

**ITL falls by about 2.7×, and stays flat.** That is the cache doing its job: per-token cost no
longer grows with sequence length.

**Capacity roughly doubles**, and the effect on TTFT at load is much larger than the ITL
improvement alone suggests. At 8 requests per second the median wait for a first token drops from
seconds to a fraction of one. This is a queueing effect rather than a compute one: faster service
means a shorter queue, and at high utilisation a small service improvement produces a large
latency improvement. {ref}`ch10` makes that relationship explicit.

**And 2.7× is not 27.5×.** {ref}`ch03` predicted an order of magnitude more than we got, and the
explanation is the two-bound model. A decode step on this model reads 23 MB of weights whether it
processes one token or ninety-six. Removing the redundant arithmetic removed work the machine was
doing in time it was spending on memory anyway.

That gap is not a disappointment, it is the lesson: **on a bandwidth-bound operation, saving FLOPs
buys less than arithmetic suggests.** Scale the model up and the ratio shifts — a 7B model moves
far more bytes per step, but its redundant prefix computation grows faster still. The prediction
was not wrong so much as it was answering a question about a resource that was not scarce.

## The cost

The cache is close to a free lunch, but not quite:

- **Memory is now the binding constraint.** Concurrency is capped by KV cache, not compute. The
  rest of Part III is about that ceiling.
- **The cache must be reserved for a sequence's whole lifetime**, and a sequence's final length is
  not known when it starts. Reserve for the maximum and most of it is wasted; reserve less and a
  long generation has nowhere to go. {ref}`ch08` is the answer.
- **The cache is state, and state can go stale.** A position off by one, a cache reused across
  requests, an entry not evicted on cancellation — none of these crash. They produce slightly
  worse output. This is exactly why {ref}`ch04`'s equivalence test arrives before this chapter
  rather than after it.

## Key takeaways

- Keys and values depend only on a token and its position, so they can be computed once and kept.
  Queries cannot and need not be.
- The cache converts a quadratic amount of recomputation into a linear amount of memory. Memory
  then becomes the limit on concurrency.
- Passing the wrong position during decode fails silently, degrading output without any error.
- RoPE makes a cached key self-describing, which is what lets {ref}`ch09` share cache entries
  between different requests.
- A FLOPs-based prediction overstated this speedup by 10×, because the operation was bound by
  bytes moved rather than arithmetic. Predict with the right resource, then measure.

## Looking ahead

The engine is much faster per request and still serves exactly one request at a time. Meanwhile it
reads all 23 MB of weights to produce a single token for a single user — and would read the same
23 MB to produce a token for thirty users. {ref}`ch06` starts collecting that free lunch, and runs
straight into why the obvious way of doing it wastes most of the gain.

## Further reading

The KV cache is old enough to be folklore rather than a paper; it appears in the original
transformer decoding implementations without much ceremony. Its consequences are the subject of
the vLLM paper (Appendix E), which is best read after {ref}`ch08` when the fragmentation problem
it solves is something you have already run into.
