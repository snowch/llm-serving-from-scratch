---
title: "RAG and Long Context"
short_title: "ch21 RAG and Long Context"
---

(ch21)=
# ch21 · RAG and Long Context

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch09](#ch09), [ch10](#ch10), [ch14](#ch14) |
| **Scorecard** | The worst row in Part VI's table, and why none of the obvious fixes work. |
:::

## The problem

Go back to {ref}`ch20`'s framing table and compare two rows. Retrieval and agent traffic have prompts
of almost the same length. Their time to first token differs by two orders of magnitude.

The difference is entirely in the reuse column, and the reason is in the trace:

```{literalinclude} ../bench/traces.py
:language: python
:start-at: def make_rag_trace
:end-before:     rng = np.random.default_rng(seed)
```

Every request opens with the same instruction, so there is *something* to reuse — a block or two.
Then each carries a different set of retrieved passages in a different order, and from that point
the prompts have nothing in common. A prefix cache can only reuse up to the first difference, and
in retrieval that comes early.

**This is a property of the workload, not a shortcoming of the cache.** Retrieval exists to put
*different* context in front of the model for each query. A retrieval system whose prompts were
largely identical would be a retrieval system that was not retrieving.

So {ref}`ch09`, the best return in this book, is worth comparatively little here. That leaves the
prefill bill to be paid in full, on every request, and retrieval prompts are long.

## What does not fix it

The reflex, having read {ref}`ch10`, is chunked prefill: split the long prompt across steps so it
cannot monopolise one. Here is that, at two arrival rates:

```{include} _generated/ch21-interference.md
```

**Below saturation, chunking does what {ref}`ch10` said it would.** At the lower arrival rate the
short interactive requests get a clearly better tail, and so do the long ones — a moderate budget is
also the faster way to prefill, per {ref}`ch10`'s microbenchmark, so both halves of the win land at
once. Throughput gives up a little, which is the trade.

**Above saturation, chunking does nothing at all.** At the higher rate every column is the same to
within noise, because at that rate the engine simply cannot keep up and the queue grows without
bound. Chunked prefill redistributes latency between requests; it does not create capacity. Applying
it to an overloaded engine is treating a symptom of the wrong problem.

Note also which column moves least: the inter-token latency of the short requests is barely affected
either way. That is worth knowing before reaching for chunking to fix a streaming-smoothness
complaint — on this workload the stall is in the queue, not in the steps.

This is worth dwelling on because it generalises. **Almost every scheduling technique in Parts II
and III redistributes; almost none creates capacity.** When the arrival rate exceeds what the
hardware can serve, the only things that help are fewer tokens, more hardware, or refusing work
({ref}`ch26`). A scheduler is not a capacity plan.

## What does help

Three things, in descending order of how much they move the number.

**Cache what is actually shared, which is the passages.** A prefix cache keyed on the prompt from
the beginning cannot reuse a passage that appears second in one request and fourth in another —
the prefix differs, so everything after it differs. The fix is architectural: put the retrieved
passages in a fixed order, or use a cache that can reuse non-prefix spans. Fixed ordering is free
and surprisingly effective; the general version needs an attention implementation that tolerates a
sequence assembled from independently-cached pieces, which is real research and out of scope here.

**Quantise the KV cache** ({ref}`ch14`). Retrieval is where the KV footprint binds first, because
every request holds blocks proportional to its very long prompt. Halving the bytes per token nearly
doubles concurrency, and {ref}`ch14`'s measurement is that the quality cost of INT8 KV is small.
This is the single most useful thing to do to a retrieval deployment, and it is a configuration
change.

**Separate the phases** ({ref}`ch11`). Retrieval is the workload disaggregation was designed for:
prefill-dominated, with a prefill so large it damages everyone else's decoding. Whether it pays is
decided by {ref}`ch11`'s handoff arithmetic and the interconnect, not by anything in this chapter.

## Co-serving the retrieval stack

A RAG deployment is not one model. There is an embedding model on the ingest path and usually a
reranker between retrieval and generation, and the temptation is to put them on the same
accelerator as the generator, since neither is large.

The arithmetic says be careful. Embedding and reranking are **prefill-shaped**: short, compute-bound,
parallel, no decode phase at all. They compete directly for the resource the generator's prefill
needs, and they arrive in bursts correlated with exactly the requests that are about to need a large
prefill. Co-serving them makes the generator's worst moment worse.

The two workable arrangements are separate hardware, or the same hardware with the small models
under a strict token budget ({ref}`ch10`) so they cannot take more than a slice of any step. What
does not work is running them unmanaged and hoping, which is what most first implementations do.

## The cost

- **The prefix cache stops being the answer**, and a deployment tuned on chat traffic will be tuned
  wrong. Cache size is nearly free to grow for chat and close to useless for retrieval.
- **KV memory is the binding constraint**, not compute. Concurrency is set by how many long prompts
  fit, which is why {ref}`ch14`'s quantisation matters more here than in any other workload.
- **Chunked prefill's benefit is conditional** on not being saturated, and the condition is easy to
  miss because the technique is usually described without one.
- **Prompt assembly becomes a serving concern.** The order in which a retriever emits passages is
  normally an application detail; here it decides the cache hit rate, which puts a performance
  constraint on a component that does not think of itself as part of the serving stack.
- **Quality and cost pull in opposite directions.** Retrieving more passages improves answers and
  linearly increases the thing that is already the bottleneck. `top_k` is a serving parameter as
  much as a retrieval one, and it is usually owned by someone who does not know that.

## Key takeaways

- Retrieval prompts share an instruction and then diverge, so prefix caching is worth far less than
  it is on chat — by design, not by defect.
- **Chunked prefill helps below saturation and does nothing above it.** Scheduling redistributes
  latency; it does not create capacity.
- The binding constraint is KV memory, so quantising the cache is the highest-value change
  available, and it is a configuration change rather than an architecture one.
- Fix the passage *order* before reaching for a cleverer cache. A stable order restores a prefix
  that a shuffled one destroys.
- Embedding and reranking models are prefill-shaped and burst in sync with the generator's worst
  case. Co-serve them under a token budget or not at all.
- `top_k` is a serving parameter. Whoever tunes retrieval quality is also tuning your prefill bill.

## Looking ahead

Retrieval was the workload where reuse was worth least. {ref}`ch22` is the other extreme: agent
traffic replays its entire transcript on every step, so reuse is worth more there than anywhere else
in this book — and the chapter's real subject turns out to be something else entirely, because a
user waiting on an agent is waiting on a *chain* of requests rather than one.

## Further reading

Sarathi-Serve (Appendix E) is the chunked-prefill source and is careful about the conditions under
which it helps — more careful than most summaries of it. For the KV side, the quantisation papers in
{ref}`ch14`'s list apply directly, and the cache-quantisation results are the relevant ones here
rather than the weight-quantisation ones.
