---
title: "Paged Attention and the Block Manager"
short_title: "ch08 Paged Attention"
---

(ch08)=
# ch08 · Paged Attention and the Block Manager

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch05](#ch05), [ch07](#ch07) |
| **Scorecard** | Concurrency ceiling raised. Throughput **falls** — and that is the honest result. |
:::

## The problem

{ref}`ch07`'s scheduler wants to admit more sequences than memory allows. To see why that limit
is lower than it should be, ask a question the engine cannot answer: **how much cache does this
request need?**

Nobody knows. The request will generate somewhere between one token and its budget, and which one
is not determined until it happens. A contiguous allocator has to commit before finding out, and
both available answers are bad:

- **Reserve the maximum.** Almost all of it is wasted on almost every request — reserved, not
  free, and unusable by anyone else.
- **Reserve less and grow.** Growing means finding a larger contiguous region and copying, while
  the old region leaves a hole that only a similarly sized sequence can use.

The first is internal fragmentation, the second external. Together they mean a server can be out
of memory while most of its memory is doing nothing.

## The idea

This problem is old, and operating systems solved it in the 1960s: stop allocating contiguous
ranges.

Split KV memory into fixed-size **blocks**. Give each sequence a **block table** mapping its
logical token positions to physical blocks. Allocate one block at a time, as the sequence actually
grows. Three things follow immediately:

- A sequence wastes at most one partly filled block, regardless of how long it might have become.
- Any free block fits any sequence, so external fragmentation disappears entirely.
- Two sequences can *share* a block, because nothing about a block ties it to one sequence. That
  is {ref}`ch09`'s entire mechanism, and it is free here.

This is PagedAttention, introduced by vLLM, and the analogy to virtual memory is exact enough that
the terminology carries over wholesale.

### Running out of memory becomes a scheduling decision

A contiguous engine that exhausts memory has no move available. A paged engine does: **preempt**
a running sequence, return its blocks to the pool, and requeue it.

Two ways to bring it back, and the choice is a real trade:

- **Recompute** — throw the cache away and prefill it again on re-admission. Costs compute.
- **Swap** — copy the cache to host memory and back. Costs bandwidth, twice.

We recompute, and preempt the *newest* sequence because it has generated least, so the least work
is lost.

One detail matters enormously here and is easy to get backwards. Preemption discards the **cache**,
never the **output**. A sequence that has already streamed twenty tokens to a caller keeps them;
only its keys and values are rebuilt, from the tokens it has already produced. Clearing the output
instead would make the caller receive those twenty tokens a second time — a correctness bug
wearing a scheduling costume, and one our tests caught precisely because they check what the
caller receives rather than what the engine does.

## The build

The allocator is a free list with reference counts. The counts are unused until {ref}`ch09`, and
present now because retrofitting them later would mean touching every call site:

```{literalinclude} ../llmserve/cache/blocks.py
:language: python
:start-at: class BlockAllocator
:end-before:     @property
```

Storage is per layer and indexed by *block*, not by sequence — which is what makes a sequence
merely a list of numbers:

```{literalinclude} ../llmserve/cache/blocks.py
:language: python
:start-at: class PagedKVCache
:end-before:     def __init__
```

Growth allocates only what the next token needs:

```{literalinclude} ../llmserve/engines/paged.py
:language: python
:start-at:     def _grow
:end-before:     def _release
```

And admission has to account for what it has already promised within the same step:

```{literalinclude} ../llmserve/engines/paged.py
:language: python
:start-at:         admitted: list[RequestState] = []
:end-before:         outputs: list[StepOutput] = []
```

That `reserved` counter fixes a bug worth naming, because the shape of it recurs throughout
scheduling. Blocks are not actually taken until prefill runs, so checking each candidate against
the same free-block count admits a batch that cannot fit — and prefill then fails on a request the
scheduler had already accepted. **A scheduler must account for its own promises, not just for the
current state.**

## The measurement

First, what paging bought. Same KV budget, allocated two different ways:

```{include} _generated/ch08-memory.md
```

A reserve-max allocator fits **nothing at all** in this budget: one sequence would claim the
model's entire maximum context before generating a single token. Paging fits several, because it
only ever allocates what has actually been used. Scale that up and it is the difference between a
GPU serving a handful of conversations and serving hundreds.

Now what it cost:

```{include} _generated/ch08-paged.md
```

**Throughput fell.** At the highest load the paged engine is meaningfully slower than {ref}`ch07`,
with worse TTFT and worse ITL. This is not a mistake in the measurement and it is not a bug to be
fixed later — it is what this implementation actually costs.

The cause is in the gather. To run a step, each sequence's cache is copied out of its blocks into
a contiguous tensor, the batch is padded, and the result is written back:

```{literalinclude} ../llmserve/cache/blocks.py
:language: python
:start-at:     def gather
:end-before:         n_blocks = self.allocator
```

That is a full copy of every running sequence's cache on every single step, in Python, one block
at a time. Contiguous storage needed no such copy.

So the honest summary of this chapter is: **paging raises the concurrency ceiling and lowers
throughput.** Whether that is a good trade depends entirely on which one you are short of — and in
production you are nearly always short of memory, which is why every serious engine pages. But the
trade is only worth it once the gather stops costing this much, and making it stop is a kernel
problem rather than a design problem. {ref}`ch14` writes that kernel.

It is worth stating plainly that a book which reported only the memory win here would be
misleading you, and a book which reported only the throughput regression would be missing the
point.

## The cost

- **Throughput and latency both regress**, measurably, until a real kernel replaces the gather.
- **Indirection everywhere.** A position is now a block plus an offset. Every access goes through
  a table, and debugging a wrong answer means reasoning about mappings rather than ranges.
- **Preemption is a new failure mode.** Under sustained pressure an engine can spend its time
  preempting and recomputing rather than making progress — thrashing, exactly as an
  over-committed OS does. Our tests assert that preempted requests still finish; a production
  engine needs a policy that guarantees it.
- **Block size is a new tuning parameter.** Too small and the tables and per-block overhead grow;
  too large and internal fragmentation returns.
- **Leaks are silent.** A block not returned is invisible until the pool empties and throughput
  collapses, which is why one test does nothing but check that the pool is whole after a drain.

## Key takeaways

- A sequence's final length is unknowable at admission, so contiguous allocation must either
  over-reserve or copy. Both waste memory that is then unavailable to anyone.
- Fixed-size blocks plus a per-sequence block table remove external fragmentation entirely and cap
  internal fragmentation at one partial block.
- Exhausting memory becomes a scheduling decision — preempt and recompute — rather than a failure.
- Preemption must discard the cache and keep the output. Discarding the output makes a caller
  receive the same tokens twice.
- A scheduler must count the memory it has already promised this step, not only what is free now.
- **Paging costs throughput when the gather is naive.** The memory win is real, the performance
  loss is real, and only a kernel resolves the tension.

## Looking ahead

Blocks can be shared, and reference counting is already in place. {ref}`ch09` uses that to stop
recomputing prompts the engine has already seen — which on a realistic chat workload is the
largest remaining waste in the system, and unlike this chapter, it costs nothing.

## Further reading

Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*
(SOSP 2023, Appendix E) is the primary source and is now worth reading in full: you have
implemented its block manager and run into the gather problem its kernel exists to solve.
