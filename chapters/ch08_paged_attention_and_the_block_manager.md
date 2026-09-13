---
title: "Paged Attention and the Block Manager"
short_title: "ch08 Paged Attention and the Block Mana…"
---

(ch08)=
# ch08 · Paged Attention and the Block Manager [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch05](#ch05), [ch07](#ch07) |
| **Scorecard** | Concurrency and throughput row up, via memory recovered from fragmentation. |
:::

## The problem

ch07's scheduler wants to admit more sequences than memory allows — because a contiguous per-sequence cache must reserve for the *maximum* length, not the actual one. Measure the waste.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Internal and external fragmentation in a contiguous KV cache
- Fixed-size blocks plus a block table: the OS-paging analogy, and where it breaks down
- Copy-on-write for forked sequences (beam search, n-sample requests)
- Preemption when memory runs out: recompute vs swap-to-CPU, and how to choose

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- `llmserve/cache/blocks.py` — block pool, allocator, per-sequence block tables
- `llmserve/attention/paged.py` — start with a readable PyTorch gather
- Preemption path in the scheduler, with a preemption-rate metric

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Higher concurrency at the same memory. Report peak KV utilisation and preemption rate alongside throughput.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Indirection costs a little per-step time and a lot of clarity. The naive gather kernel is slow — ch13 fixes that for GPU readers.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
