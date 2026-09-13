---
title: "Writing a Paged Attention Kernel in Triton"
short_title: "ch13 Writing a Paged Attention Kernel i…"
---

(ch13)=
# ch13 · Writing a Paged Attention Kernel in Triton [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 2 — single GPU required |
| **Prerequisites** | [ch08](#ch08), [ch12](#ch12) |
| **Scorecard** | Latency row improves over the ch08 gather; nothing downstream depends on this chapter. |
:::

## The problem

ch08's PyTorch gather materialises far too much and launches too many kernels. To fix it we have to write the kernel.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Triton's programming model: programs, blocks, and what the compiler handles for you
- Mapping the paged decode-attention problem onto it: one program per (sequence, head)
- Reading the block table inside a kernel; masking partial blocks

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- A paged decode-attention kernel in Triton, built up in stages, each stage tested
- Benchmark against the ch08 gather and against FlashAttention
- Numerical comparison against the reference — bitwise equality is not the bar; bounded error is

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Kernel-level microbenchmark plus the end-to-end scorecard row.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

**Optional by design** (PLAN.md §13, decision 4). Raises the hardware floor and is the least portable code in the book. Nothing after this chapter assumes it.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
