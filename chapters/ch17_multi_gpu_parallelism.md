---
title: "Multi-GPU: Tensor, Pipeline and Expert Parallelism"
short_title: "ch17 Multi-GPU"
---

(ch17)=
# ch17 · Multi-GPU: Tensor, Pipeline and Expert Parallelism [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 3 — 2–4 GPUs required (rented hourly; cost stated below) |
| **Prerequisites** | [ch03](#ch03), [ch12](#ch12) |
| **Scorecard** | Enables models that previously did not fit at all; per-GPU efficiency drops. |
:::

## The problem

A 70B model in FP16 does not fit on one device. Quantisation (ch14) buys ~4x; past that, the model must be split.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Tensor parallelism: splitting a layer, and exactly where the all-reduce lands
- Pipeline parallelism: stages, microbatches, and bubbles
- Expert parallelism for MoE, and why routing makes load uneven
- Collective cost arithmetic: when communication eats the win
- What breaks in practice: head counts not divisible, uneven splits, one slow rank

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- Tensor-parallel attention and MLP blocks in `llmserve/parallel/`
- NCCL setup, rank/world plumbing, and a correctness check across ranks
- Scaling measurements at 1, 2 and 4 GPUs

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Throughput and latency vs GPU count, plus **scaling efficiency** — the number that shows what communication costs.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

**States the hourly rental cost to reproduce.** Operationally much harder: one failed rank kills the replica. Often quantisation plus a smaller model is the better answer — say when.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
