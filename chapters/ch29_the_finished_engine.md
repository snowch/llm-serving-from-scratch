---
title: "The Finished Engine"
short_title: "ch29 The Finished Engine"
---

(ch29)=
# ch29 · The Finished Engine [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | None — start here |
| **Scorecard** | **The whole scorecard**, ch01 to ch29, in one table. |
:::

## The problem

No problem left to solve — this chapter accounts for what was solved.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- The full scorecard: baseline to finished engine, every row, with the chapter that moved it
- Design retrospective: which changes paid for their complexity and which did not
- What we deliberately did not build — multi-node, MoE routing at scale, custom CUDA beyond ch13, disaggregation at real scale — and what each omission costs
- **A guided map into the real codebases**: where each concept in this book lives in vLLM, SGLang and TensorRT-LLM, so the reader can now read them fluently

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- Nothing new. Consolidate, tag the final checkpoint, and point outward

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

The complete table, plus the ratio between the first and last rows.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

The engine is a teaching artefact, not production software. State clearly what a reader should use in production instead, and why that is the right call.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
