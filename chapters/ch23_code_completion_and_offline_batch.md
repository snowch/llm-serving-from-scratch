---
title: "Code Completion and Offline Batch"
short_title: "ch23 Code Completion and Offline Batch"
---

(ch23)=
# ch23 · Code Completion and Offline Batch [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch15](#ch15), [ch10](#ch10) |
| **Scorecard** | Two opposite configurations of the same engine, at both extremes of the frontier. |
:::

## The problem

Two workloads break our assumptions in opposite directions: completion needs TTFT humans cannot perceive, and offline batch does not care about latency at all.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Code completion: fill-in-the-middle prompting, single-digit-millisecond TTFT targets, aggressive speculation (ch15 wins big at batch 1)
- Why completion is the best case for speculation and the worst case for large batches
- Offline batch: latency is irrelevant, so maximise tokens per dollar — huge batches, maximum quantisation, full device saturation
- Sorting and bucketing offline work by length to eliminate ragged waste

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- A completion configuration: low batch, speculation on, tight token budget
- A batch configuration: maximum batch, sorted by length, quantised
- Measure both against the same engine build

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Two rows at opposite ends: minimum TTFT achieved, and maximum tokens/dollar achieved.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Neither configuration is acceptable for the other workload. If you must serve both, you need two pools — which is a capacity decision (ch27), not a tuning one.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
