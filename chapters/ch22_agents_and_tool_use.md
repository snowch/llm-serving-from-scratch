---
title: "Agents and Tool Use"
short_title: "ch22 Agents and Tool Use"
---

(ch22)=
# ch22 · Agents and Tool Use [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch09](#ch09), [ch24](#ch24) |
| **Scorecard** | Cancellation handling and prefix reuse dominate; raw throughput barely matters. |
:::

## The problem

Agent workloads are many short, highly repetitive calls, and a large fraction are abandoned mid-generation when the orchestrator changes its mind.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- The shape: short outputs, heavy prefix overlap, high request rate, frequent cancellation
- **Tail latency amplification**: a chain of N calls is gated by the slowest, so p99 becomes the number that matters
- Cheap cancellation as a throughput feature — abandoned work is pure waste
- Prefill-heavy economics again, and why prompt caching is the whole ballgame

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- An agent trace with realistic cancellation rates
- Measure wasted GPU time from late cancellation, then fix it (ties to ch24)
- Tune prefix cache retention for high-overlap, short-output traffic

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Goodput plus a **wasted-work** metric: fraction of generated tokens nobody consumed.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Optimising for cancellation adds scheduler paths that are easy to get wrong and hard to test. Aggressive cache retention for agents starves other workloads.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
