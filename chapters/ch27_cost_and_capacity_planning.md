---
title: "Cost and Capacity Planning"
short_title: "ch27 Cost and Capacity Planning"
---

(ch27)=
# ch27 · Cost and Capacity Planning [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch03](#ch03), [ch14](#ch14), [ch23](#ch23) |
| **Scorecard** | No engine change. Converts every previous row into money. |
:::

## The problem

Everything so far is measured in tokens and milliseconds. Decisions get made in currency.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Deriving $/million tokens from hardware cost, utilisation and token mix — input and output priced separately, because ch21 showed they differ
- The batch-size vs SLO frontier: where to sit, and what each step along it costs
- Hardware selection using ch03's arithmetic instead of vendor benchmarks
- Spot and preemptible economics against the reliability requirements of ch26
- **Honest build-vs-buy** against hosted APIs, including the engineering time this book represents

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- A capacity model: workload shape in, hardware and cost out
- Apply it to the four Part VI workloads and compare
- A break-even analysis versus current hosted pricing, with the date stated

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

$/million tokens for each Part VI workload, on each hardware option considered.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Prices and hardware move constantly. The model is the durable artefact; the numbers are a snapshot, and every one carries its date.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
