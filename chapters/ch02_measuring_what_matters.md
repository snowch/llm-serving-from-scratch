---
title: "Measuring What Matters"
short_title: "ch02 Measuring What Matters"
---

(ch02)=
# ch02 · Measuring What Matters [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch01](#ch01) |
| **Scorecard** | Builds the harness that produces **every scorecard row in the book**. |
:::

## The problem

Chapter 1 was hand-timed. Hand-timing cannot distinguish a p50 regression from a p99 catastrophe, and cannot generate load at a controlled rate at all.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- TTFT, ITL/TPOT, end-to-end latency: what each one means to a user, and which the reader will feel
- Throughput vs **goodput** — requests/sec that met an SLO, the only number worth optimising
- Why means lie: show a distribution where the mean is fine and p99 is unusable
- Open-loop vs closed-loop load generation, and **coordinated omission** — why a closed-loop harness silently hides overload

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- `bench/harness.py`: Poisson arrivals at a target rate, per-request timing records
- `bench/traces/`: fixed request traces with realistic input/output length distributions
- Scorecard emitter: JSON to `bench/results/`, stamped with model, hardware, versions, date (PLAN.md §6.3)
- The scorecard table renderer that every later chapter calls

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Re-measure ch01's server properly. This is the true baseline row.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

The harness is now a dependency of every chapter. Keep it boring and stable; a harness change invalidates every committed result.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
