---
title: "Continuous Batching"
short_title: "ch07 Continuous Batching"
---

(ch07)=
# ch07 · Continuous Batching [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch06](#ch06) |
| **Scorecard** | Expected to be the **single largest jump** in the book. |
:::

## The problem

ch06 measured it: sequences that finished early sit padded in the batch while one long generation runs. The batch boundary is the problem.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Iteration-level scheduling (after Orca): admit and retire sequences *per step*, not per batch
- The engine as a state machine: waiting / running / preempted / finished
- Why this is a scheduling change, not a kernel change

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- Restructure into `llmserve/engine.py` with a `step()` that advances all running sequences one token
- `llmserve/scheduler/` — the running set, the waiting queue, admission per step
- Equivalence tests must still pass: reordering must not change any output

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Throughput up sharply and TTFT down, on the same hardware. Note where mean batch size now sits.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Real complexity arrives: the engine is now a scheduler. TTFT variance grows, and a burst of admissions can spike memory.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
