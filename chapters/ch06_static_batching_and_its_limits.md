---
title: "Static Batching and Its Limits"
short_title: "ch06 Static Batching and Its Limits"
---

(ch06)=
# ch06 · Static Batching and Its Limits [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch05](#ch05) |
| **Scorecard** | Throughput row up, utilisation row poor. |
:::

## The problem

One sequence at a time leaves the device almost idle: per ch03, decode reads all the weights to produce a single token. Those weight reads should be amortised over many sequences.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Batching amortises weight reads — the direct consequence of decode being bandwidth-bound
- Padding, attention masks, position IDs; left vs right padding and when each is wrong
- **The ragged-completion problem**: the batch is held hostage by its longest generation

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- Batched prefill and batched decode over a fixed batch
- Instrumentation: per-step count of sequences doing useful work vs padding
- A trace with deliberately mixed output lengths, to make the waste visible

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Throughput up several-fold, plus a new utilisation metric showing how much of the batch is wasted work.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

TTFT now depends on batch assembly, and a single long request degrades everyone behind it. Both are fixed in ch07.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
