---
title: "RAG and Long Context"
short_title: "ch21 RAG and Long Context"
---

(ch21)=
# ch21 · RAG and Long Context [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch09](#ch09), [ch10](#ch10), [ch14](#ch14) |
| **Scorecard** | Prefill-dominated workload; KV pressure is the binding constraint. |
:::

## The problem

RAG inverts the ratio: prompts of 8-32K tokens producing 200-token answers. Almost all the work is prefill, and the KV cache fills immediately.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Prefill-dominated economics: cost is driven by input tokens, so retrieved-context size is a cost decision
- KV pressure at long context, and why ch14's KV quantisation matters most here
- Chunked prefill (ch10) plus prefix caching (ch09) plus KV quantisation together — they compose
- Co-serving embedding and reranker models on the same hardware, and how to budget between them

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- A RAG trace with realistic retrieved-context sizes
- Stack the three techniques and attribute the gain to each
- An embedding/reranker co-serving configuration with its own accounting

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Goodput against a RAG SLO, with a breakdown of which technique contributed what.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Longer context is not free even when it fits: quality does not rise monotonically with retrieved tokens, and cost certainly does.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
