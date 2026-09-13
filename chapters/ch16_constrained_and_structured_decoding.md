---
title: "Constrained and Structured Decoding"
short_title: "ch16 Constrained and Structured Decoding"
---

(ch16)=
# ch16 · Constrained and Structured Decoding [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch04](#ch04), [ch09](#ch09) |
| **Scorecard** | Throughput cost if implemented naively; near-zero if done properly. Show both rows. |
:::

## The problem

Callers want valid JSON. Retrying until the model produces it wastes tokens and still fails sometimes.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Grammars and JSON Schema as finite-state machines over the token vocabulary
- Logit masking at each step; precomputing masks per FSM state (the Outlines/XGrammar idea)
- **Token healing**: why a constraint that splits a token produces garbage
- Interaction with speculation (ch15) and prefix caching (ch09) — constraints change what is cacheable

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- `llmserve/constrain/` — schema -> FSM -> per-state token mask, with caching
- Naive per-step mask construction first, to measure the cost being avoided
- Validity tests: 100% schema-valid output across a broad schema set

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Two rows: naive masking, then cached masking. The gap is the chapter's point.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Constrained output can be valid and wrong — a schema guarantees shape, never truth. Also narrows the token distribution in ways that interact badly with low temperature.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
