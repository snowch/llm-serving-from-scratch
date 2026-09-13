---
title: "Generating Tokens Correctly"
short_title: "ch04 Generating Tokens Correctly"
---

(ch04)=
# ch04 · Generating Tokens Correctly [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch01](#ch01) |
| **Scorecard** | No performance change. Establishes the **correctness baseline** that every later optimisation is tested against. |
:::

## The problem

`generate()` hides all the decisions. Before optimising anything we need our own decode path whose output we can hold fixed.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Logits -> probabilities: temperature, top-k, top-p, min-p; numerically stable softmax and why naive log-sum-exp bites
- Repetition, presence and frequency penalties, and how they interact
- Determinism: seeding, and why 'same seed, same output' is harder than it looks
- Stop conditions: EOS, `max_tokens`, stop strings
- **Incremental detokenisation**: partial UTF-8 sequences and BPE boundaries. Under-documented and a genuine source of production bugs

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- `llmserve/sampling.py` — every sampler from scratch, no library helpers
- `llmserve/tokenizer.py` — streaming-safe incremental detokeniser
- `tests/test_equivalence.py` — greedy output must be token-identical to the HF reference

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Correctness, not speed: the equivalence suite passes. Note any throughput cost from leaving the fast path.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

We now own code HuggingFace used to own. Every bug here is ours; hence the test suite lands in this chapter, not later.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
