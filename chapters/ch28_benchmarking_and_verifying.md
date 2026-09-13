---
title: "Benchmarking and Verifying a Serving Stack"
short_title: "ch28 Benchmarking and Verifying a Servi…"
---

(ch28)=
# ch28 · Benchmarking and Verifying a Serving Stack [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU runnable; Tier 2 GPU for the published numbers |
| **Prerequisites** | [ch02](#ch02), [ch04](#ch04) |
| **Scorecard** | The methodology chapter. Validates every row in the book, and the framework comparison. |
:::

## The problem

The book has been making performance claims for 27 chapters. This chapter is where they are audited — including against the possibility that our own harness has been flattering us.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Trace-driven load: why input/output length distributions dominate results, and how a wrong distribution invents a winner
- Warmup, steady state, run length, and reporting statistics rather than a single number
- Version pinning and hardware disclosure as a minimum standard for a credible claim
- **Correctness verification**: output-distribution equivalence, not eyeballing — the discipline from ch04 applied to whole stacks
- How published benchmarks mislead, and the questions to ask of any vendor number

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- Harden `bench/` into a tool that can drive any OpenAI-compatible endpoint
- Run our engine against vLLM, SGLang, TGI and TensorRT-LLM on identical traces
- Cross-check correctness: same prompts, compare output distributions across all stacks

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

The comparison table, with full methodology disclosure — including where our engine loses, which will be in several places.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Benchmarking well is slow, and results expire. Publish the method so others can rerun it; do not defend the numbers past their date.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
