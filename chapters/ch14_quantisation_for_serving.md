---
title: "Quantisation for Serving"
short_title: "ch14 Quantisation for Serving"
---

(ch14)=
# ch14 · Quantisation for Serving [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU runnable; Tier 2 GPU for the published numbers |
| **Prerequisites** | [ch03](#ch03), [ch12](#ch12) |
| **Scorecard** | Memory row down sharply, throughput up, **quality row moves for the first time**. |
:::

## The problem

Per ch03, decode reads all the weights every token. The cheapest way to read fewer bytes is to store fewer bytes.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Symmetric and asymmetric quantisation, scales and zero-points, per-tensor vs per-channel vs per-group
- Weight-only INT8 and INT4 (GPTQ, AWQ); outliers and why naive INT8 fails (LLM.int8(), SmoothQuant)
- FP8 on recent hardware: what changes when the format is native
- **KV-cache quantisation** — usually the bigger win for long context, and usually forgotten
- Calibration: what data, how much, and how to tell when it is wrong

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- `llmserve/quant/` — INT8 weight-only from scratch, then library-backed INT4
- KV-cache quantisation in the ch08 block store
- A quality harness: perplexity **and** a task eval, not just perplexity

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Three axes at once: quality, speed, memory. A table the reader can use to choose, not a single headline number.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

The first change that makes output *worse*. Every quantisation claim needs a quality number beside it, and 'negligible' is not a number.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
