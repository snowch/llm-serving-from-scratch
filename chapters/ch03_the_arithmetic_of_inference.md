---
title: "The Arithmetic of Inference"
short_title: "ch03 The Arithmetic of Inference"
---

(ch03)=
# ch03 · The Arithmetic of Inference [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch01](#ch01), [ch02](#ch02) |
| **Scorecard** | No engine change — this chapter makes every later row **predictable in advance**. |
:::

## The problem

The ch02 numbers are facts without explanation. Without a model of *why* they are what they are, every optimisation is guesswork.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- FLOPs per token (~`2 x params`) vs bytes moved per token; arithmetic intensity and the roofline
- **Prefill is compute-bound. Decode is memory-bandwidth-bound.** The book's central claim (PLAN.md §3.1)
- Derive the single-stream decode ceiling: roughly `HBM bandwidth / bytes read per forward pass`
- Derive KV cache per token: `2 x n_layers x n_kv_heads x head_dim x bytes_per_element`

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- A small calculator module: given a model config and a device, predict decode tok/s and KV bytes/token
- Apply it to the book's reference models, and to one 7B and one 70B model for scale

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

**Predict** ch01's throughput from arithmetic alone, then compare to the measured value. Explain the gap.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

A model, not the truth. Name what it ignores (kernel launch overhead, attention cost growth, memory-allocator behaviour) so the reader is not surprised when it is 20% off.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
