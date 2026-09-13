---
title: "Attention at Speed"
short_title: "ch12 Attention at Speed"
---

(ch12)=
# ch12 · Attention at Speed [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU runnable; Tier 2 GPU for the published numbers |
| **Prerequisites** | [ch08](#ch08) |
| **Scorecard** | Latency row improves; KV-cache footprint row improves a lot with GQA. |
:::

## The problem

ch08's paged gather is correct and slow, and ch05's KV cache is large. Both are attention problems.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Online softmax and tiling from first principles — derive why FlashAttention is **IO-aware**, not fewer-FLOPs
- Recomputation in the backward pass, and why inference only needs the forward story
- KV-shrinking architectures: MQA, GQA, sliding-window, and MLA-style latent compression
- What to write yourself vs what to call — and how to tell the difference

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- Swap in a FlashAttention backend behind our attention interface
- Configure GQA on a model that supports it; recompute KV bytes/token with the ch03 formula
- Keep the equivalence tests green across every backend

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Latency down, KV footprint down. Publish GPU-tier numbers; note what the CPU path can and cannot show.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

A dependency on a fast-moving library, and an attention path that is now much harder to debug. Pin the version and keep the readable backend as a reference oracle.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
