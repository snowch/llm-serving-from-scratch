---
title: "What an Inference Server Actually Does"
short_title: "ch01 What an Inference Server Actually …"
---

(ch01)=
# ch01 · What an Inference Server Actually Does [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | None — start here |
| **Scorecard** | Establishes the **baseline row**: every later number is relative to this one. |
:::

## The problem

Nothing yet — this chapter *creates* the problem. Show a single user waiting, then ten users waiting, and let the wait speak for itself.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- The full request lifecycle: HTTP -> chat template -> tokenise -> prefill -> decode loop -> detokenise -> stream -> disconnect
- Which of those steps touch the GPU, and which are pure bookkeeping (most of them)
- Why `model.generate()` is a *library* call, not a *server*

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- A ~100-line FastAPI server wrapping HuggingFace `generate()`
- One endpoint, no streaming yet, no batching, one request at a time
- Deliberately naive: this is the strawman the whole book dismantles

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Hand-timed for now — ch02 builds the real harness. Record wall-clock for 1 and 10 concurrent requests.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Nothing to trade off yet. Name the four things this server does wrong; each becomes a later chapter.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
