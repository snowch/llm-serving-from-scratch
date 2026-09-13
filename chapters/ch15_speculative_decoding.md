---
title: "Speculative Decoding"
short_title: "ch15 Speculative Decoding"
---

(ch15)=
# ch15 · Speculative Decoding [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch07](#ch07), [ch10](#ch10) |
| **Scorecard** | Large ITL improvement at low batch; **negative** at high batch. Both rows get published. |
:::

## The problem

Decode is serial by construction: one token per forward pass. At low load the device is idle between dependencies.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Draft-and-verify: generate k cheap tokens, verify them in one target pass
- **Why rejection sampling preserves the target distribution exactly** — the chapter's real payload, worked through properly
- Acceptance rate: how it is measured, what drives it, and the expected-speedup arithmetic
- Variants: draft model, self-speculation (Medusa, EAGLE), prompt-lookup n-gram

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- `llmserve/speculative/` — one full variant, with the acceptance test implemented exactly
- Distributional equivalence test: empirical token distributions must match the unspeculated path
- Acceptance-rate metric on the scorecard

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Sweep batch size. Show the crossover where speculation stops helping and starts hurting.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Extra compute per token and a second model to manage. **It loses at high batch** — say so in the same breath as the low-batch win.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
