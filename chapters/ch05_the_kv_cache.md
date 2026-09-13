---
title: "The KV Cache"
short_title: "ch05 The KV Cache"
---

(ch05)=
# ch05 · The KV Cache [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch03](#ch03), [ch04](#ch04) |
| **Scorecard** | First real speedup row. |
:::

## The problem

Our decode loop re-reads the entire prefix on every single token. Show the quadratic cost: time per token climbing as the sequence grows.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- What is actually recomputed, and why caching K and V is sufficient
- Recompute O(n^2) -> O(n), and what that does *not* fix
- The cost side: apply the ch03 formula to get bytes per token, per sequence, per concurrent user

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- `llmserve/cache/kv.py` — a per-sequence contiguous KV cache
- Wire it into the decode loop; keep the equivalence tests green
- Then compute how many concurrent sequences fit in memory — and hit the wall deliberately

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Large speedup, flat per-token latency. Then a second row showing the concurrency ceiling memory now imposes.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Traded compute for memory, and memory is now the binding constraint. This is the setup for all of Part III.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
