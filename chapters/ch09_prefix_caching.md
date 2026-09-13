---
title: "Prefix Caching"
short_title: "ch09 Prefix Caching"
---

(ch09)=
# ch09 · Prefix Caching [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch08](#ch08) |
| **Scorecard** | Large TTFT improvement on realistic traces; little change on synthetic ones. |
:::

## The problem

Every request in a chat workload re-prefills the same system prompt. On a realistic trace, measure how much prefill compute is pure duplication.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Content-hashed blocks: identical prefix -> identical blocks -> share them
- From a flat hash map to a **radix tree** (à la SGLang RadixAttention) for partial prefix sharing
- Eviction: LRU over a reference-counted block pool, and why refcounts are mandatory
- The invariant that matters: a cache hit must never change output

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- `llmserve/prefix/` — hashing, the radix tree, refcounting, LRU eviction
- Integrate with ch08's allocator; add a cache-hit-rate metric
- A trace with a long shared system prompt plus multi-turn conversations

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

TTFT drops sharply on the shared-prefix trace. Report hit rate next to it — the two move together.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Memory spent on cache is memory not spent on running sequences. Eviction policy is now a tuning knob, and a bad one is worse than no cache.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
