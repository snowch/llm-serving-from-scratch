---
title: "Chat and Assistants"
short_title: "ch20 Chat and Assistants"
---

(ch20)=
# ch20 · Chat and Assistants [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch09](#ch09), [ch10](#ch10) |
| **Scorecard** | Same engine, tuned for this workload. Show the gain from configuration alone. |
:::

## The problem

Our defaults were tuned on a synthetic trace. A real chat workload has a different shape: long shared system prompts, growing multi-turn context, and a human watching tokens appear.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- The workload's shape: prefix reuse is high, output lengths are moderate, and ITL is perceptible to a human
- Why ITL matters more than throughput here, and what ITL target actually feels smooth
- Session affinity and its tension with load balance (ch18)
- Multi-turn growth: context expanding every turn, and when to summarise rather than resend

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- A representative chat trace and an SLO appropriate to it
- Tune ch09 cache size, ch10 token budget and ch18 routing for *this* workload
- Compare tuned vs default configuration

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Goodput against a chat SLO, tuned vs default. No code change — configuration alone.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Tuning for chat makes the same engine worse at batch (ch23). That is the point of this part: there is no single best configuration.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
