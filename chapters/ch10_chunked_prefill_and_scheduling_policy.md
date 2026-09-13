---
title: "Chunked Prefill and Scheduling Policy"
short_title: "ch10 Chunked Prefill and Scheduling Pol…"
---

(ch10)=
# ch10 · Chunked Prefill and Scheduling Policy [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch07](#ch07), [ch09](#ch09) |
| **Scorecard** | ITL p99 improves markedly; prefill gets slightly slower. |
:::

## The problem

One 8K-token prompt stalls every streaming response in the batch. Show the ITL spike: a user mid-sentence stops for hundreds of milliseconds because someone else arrived.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Prefill and decode compete: one is compute-bound, the other bandwidth-bound (ch03)
- Chunked prefill: split a long prefill across steps under a per-step **token budget**
- Policy: FCFS vs priority vs fair-share; admission control and queue limits
- Just enough queueing theory to explain why p99 explodes as utilisation approaches 1

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- Token-budget scheduling in `llmserve/scheduler/policy.py`
- Chunked prefill with correct positional handling across chunk boundaries
- A load sweep: latency vs utilisation, to locate the knee

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

ITL p99 down, TTFT for long prompts slightly up. Show the latency-vs-utilisation curve and mark the operating point.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

A new knob that trades TTFT for ITL with no universally right setting — it depends on workload, which is why Part VI exists.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
