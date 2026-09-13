---
title: "The API Surface"
short_title: "ch24 The API Surface"
---

(ch24)=
# ch24 · The API Surface [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch07](#ch07) |
| **Scorecard** | No throughput change from the API itself; **cancellation handling recovers real capacity**. |
:::

## The problem

Our engine has no usable interface. Every client library in existence already speaks one protocol, so the choice is made for us.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- OpenAI-compatible `/v1/chat/completions` and `/v1/completions`: the fields that matter and the ones that lie
- SSE streaming: framing, flushing, keepalives, and usage accounting on a stream
- **Client disconnect and cancellation** — wasted GPU that nobody monitors (measured in ch22)
- Backpressure, queue limits, timeouts, request size caps: what to return when full, and why 503 beats a 30-second hang
- Chat templates, and the many quiet ways they are wrong

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- `llmserve/server/` — FastAPI app, streaming, cancellation propagated into the scheduler
- Chat-template handling with a test per supported model family
- Admission control wired to ch10's queue limits

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Cancellation recovery: wasted tokens before and after. Plus a compatibility test against a real OpenAI client.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Protocol compatibility means inheriting someone else's semantics, including the awkward parts. Document every deviation.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
