---
title: "Reliability and Operations"
short_title: "ch26 Reliability and Operations"
---

(ch26)=
# ch26 · Reliability and Operations [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch10](#ch10), [ch24](#ch24), [ch25](#ch25) |
| **Scorecard** | Behaviour under overload and failure, rather than a throughput number. |
:::

## The problem

Everything so far assumed the engine is healthy and load is within capacity. Neither holds for long.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Load shedding and graceful degradation: shed early, shed cheaply, shed visibly
- Draining and rolling upgrades without dropping in-flight streams
- Failure modes: OOM under a length spike, NaN/inf, GPU fault, stuck step, slow memory leak
- **Silent quality regression** — the worst failure, because nothing alerts. How to detect it (canary prompts, output distribution monitoring)

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- Shedding and drain in the server; a health endpoint that means something
- Fault injection tests: OOM, rank loss, poisoned config
- A canary-prompt check that runs continuously and alerts on drift

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Behaviour curves past saturation, and recovery time after each injected fault.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Shedding means deliberately failing requests; that is a product decision, not only an engineering one. Get agreement before choosing the threshold.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
