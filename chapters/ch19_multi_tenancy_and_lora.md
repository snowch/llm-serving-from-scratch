---
title: "Multi-Tenancy and LoRA at Serving Time"
short_title: "ch19 Multi-Tenancy and LoRA at Serving …"
---

(ch19)=
# ch19 · Multi-Tenancy and LoRA at Serving Time [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch07](#ch07), [ch08](#ch08) |
| **Scorecard** | Serves many adapters on one base model; per-request overhead rises modestly. |
:::

## The problem

Fifty fine-tunes of one base model would mean fifty replicas. The weights are ~99% identical.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- LoRA as a low-rank delta; why adapters can be applied per-request inside one batch (S-LoRA pattern)
- Adapter memory management and paging, reusing ch08's allocator
- Fairness between tenants: a shared KV cache is a shared resource, so noisy neighbours are inevitable without policy
- Quotas, per-tenant rate limits, and priority interaction with ch10's scheduler

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- `llmserve/lora/` — batched adapter application with mixed adapters in one step
- Per-tenant accounting and quota enforcement in the scheduler
- A multi-tenant trace with deliberately unequal load

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Throughput with mixed adapters vs single-adapter baseline, plus a per-tenant fairness measure.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Per-step overhead and considerable bookkeeping. Isolation is best-effort: one tenant can still degrade another through cache pressure alone.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
