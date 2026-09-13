---
title: "Multi-Replica: Routing, Autoscaling and Cold Starts"
short_title: "ch18 Multi-Replica"
---

(ch18)=
# ch18 · Multi-Replica: Routing, Autoscaling and Cold Starts [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch09](#ch09), [ch10](#ch10) |
| **Scorecard** | Aggregate throughput scales; tail latency depends almost entirely on routing quality. |
:::

## The problem

Two replicas behind round-robin deliver worse than 2x, because round-robin destroys the prefix cache ch09 built.

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Why round-robin is the wrong LLM load balancer: requests are not interchangeable units of work
- Cache-aware / prefix-affinity routing; least-outstanding-**tokens** rather than least-connections
- Autoscaling on queue depth and KV utilisation, never on CPU
- Cold starts: weight loading (safetensors mmap), warmup, CUDA-graph capture — why the first request after a deploy is always terrible

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- A router in front of N engine replicas, with pluggable policies
- Measure round-robin vs least-outstanding-tokens vs prefix-affinity on the ch09 trace
- A cold-start timeline, broken down by phase

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Scaling efficiency and tail latency per routing policy. The spread between policies is the chapter's headline.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

A router is a new failure domain and a new thing to observe. Prefix affinity trades load balance for cache hits, and can hot-spot.

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
