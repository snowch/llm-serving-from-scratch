---
title: "Disaggregating Prefill and Decode"
short_title: "ch11 Disaggregating Prefill and Decode"
---

(ch11)=
# ch11 · Disaggregating Prefill and Decode [DRAFT]

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch10](#ch10) |
| **Scorecard** | Better TTFT/ITL separation at meaningful complexity cost. Be honest if it does not pay at our scale. |
:::

## The problem

ch10 reduced prefill/decode interference but could not eliminate it — they still share one device. What if they did not?

[To write: open with the measurement from the previous chapter that is unacceptable. Show it,
do not assert it.]

## The idea

- Separate prefill and decode pools (DistServe / Splitwise pattern)
- The KV handoff: what must be transferred, and why interconnect bandwidth decides viability
- Independent scaling of the two pools, and the arithmetic for sizing each

[To write: develop the mechanism from first principles, with the arithmetic that predicts the
result.]

## The build

- A simplified two-pool engine with an explicit KV transfer step
- Measure transfer cost directly; compare against the ch10 single-pool numbers

[To write: incremental implementation. Quote code from `llmserve/` with `{literalinclude}`
and `:start-at:` / `:end-at:` anchors — never paste it inline.]

## The measurement

Compare against ch10 on the same trace. Report the transfer overhead as its own line.

[To write: re-run the harness, append the scorecard row, and explain in one paragraph why the
number moved. Read the figures from `bench/results/` — never type a number into prose
(PLAN.md §6.3).]

## The cost

Substantially more moving parts, and a hard dependency on fast interconnect. **State plainly the scale below which this is the wrong choice.**

[To write: what did this make worse? Complexity, latency variance, quality, memory, operational
burden. This section is mandatory — the chapter is not done without it (PLAN.md §12.3).]

## Key takeaways

- [To write: 3–5 bullets, each a claim the chapter earned.]

## Looking ahead

[To write: the transition that sets up the next chapter's problem.]

## Further reading

[To write: primary sources, cited with MyST citation syntax against `references.bib`.]
