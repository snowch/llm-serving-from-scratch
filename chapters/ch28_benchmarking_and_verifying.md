---
title: "Benchmarking and Verifying a Serving Stack"
short_title: "ch28 Benchmarking and Verifying"
---

(ch28)=
# ch28 · Benchmarking and Verifying a Serving Stack

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch02](#ch02) |
| **Scorecard** | Same engine, same work, two load generators, a hundredfold difference in the reported tail. |
:::

## The problem

Every number in this book is a claim about a workload, a baseline and a service objective, and a
claim that omits any of the three is not checkable. This chapter is about running a comparison
nobody can dismiss — and about the specific way most published serving comparisons are wrong.

## The load generator is part of the measurement

{ref}`ch02` chose an open loop and gave the reasoning. Here is the alternative, implemented
faithfully rather than as a straw man: a fixed number of workers, each submitting one request,
waiting for it, and submitting the next.

```{literalinclude} ../bench/closed_loop.py
:language: python
:start-at: def run_closed_loop
:end-before:     slo = slo or SLO()
```

It is how most load tests are written. Now the same engine, the same workload, both generators:

```{include} _generated/ch28-generators.md
```

Look at the four-worker row against the open-loop row. **They achieve the same throughput.** The
engine did the same amount of work, at the same rate, on the same requests. The reported tail
latency differs by two orders of magnitude.

Neither number is a lie. They measure different things, and only one of them is the thing anyone
cares about.

The closed loop's defect is structural. When the server slows down, the generator slows with it — a
worker that is waiting is not submitting — so the offered load is *defined by the server's
throughput* and the queue can never grow. Every latency it records is measured from a moment the
server was already ready to receive the request. This is **coordinated omission**, and it does not
make latencies slightly optimistic; it deletes the entire tail it was supposed to measure.

The tell is in the last column: a closed-loop run cannot report an offered rate, only an achieved
one, because there is no arrival process. If a benchmark reports "N concurrent users" rather than "N
requests per second", it is measuring a concurrency level and its latencies mean something other than
what they appear to.

## Verifying correctness, not just speed

A faster engine that produces different output is not a faster engine. Three kinds of check, in
ascending order of what they can prove:

**Equivalence.** For anything deterministic, the optimised path must produce the *same tokens* as the
naive one. {ref}`ch05`'s KV cache, {ref}`ch07`'s batching and {ref}`ch08`'s paging are all checked
this way, and the checks caught real bugs — a padded row leaking into another sequence's attention,
a preempted request receiving its tokens twice.

**Distributional.** For anything sampled, "the same tokens" is the wrong standard. {ref}`ch15`'s
speculative decoding is only correct if it samples from the *same distribution*, which is a claim
about many samples rather than one — and the chapter shows a plausible-looking implementation whose
output is biased by an amount no eyeballing would ever find.

**Property.** Invariants a scalar cannot see: no block is leaked, no sequence attends to another's
context, a trace's declared prompt length matches its tokens.

Eyeballing a few completions proves none of these. It is the industry standard and it is worth
saying plainly that it is not a verification method.

## How to run a comparison nobody can dismiss

Ten things, and the first four are where comparisons actually go wrong:

1. **State the workload.** Prompt and output length *distributions*, not means. {ref}`ch20`'s table
   exists to show that the distribution decides the result.
2. **Use an open loop** and report the offered rate alongside the achieved one. If they differ, the
   system is saturated and every latency is a queueing measurement.
3. **State the SLO** and report goodput against it. Throughput and latency in isolation can both look
   fine while the service fails.
4. **Report percentiles, never means.** Nobody experiences the mean, and it hides exactly the
   behaviour that matters.
5. **Warm up.** First-request latency includes allocator growth, and on a GPU, graph capture. Discard
   it deliberately rather than averaging it in.
6. **Pin versions,** and record them in the result. This book's harness stamps the model, the
   hardware, the library versions and a content hash of the code that produced each figure, which is
   why regenerating a result can invalidate a chapter.
7. **Tune both sides.** A comparison against somebody else's defaults is a comparison against their
   documentation, not their engine.
8. **Run each configuration more than once** and report the spread. A single run of a scheduler is a
   sample from a distribution.
9. **Verify the outputs**, per the section above.
10. **Publish the harness.** A number nobody can reproduce is an assertion.

## What is not in this book

PLAN.md promised a fair run of this engine against vLLM, SGLang, TGI and TensorRT-LLM, and this
chapter does not have one.

The reason is rule 7 combined with the book's default tier. A fair comparison needs each engine tuned
on the same GPU, and this book's measurements are all CPU-only by design ({ref}`appendix-b` explains
why). A cross-engine comparison run here would compare four engines' CPU fallback paths, which is a
measurement of nothing.

What is here instead is the machinery to run it. `run_benchmark` needs three methods — `add_request`,
`has_work`, `step` — so a thin client wrapping any OpenAI-compatible server is measurable by the same
harness, against the same traces and the same stamping, as every engine in this book. That is the
intended path from this book to a real one, and it is left as the reader's exercise with the tools
provided rather than presented as a result nobody could reproduce.

## The cost

- **Doing this properly is slow.** Ten rules, several runs each, both engines tuned — a real
  comparison is days, not an afternoon, which is why most published ones are not.
- **An open-loop harness needs somewhere to put the queue**, so the generator's own memory becomes a
  limit at high offered rates. It is a real constraint and it is the right side to be constrained on.
- **Stamping results makes them brittle on purpose.** This book's code fingerprint invalidates every
  figure when the engine changes, which has forced full regeneration several times. That is the
  feature working.
- **Verification suites are expensive to build** and are the first thing cut. The distributional
  tests in {ref}`ch15` took longer to get right than the speculation implementation they check.

## Key takeaways

- **The load generator is part of the measurement.** The same engine, on the same work, reported a
  tail two orders of magnitude apart between an open and a closed loop.
- A closed loop cannot report an offered rate. "N concurrent users" is a concurrency level, and its
  latencies are not what they look like.
- Report goodput against a stated SLO, with percentiles, at a stated offered rate, on a stated
  length distribution. Anything less is not checkable.
- A faster engine that changes the output is not a faster engine. Equivalence tests for deterministic
  paths, distributional tests for sampled ones.
- Tune both sides or you are benchmarking documentation.
- Publish the harness. A number nobody can reproduce is an assertion.

## Looking ahead

{ref}`ch29` puts the whole journey in one table, accounts for which changes paid for their
complexity, and maps this book's chapters onto the source of the engines you would actually deploy.

## Further reading

Gil Tene's "How NOT to Measure Latency" (Appendix E) is the clearest explanation of coordinated
omission there is, and it is a conference talk rather than a paper, which tells you something about
where this knowledge lives. For the verification side, the speculative decoding papers in
{ref}`ch15`'s list contain the proofs that make a distributional test meaningful — without them you
would not know what distribution to test against.
