---
title: "Choosing a Serving Framework"
short_title: "ch30 Choosing a Framework"
---

(ch30)=
# ch30 · Choosing a Serving Framework

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — no hardware needed |
| **Prerequisites** | Most of the book. This chapter is the payoff for the rest of it. |
| **Scorecard** | No new numbers. A rubric, and a method for filling it in that does not depend on anyone's README. |
:::

## The problem

You are not going to deploy the engine in this book ({ref}`ch32` is explicit about why). You are
going to pick one, and the usual way that decision gets made is a throughput chart from a vendor
blog and whichever name is most familiar.

Throughput is rarely what decides it. Most teams that regret a choice regret it because the engine
could not do something they needed — a quantisation format, an adapter workflow, a scheduling
behaviour, an operational hook — not because it was ten percent slower. Capability fit and
operational maturity are the real axes, and they are the ones nobody publishes a chart for.

Which is convenient, because you have just spent thirty chapters learning exactly what to ask.

## The rubric

Every mechanism in this book is a question. That is the whole idea: you now know what these things
*are*, so you can ask whether a framework has them and understand the answer.

| From | Ask | Why it decides things |
|---|---|---|
| {ref}`ch07` | Continuous batching, and what the scheduler does at the step boundary | Everything else rests on it. An engine without it is not a contender |
| {ref}`ch08` | Paged KV, and **what happens when blocks run out** — preempt, swap, or refuse | Determines behaviour at the edge, which is where you will live |
| {ref}`ch09` | Prefix caching: enabled by default? keyed how? shared across requests? | Worth the most of anything here on chat and agent traffic |
| {ref}`ch10` | Chunked prefill, and whether the token budget is tunable | Decides whether long prompts stall short ones |
| {ref}`ch11` | Prefill/decode disaggregation | Only matters above a scale and an interconnect; ask if you are near it |
| {ref}`ch12` | KV offload to host or disk, and whether a tier is shared across replicas | Turns evictions from prefills into transfers |
| {ref}`ch13` | Attention implementation, and which are available for **your** hardware | A framework that only has a kernel for the GPU you do not own is not fast for you |
| {ref}`ch15` | Which quantisation formats load, and whether kernels exist or it dequantises | "Supports INT4" can mean a memory win with no speed win. {ref}`ch15` explains why |
| {ref}`ch16` | Bounded context, sliding window, sink handling | Only route to long context at a fixed memory cost |
| {ref}`ch17` | Speculative decoding, and which drafter styles | Real win at low batch; check the acceptance reporting exists |
| {ref}`ch18` | Structured output, and whether it composes with speculation and caching | Frequently bolted on and frequently interacts badly |
| {ref}`ch19` | Tensor and pipeline parallelism, and the shape constraints they impose | Decides whether your model fits at all |
| {ref}`ch20` | Cache-aware routing, if you run more than one replica | A policy change worth more than most engine features |
| {ref}`ch21` | Multi-adapter serving and per-tenant fairness or quotas | If you are multi-tenant, this is not optional |
| {ref}`ch24` | Cancellation that actually reaches the engine | Pure throughput on agent workloads, and often missing |
| {ref}`ch26` | API surface, streaming, **and the chat template you will get** | The template silently decides your cache hit rate |
| {ref}`ch27` | Which of the six signals it exports | You cannot operate what you cannot see |
| {ref}`ch28` | Load shedding and drain-on-shutdown | Decides whether overload degrades or fails |

Three questions that are not in any chapter and belong on the same sheet: **how is it released and
how often do releases break**, **what happens when you file a bug**, and **who else runs it at your
scale**. These are not engineering properties and they decide more deployments than half the rows
above.

## Filling it in honestly

A rubric is only as good as the evidence you put in it, and the available evidence is mostly
marketing. Three rules, in ascending order of cost and reliability.

**Do not trust the README.** Feature lists describe intent. Check the issue tracker for the feature
you care about, sorted by most recent — you will learn more from three open issues than from a
paragraph of documentation.

**Read the source for the mechanism, not for the feature.** You can now do this, which is the point
of {ref}`ch32`'s map. "Supports prefix caching" is a claim; the code that decides *when a block is
published* is the truth, and {ref}`ch09` found out the hard way that publishing at the wrong moment
gives a cache with a hit rate of zero and no error message.

**Test the claim on your own workload.** The most valuable column in the whole rubric is the one you
fill in by running it. {ref}`ch31` is how to do that without fooling yourself, and its harness
measures anything with three methods — including a thin client wrapping a real server.

A worked order of operations: score the rubric on documentation and source, which is a day and
eliminates most of the field; then benchmark the two or three survivors on your own traces, which is
the week that actually decides it.

## Why there is no comparison table in this chapter

You will have noticed that this chapter names frameworks and says nothing about what any of them can
do.

That is deliberate, and it is the same decision {ref}`ch29` made about hardware prices. A table of
capabilities is wrong within months — these projects ship weekly, and the rows that matter are
exactly the ones changing fastest. A reader finding this chapter a year from now would be worse off
with a confident stale table than with a rubric and a method.

There is a second reason, and it is the one that actually settles it. **I cannot verify those claims
from here.** This book's standard is that every figure is measured, stamped and reproducible;
{ref}`ch14` refused to ship a Triton kernel it could not run, and {ref}`ch31` refused to publish a
cross-engine benchmark it could not run fairly. Writing "framework X supports Y" from memory would
fail the same test, and would do it while misrepresenting somebody else's work.

So the deliverable is the sheet, not my answers on it. Copy the table, add a column per candidate,
and fill it in against the version you can actually download today.

## The cost

- **A rubric invites false precision.** Eighteen rows with ticks in them produce a score, and a score
  invites a decision without judgement. Three rows usually dominate for any given team; find out
  which three before you count anything.
- **It is biased toward what this book covers.** Distributed tracing, model-registry integration,
  auth, tokenizer compatibility and deployment tooling are absent from these rows and present in
  real decisions.
- **Capability is not maturity.** A framework can have every row and be six months from stable. The
  three unglamorous questions above carry more weight than they look like they should.
- **Evaluating properly costs a week**, and the pressure to skip it is exactly proportional to how
  urgently the model is needed.

## Key takeaways

- **Capability fit and operational maturity decide framework choices; throughput rarely does.**
- Every mechanism in this book is a question you can now ask and understand the answer to. That is
  what the previous twenty-nine chapters bought you.
- Check the issue tracker, then the source, then your own workload — in that order of cost and of
  reliability.
- Read the *mechanism*, not the feature list. "Supports prefix caching" is a claim; when a block is
  published is the truth.
- Score on paper to eliminate the field, benchmark the survivors to decide between them.
- Anyone's capability table, including one I could have written here, is wrong within months. The
  method outlasts it.

## Looking ahead

The rubric narrows the field; {ref}`ch31` decides between what is left, and is mostly about the ways
a benchmark can flatter the engine running it.

## Further reading

Each framework's own architecture documentation, read with {ref}`ch32`'s map beside it — the concepts
now have names you recognise, and the differences between the three major engines are more
interesting than their similarities. For the procurement side rather than the engineering side, the
most useful thing is to find someone running it at your scale and ask them what broke.
