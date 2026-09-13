---
title: "The Finished Engine"
short_title: "ch32 The Finished Engine"
---

(ch32)=
# ch32 · The Finished Engine

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | None — start here |
| **Scorecard** | **The whole scorecard**, ch01 to ch32, in one table. |
:::

## Where we started and where we got to

```{include} _generated/ch32-journey.md
```

The first five rows are directly comparable: same trace, same rate, same machine, and they are the
spine of the book. Everything after them is listed with its workload because it is *not* comparable,
and pretending otherwise would undo the discipline the rest of the book insists on.

A few things are worth pointing at.

**The largest single jump is {ref}`ch07`,** and it is not close. Continuous batching is not an
optimisation of static batching; it is a different answer to the question of what a batch is. Every
chapter after it is a refinement of a loop that chapter established.

**{ref}`ch08` is slower than {ref}`ch07` on this row**, and shipped anyway. Paging costs indirection
on every access and buys concurrency — a thing the uniform trace at this rate does not need and
every real workload does. A book that only reported wins would have quietly dropped this row, and it
is one of the most useful rows in the table.

**{ref}`ch09`'s gain lives in the TTFT column, not the throughput column**, and only on a workload
with shared text. On the uniform trace it does nothing whatsoever, correctly.

**{ref}`ch10` looks terrible here** for a reason worth restating: its row is a *mixed-length* trace
that no other row ran, chosen because chunking only matters when long prompts and short ones compete.
The comparison chunked prefill actually won is inside {ref}`ch10`, against itself at a different
budget.

## What paid for its complexity, and what did not

Honest accounting, in rough order of value per line of code:

| Change | Verdict |
|---|---|
| KV cache ({ref}`ch05`) | Unambiguous. Nothing else in the book has this ratio of benefit to complexity. |
| Continuous batching ({ref}`ch07`) | Unambiguous, and the loop it establishes carries every later chapter. |
| Prefix caching ({ref}`ch09`) | Nearly free, and enormous on workloads with shared prompts. The rare optimisation with no trade in it. |
| KV offload ({ref}`ch12`) | Restores reuse a small pool destroys, and the break-even bandwidth is low enough that any real tier clears it. Costs latency on a rig with no second device, which is exactly what its chapter measured. |
| Bounding the context ({ref}`ch16`) | The only thing here that turns a linear cache into a constant one, and the only optimisation in the book that changes what the model says. Priced in perplexity, not milliseconds. |
| Paged attention ({ref}`ch08`) | Pays for itself through concurrency, and everything after it depends on the block allocator. Cancellation ({ref}`ch24`) and quotas ({ref}`ch21`) are cheap only because it exists. |
| Cache-aware routing ({ref}`ch20`) | A policy change, no new machinery, and it turned zero goodput into real goodput. Best value in Part V. |
| Fair queueing ({ref}`ch21`) | One overridden method. Does not create capacity; decides who waits, which was the entire problem. |
| Load shedding ({ref}`ch28`) | Small, and the difference between degrading and failing. |
| Chunked prefill ({ref}`ch10`) | Real, and narrower than its reputation. It buys tail latency on mixed workloads and costs throughput; on uniform traffic it is a pure loss. |
| Quantisation ({ref}`ch15`) | Worth it for memory and for cold starts ({ref}`ch20`). Sold as a speedup, delivered as capacity. |
| Speculative decoding ({ref}`ch17`) | Genuinely useful at low batch size, and the correctness argument is subtle enough that most of the value of the chapter is knowing what to check. |
| Disaggregation ({ref}`ch11`) | Cannot be evaluated on one machine. Its viability is a property of the interconnect, which is why that chapter's decisive table is arithmetic. |
| Constrained decoding ({ref}`ch18`) | Cheap, and buys a guarantee weaker than the one people assume they are buying. |

Two of those rows are the same lesson from opposite ends. {ref}`ch12` keeps more of the cache by
finding somewhere else to put it; {ref}`ch16` keeps less of it on purpose. Which one a deployment
needs is decided by whether its problem is capacity or context length, and a deployment that reaches
for the second when it needed the first has paid in quality for nothing.

The pattern across the table: **the changes that paid best were changes to scheduling and memory
policy, not to arithmetic.** Almost nothing in this book made a matrix multiply faster. Nearly all
of it made the engine do fewer of them, or do them on behalf of more requests at once.

## What we deliberately did not build

Each of these is a real gap, and each is listed with what the omission costs rather than waved away.

- **Multi-node serving.** Everything here fits on one machine. Crossing a node boundary changes the
  interconnect by an order of magnitude, which changes which of {ref}`ch11` and {ref}`ch19` are
  viable at all. The mechanisms transfer; the arithmetic that decides whether to use them does not.
- **Mixture-of-experts routing at scale.** Expert parallelism has a scheduling problem this book does
  not have: the batch's expert assignment is data-dependent, so load across devices is uneven in a
  way no static plan fixes. {ref}`ch19` sketches it; serving a real MoE needs more.
- **Custom CUDA beyond {ref}`ch14`'s Triton kernel.** The kernel chapter shows the shape of the work
  and stops. Production kernels are where a large fraction of real engines' performance comes from,
  and writing them well is a different skill from everything else in this book.
- **A real model.** The reference model is built from seeded random weights so that every
  measurement reproduces anywhere ({ref}`appendix-b` explains the trade). The consequence is that no
  output in this book is meaningful text, and quality claims ({ref}`ch15`, {ref}`ch17`,
  {ref}`ch21`) are made against a small model trained on a synthetic corpus rather than against a
  production one.
- **A cross-engine comparison.** {ref}`ch31` builds the methodology and deliberately does not report
  numbers against vLLM, SGLang or TensorRT-LLM, because a fair run needs hardware the book's default
  tier does not have. The harness is engine-agnostic; running it is left to a reader with a GPU.
- **Security and abuse.** Rate limits appear as quotas in {ref}`ch21` and nothing else does. Prompt
  injection, data exfiltration through the cache, and adversarial inputs that trigger worst-case
  scheduling are all real and all absent.

## Reading the real engines

The point of this book is to make three large codebases readable. Here is roughly where each concept
lives; treat the names as search terms rather than paths, because directory layouts move and these
projects move quickly.

**vLLM** is the closest to this engine, and the best first read. Look for the scheduler — the
`schedule()` step that decides which sequences run this iteration — which is {ref}`ch07` and
{ref}`ch08` together. The block manager and block tables are {ref}`ch08`; the prefix caching built on
top of them is {ref}`ch09`; the chunked-prefill token budget is {ref}`ch10`. The engine's `step()` is
the function this book has been circling since chapter 7, and you will recognise it.

**SGLang** is where to look for {ref}`ch09` and {ref}`ch18` done seriously. RadixAttention is prefix
caching with a radix tree instead of a flat hash map, which handles branching conversations that this
book's cache handles poorly. Its structured-output path is {ref}`ch18` with the token-lifting problem
actually solved, and its router is {ref}`ch20`.

**TensorRT-LLM** is the least like this book and the most instructive because of it. Much of what is
runtime scheduling here is a build-time decision there: the model is compiled into an engine plan,
kernels are selected ahead of time, and shapes are fixed at build. Read it to see which of this
book's decisions are essential and which are consequences of choosing a runtime-flexible design. Its
in-flight batching is {ref}`ch07` under a different name.

Three things to look for in any of them, because they are where the real differences are:

1. **What the scheduler does when memory runs out.** Preempt-and-recompute, swap-to-host, or refuse.
   {ref}`ch08` and {ref}`ch28`.
2. **Where the prefix cache is keyed, and when entries are published.** Publishing on completion
   rather than on block fill is the difference between a cache that works and one that reports a
   hit rate of zero, which this book found out the hard way.
3. **Whether prefill and decode share a step.** {ref}`ch10`'s answer decides the entire latency
   profile, and every engine answers it differently.

## The cost

**This engine is a teaching artefact and you should not run it in production.** That is not modesty;
it is the accurate description of what it is for. It has no kernel optimisation, no multi-node
support, no security posture, no CUDA graphs, no quantised kernels, and a model implementation that
exists to be read rather than to be fast. Every one of those gaps is load-bearing in a real
deployment.

Use vLLM, SGLang or TensorRT-LLM. What this book buys you is the ability to read them, to know which
knobs matter for your workload, to recognise from a latency profile what the scheduler is doing, and
to tell a benchmark that means something from one that does not. That is a better return than a
slightly worse engine of your own.

## Key takeaways

- **Scheduling and memory policy beat arithmetic.** Almost nothing in this book made a matrix
  multiply faster, and the throughput went up by most of an order of magnitude anyway.
- **The two-phase model explains nearly everything.** Prefill is compute-bound and parallel, decode
  is memory-bound and sequential; most surprising serving results follow directly.
- **A number without its workload and its SLO is not a result.** Half this book's tables exist to
  make that concrete, and the journey table above needs a workload column for exactly that reason.
- **Every optimisation has a trade, and the chapters that found none were the suspicious ones.**
  Prefix caching genuinely has no trade. Paging, chunking, quantisation, speculation, affinity
  routing and fair queueing all do, and knowing which one you are paying is the job.
- **The gap between this engine and a production one is mostly kernels and operations**, not ideas.
  You now have the ideas.

## Where to go next

Pick the thing that is actually hurting you.

If your tail latency is bad and your throughput is fine, you have a scheduling problem: re-read
{ref}`ch07`, {ref}`ch10` and {ref}`ch27`, and go look at your queue depth. If your throughput is bad,
you have a memory problem: {ref}`ch03`, {ref}`ch08` and {ref}`ch15`, and go compute your KV bytes per
token. If your costs are bad, {ref}`ch29` and go find your utilisation — it is almost certainly lower
than you think, and that is almost certainly the whole answer.

And if you are about to publish a performance comparison, {ref}`ch31` first.

## Further reading

Everything in {ref}`appendix-e`, and then the source of one engine. Reading a real scheduler after
finishing this book is a different experience from reading one before, which was the entire point.
