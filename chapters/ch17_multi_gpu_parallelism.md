---
title: "Multi-GPU: Tensor, Pipeline and Expert Parallelism"
short_title: "ch17 Multi-GPU"
---

(ch17)=
# ch17 · Multi-GPU: Tensor, Pipeline and Expert Parallelism

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 3 for the timings — 2–4 GPUs. The split itself, and its correctness, run on any laptop. |
| **Prerequisites** | [ch03](#ch03), [ch12](#ch12) |
| **Scorecard** | Enables models that did not fit at all. Per-device efficiency drops, and how much is a property of the interconnect rather than the software. |
:::

:::{warning} What this chapter measures, and what it does not
This book's default tier has one device. A simulated all-reduce on one device measures a memory
copy, not a network, and reporting it as a speedup would be exactly the kind of number
{ref}`ch28` warns about.

So this chapter does two things it *can* do honestly. It builds the split and proves it is
**exactly correct** — bit-for-bit the same output as the unsharded model — which is the part
implementations get wrong and which needs no GPU at all. And it computes the communication cost
**exactly from the model's shape and the link's two numbers**, which is what decides whether any
of this is viable and is more trustworthy than a measurement on the wrong hardware.

What is missing is a wall-clock speedup on real devices. If you have them, the code here runs;
the arithmetic below tells you what to expect before you rent anything.
:::

## The problem

A 70B model in fp16 is 140 GB of weights. No single accelerator holds that, and the KV cache still
has to fit alongside it. At that point the question stops being "how do we serve this faster" and
becomes "how do we serve this at all".

There are two ways to cut a model, and they are not interchangeable.

## Tensor parallelism: cut each layer

Every device holds a slice of every matrix, computes a partial result, and the partials are combined.

The combining is the whole story. Done naively you need a collective after *every* matrix multiply,
which is unaffordable. Done correctly you need one per pair, and the trick is which way you cut:

```{literalinclude} ../llmserve/parallel.py
:language: python
:start-at: def shard_columns
:end-before:     if weight.shape[0] % n_shards:
```

```{literalinclude} ../llmserve/parallel.py
:language: python
:start-at: def shard_rows
:end-before:     if weight.shape[1] % n_shards:
```

Column-split the first matrix of a pair, row-split the second. Each device then produces a slice of
the intermediate, feeds it straight into its own row-slice of the second matrix with no
communication, and only the final output needs summing:

```{literalinclude} ../llmserve/parallel.py
:language: python
:start-at: class TensorParallelPair
:end-before:     def __init__
```

Cut it the other way and you need a collective *between* the two matrices as well, doubling the
communication for nothing. Every production implementation makes this choice, and recognising it is
the single most useful thing when reading one — it is why attention's QKV projections are
column-parallel and its output projection is row-parallel, and why the MLP is the same shape.

### It has to be exactly right

An approximately-correct split is worse than no split: it produces a different model, silently, and
nothing in a serving stack will tell you.

```{literalinclude} ../tests/test_parallel.py
:language: python
:start-at: @pytest.mark.parametrize("n_shards", [2, 4, 8])
:end-before: def test_one_collective_per_pair_not_two
```

Tested at several shard counts deliberately. A two-way split is symmetric enough to hide an
off-by-one that a four-way split exposes immediately.

### What it costs

```{include} _generated/ch17-collectives.md
```

Two things in this table are worth more than the numbers.

**Decode is latency-bound, not bandwidth-bound.** A decode step all-reduces one vector per layer — a
few kilobytes — and a few kilobytes over a fast link takes nanoseconds. What it actually costs is the
*fixed* cost of issuing a collective, paid twice per layer, which for a 32-layer model is 64
collectives per token. That is why the decode column does not improve when you use fewer shards: you
are not paying for the data.

**Prefill is the opposite.** Its payload scales with the prompt, so it is genuinely bandwidth-bound
and the interconnect column matters enormously. {ref}`ch03`'s split between the two phases turns up
here too, as it does everywhere else in this book.

The practical reading: on NVLink, tensor parallelism is routine. On PCIe it is a real tax, worst on
long prompts. Across a network it is usually a mistake, and the row exists so you can see why rather
than be told.

### The wall you actually hit

```{literalinclude} ../llmserve/parallel.py
:language: python
:start-at: def max_tensor_parallel_shards
:end-before:     return model.n_kv_heads
```

Attention splits by head, so a device must get whole heads — and with grouped-query attention
({ref}`ch12`) the binding constraint is the *key/value* head count, not the query head count. A model
with eight KV heads cannot be split sixteen ways however large it is or however many devices you
have. This is a surprisingly common wall, it is hit at load time, and the error message is usually
about a tensor shape rather than about what actually went wrong.

## Pipeline parallelism: cut the stack

Each device holds some layers whole, and activations flow between stages. The communication is once
per stage boundary rather than twice per layer, which is dramatically less — and it buys that with
idle time instead:

```{literalinclude} ../llmserve/parallel.py
:language: python
:start-at: def pipeline_bubble_fraction
:end-before:     if n_stages < 1
```

```{include} _generated/ch17-bubbles.md
```

**Serving sits at the left-hand side of that table**, which is the problem. The cure for bubbles is
more microbatches in flight, and a decode step produces one token per sequence — there is very little
to split. Training, which processes large batches of long sequences, sits comfortably at the right.
That asymmetry is why pipeline parallelism is standard in training and awkward in inference.

Where it does earn its place is across a slow link. Pipelining across nodes and tensor-parallelising
within them is the standard arrangement, precisely because it puts the chatty collective on the fast
link and the quiet one on the slow link.

## Expert parallelism, briefly

A mixture-of-experts model routes each token to a few of many expert MLPs, so the natural split is to
put different experts on different devices. The arithmetic is appealing — parameters grow without
per-token compute growing — and the serving problem is one this book does not otherwise have: **the
routing is data-dependent**, so which device is busy depends on the batch's contents. A batch whose
tokens all prefer the same expert leaves most of the fleet idle, and no static plan fixes it.

The mitigations are all forms of the same thing: capacity factors that cap how much any expert takes,
dropping or rerouting the overflow, and replicating hot experts. All of them trade quality or memory
for balance. This is the one topic in the book where I will say plainly that a serious treatment is
out of scope; {ref}`ch29` lists it among the deliberate omissions.

## The cost

- **Per-device efficiency always drops.** Two devices never give twice the throughput of one, and the
  gap is the collectives. Tensor parallelism is what you do when the model does not fit, not what you
  do to go faster — for that, {ref}`ch18`'s replicas are strictly better, because independent
  replicas communicate nothing.
- **The interconnect becomes a hard dependency.** A deployment that works on an NVLink box and is
  moved to a PCIe box does not degrade gracefully; the decode path gets a fixed tax per token that
  no software change removes.
- **A new failure mode: one slow rank.** Every collective is a synchronisation point, so the slowest
  device sets the pace for all of them. A single throttling GPU degrades the whole group, and the
  symptom is uniform slowness rather than an error anywhere.
- **Debugging gets much harder.** A numerical discrepancy is now a distributed-systems problem, and
  the equivalence test above is the only cheap way to catch a bad split before it becomes a quality
  regression nobody can explain.
- **Shapes constrain deployment.** Head counts, hidden sizes and vocabulary sizes all have to divide
  by the shard count, so the set of usable device counts is a property of the model.

## Key takeaways

- Cut each layer's pair of matrices **column-then-row**, so one collective per pair suffices. Any
  other cut doubles the communication for nothing.
- The split must be **exactly** correct, and that is testable on one device with no GPU at all.
- **Decode's collective cost is latency, prefill's is bandwidth.** They are different problems and
  respond to different fixes.
- Grouped-query attention caps how far a model can be tensor-parallelised, and the cap is the KV
  head count.
- Pipeline parallelism communicates far less and idles instead. Serving has too few microbatches to
  hide the bubble, which is why it is a cross-node tool rather than a within-node one.
- **Parallelism is how you fit a model, not how you make it fast.** If it already fits, run replicas.

## Looking ahead

{ref}`ch18` takes exactly that last point seriously. If the model fits on one device, the way to
serve more traffic is more devices running independent copies — and the interesting question becomes
not how to split a model but how to decide which replica each request goes to. That turns out to
matter far more than it sounds.

## Further reading

Megatron-LM (Appendix E) is the origin of the column-then-row split and is written for training,
which is where all of this came from. For the serving side, vLLM and TensorRT-LLM both implement
tensor parallelism and disagree about almost everything else, which makes reading them side by side
unusually informative.
