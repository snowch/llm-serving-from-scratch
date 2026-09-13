---
title: "GPU and Hardware Primer for Serving"
short_title: "Appendix A"
---

(appendix-a)=
# Appendix A · GPU and Hardware Primer for Serving

This appendix covers the hardware facts the rest of the book leans on, and deliberately stops
there. It is not a GPU architecture course. The test for including something was whether it changes
a serving decision; most of what a graphics or training text would cover does not.

## The one paragraph version

Serving an LLM is mostly a memory problem. Decode reads the entire weight matrix to produce one
token, so its ceiling is set by how fast the device can read memory, not by how fast it can
multiply. Two numbers therefore predict most of what an accelerator will do for you: **memory
bandwidth** and **memory capacity**. Everything else on the spec sheet matters less than those, and
FLOPS matters least of all for the phase that dominates a chat workload.

{ref}`ch03` derives this rather than asserting it, and every chapter after it is an attempt to
either move work out of the memory-bound phase or get more useful tokens out of each pass over the
weights.

## The memory hierarchy

Three tiers, each roughly an order of magnitude faster and smaller than the one below:

| Tier | Typical size | Typical bandwidth | What lives here |
|---|---|---|---|
| Registers | tens of KB per SM | effectively free | the values a thread is working on right now |
| SRAM (shared memory / L1) | ~100–250 KB per SM | ~10–20 TB/s | a tile of a matrix, mid-computation |
| HBM (device memory) | 24–192 GB | ~1–8 TB/s | model weights, KV cache, activations |

The gap between SRAM and HBM is the entire reason {ref}`ch12` exists. FlashAttention does not reduce
the number of floating-point operations attention performs — it performs slightly more — and it is
much faster anyway, because it avoids writing the attention score matrix to HBM and reading it back.
An algorithm that does more arithmetic and less memory traffic wins, and that sentence is the most
useful thing in this appendix.

A useful habit when reading any kernel: ask what it reads from HBM and what it writes back, before
asking what it computes.

## SMs, warps and occupancy

A GPU is a few dozen to a few hundred **streaming multiprocessors**, each running many threads in
lockstep groups of 32 called **warps**. Threads in a warp that take different branches execute both
sides serially, which is why GPU code avoids data-dependent branching.

**Occupancy** is how many warps are resident per SM, and it exists to hide memory latency: while one
warp waits on HBM, another computes. Low occupancy usually means a kernel asked for too much shared
memory or too many registers per thread.

This is as deep as serving goes. If you are writing kernels — {ref}`ch13` — you will need more than
this, and the Triton documentation is a better place to get it than any summary here. If you are
choosing hardware or debugging a throughput number, occupancy is almost never the answer; the
batch size, the KV budget or the scheduler almost always is.

## Tensor cores and dtypes

Tensor cores are fixed-function matrix-multiply units, and they are why FLOPS figures vary so
wildly with precision on one device. A modern accelerator supports some subset of fp32, tf32, fp16,
bf16, fp8 and int8, at roughly doubling throughput as the width halves.

Two consequences for {ref}`ch14`:

- **A quantised format is only fast if the hardware has a matching instruction.** INT4 weights are
  usually dequantised to fp16 before the multiply, so INT4's win is memory, not arithmetic. Sold as
  a speedup, delivered as a capacity increase — which is still worth having, and is a different
  thing.
- **bf16 and fp16 differ in where they spend their bits.** bf16 keeps fp32's exponent range and
  loses mantissa precision, which makes it far harder to overflow and is why it became the default
  for training. For inference either is usually fine; a model trained in bf16 and served in fp16
  can overflow on outlier activations, which presents as occasional garbage output rather than an
  error.

## Interconnect

| Link | Typical bandwidth | Where it binds |
|---|---|---|
| PCIe 4.0 x16 | ~32 GB/s | host-to-device transfers, and GPU-to-GPU without NVLink |
| PCIe 5.0 x16 | ~64 GB/s | as above |
| NVLink (recent) | ~400–900 GB/s | tensor parallelism ({ref}`ch17`), KV handoff ({ref}`ch11`) |
| 100 GbE | ~12.5 GB/s | cross-node anything |

The interconnect decides which architectures in Part III and Part V are viable. Tensor parallelism
does an all-reduce **per layer**, so on PCIe it can be slower than not splitting the model at all;
on NVLink it is routine. {ref}`ch11`'s prefill/decode disaggregation has the same character: the KV
handoff is a real payload, and whether it is free or fatal is a property of the link, which is why
that chapter's table is arithmetic over bandwidth rather than a measurement.

## MIG and partitioning

Multi-Instance GPU splits one device into hardware-isolated slices with their own memory and SMs. It
is worth knowing about as the *alternative* to {ref}`ch19`'s software multi-tenancy, and the trade is
clean: MIG gives genuine isolation — a noisy neighbour physically cannot touch your slice — and gives
up all sharing. Each slice has its own copy of the weights and its own KV budget, so a fleet of
slices serves far fewer concurrent sequences than one undivided device running a shared engine.

Use MIG when isolation is a requirement rather than a preference. Use {ref}`ch19` when utilisation is.

## The spec sheet, ranked

When comparing accelerators for serving, in order:

1. **HBM bandwidth.** Sets the decode ceiling. {ref}`ch03` computes it: tokens per second is roughly
   bandwidth divided by the bytes read per token, and for a single stream that is the whole story.
2. **HBM capacity.** Decides how many sequences fit — weights plus KV cache — and therefore the
   batch size, and therefore throughput. Capacity buys concurrency, and concurrency is how a
   memory-bound engine gets throughput at all.
3. **Interconnect bandwidth**, if you will use more than one device.
4. **FLOPS at your serving dtype.** Binds prefill, which matters enormously for {ref}`ch21`'s
   workload and barely at all for {ref}`ch20`'s.
5. Everything else.

A concrete way to use this: before buying or renting anything, compute the decode ceiling from
bandwidth and the concurrency from capacity, for your model and your context length. If the answer
is far from what you need, no amount of software in this book will close the gap — and if it is
close, Parts II and III are how you get there.
