---
title: "Quantisation for Serving"
short_title: "ch15 Quantisation"
---

(ch15)=
# ch15 · Quantisation for Serving

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU. The model is **trained** for this chapter, because quality must be able to fall |
| **Prerequisites** | [ch03](#ch03), [ch13](#ch13) |
| **Scorecard** | Memory falls sharply. Quality moves for the first time. Decode gets **slower** — read why. |
:::

## The problem

Per {ref}`ch03`, a decode step reads every weight in the model from memory and does very little
arithmetic with each one. The way to make a memory-bound operation faster is to move fewer bytes,
and the most direct way to move fewer bytes is to store fewer of them.

Every optimisation so far has been free in the sense that mattered: output was token-identical, and
only time or memory changed. Not this one. **Quantisation makes the model's answers worse**, and
the whole engineering question is whether the loss is small enough to be worth what it buys.

Which means a quantisation chapter without a quality number is not a chapter, it is a brochure.

## The idea

### Three axes, never one

Quality, speed, memory. Any claim about one is worthless without the other two. A scheme that
halves memory and ruins output is not a compression technique, and a scheme that preserves quality
while running slower is not an optimisation.

### Weight-only quantisation

Store weights at 8 or 4 bits; convert them back to floating point on the way into the matmul. The
arithmetic still happens in fp32 — only the *storage* changes.

That sounds like it should not help, and on a compute-bound operation it does not. On decode,
which {ref}`ch03` showed is bound by bytes read, reading a quarter as many bytes is precisely the
win. This is why weight-only quantisation is the dominant serving technique and why it does
nothing for prefill.

### How you scale matters more than how many bits

Mapping floats to integers needs a scale. The naive choice is one scale for the whole tensor, set
by its largest absolute value. The better choice is one per output channel.

```{literalinclude} ../llmserve/quant.py
:language: python
:start-at: def quantize_int8_symmetric
:end-before:     scale = weight.abs().amax(dim=-1
```

Four bits gives sixteen levels, too few to cover a channel's range, so INT4 subdivides further
into *groups* along the input dimension. Smaller groups mean tighter ranges and more scales to
store — that trade is the entire design space of INT4 formats.

### And the cache is a separate decision

Weights are a fixed cost. The KV cache is per concurrent request, and {ref}`ch13` showed it is what
limits concurrency at long context. Quantising it is a different lever with a different payoff, and
it is the one people forget.

## The build

```{literalinclude} ../llmserve/quant.py
:language: python
:start-at: class QuantizedLinear
:end-before:     def __init__
```

A note on what this chapter needed that no other did: **a trained model**. The book's reference
model has random weights, which is fine for measuring latency and scheduling but useless here —
random weights sit at chance perplexity, so quantising them changes one meaningless number into
another.

So we train it. A first attempt used four sentence templates; the model reached near-perfect
perplexity and four-bit weights still solved it, because the task left so much capacity spare that
damage did not show. The corpus is now a Zipf-distributed word process over six hundred word types,
which puts the model near its limit, where quantisation has something to break.

## The measurement

```{include} _generated/ch15-quantisation.md
```

Three things, and two of them are not what the technique's reputation predicts.

**Memory falls as advertised.** Four times smaller at INT8, seven at INT4. No surprises.

**Quality barely moves, and INT4 costs something real.** The ordering is right — fp32 best, INT8
indistinguishable, INT4 measurably worse — and the magnitudes are much smaller than a production
model would show. That is a property of our model, not of quantisation: a 5.8M-parameter model on a
learnable task has capacity to spare. Published ablations on 7B-class models, not this table, are
the evidence for what INT4 costs in practice.

**Decode gets slower.** Every quantised configuration loses throughput, and the more aggressive the
scheme the worse it gets.

That last one deserves more than a footnote, because it looks like a refutation and is not. We
dequantise on every forward pass, so each matmul now does extra work to reconstruct the weights.
The whole premise is that this is worth it because reading the weights was the expensive part — and
on this machine, with a 23 MB model, it is not. The weights fit comfortably in cache; there was
never a memory bottleneck to relieve.

On a GPU serving a 7B model, the weights are 14 GB, they are read from HBM every single step, and
the dequantisation is fused into the kernel. There the arithmetic is real and the win is real. Here
we have implemented the mechanism correctly and measured it on hardware where its premise does not
hold.

**Say so plainly: this chapter's speed column is a negative result produced by the test rig, not by
the technique.** It is the clearest example in the book of why a measurement needs its conditions
attached.

### The cache

```{include} _generated/ch15-kv-quantisation.md
```

Quantising the cache is nearly free in quality and halves or quarters the per-token footprint.
Combined with {ref}`ch13`'s arithmetic — where an 8B model at 32k context holds ten concurrent
sequences under GQA — INT8 KV takes that to twenty. For long-context serving this is usually a
better return than quantising weights, and it is the one most often skipped.

### Why per-channel scaling exists

The table above shows per-tensor and per-channel INT8 performing identically, which would suggest
the distinction does not matter. It does; our model simply does not have the property that makes it
matter. Demonstrated directly on a matrix that does:

```{include} _generated/ch15-outliers.md
```

One channel two hundred times larger than its neighbours sets the single global scale, and every
other channel is crushed into a handful of levels. Large transformers reliably develop exactly such
outlier channels — that is the entire reason LLM.int8() and SmoothQuant exist.

This is a case where the honest end-to-end measurement and the correct engineering advice point
different ways, and resolving it means understanding the mechanism rather than trusting the number
in front of you.

## The cost

- **Output gets worse.** Every quantisation claim needs a quality figure beside it, and
  "negligible" is not a figure.
- **Quality loss is workload-dependent.** Perplexity on one corpus can hide damage that shows up on
  code, other languages, or long context. One number is not an evaluation.
- **The speed win depends entirely on being memory-bound.** Measure on the hardware you will deploy
  on; ours says the opposite of what a GPU would.
- **Calibration adds a pipeline step**, and the better schemes (GPTQ, AWQ) need calibration data
  representative of production traffic — which you may not be allowed to keep.
- **More formats, more surface.** Each combination of scheme, group size and kernel is a
  configuration that can be silently wrong.

## Key takeaways

- Quality, speed and memory are one measurement. Never report a subset.
- Weight-only quantisation attacks bytes read, so it helps decode and does nothing for prefill.
- How you choose scales matters more than the bit width. Per-channel is not a refinement; without
  it a single outlier channel destroys everything else.
- Quantising the KV cache is often the better return at long context, and the most frequently
  forgotten.
- Dequantisation overhead makes quantisation a *loss* on hardware that was not memory-bound. Our
  measurement shows precisely that, and it is about the rig, not the technique.
- When an end-to-end measurement and the established advice disagree, find the mechanism before
  believing either.

## Looking ahead

Quantisation made each step cheaper. {ref}`ch17` attacks something else entirely: the fact that
decode produces one token at a time *by construction*. Speculative decoding breaks that
serialisation — and, unusually for this book, does so while provably not changing the output
distribution at all.

## Further reading

LLM.int8() (Dettmers et al.) is the paper that identified outlier channels; SmoothQuant addresses
them by moving the difficulty into the activations. GPTQ and AWQ are the two INT4 schemes worth
knowing, and both are in Appendix E. For the hardware side, FP8 on recent accelerators changes this
calculus again, because the format is native rather than emulated.
