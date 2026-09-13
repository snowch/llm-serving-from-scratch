---
title: "The Arithmetic of Inference"
short_title: "ch03 The Arithmetic of Inference"
---

(ch03)=
# ch03 · The Arithmetic of Inference

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch01](#ch01), [ch02](#ch02) |
| **Scorecard** | No engine change. Makes every later row **predictable in advance**. |
:::

## The problem

Chapter 2 established that the naive engine saturates at a particular throughput. It did not
explain why that number, rather than one ten times higher or lower.

Without a model of where the time goes, every optimisation is guesswork — and guesswork in this
field is expensive, because the obvious answer is usually wrong. The rest of this book is a
sequence of specific interventions, and you should be able to predict roughly what each will buy
before implementing it.

## The idea

### Two operations, two different bottlenecks

Recall the split from {ref}`ch01`. Prefill processes the whole prompt at once; decode produces one
token at a time. The crucial fact is that these are limited by *different resources*.

**Prefill is compute-bound.** The prompt's tokens all exist already, so they go through the model
together as a matrix multiplication with plenty of parallel work. The cost is roughly
`2 × params` FLOPs per prompt token — one multiply and one add per parameter.

**Decode is memory-bandwidth-bound.** To produce a single token, the model must read every one of
its weights out of memory, and then does barely any arithmetic with each. The work is trivial;
the data movement is not. Decode speed is therefore set by how fast the device can stream weights,
not by how fast it can multiply.

That asymmetry explains almost everything that follows, so it is worth stating as a rule:

> Prefill is limited by FLOPs. Decode is limited by bytes moved.

Two consequences that will keep recurring:

- Making decode faster usually means **moving fewer bytes** — smaller weights ({ref}`ch14`),
  a smaller KV cache ({ref}`ch12`), or reading the same weights for more sequences at once
  ({ref}`ch07`).
- Adding sequences to a decode batch is nearly free. The weights are read once per step whatever
  the batch size, so the second sequence costs only its own KV cache. **This is the single most
  important economic fact in LLM serving**, and continuous batching exists to exploit it.

### The two formulas worth memorising

**KV cache per token.** Each token's keys and values are stored for every layer:

```
kv_bytes_per_token = 2 × n_layers × n_kv_heads × head_dim × bytes_per_element
```

The 2 is for K and V. Note `n_kv_heads`, not `n_heads` — grouped-query attention shrinks this
term directly, which is why every modern model uses it.

**The decode ceiling.** Per step the device must read the weights, plus the KV cache of every
sequence in the batch:

```
decode_tok_per_s ≤ bandwidth / (weight_bytes + kv_bytes_per_token × context × batch)
```

No kernel beats this. It is a ceiling, not a forecast.

## The build

Both formulas, written out so predictions cannot drift from the model they describe — the same
`ModelConfig` object builds the model and feeds the arithmetic:

```{literalinclude} ../llmserve/arithmetic.py
:language: python
:start-at: def kv_bytes_per_token
:end-before: def kv_bytes_for
```

```{literalinclude} ../llmserve/arithmetic.py
:language: python
:start-at: def decode_bytes_per_step
:end-before: def decode_ceiling_tokens_per_second
```

That the prediction and the model share a source is worth more than it looks: a test asserts that
the predicted parameter count equals the built model's, so the arithmetic cannot quietly describe
a model we are not running.

## The measurement

Apply it to the reference model, then check it against chapter 2's measurements:

```{include} _generated/ch03-arithmetic.md
```

Three things to take from that table.

**The KV cache is small here, and that is a property of this model, not of serving.** Weights are
98.8% of the bytes read per decode step. Scale to a 7B model in fp16 and the picture inverts at
long context: weights are fixed at about 14 GB, while KV grows with every token of every
concurrent sequence. That crossover is why Part III is mostly about memory.

**The implied bandwidth is a useful number to keep.** Dividing bytes-per-step by measured
time-per-token gives roughly 5 GB/s achieved on this machine. That figure now predicts things —
{ref}`ch06` uses it to forecast batched throughput before implementing batching.

**The FLOPs model over-predicts by an order of magnitude, and that is the most instructive line in
the table.** Counting arithmetic alone says removing the KV cache should cost 27.5×: without a
cache, generating token *n* redoes all *n* previous tokens. Measured, it costs 2.76×.

The explanation is the rule above. At this model size, a decode step is dominated by reading 23 MB
of weights, and that cost is paid whether we process one token or ninety-six. The extra arithmetic
rides along in time the machine was spending on memory anyway. The FLOPs model counted the work;
it did not count what was actually scarce.

This is worth dwelling on because it generalises. Any optimisation that reduces FLOPs without
reducing bytes moved will disappoint you during decode. Most of the ones that work in this book —
batching, quantisation, GQA, prefix caching — reduce bytes, or amortise them over more useful
output.

## The cost

The arithmetic is a model, and it ignores:

- **Attention's quadratic term.** Fine at 96 tokens, badly wrong at 32k. {ref}`ch21` is where this
  stops being safe.
- **Kernel launch and framework overhead.** At this model size a meaningful share of each step is
  Python and dispatch, not memory traffic. That is one reason the 27.5× did not materialise.
- **Cache hierarchy.** "Bandwidth" is one number standing in for registers, several cache levels
  and DRAM, each an order of magnitude apart.
- **Everything except the model.** Tokenisation, scheduling and HTTP are all invisible here.

Use it to predict orders of magnitude and to decide which of two optimisations is worth trying.
Do not use it to predict a percentage. Then measure, and when the measurement disagrees, the
disagreement is the interesting part — as it was above.

## Key takeaways

- **Prefill is compute-bound; decode is memory-bandwidth-bound.** Every technique in this book
  attacks one of those two, and knowing which tells you when it will help.
- KV cache per token is `2 × n_layers × n_kv_heads × head_dim × bytes`. It is the budget all of
  Part III competes over.
- Decode throughput is bounded by `bandwidth / bytes read per step`. Weights are read once per
  step regardless of batch size, which is why batching is close to free and why {ref}`ch07` works.
- A FLOPs count predicts prefill reasonably and decode badly. When the two disagree, the scarce
  resource is bytes, not arithmetic.

## Looking ahead

We can now predict, measure, and explain the gap. Time to fix something. {ref}`ch04` takes
ownership of the decode loop itself — sampling, stop conditions, and the streaming-detokenisation
bug that makes non-ASCII output look like model corruption — so that every optimisation afterwards
can be tested for producing identical output.

## Further reading

The roofline model is the general form of the argument here, and reading the original Williams,
Waterman and Patterson paper will make the prefill/decode split feel inevitable rather than
particular to transformers. For the same arithmetic applied to production-scale models, the
FlashAttention paper (Appendix E) opens with an unusually clear statement of why attention is
IO-bound rather than compute-bound.
