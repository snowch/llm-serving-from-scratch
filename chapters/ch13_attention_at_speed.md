---
title: "Attention at Speed"
short_title: "ch13 Attention at Speed"
---

(ch13)=
# ch13 · Attention at Speed

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU runnable; the kernel work in [ch14](#ch14) needs a GPU |
| **Prerequisites** | [ch03](#ch03), [ch08](#ch08) |
| **Scorecard** | KV footprint falls sharply. Throughput moves only a little here, and the reason matters. |
:::

## The problem

Part III organised work better and better, until {ref}`ch11` ran out of organising to do. What
remains is the arithmetic itself, and attention is where it hurts twice.

**It materialises something enormous.** The standard formulation builds a score matrix of shape
`seq × keys` per head per layer, softmaxes it, and multiplies by V. Here is what that costs:

```{include} _generated/ch13-score-matrix.md
```

At 32k context a *single layer's* score matrix is larger than most models' weights, written to
memory and read back for no reason other than that softmax appears to need all its inputs at once.

**And its cache is the concurrency ceiling.** {ref}`ch05` established that KV cache, not weights,
limits how many conversations a device can hold. Attention's shape decides how big that cache is.

## The idea

### Softmax does not need all its inputs at once

The apparent constraint — that you cannot normalise until you have seen every score — is false.
Softmax can be computed incrementally, one block of keys at a time, by carrying a running maximum
and a running sum, and rescaling what you have accumulated whenever the maximum moves.

```{literalinclude} ../llmserve/attention.py
:language: python
:start-at: def online_softmax_attention
:end-before:     scale = scale or
```

The correction factor `exp(m_old - m_new)` is the entire trick. Everything accumulated under the
old maximum is rescaled to the new one, so the result is *identical* to having seen all scores
together — not an approximation, exactly equal within floating-point error.

Peak score memory becomes `seq × block_size` instead of `seq × keys`. The quadratic term does not
disappear from the arithmetic, but it disappears from memory.

Subtracting the running maximum is also what keeps `exp` from overflowing. That is not a
refinement; a naive incremental softmax produces infinities on ordinary inputs.

**This is what FlashAttention is.** The published implementation adds GPU-specific tiling, keeps
the running state in SRAM rather than HBM, and fuses the whole thing into one kernel — but the
mathematics above is the substance. It is called IO-aware rather than faster because it does not
reduce FLOPs at all; it reduces bytes moved, which per {ref}`ch03` is what attention was actually
limited by.

### Shrinking the cache: MQA and GQA

The second problem needs a different fix, because no kernel makes a cache smaller.

Standard multi-head attention gives every query head its own key and value heads. **Multi-query
attention** gives all query heads a single shared KV head. **Grouped-query attention** sits between
them: a small number of KV heads, each serving a group of query heads.

The KV cache shrinks by exactly the group ratio, because `n_kv_heads` appears directly in the
formula from {ref}`ch03`. That is the entire mechanism. It is not subtle, and it is why essentially
every model released since 2023 uses GQA.

The cost is quality, paid at training time: fewer distinct key/value projections is less capacity.
GQA exists because the quality loss is small and the serving win is enormous — which is a choice
about serving made by whoever trained the model, and one you cannot revisit afterwards.

### Shrinking it further: latent attention

GQA is a coarse dial. The sharing is all-or-nothing per group and the floor is one KV head, at which
point MQA has taken everything head-sharing has to give.

**Multi-head latent attention** reaches the same goal by a different route: cache a single low-rank
*latent* vector per token, and reconstruct each head's keys and values from it on the fly.

```{literalinclude} ../llmserve/latent.py
:language: python
:start-at: class LatentKV
:end-before:     def __init__
```

The cache stops scaling with head count at all, which is what makes this a different technique
rather than a more aggressive setting of the same one:

```{include} _generated/ch13-latent.md
```

What it costs is a matrix multiply per layer to project back up — work that {ref}`ch03`'s
memory-bound decode phase has to spare, which is the whole reason the trade is favourable. What it
costs *you* is that it is not available: like GQA, this is a property of how the model was trained,
and no serving decision introduces it after the fact. It appears here because it explains why some
models' caches are so much smaller than their head counts suggest, and because {ref}`ch30`'s rubric
asks whether your engine supports the ones that have it.

One honest limit on the table above: it is arithmetic over shapes, not a quality measurement. A
randomly-initialised latent projection would tell you nothing about what the compression costs in
accuracy, and this book has no trained MLA model to ask. The shape of the trade is exact; the
quality half is a question for whoever trains one.

## The build

Both implementations live side by side so the equivalence can be asserted rather than assumed:

```{literalinclude} ../llmserve/attention.py
:language: python
:start-at: def standard_attention
:end-before: def online_softmax_attention
```

The engine has had GQA since {ref}`ch01`, in one asymmetry that is easy to miss:

```{literalinclude} ../llmserve/model.py
:language: python
:start-at:         self.q_proj
:end-before:         self.o_proj
```

The KV projections are narrower than the query projection by the group ratio. That is all of GQA.

## The measurement

Vary only `n_kv_heads`, from full multi-head down to multi-query, changing nothing else:

```{include} _generated/ch13-gqa.md
```

The KV cache per token halves at every step — an eightfold reduction from MHA to MQA — while
throughput improves only slightly.

That gap is worth explaining rather than glossing. {ref}`ch03` measured that weights are the
overwhelming majority of bytes read per decode step on this model, so shrinking the KV term moves
a small share of the total. **On this model, at these context lengths, KV is not the bottleneck.**

It becomes the bottleneck as soon as the model and the context grow:

```{include} _generated/ch13-footprint.md
```

On an 8B-class model at 32k context, the same choice is the difference between holding two
conversations on an accelerator and holding eighty. Nothing about the serving code changes; the
model's attention shape decides it.

This is the clearest case in the book of a measurement being *correct and not generalisable*. Our
numbers say GQA barely matters. The arithmetic says it decides whether a deployment is viable. Both
are true, about different regimes, and only one of them is the regime you will deploy in.

## The cost

- **Online softmax is not free on CPU.** It trades one large matmul for many small ones plus
  bookkeeping. The win is memory, and on hardware where memory was not the constraint it is a
  loss — which is why FlashAttention is a GPU technique.
- **GQA is not a serving decision.** It is fixed at training time. You inherit it.
- **Quality is genuinely traded**, at training time, in a way this book cannot measure with random
  weights. Treat the published ablations as the evidence, not us.
- **Fused kernels are hard to debug.** When a fused attention path disagrees with the reference by
  1e-3, finding out why is real work — hence keeping `standard_attention` as an oracle.

## Key takeaways

- The score matrix is the quadratic term, and at long context it is larger than the model. Online
  softmax removes it from memory without changing the result.
- The running-maximum correction makes the incremental result exactly equal to the batch one, and
  is also what prevents overflow.
- FlashAttention is IO-aware, not FLOP-reducing. It attacks bytes moved, which is what attention
  was limited by.
- MQA and GQA shrink the KV cache by the group ratio, directly from the ch03 formula. That is the
  whole mechanism.
- Our measurement shows GQA barely mattering, and it is right: on a small model with short
  contexts, weights dominate. At 8B and 32k the same choice is 2 concurrent sequences versus 80.
- Distrust any single measurement's generality — including this book's. Check which term dominates
  in *your* regime before concluding.

## Looking ahead

The arithmetic is now as cheap as restructuring can make it. What remains is to make the numbers
themselves smaller: {ref}`ch15` stores weights and cache in fewer bits, and is the first chapter
where an optimisation makes output measurably *worse* as well as faster.

{ref}`ch14` writes the paged-attention kernel that {ref}`ch08` has been waiting for. It needs a
GPU, and nothing later depends on it.

## Further reading

Dao et al., *FlashAttention* (Appendix E) is now readable in full — you have implemented its core.
FlashAttention-2 and -3 are refinements of the same idea for newer hardware. For the cache side,
Shazeer's multi-query attention paper is three pages and says everything; Ainslie et al. on GQA
covers the interpolation and the quality ablations this chapter cannot run.
