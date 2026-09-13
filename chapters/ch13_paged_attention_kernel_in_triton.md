---
title: "Writing a Paged Attention Kernel in Triton"
short_title: "ch13 Paged Attention Kernel"
---

(ch13)=
# ch13 · Writing a Paged Attention Kernel in Triton

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 for the algorithm and the arithmetic. Tier 2 — a GPU — for the kernel itself. |
| **Prerequisites** | [ch08](#ch08), [ch12](#ch12) |
| **Scorecard** | Doubles the decode ceiling on paper. Optional by design: nothing after this chapter depends on it. |
:::

:::{warning} What this chapter contains, and what it does not
**There is no Triton kernel in this chapter.** The book's default tier has no GPU, and a Triton
kernel cannot be compiled or run without one — so shipping a kernel here would mean shipping
untested code, in a book whose entire argument is that untested performance claims are worthless.
That would be the wrong trade.

What is here instead is everything that *can* be verified: the arithmetic that says how much there
is to win, the algorithm the kernel has to implement, written in PyTorch and checked against
{ref}`ch12`'s reference on every shape that matters, and a precise account of the translation. The
reference implementation is a specification, and a kernel that matches it on the tests in
`tests/test_kernels.py` is a correct kernel.

If you have a GPU, this chapter is the exercise it looks like. If you do not, the first two sections
are still the most important part of it.
:::

## The problem

{ref}`ch08` bought concurrency by storing the KV cache in blocks, and paid for it with a gather.
Every decode step, for every sequence, the engine copies the whole cache into one contiguous tensor,
pads it to the batch's longest, and hands that to the model — which then concatenates the new key and
value onto it, writing the same bytes again.

The comment in that code has been waiting for this chapter:

```{literalinclude} ../llmserve/cache/blocks.py
:language: python
:start-at:     def gather(
:end-before:         n_blocks = self.allocator.blocks_needed(length)
```

Here is what "deliberately wasteful" costs:

```{include} _generated/ch13-gather-cost.md
```

**The gather roughly halves the decode ceiling.** {ref}`ch03` established that decode is bound by
memory bandwidth, so the number of bytes moved per step *is* the performance. Moving the cache twice
to produce one token per sequence means half the tokens per second the hardware could deliver.

It is worth noticing what this does *not* depend on. Not the model's arithmetic, not the attention
algorithm, not the scheduler. It is pure data movement introduced by a convenience in the
implementation, which is the most satisfying kind of thing to remove.

## The idea

{ref}`ch12` already contains the key. Online softmax processes keys one block at a time, maintaining
a running maximum and a running denominator, and never needs to see all the scores at once. So the
blocks do not have to be adjacent, and they do not have to be gathered — they can be walked through
the block table and read in place, each exactly once, straight into the accumulator.

That is the whole idea, and it composes the two previous chapters exactly: {ref}`ch08`'s indirection
plus {ref}`ch12`'s streaming softmax.

```{literalinclude} ../llmserve/kernels.py
:language: python
:start-at: def paged_decode_attention(
:end-before:     n_seqs, n_heads, head_dim = query.shape
```

Three details in that implementation are the ones a kernel gets wrong.

**Grouped-query attention maps heads; it does not expand the cache.** With eight KV heads and
thirty-two query heads, the temptation is to repeat each KV head four times so the shapes line up.
That reintroduces exactly the copy this chapter exists to remove. Index the KV head instead.

**The partial final block must be masked, with a finite floor.** A sequence of seventeen tokens in
blocks of sixteen has fifteen slots of whatever was there before. Masking them with negative infinity
sets up {ref}`ch06`'s NaN — an all-masked row makes softmax produce NaN, and one NaN row corrupts
the batch. `torch.finfo(dtype).min` instead.

**The rescale on the first block is a special case.** The running maximum starts at negative
infinity, and `exp(-inf - -inf)` is NaN rather than zero. The implementation checks for it; a kernel
must too, and this is the single most common way a hand-written online-softmax loop produces NaN on
its first iteration.

The test that all of this is right is not optional, and it is the reason the PyTorch version exists:

```{literalinclude} ../tests/test_kernels.py
:language: python
:start-at: @pytest.mark.parametrize("length", [1, 7, 8, 9, 16, 20])
:end-before: def test_blocks_may_be_in_any_physical_order
```

Lengths on both sides of a block boundary, and a length of one. Every one of those was a bug in
somebody's kernel.

## The build: mapping it onto Triton

The translation is mechanical, and worth stating precisely enough to follow without a GPU in front
of you.

**The grid is `(n_sequences, n_query_heads)`.** One Triton program per sequence-head pair, which is
the natural unit: each one owns a single query vector and reduces over that sequence's blocks. There
is no parallelism *within* a program's reduction, and that is fine — the parallelism is in the grid,
and a decode step has batch × heads of it, which is thousands.

**The query vector is loaded once**, into registers, and stays there. It is `head_dim` floats — a few
hundred bytes — and every block it meets is multiplied against the same values, so the arithmetic
intensity of the inner loop is as good as it gets for decode.

**The block table is read inside the kernel.** Program `(seq, head)` loads `block_table[seq, i]` to
find the physical block, then loads that block's keys and values directly from the cache tensor.
Nothing is materialised. This is the step that makes it a *paged* kernel rather than an attention
kernel that happens to be fast.

**The loop is bounded by a compile-time constant.** Triton needs the trip count to be a
`tl.constexpr`, so the kernel loops to the maximum block count and skips blocks past the sequence's
length. That wastes a predicate per iteration and nothing else.

**The online-softmax state lives in registers**: a running maximum, a running denominator, and an
accumulator of `head_dim` floats. That is why the kernel is fast — the reduction never touches
memory, and it is the same state the PyTorch version above maintains, one block at a time.

Then benchmark it against three things, in this order: the {ref}`ch08` gather it replaces, the
PyTorch reference for *numerical* agreement, and a production kernel such as FlashAttention's paged
decode path for a reality check on how much is left on the table. Expect the third comparison to be
humbling; expect it to be worth doing anyway.

On numerical agreement, one warning: **bitwise equality is not the bar and will not be met.** A
different reduction order gives different floating-point rounding, and demanding exactness will send
you looking for a bug that is not there. Bounded relative error against the reference is the right
test — the tests in `tests/test_kernels.py` use `atol=1e-5` on fp32 for exactly this reason.

## The cost

- **This is the least portable code in the book.** A Triton kernel targets a specific vendor's
  hardware and a specific compiler version, and the book's promise that everything runs anywhere
  stops here. That is why it is optional and why nothing after it depends on it.
- **It raises the hardware floor** for anyone wanting to reproduce the chapter, from "any laptop" to
  "a GPU you can get a Triton toolchain onto".
- **Kernel bugs are quiet.** A masking error changes the output rather than raising, and the wrong
  answer looks like a bad model. The test suite above is the whole defence, and writing it before the
  kernel is not optional discipline — it is the only way to know.
- **It is easy to write a kernel that is slower** than the PyTorch it replaces. PyTorch's operations
  are already calling well-tuned kernels; the win here comes from removing a copy, not from being a
  better matrix-multiplier, and a kernel that removes the copy while getting the memory access
  pattern wrong can easily lose.
- **Two implementations of one thing.** The reference and the kernel must agree forever, which means
  every future change happens twice. That is the standing cost of having a fast path.

## Key takeaways

- {ref}`ch08`'s gather roughly halves the decode ceiling, and the cost is pure data movement rather
  than arithmetic.
- **Online softmax is what makes a paged kernel possible.** Because {ref}`ch12`'s algorithm never
  needs all the scores at once, the blocks never need to be adjacent or gathered.
- Map grouped-query heads to KV heads; do not expand the cache. Expanding it restores the copy.
- Mask partial blocks with a finite floor, and special-case the first block's rescale. Both failures
  are NaN, and both are silent.
- **Write the reference and its tests first.** A kernel is correct when it matches a specification,
  and bitwise equality is not the specification — bounded error is.
- This is the one place in the book where the code is not portable, which is why nothing else depends
  on it.

## Looking ahead

{ref}`ch14` goes after the same bottleneck from the other direction. This chapter halved the bytes
moved per step by not copying them; the next one halves the bytes themselves.

## Further reading

The Triton documentation's fused-attention tutorial is the right starting point, and vLLM's paged
attention kernels are the reference implementation to read once you have written your own — reading
them first tends to produce a copy rather than an understanding. FlashAttention's decode path
(Appendix E) solves the same problem with considerably more sophistication about memory hierarchy,
and its papers are clearer than most about *why* each choice was made.
