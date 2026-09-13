---
title: "Bounding the Context"
short_title: "ch16 Bounding the Context"
---

(ch16)=
# ch16 · Bounding the Context

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch05](#ch05), [ch13](#ch13), [ch15](#ch15) |
| **Scorecard** | The cache stops growing with the conversation. The first optimisation in this book that can change what the model says. |
:::

## The problem

Every chapter so far has made the cache cheaper *per token*. {ref}`ch13` gave several query heads one
KV head; {ref}`ch15` gave each entry fewer bits; {ref}`ch14` stopped copying it twice; {ref}`ch12`
gave it somewhere else to live. All of them leave one thing untouched:

**The cache still grows linearly with the conversation.** A long enough sequence exhausts any budget,
any number of bits, any number of tiers. {ref}`ch23`'s retrieval workload runs into this, and so does
any agent whose transcript keeps growing — which is all of them.

The other lever is to stop keeping all of it.

## The idea, and why it is different from everything before it

Drop the keys and values for tokens far enough back, and the cache becomes constant-size regardless
of how long the conversation runs:

```{literalinclude} ../llmserve/cache/window.py
:language: python
:start-at: class WindowPolicy
:end-before:     window: int = 256
```

```{include} _generated/ch16-window-memory.md
```

Linear becomes flat. At long context that is the difference between a handful of concurrent
sequences and a fleet's worth.

**This is the first optimisation in this book that can change what the model says.** Everything
before it was equivalence-testable: {ref}`ch05`'s cache, {ref}`ch07`'s batching and {ref}`ch08`'s
paging all produce identical tokens, and {ref}`ch17`'s speculation produces an identical
*distribution*. Dropping a token's keys and values removes information the model would have attended
to. So the measurement here has to be a quality one, and the honest framing is not "how much memory
did we save" but "what did it cost".

## The measurement

Perplexity on held-out text, decoded one token at a time so the cache is genuinely evicted as
generation proceeds — a batched evaluation would quietly hand every token the full context the policy
is supposed to have taken away.

Three lengths, and the first one exists because without it this chapter would have reported the
opposite of the truth.

```{include} _generated/ch16-window.md
```

**At a length the model was trained for, bounding costs almost nothing.** That is the top block, and
it is the closest thing here to an honest price: a window that retains most of a short sequence
retains most of the information, and perplexity barely moves.

**Past the training length, bounding appears to win — and that is the baseline failing, not the
window succeeding.** In the middle block the full-context row is much worse than either bounded one.
Nothing about the bounded configurations improved; the unbounded one degraded, because this model was
trained on 128-token sequences and at 768 tokens it is extrapolating. A window of a few dozen tokens
puts it back inside the distribution it knows.

This is the trap, and it is worth more than the number: **measure only at long context and you would
conclude that throwing away the cache is free.** It is not. What you measured was a baseline out of
its depth. Any claim that a bounded window is quality-neutral needs to say what the model's training
sequence length was, and almost none of them do.

**Past the rotary table, nothing runs at all.** The bottom block is empty for every configuration.
A model's rotary table is sized to the context it was trained on, so a position past the end has no
embedding to look up — the failure is an exception, not a degradation.

### The fix for that last row, and why it is not here

The standard answer is to renumber: give the surviving tokens positions *within the cache* rather
than within the text, so every index stays inside the table. That is what makes a bounded window
work at unbounded length.

It is also not implementable on this engine, and the reason is worth understanding. **A key is
rotated when it is computed**, so its position is baked into the cached value. Renumbering the
survivors means re-rotating them, which means caching keys *before* rotation and applying the
rotation at attention time against a position that can change. That is a model-level change, not a
cache policy, and this book's model does not make it.

I had a version of this measurement that appeared to renumber and did not — it handed each *new*
token a compacted index while the cached keys kept their old rotations, which after the first
eviction meant every subsequent token claimed the same position. It produced numbers. They were
meaningless, and they are not in the table above.

### Does this model even have attention sinks?

The case for keeping the opening tokens is that softmax must put its mass somewhere, and in a trained
model those positions become where it goes. That is a claim about the *model*, not about the eviction
policy, and it is directly measurable:

```{include} _generated/ch16-sinks.md
```

The sinks are unambiguously there — an order of magnitude above an even spread.

And keeping them buys nothing measurable. Compare the two bounded rows in any block above: with
sinks and without are the same to three decimal places. The phenomenon is real in this model and
recovering it does not help it, which is a more interesting result than either "sinks work" or "this
model has no sinks", and it is only visible because the mechanism was measured separately from the
outcome.

The honest limit on all of this: a model trained on 128-token sequences cannot use distant context,
so this rig cannot show what bounding costs a model that can. The mechanism, the memory saving and
the hard wall past the rotary table are all real and measured. The quality cost at long context is
the one thing here that needs a bigger model to price, and {ref}`ch32` lists it among the things this
book could not measure.

## The cost

- **You deleted information.** This is the cost, and it is not recoverable by tuning. Every other
  chapter's trade was latency against throughput or memory against compute; this one trades memory
  against what the model knows.
- **The failure is invisible and plausible.** A model missing distant context does not error — it
  produces a fluent answer that has forgotten something. No latency metric moves. Only an evaluation
  catches it, which makes this the technique most in need of {ref}`ch28`'s "silent quality
  regression" discipline.
- **Eviction is per block, not per token**, so the real budget is a little larger than the policy
  asks for. Rounding outwards is deliberate; rounding inwards would silently drop tokens the policy
  promised to keep.
- **Prefix caching and bounded windows interact badly.** {ref}`ch09` shares blocks between sequences
  by content; a window evicts blocks by *position within a sequence*. Two requests with the same
  prefix but different lengths now want different subsets of the same blocks, and the reference
  counting that made sharing safe was not designed for that.
- **The position rule depends on the model's trained context**, which is a number the serving layer
  has to know and frequently does not. Getting it wrong is a silent quality loss in one direction
  and a crash in the other.

## Key takeaways

- Every other cache optimisation makes entries cheaper; this one makes there be fewer of them. It is
  the only one that turns a linear cache into a constant one.
- **It is the first optimisation in the book that changes the output**, so it is measured on quality
  and not on latency.
- At a length the model was trained for, a window costs almost nothing. Past that length it appears
  to *help* — because the unbounded baseline is extrapolating, not because the window is good.
  **A quality-neutral claim about bounded context is meaningless without the training sequence
  length.**
- Past the rotary table nothing runs at all. The fix is to renumber positions, and that requires
  caching keys before rotation — a model-level change, not a cache policy.
- This model has attention sinks, measurably and by an order of magnitude, and keeping them buys
  nothing at this window size. Measuring the mechanism separately from the outcome is what lets you
  tell "no sinks" apart from "sinks are not the binding loss".

## Looking ahead

{ref}`ch17` goes after latency rather than memory, and does it by a route that sounds impossible:
having a second, worse model write the answer, and checking its work cheaply enough that the result
is provably identical to what the real model would have said.

## Further reading

StreamingLLM (Appendix E) is the attention-sink result and is worth reading for the figure showing
where attention actually goes — it is the observation the technique rests on, and it was found by
looking rather than by theorising. The KV-eviction literature that followed it (H2O, SnapKV and
relatives) attacks the same problem by scoring which tokens to keep rather than taking the most
recent ones, and trades a little compute for a better choice.
