---
title: "Generating Tokens Correctly"
short_title: "ch04 Generating Tokens Correctly"
---

(ch04)=
# ch04 · Generating Tokens Correctly

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch01](#ch01) |
| **Scorecard** | No performance change. Establishes the **correctness baseline** every later optimisation is tested against. |
:::

## The problem

We are about to spend twenty-five chapters making this engine faster. Every one of those changes
is an opportunity to make it *wrong* — and wrong in the particular way that is hardest to catch,
where the output is still fluent, still plausible, just no longer what the model would have said.

Before optimising, we need a decode path we own completely and can hold fixed. That means taking
back the decisions `model.generate()` was making on our behalf. There are more of them than you
would expect, and one of them is a bug that will otherwise ship to production and look like the
model's fault.

## The idea

### Sampling is a pipeline, and the order matters

Turning logits into a token is four decisions applied in sequence:

1. **Penalties** adjust the logits of tokens already seen.
2. **Temperature** rescales the whole distribution: below 1 sharpens it, above 1 flattens it.
3. **Truncation** — top-k, top-p, min-p — removes the tail.
4. **Draw** one token from what survives.

The order is not arbitrary, and getting it wrong is a classic source of two implementations
disagreeing while both look correct. Apply temperature *after* truncation and you change which
tokens survived the cut — high temperature will have widened a distribution that was already
narrowed, so the nucleus you kept was computed against different probabilities than the ones you
sample from.

A note on the repetition penalty, which looks like a typo when you first meet it:

```{literalinclude} ../llmserve/sampling.py
:language: python
:start-at: def apply_repetition_penalty
:end-before: def apply_top_k
```

Dividing a *negative* logit makes it larger, and therefore the token more likely — the opposite
of a penalty. The sign has to be handled explicitly, and implementations that forget it quietly
encourage exactly the repetition they were meant to suppress.

The truncation filters are worth distinguishing, because they fail differently:

- **top-k** keeps a fixed number of tokens. Simple, and wrong in both directions: too permissive
  when the model is confident, too restrictive when it genuinely is not.
- **top-p** (nucleus) keeps the smallest set whose probability mass reaches *p*. Adapts to the
  distribution's shape. The detail people get wrong is the boundary: the token that *crosses* the
  threshold is kept, not dropped.
- **min-p** keeps tokens at least some fraction as probable as the best one. Adapts to
  *confidence* rather than mass: a peaked distribution keeps almost nothing, a flat one keeps a
  lot.

### Stopping is not one condition

A request ends for one of several reasons, and a serving engine has to distinguish them because
the caller needs to know which happened:

- An **end-of-sequence token** was produced — the model chose to stop.
- The **token budget** ran out — we stopped it.
- A **stop string** appeared in the output.
- The **caller disconnected** — nobody is waiting for the rest ({ref}`ch26`).

Only the first is the model finishing a thought. Reporting a budget exhaustion as a normal
completion is how truncated JSON reaches a caller that had no reason to suspect it.

Stop *strings* are harder than stop tokens, because a string may span several tokens, and may be
only partially emitted when you check. That is the same boundary problem as the next section,
which is the one that actually bites.

### The bug: streaming detokenisation

Here is a serving bug that looks exactly like a model bug.

The obvious way to stream is to decode each token to text as it is produced and send it. For
ASCII this works perfectly, so it survives testing.

Then someone types a prompt in Japanese. A character like `日` is three bytes; with a byte-level
tokenizer that is three tokens, and with a BPE tokenizer it is frequently split too. Decode the
first of those bytes on its own and it is not valid UTF-8, so the decoder emits `�`. The user
watches replacement characters appear mid-word and concludes the model is broken.

The fix is to hold back bytes that do not yet form a complete character:

```{literalinclude} ../llmserve/tokenizer.py
:language: python
:start-at: class IncrementalDetokenizer
:end-before:     def __init__
```

The subtlety — and the reason our first attempt at this was wrong — is telling a **truncated**
sequence from an **invalid** one. Both raise the same exception. The first should be held back
until more bytes arrive; the second should be emitted as a replacement character immediately,
because waiting for it to become valid means waiting forever. Python's incremental codec already
draws that line correctly, so we use it rather than re-deriving the UTF-8 state machine and
getting it subtly wrong.

## The build

Everything above lives in two modules. The engine now owns its decode path end to end:

```{literalinclude} ../llmserve/sampling.py
:language: python
:start-at: def sample(
:end-before:     if logits.dim() != 2
```

And the stopping logic is explicit about *why* a request ended:

```{literalinclude} ../llmserve/request.py
:language: python
:start-at:     def check_finished
:end-before:         if len(self.output_token_ids)
```

## The measurement

This chapter's measurement is not a latency number — it is a test suite. From here on, every
optimisation must produce identical output to the path it replaces, and that claim is checked
rather than asserted.

The pattern that matters most:

```{literalinclude} ../tests/test_engines.py
:language: python
:start-at: def test_kv_cache_does_not_change_output
:end-before: def test_engine_produces_exactly_max_tokens
```

Two requirements this places on the engine, both easy to lose later:

- **Greedy decoding must be deterministic.** With temperature at zero there is no sampling, so
  any difference between two runs is a bug, not noise. This makes greedy the workhorse of every
  equivalence test in the book.
- **Sampled decoding must be reproducible given a seed.** Otherwise optimisations that change
  sampling — speculative decoding in {ref}`ch17` above all — cannot be checked at all, and that
  is precisely the one whose correctness argument is subtle enough to need checking.

The streaming contract gets its own tests, because the failure is silent:

```{literalinclude} ../tests/test_tokenizer.py
:language: python
:start-at: def test_partial_multibyte_is_held_back_not_emitted
:end-before: def test_invalid_byte_becomes_a_replacement_character
```

## The cost

We now own code that HuggingFace used to own, and that has a price:

- **Every bug here is ours.** Hence the test suite arriving in this chapter rather than later.
- **We will diverge from reference implementations in small ways.** Filter order, tie-breaking in
  `argmax`, float accumulation order — all defensible, none identical. When comparing against
  another engine ({ref}`ch31`), expect to reconcile these before concluding anything.
- **Determinism constrains optimisation.** Some fast paths reorder floating-point accumulation and
  change results in the last bits. Our equivalence tests will flag that, which is the intent, but
  it does mean some tests need tolerances rather than exact equality — and deciding which is a
  judgement call each time.

## Key takeaways

- Sampling is a pipeline, and applying its stages out of order produces a plausible-looking
  distribution that is not the one you configured.
- Dividing a negative logit raises its probability. A repetition penalty that ignores the sign
  encourages what it was meant to suppress.
- Stop reasons are not interchangeable. A caller needs to know whether the model finished or the
  budget ran out.
- Decoding each token to text as it arrives breaks every multi-byte character. Hold back
  incomplete sequences — but distinguish *incomplete* from *invalid*, or the stream stalls forever
  on a corrupt byte.
- Greedy determinism plus seeded reproducibility is what makes every later optimisation testable.

## Looking ahead

The decode path is correct and ours. It is also doing an enormous amount of redundant work:
{ref}`ch01`'s engine re-runs the entire prefix for every single token. {ref}`ch05` caches what it
already computed, measures the difference, and immediately runs into the constraint that shapes
all of Part III.

## Further reading

Nucleus sampling was introduced in Holtzman et al., *The Curious Case of Neural Text
Degeneration*, which is also the clearest explanation of why pure greedy decoding produces
degenerate repetition. For stop-string handling and chat-template edge cases, {ref}`ch26` returns
to the topic once there is an API surface to break.
