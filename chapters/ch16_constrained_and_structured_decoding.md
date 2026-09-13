---
title: "Constrained and Structured Decoding"
short_title: "ch16 Constrained Decoding"
---

(ch16)=
# ch16 · Constrained and Structured Decoding

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch04](#ch04) |
| **Scorecard** | Validity becomes a guarantee — with one caveat that matters more than the guarantee. |
:::

## The problem

A caller asks for JSON. The model produces JSON *usually*. Prompting raises the odds, retrying
covers some of the remainder, and the caller still has to handle a parse failure — which is the
one branch nobody tests.

All of that is paying, repeatedly, for something you can simply have.

## The idea

At every step the engine already has the full distribution over next tokens. If a token cannot
legally come next, set its logit to negative infinity before sampling. Invalid output stops being
unlikely and becomes *unreachable*.

The grammar is a finite-state machine. Each state knows which tokens keep the output parseable:

```{literalinclude} ../llmserve/constrain.py
:language: python
:start-at:     def allowed_bytes
:end-before:         if state is State.EXPECT_COLON
```

Generation then interleaves two machines: the model proposes a distribution, the FSM says which
part of it is legal, and the sampler draws from what survives.

### One place this book gets an unfair advantage

Our tokenizer is byte-level, so a token *is* a grammar symbol and the FSM transitions directly.
With BPE that is false: a single token can span several grammar symbols — `":"` and `": "` and
`": {"` may all be distinct tokens — so the character-level FSM has to be lifted into token space,
which means precomputing, for every state, which of 128,000 tokens keep the parse alive.

**That lifting, not the grammar, is where production implementations spend their complexity.**
Outlines and XGrammar are largely engineering around that problem. Chapter 16 gets to skip it, and
it would be dishonest not to say so.

### Masking has a sharp edge

```{literalinclude} ../llmserve/constrain.py
:language: python
:start-at:     def apply
:end-before:         allowed = self.mask
```

The same trap as {ref}`ch06`. A grammar that permits nothing produces an all-`-inf` row, softmax
turns it into NaN, and the corruption spreads. A finite floor makes that failure show up as a
strange token rather than as poison in the batch.

## The measurement

```{include} _generated/ch16-validity.md
```

Unconstrained, **nothing** parses. Constrained, everything that finished parses — a hundred
percent, by construction rather than by luck.

But look at the middle row. Across all attempts, well under half produced valid JSON, and the
grammar was working perfectly the entire time.

**A grammar constrains shape, not length.** Every token emitted was legal; the generation simply
ran out of budget before closing its braces. A prefix of a valid document is not a valid document,
so the caller receives something unparseable despite a guarantee that sounds like it should have
prevented exactly that.

This is the most important thing in the chapter, and it is easy to miss because it is not a bug in
anything. The guarantee you get is "no illegal token is ever emitted". The guarantee people assume
they are buying is "the output parses". Those differ whenever generation can be truncated — which
is always, since `max_tokens` exists. Production systems need the FSM's accepting state wired to
the stopping condition, and a budget large enough to reach it.

### What enforcement costs

```{include} _generated/ch16-mask-cost.md
```

The cost is flat in vocabulary size, which is not what you would expect, and the reason is in our
implementation: `build_mask` iterates the *allowed* set — under a hundred bytes — rather than the
vocabulary. An implementation that scanned all 128,000 tokens per step would scale with the
vocabulary and be far slower. **Iterating the small side is the optimisation**, and it is invisible
until you look at which loop you wrote.

Even so, rebuilding on every step costs real time over a long generation, and there are only a
handful of states. Caching turns a per-step rebuild into a dictionary lookup. Our end-to-end timing
shows no difference between cached and uncached, because 337 µs vanishes next to a forward pass on
this model — another case where the rig cannot show what the arithmetic does.

## The cost

- **Valid is not correct.** A schema guarantees shape and says nothing about truth. Constrained
  output can be perfectly formed and entirely wrong, and it looks more authoritative for it.
- **The distribution is narrowed**, sometimes to a single token. At low temperature a grammar can
  force a path the model assigned almost no probability to — syntactically fine, semantically odd.
- **Truncation still produces garbage**, per the measurement above.
- **Interaction with {ref}`ch15`** is awkward: a drafter that does not know the grammar proposes
  tokens the mask forbids, and acceptance collapses. Speculation and constraints need to share the
  FSM.
- **Interaction with {ref}`ch09`** too: a constrained request's logits depend on grammar state as
  well as prefix, so two requests with identical prefixes are no longer interchangeable.
- **Grammar bugs are silent.** An over-permissive grammar produces output that parses and is
  subtly wrong; an over-restrictive one quietly forces the model into strange corners.

## Key takeaways

- Masking illegal tokens makes invalid output unreachable rather than unlikely.
- The grammar is a finite-state machine; only the *lifting* from characters to tokens is hard, and
  byte-level tokenization lets this book skip that.
- Mask with a finite floor, never `-inf`, or an empty permitted set produces NaN.
- **A grammar constrains shape, not length.** Truncation yields unparseable output despite every
  token being legal, so wire the accepting state to the stopping condition.
- Build masks by iterating the allowed set, not the vocabulary, and cache them per state.
- Constraints interact badly with speculation and with prefix caching. Both need to know about the
  grammar.

## Looking ahead

Part IV has made the arithmetic cheaper, the weights smaller, the serialisation shorter and the
output well-formed. Part V asks what happens when one accelerator is not enough: splitting a model
across devices, routing across replicas, and serving many tenants from one base model.

{ref}`ch17` needs more than one GPU and is written against hardware this book's default tier does
not have.

## Further reading

Willard and Louf's *Efficient Guided Generation* (Appendix E) is the Outlines paper and covers the
token-lifting problem this chapter sidesteps. XGrammar attacks the same problem with a faster
precomputation. For the interaction with speculation, the literature is thin and the engineering is
mostly folklore — which is a reasonable signal about how often it is got wrong.
