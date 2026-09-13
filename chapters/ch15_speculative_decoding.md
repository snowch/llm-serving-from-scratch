---
title: "Speculative Decoding"
short_title: "ch15 Speculative Decoding"
---

(ch15)=
# ch15 · Speculative Decoding

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch03](#ch03), [ch04](#ch04) |
| **Scorecard** | Several tokens per target forward pass at low batch. Nothing at high batch. |
:::

## The problem

Decode is serial by construction. Token *n+1* depends on token *n*, so the model runs, produces one
token, and runs again. {ref}`ch03` showed why that is the worst possible ratio: a forward pass reads
every weight in the model, and we are spending that entire read on a single token.

{ref}`ch07` fixed this for *many* requests by putting them in the same pass. But one user, alone on
a machine, still gets one token per full weight read. Batching cannot help them — there is nobody
to batch with.

## The idea

Something cheap proposes several tokens. The real model checks all of them **in one forward pass**,
because verification is prefill-shaped and prefill is parallel. Proposals the model agrees with are
free; the rest are thrown away.

The economics are simple and unusual: a round costs one target forward pass and yields somewhere
between one and *k+1* tokens. It can never yield zero, because even total disagreement leaves the
target's own token at the first mismatch. **Speculation cannot produce fewer tokens per pass than
not speculating.**

### What proposes the tokens

Two options, and the cheap one is more interesting than it sounds.

A **draft model** is the canonical approach: a smaller model of the same vocabulary, run
autoregressively. It costs real compute per proposal, so it must be much faster than the target.

An **n-gram drafter** costs nothing at all — it looks for where the recent context appeared before
and copies whatever followed:

```{literalinclude} ../llmserve/speculative.py
:language: python
:start-at: class NgramDrafter
:end-before:     def __init__
```

No model, no training, no memory. It works exactly when output repeats input — summarising,
editing code, answering from a quoted document — which is a large share of real traffic. Where
nothing repeats, acceptance falls to zero and it costs nothing to have tried.

### Why the result is exactly the target model

This is the part worth slowing down for, because it is what separates speculation from a
heuristic.

For greedy decoding the argument is trivial: accept a proposal only if it is what the target would
have chosen. Same tokens, fewer passes.

For sampling it is subtler, and the rule looks arbitrary until you see why it works. Accept a
proposed token `x` with probability `min(1, p_target(x) / p_draft(x))`. On rejection, **do not**
resample from `p_target`. Sample from the normalised positive part of `p_target - p_draft`:

```{literalinclude} ../llmserve/speculative.py
:language: python
:start-at: def accept_sampled
:end-before:     accepted: list[int] = []
```

Proposing from `p_draft` and accepting at that ratio leaves each token under-sampled by exactly
`max(0, p_target(x) - p_draft(x))`. Drawing the replacement from precisely that shortfall puts the
missing mass back where it belongs. The combined procedure samples from `p_target` — not
approximately, exactly.

Resampling from `p_target` instead is the obvious-looking simplification, and it is wrong in a way
that is almost impossible to notice: the drafted token gets its mass counted twice, once through
acceptance and again through the correction.

## The build

```{literalinclude} ../llmserve/speculative.py
:language: python
:start-at: def speculative_generate
:end-before:     stats = stats if stats is not None
```

One deliberate omission: the target has no KV cache across speculative rounds. Rolling a rejected
proposal back out of a cache is fiddly and would bury the idea under bookkeeping. The measurement
therefore counts **target forward passes**, which is the quantity speculation actually reduces.

## The measurement

### Does it pay?

```{include} _generated/ch15-acceptance.md
```

At the largest *k* the engine produces eight tokens per target forward pass. Every configuration
produces output identical to greedy decoding, which is the claim being checked rather than assumed.

Two conditions made this measurable, and both are worth naming.

**The model had to be trained.** Against random weights the target's next token is arbitrary, so an
n-gram drafter matches almost nothing and acceptance sits at zero — measuring the drafter against
noise rather than measuring speculation. The first version of this benchmark did exactly that.

**Acceptance is a property of the workload, not of the technique.** This corpus repeats, so copying
earlier text works extremely well. On text that never repeats, the same drafter proposes nothing
useful. Any acceptance rate quoted without its workload is meaningless.

### Is it the same model?

```{include} _generated/ch15-distribution.md
```

The correct rule lands within half a standard error of the target. The plausible-looking bug lands
nearly seven standard errors away, at almost exactly the probability theory predicts for it.

That negative control is the point. A correctness argument you cannot falsify is not worth much, so
the benchmark implements the wrong rule too and shows the test detects it. Without that, "our
speculation is distribution-preserving" is a claim about code nobody checked.

Note also what the test had to avoid. Measured on the *trained* model, the drafted token has
probability 0.9998 — a distribution that close to a point mass cannot reveal a sampling bias,
because every rule returns that token essentially always. The distribution test runs on the
untrained model for that reason. **A measurement has to be taken where it can discriminate**, and
choosing those conditions honestly is different from choosing the conditions that flatter you.

### When it does not pay

Speculation spends extra compute to save serialisation. At batch one, that compute was idle anyway.
As the batch grows, {ref}`ch07` is already amortising the weight read across many sequences, the
device is no longer waiting on a dependency, and the extra verification work competes with real
work. The technique stops helping and starts costing.

This makes speculation a *low-load* optimisation, which is an unusual shape: it improves exactly the
regime where the server has capacity to spare, and stops when it does not. That is still valuable —
single-user interactive latency is what most people judge a deployment by — but it is not a
throughput technique, and deploying it as one disappoints.

## The cost

- **Extra compute per token**, wasted whenever a proposal is rejected.
- **The win evaporates at high batch**, and can become a loss.
- **A draft model is a second model** to load, version, and keep in memory.
- **Acceptance is workload-dependent** and can collapse without warning when traffic shifts.
- **The correctness argument is easy to get subtly wrong**, and the failure mode is fluent,
  plausible text drawn from the wrong distribution. Only a distributional test catches it.
- **KV cache interaction is genuinely awkward** — rolling back rejected proposals is where real
  implementations get complicated.

## Key takeaways

- Verification is prefill-shaped and therefore parallel: *k* proposals cost one target forward pass.
- A round always yields at least one token, so speculation never reduces tokens per pass.
- The residual acceptance rule makes the output distribution *exactly* the target's. Resampling
  from `p_target` on rejection double-counts the drafted token and biases the result invisibly.
- Test the correctness claim with a negative control. An argument you cannot falsify is not
  evidence.
- Acceptance depends on the workload. Quote it with the trace or not at all.
- Speculation helps at low batch and hurts at high batch. It is a latency technique, not a
  throughput one.

## Looking ahead

Speculation constrains *which* tokens appear by proposing them. {ref}`ch16` constrains which tokens
are *allowed* — grammars and JSON schemas enforced during decoding — and runs into the same
question from the other side: what does restricting the distribution do to the output, and what does
enforcing it cost per step?

## Further reading

Leviathan et al. and Chen et al. (Appendix E) introduced speculative decoding independently; both
contain the proof sketched above, and it is short enough to read in full now that you have
implemented it. Medusa and EAGLE replace the separate draft model with extra heads on the target,
removing the second-model problem. Prompt lookup — the n-gram drafter here — is folklore rather than
a paper, and is the one to try first because it costs nothing.
