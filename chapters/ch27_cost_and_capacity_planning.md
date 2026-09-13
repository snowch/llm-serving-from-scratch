---
title: "Cost and Capacity Planning"
short_title: "ch27 Cost and Capacity"
---

(ch27)=
# ch27 · Cost and Capacity Planning

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — arithmetic only |
| **Prerequisites** | [ch03](#ch03), [ch07](#ch07), [ch18](#ch18) |
| **Scorecard** | Nine times the cost per token from one number nobody states. |
:::

## The problem

Cost per token is arithmetic over four inputs: what the hardware costs per hour, how many tokens per
second it sustains, how busy it is, and what mix of prompt and output tokens the traffic has.

Cost estimates are usually wrong, and not because the arithmetic is hard. They are wrong because one
of the four is quietly assumed — and it is almost always the same one.

## Utilisation is the whole answer

```{include} _generated/ch27-cost.md
```

Same hardware. Same engine. Same throughput. Nine times the cost per token, from the fraction of
wall-clock time the machine spends doing work.

The trap is that a throughput number from a benchmark is a **peak**. A service with a daily traffic
curve runs far below it most of the time, and the cost per token is the hardware bill divided by the
tokens actually produced, not by the tokens that could have been produced. A fleet sized for the daily
peak is idle at night; a fleet sized for the mean misses its objective at the peak; and the fleet you
have is somewhere between, at a utilisation nobody has measured.

```{literalinclude} ../llmserve/cost.py
:language: python
:start-at: class Deployment:
:end-before:     name: str
```

**If you take one thing from this chapter: go and measure your utilisation before you optimise
anything.** It is almost certainly lower than you think, and if it is, it is the largest term
available and no amount of work from Parts II and III competes with it.

The corollary is that most of {ref}`ch18` is a cost chapter in disguise. Routing, autoscaling and
warm pools all move utilisation, which moves the number above by more than a kernel ever will.

## Prompt tokens and output tokens are not the same thing

{ref}`ch03` established that prefill is compute-bound and parallel across the prompt, while decode is
memory-bandwidth-bound and strictly sequential. A prompt token therefore costs a small fraction of
what an output token costs in machine time:

```{literalinclude} ../llmserve/cost.py
:language: python
:start-at: def blended_cost_per_million
:end-before:     if prefill_speedup <= 0:
```

Any pricing model that charges the same for both is either overcharging for prompts or undercharging
for outputs — which is why every hosted API prices them separately, and why a self-hosting comparison
that uses a single blended rate is comparing the wrong quantities.

It also means **your token mix changes your cost per token** without anything about the deployment
changing. {ref}`ch21`'s retrieval workload is enormously prompt-heavy and is therefore cheaper per
billed token than {ref}`ch20`'s chat, on identical hardware, while being harder to serve. The two
statements are not in tension: one is about machine time, the other about what you count.

## Build versus buy

```{include} _generated/ch27-break-even.md
```

Note what does not appear in that table: utilisation. The hardware bill is the same whether the
machine is busy or idle, so the break-even *volume* depends only on that bill and the hosted price.
What utilisation changes is the cost of the tokens you do serve — the other half of the comparison,
and the reason a fleet that looks cheap on paper is not.

```{literalinclude} ../llmserve/cost.py
:language: python
:start-at: def break_even_tokens_per_month
:end-before:     if api_dollars_per_million <= 0:
```

The honest reading of that table is that the crossover is at a **large and specific volume**, and
below it the hosted API is cheaper *and* somebody else operates it. This book has spent twenty-nine
chapters on how to operate one, so it should be the one to say that operating it is a cost the
arithmetic above does not contain: the engineers, the on-call rotation, the incidents in
{ref}`ch26`, and the capacity headroom you pay for and do not use.

The reasons to self-host that survive this arithmetic are usually not cost: data residency, a
fine-tuned model, latency requirements a shared API cannot meet, or a volume far enough past the
crossover that the margin is worth the operational burden.

## Sizing a fleet

```{literalinclude} ../llmserve/cost.py
:language: python
:start-at: def instances_for
:end-before:     per_instance = deployment.tokens_per_second
```

Dividing demand by peak throughput is the standard mistake, and it sizes the fleet for a world in
which every instance is always saturated — which is the world where every request also waits. The
peak throughput and the latency objective are not simultaneously achievable, and the batch size that
reaches one is not the batch size that meets the other ({ref}`ch20`).

So sizing has three steps, in this order:

1. **Pick the batch size that meets the SLO**, from {ref}`ch20`'s table for your workload.
2. **Measure sustained throughput at that batch size**, not at the maximum.
3. **Divide demand by that, then by your real utilisation**, and add headroom for the cold start in
   {ref}`ch18` — because an autoscaler is late by exactly that much.

## The cost

- **Every number here is an input you have to supply**, and the ones people have are usually the
  wrong ones: a vendor's peak throughput, an assumed utilisation, and a blended token price.
- **The arithmetic omits operations entirely.** Engineers, on-call and incident cost are real and are
  frequently larger than the hardware bill at the volumes where self-hosting first looks attractive.
- **Optimising cost and optimising latency pull apart.** Higher utilisation is cheaper and queues
  more; {ref}`ch20`'s chat tuning is deliberately more expensive per token than {ref}`ch23`'s batch
  tuning, and that is correct rather than wasteful.
- **Spot and preemptible capacity change the model** in a way this chapter does not cover: they cut
  the hourly rate substantially and add a failure mode that {ref}`ch26`'s draining is the only
  defence against, on a timescale shorter than a long generation.
- **These numbers age fast.** Hardware prices, hosted prices and model efficiency all move quickly,
  which is why this chapter computes from inputs you supply rather than quoting any.

## Key takeaways

- **Utilisation dominates cost per token**, by nearly an order of magnitude across a realistic range.
  Measure yours before optimising anything else.
- A benchmark's throughput is a peak. Sizing from it produces a fleet that meets its throughput
  target and misses its latency one.
- Prompt tokens and output tokens cost different amounts of machine time, so your traffic's mix
  changes your cost per token without anything else changing.
- The break-even volume against a hosted API depends on the fixed hardware bill and the hosted price,
  not on utilisation — but the cost of what you serve depends on almost nothing else.
- The arithmetic leaves out operations, and at the volumes where self-hosting first looks attractive,
  operations is usually the larger number.
- Size the fleet from the batch size that meets the SLO, not the one that maximises throughput.

## Looking ahead

Every number in this chapter came from a throughput figure, and {ref}`ch28` is about where that figure
comes from — including a demonstration that the same engine, on the same workload, can honestly report
a tail latency two orders of magnitude apart depending on how the load was generated.

## Further reading

There is no literature here worth citing, and that is itself informative: cost modelling for serving
is done in spreadsheets inside companies and published rarely. The closest useful reading is capacity
planning from classical systems work, where the relationship between utilisation, queueing and latency
is derived properly rather than assumed — and the result, that latency rises sharply as utilisation
approaches one, is the reason the cheapest configuration is never the fastest.
