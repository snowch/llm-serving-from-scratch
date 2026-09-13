---
title: "Chunked Prefill and Scheduling Policy"
short_title: "ch10 Chunked Prefill"
---

(ch10)=
# ch10 · Chunked Prefill and Scheduling Policy

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch07](#ch07), [ch09](#ch09) |
| **Scorecard** | A dial, not an improvement. On our workload, turning it costs more than it saves. |
:::

## The problem

Every engine so far treats a prefill as indivisible. Admitting a request means running its entire
prompt in one step, and everything already streaming waits for that step to end.

With short prompts nobody notices. With an eight-thousand-token prompt, every user mid-sentence
stops for as long as that prefill takes. They are not queued behind a busy server; they are queued
behind one other person's paste.

## The idea

Stop letting a step be as large as the work that arrives. Give each step a **token budget**, spend
it on decode first — those tokens are what someone is watching — and fill whatever is left with a
*piece* of a pending prefill. A long prompt then arrives over several steps instead of stopping
the world once.

```{literalinclude} ../llmserve/engines/chunked.py
:language: python
:start-at:         # 1. Decode first
:end-before:         # 3. Admit new work
```

The ordering encodes the policy: decode before prefill, prefill-in-progress before new admissions.
Reverse any of those and you get a different, defensible engine with different victims.

### What it actually trades

The usual description — "chunking smooths out prefill stalls" — is true and incomplete. Three
things change at once, and only the first is the advertised one:

1. **The worst-case stall shrinks.** No single step contains an entire long prefill.
2. **Typical ITL rises.** Every step now carries a prefill chunk, so *all* tokens get slower
   rather than a few getting much slower. Chunking converts a rare large delay into a constant
   moderate one.
3. **The prefilling request's own TTFT grows**, and badly. Its prompt now takes many steps, and it
   produces nothing until the last of them. Chunking protects everyone except the request being
   chunked.

Whether that is a good deal depends entirely on how often prefills are in flight. If they are
rare, you trade a bad p99 for a slightly worse p50 and win. If nearly every step has a prefill
pending, there is no quiet time to spread the cost into, and you simply pay more.

## The build

A sequence mid-prefill is not the same as one mid-decode, and distinguishing them is fiddlier than
it looks:

```{literalinclude} ../llmserve/request.py
:language: python
:start-at:     #: how many tokens must be cached before this sequence can decode
:end-before:     finished: bool
```

That comment records a bug. The obvious test for "still prefilling" is `stored < len(all_token_ids)`
— and it is wrong, because `all_token_ids` grows with every generated token, so a sequence that
has decoded once looks like it needs prefilling again. It then takes the prefill path, which never
grows the block table, and writes off the end of it.

A second, similar bug: a sequence that finishes during the decode phase has already had its blocks
released, so it must leave the running set *before* the prefill phase looks at it. Otherwise it
appears as a sequence needing prefill against an empty block table.

Both are the same mistake in different clothes — using a derived quantity to represent a state
that deserves to be explicit.

Finally, carrying the cache between chunks matters more than it appears:

```{literalinclude} ../llmserve/engines/chunked.py
:language: python
:start-at:         # Carry the cache forward between chunks
:end-before:         if state.past is None
```

Re-reading a sequence's cache from its blocks on every chunk makes a chunked prefill quadratic in
prompt length. We wrote it that way first, measured, and found chunking catastrophically slow — a
result that would have been a libel on the technique rather than a finding about it.

## The measurement

First, something genuinely surprising. How long does one long prefill take, split different ways?

```{include} _generated/ch10-chunk-cost.md
```

**Splitting the prefill makes it nearly twice as fast.** Not slower — faster. A single pass over
1536 tokens materialises a 1536×1536 attention matrix per head per layer, which is hostile to
every cache in the machine. Several smaller passes, each attending to the accumulated KV cache,
do the same arithmetic with far better locality.

So chunking is not merely a scheduling tool. It is also, at this size, a faster way to prefill.

Now the serving result, which is not a straight line:

```{include} _generated/ch10-token-budget.md
```

**The budget is a dial with an interior optimum, not a direction.** A moderate budget beats no
chunking on every axis in that table — better TTFT at both percentiles, lower median ITL, more
throughput and more goodput. An aggressive budget is worse than not chunking at all, on every axis.

Both halves have the same cause, and it is the microbenchmark above. Splitting a prefill into a few
passes is *faster* than doing it in one, so a moderate budget is getting the locality win and the
scheduling win together. Splitting it into very many passes throws that away: each step's chunk is
too small to amortise the per-step overhead, and a long request now waits many steps for its first
token. The mechanism is the same at both settings; what changes is whether the chunk is large enough
to be worth a pass.

That gives the honest statement of what chunking is:

> **Chunked prefill is a redistribution, not a saving** — plus, at the right chunk size, a locality
> win that happens to come with it. It moves latency from streaming users to arriving ones, and how
> much it moves depends on a budget that has a wrong answer in both directions.

The practical consequence is that a token budget has to be *tuned*, on the workload, and that the
default in any engine is a guess about a length distribution. Our trace is deliberately harsh —
35% of requests carry a 1536-token prompt, so a prefill is almost always in flight and there is
little quiet time to spread work into — and even here a moderate budget wins. On a workload with
occasional long prompts among many short ones, the margin is larger; {ref}`ch23` measures it on
retrieval traffic, where the prompts are long and the arrival rate decides the answer.

## The cost

- **A new dial with no safe default.** The right budget depends on prompt-length distribution,
  concurrency and which latency you are accountable for. Part VI tunes it per workload.
- **The prefilling request pays for everyone else's smoothness.** Fairness has been traded
  without saying so, which matters if long prompts come disproportionately from one tenant.
- **More steps means more per-step overhead**, all of it pure loss.
- **The scheduler now has three phases with an order that encodes policy.** Every later chapter
  modifies this loop, and mistakes here starve requests rather than crash.

## Key takeaways

- A token budget per step stops one long prefill from monopolising the engine, and makes the
  prefill/decode trade explicit rather than accidental.
- Decode before prefill: those tokens have someone waiting on them.
- Chunking redistributes latency rather than removing it, and the budget has a wrong answer in
  both directions: a moderate one beat no chunking on every axis here, an aggressive one lost on
  every axis.
- A token budget must be tuned on the workload. Any engine's default is a guess about a length
  distribution, and it may not be yours.
- Splitting a long prefill can be *faster* than doing it in one pass, because a single enormous
  attention matrix has terrible locality.
- Represent scheduler state explicitly. Both bugs in this chapter came from inferring "is this
  sequence still prefilling?" from a quantity that changes for other reasons.

## Looking ahead

Prefill and decode want different things from the hardware, and every chapter so far has made one
engine serve both. {ref}`ch11` asks what happens if they simply stop sharing — separate pools,
each scheduled for what it is good at, with the KV cache shipped between them.

## Further reading

Sarathi-Serve (Agrawal et al., Appendix E) introduced chunked prefill and the throughput-latency
framing used here; its evaluation covers the regime where it clearly wins, which is a useful
counterweight to this chapter's trace.
