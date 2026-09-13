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

Which makes the serving result harder to explain away:

```{include} _generated/ch10-token-budget.md
```

On this trace, chunking loses on every axis. Goodput falls as the budget shrinks. Median ITL
rises. TTFT gets worse, not better.

The trace is why: 35% of these requests carry a 1536-token prompt, so a prefill is almost always
in flight. There is no quiet time to spread the work into, so every step carries a chunk, every
token pays, and the long requests — a third of the workload — wait many steps for their first
token. The mechanism works exactly as designed and the design does not suit the workload.

This is worth sitting with, because it is the first chapter where the technique everyone
recommends does not help. The honest conclusion is not "chunked prefill is bad". It is:

> **Chunked prefill is a redistribution, not a saving.** It moves latency from streaming users to
> arriving ones. That is a good trade when prefills are occasional and a bad one when they are
> constant.

At production scale the usual case is the first: prompts of a few thousand tokens arriving into a
pool serving hundreds of concurrent conversations, where a single unchunked prefill would stall
every one of them for hundreds of milliseconds. Our trace is deliberately the other case, and it
shows what happens when the assumption behind a technique does not hold.

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
- Chunking redistributes latency rather than removing it — better worst case, worse typical case,
  and much worse TTFT for the request being chunked.
- It only pays when prefills are occasional enough that there is quiet time to spread them into.
  Measure your own trace; do not assume.
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
