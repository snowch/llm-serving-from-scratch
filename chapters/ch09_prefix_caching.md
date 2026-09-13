---
title: "Prefix Caching"
short_title: "ch09 Prefix Caching"
---

(ch09)=
# ch09 · Prefix Caching

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch08](#ch08) |
| **Scorecard** | Large TTFT improvement on realistic traffic, at no cost. |
:::

## The problem

Look at what an assistant actually receives. Every request carries the same system prompt —
instructions, tone, tool descriptions, safety text — and then a short unique question. In
production that preamble is routinely hundreds or thousands of tokens, and every request prefills
all of it from scratch.

The engine has computed those exact keys and values before. Often within the last second. It
throws them away every time.

## The idea

{ref}`ch08` already did the hard part without using it. A sequence's cache is a list of block
numbers, and nothing about a block ties it to a particular sequence. The allocator already
reference counts. Everything needed to *share* a prefix is in place.

So: if a request's prompt begins with tokens the engine has already processed, point its block
table at the existing blocks and prefill only the genuinely new part.

### Key by content, and by everything before it

A block's key is a hash of **every token up to and including that block**, not of the tokens
inside it. This is the detail that makes the scheme safe.

Hash only a block's own contents and two different prompts that happen to share a middle block
would collide — a request would attend to somebody else's context, and the output would be subtly
wrong with nothing to indicate it. Cumulative hashing makes a block's identity mean "the sequence
that got here", which is exactly what attention depends on.

For the same reason, a lookup stops at the first miss. A hit *after* a gap is worthless: keys and
values encode positions and what preceded them, so a prefix is only usable if every block before
it is usable too.

```{literalinclude} ../llmserve/cache/prefix.py
:language: python
:start-at:     def lookup
:end-before:         blocks: list[int] = []
```

### Only full blocks, and published immediately

Two rules follow from how blocks fill.

**Only complete blocks are shareable.** A partially filled block still has slots that will be
written by tokens which have not arrived; publishing it would let another sequence read whatever
happens to be there.

**A block is published the moment it is full, not when its author finishes.** This one we got
wrong first, and the bug is instructive: publishing on request completion produced a cache with a
zero hit rate. Requests that arrive together all prefill before any of them finishes, so every one
of them recomputed the same shared prompt and only a *later* request could ever benefit. In a real
assistant, concurrent arrivals sharing a system prompt are the entire workload.

### The cache must lose to the scheduler

There is a third rule, and unlike the first two it is not about correctness — it is about the
engine continuing to function at all.

A cached block is held with a reference, so it is not free. To {ref}`ch08`'s admission loop, which
counts free blocks, a warm cache is **indistinguishable from a full pool**. We got this wrong too,
and the failure is total rather than gradual: the scheduler refuses to admit anything; every running
sequence finishes and publishes *more* blocks into the cache; and the engine then spins forever with
a full queue and an empty batch. Not a slowdown — a stall, with work available, and with the memory
to do it held by a cache whose entire purpose was to make that work cheaper.

So admission is allowed to take the cache apart:

```{literalinclude} ../llmserve/engines/prefix.py
:language: python
:start-at:     def _make_room
:end-before:         while self.cache.allocator.n_free < n:
```

Note what it deliberately does not do. It evicts cached prefixes and stops; it never preempts a
running sequence to admit a queued one. Preemption to admit is not scheduling, it is thrash, and it
belongs in the path where a sequence that was already promised memory needs it.

The general form of this is worth carrying out of the chapter: **a cache that can refuse to yield is
not a cache, it is a leak.** Anything holding memory speculatively has to lose to something that
needs it now, and the test that this actually happens is not optional:

```{literalinclude} ../tests/test_prefix.py
:language: python
:start-at: def test_the_cache_gives_up_blocks_so_the_scheduler_can_admit
:end-before:     class WithoutMakeRoom
```

## The build

Lookup, share, and prefill only the remainder:

```{literalinclude} ../llmserve/engines/prefix.py
:language: python
:start-at:             tokens = state.all_token_ids
:end-before:             past = None
```

Note the cap on how much may be shared. If the *whole* prompt were cached there would be nothing
to run the model on, and no distribution for the next token. At least one token always goes
through.

Publishing is deliberately separated so it can happen at prefill time:

```{literalinclude} ../llmserve/engines/prefix.py
:language: python
:start-at:     def _publish
:end-before:         published = self.prefix.publish
```

And when memory runs short, cached prefixes are evicted before anyone is preempted:

```{literalinclude} ../llmserve/engines/prefix.py
:language: python
:start-at:     def _free_blocks_for
:end-before:         while self.cache.allocator.n_free
```

The ordering there is the chapter's one real policy decision. A cached prefix costs only
recomputation, and only if someone wants it again. A preempted sequence costs recomputation *and*
a latency spike for a caller already waiting. Evict the cheap thing first.

## The measurement

This needs a workload where requests genuinely share text, so the trace changes: every request
carries one system prompt, then a unique question. That is not a convenient special case — it is
what assistant traffic looks like.

```{include} _generated/ch09-prefix-caching.md
```

Three quarters of all prompt tokens were served from cache rather than recomputed. Median TTFT
falls by roughly half at every load, and throughput improves as well — the prefill work avoided is
capacity handed back to decode.

The result worth emphasising is not the size of the win but its shape: **this chapter has no
trade in it**. {ref}`ch06` bought throughput with ITL. {ref}`ch08` bought memory with throughput.
This one costs nothing anybody notices.

One honest caveat: on the uniform-random trace used by chapters 1–8, this engine and {ref}`ch08`'s
are identical, because there is nothing to reuse. A test asserts exactly that, because an
optimisation that invents savings on a workload with no shared text would be measuring its own
bookkeeping.

## The cost

Small, but not zero:

- **Cache memory competes with running sequences.** Blocks held for reuse are blocks unavailable
  to admit somebody. The eviction order above is what keeps that from hurting, it is a policy that
  can be got wrong, and when it is got wrong the engine stalls rather than slows.
- **Hashing every prompt costs something.** Trivial next to prefill, not free.
- **Correctness now depends on a hash.** A collision means a request attends to another's context.
  Cumulative keying makes this vanishingly unlikely; a production engine verifies the tokens
  match rather than trusting the hash alone, and ours does not.
- **Cache state is invisible in the output.** A bug here does not crash, it produces subtly wrong
  answers — hence the equivalence test asserting a hit changes nothing.
- **Reuse depends entirely on the workload.** {ref}`ch22` shows agent traffic where it is worth
  more than everything else in this book combined, and batch workloads where it is worth nothing.

## Key takeaways

- Identical prompt prefixes produce identical keys and values, so they can be shared rather than
  recomputed. Paging already provided the mechanism.
- Key a block by every token up to and including it. Keying by its own contents lets unrelated
  prompts collide, producing wrong output with no error.
- Stop a lookup at the first miss; a later hit across a gap is unusable.
- Publish a block when it fills, not when its author finishes — otherwise concurrent requests
  sharing a prompt all miss, which is the exact case that matters.
- Evict cached prefixes before preempting running sequences: the cached prefix is the cheaper loss.
- **A cache that can refuse to yield is not a cache, it is a leak.** Admission must be able to take
  the cache apart, or a warm cache stalls the engine outright.
- On shared-prefix traffic this is the best return in the book, and the only optimisation so far
  with no trade attached.

## Looking ahead

The engine now avoids repeating work it has already done. What it still does badly is *mix* work:
admitting a request with a long prompt means one enormous prefill in a step, and every sequence
already streaming feels it as a stall. {ref}`ch10` breaks prefill into pieces and makes the
scheduler choose deliberately between time-to-first-token and smooth streaming.

## Further reading

SGLang's RadixAttention (Zheng et al., Appendix E) generalises this from a flat hash map to a
radix tree, which shares *partial* prefixes between requests that diverge partway through — the
natural next step once you have built the version in this chapter.
