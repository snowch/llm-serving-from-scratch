---
title: "Offloading the KV Cache"
short_title: "ch12 Offloading the KV Cache"
---

(ch12)=
# ch12 · Offloading the KV Cache

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch08](#ch08), [ch09](#ch09) |
| **Scorecard** | Reuse survives a block pool too small to hold it — and on a rig with no second device, costs latency to do so. The arithmetic says why, and says it inverts on real hardware. |
:::

## The problem

{ref}`ch08`'s allocator has exactly one answer when it runs out of blocks: throw something away.
{ref}`ch09` softened it — give up a cached prefix before preempting a live sequence — but the shape
did not change. Whatever is evicted must be **recomputed** if it is wanted again, and recomputing a
long shared prefix is a full prefill nobody asked for.

Give {ref}`ch09`'s engine a block pool too small for its working set and watch what happens to the
thing that chapter was built for:

```{include} _generated/ch12-offload.md
```

Read the top row first. The prefix cache is doing almost nothing — not because the workload has no
shared text, but because every block it caches is evicted before the request that would have used it
arrives. The cache thrashes, and thrashing is worse than having no cache at all, because you paid
for the bookkeeping too.

## The idea

Device memory is not the only memory in the machine.

A block that will not fit on the device can be moved somewhere slower and larger instead of being
destroyed. Fetching it back costs a transfer; recomputing it costs a prefill. If the transfer is
cheaper, eviction stops being a loss and becomes a **demotion** — which is precisely the trade an
operating system makes when it pages rather than discards, and the reason {ref}`ch08`'s machinery
already looks like an operating system's.

The tier is keyed by content, not by block number:

```{literalinclude} ../llmserve/cache/offload.py
:language: python
:start-at: class OffloadTier
:end-before:     def __init__
```

That distinction is the one piece of this that is easy to get wrong. A physical block is a slot the
allocator hands out and reuses immediately; what makes a demoted block findable afterwards is what
*was in it*. {ref}`ch09` already computes exactly such a key for every full block, so the tier can
borrow it and nothing new has to be invented.

One more detail with teeth:

```{literalinclude} ../llmserve/cache/offload.py
:language: python
:start-at:     def store(self, key: int, layers
:end-before:         if key in self._store:
```

The tensors are cloned on the way out. Keeping a view would leave the tier pointing into a block the
allocator is about to hand to another sequence — the quiet kind of corruption that surfaces as one
request reading another's context, with no error anywhere.

## The build

The engine is {ref}`ch09`'s engine. The whole change is what happens at the eviction boundary:

```{literalinclude} ../llmserve/engines/offload.py
:language: python
:start-at:     def _demote
:end-before:     def _make_room
```

and what happens before a lookup:

```{literalinclude} ../llmserve/engines/offload.py
:language: python
:start-at:     def _promote(self, token_ids: list[int]) -> int:
:end-before:         promoted = 0
```

Restoring *before* the cache is consulted, rather than patching what the lookup returns, is what
keeps the change small: {ref}`ch09`'s prefill then finds the blocks resident and proceeds exactly as
it always has. There is no second code path to keep in step, which matters because the two would
drift and only one of them is covered by {ref}`ch09`'s tests.

**An engine that got {ref}`ch08` right grows a memory hierarchy without touching its scheduler.**
That is the structural claim of this chapter, and it is worth noticing how much of it was paid for
three chapters ago.

## The measurement

Back to the table above, and it does not say what I expected it to.

**The mechanism works.** Reuse more than triples — from a cache that is thrashing to one that is
doing its job — and the promotions column is that made visible: several hundred blocks came back
from the tier instead of being recomputed. The block pool is the same size, the trace is the same,
the scheduler is the same. The engine did not get more memory; it got somewhere to put what did not
fit.

**And on this machine it is slower anyway.** Time to first token is worse and throughput is down.
That is a real measurement and it is not a defect in the implementation — it is the implementation
being measured on hardware where its premise is false.

The premise is that the tier is *somewhere else*. On one CPU, "host memory" is the same memory the
engine is already using, reached over the same bus, and the copy is synchronous work on the same
cores that are running the model. Every one of those promotions is a `clone` and a write-back
competing with the forward pass, while the recompute it saved would have been a prefill on the same
cores. There is no second device for the transfer to overlap with, so demotion buys reuse and pays
for it in the one currency this rig has.

On real hardware that copy is a DMA across a link, overlapping with compute, on memory the
accelerator was not using anyway. {ref}`ch15` hit exactly this and said the same thing about its
speed column: the mechanism is correct, the premise does not hold here, and pretending otherwise
would be the easiest lie in the book.

### So: does it pay?

That question is entirely about the link, which makes it answerable without the hardware.

```{include} _generated/ch12-tiers.md
```

```{literalinclude} ../llmserve/cache/offload.py
:language: python
:start-at: def break_even_bandwidth
:end-before:     return bytes_per_token * prefill_tokens_per_second
```

The break-even bandwidth is the number to carry away, and it is remarkably low — comfortably below
every tier in the table, including a network. **Given a link at all, a second tier pays**, which is
why every production engine has grown one, and why the interesting engineering question is not
whether to have one but how many and how deep.

It also explains the measured table. This rig's "link" is a memcpy on the busy cores, which is the
one configuration the arithmetic does not cover: a tier whose transfer cannot overlap with anything,
competing for the resource it is trying to save. Put the tier on the other side of PCIe and the
columns invert.

Notice what cancels in that formula: block size drops out entirely. The answer is a property of the
model and the hardware, not of the allocator's geometry, so it does not move when you retune
{ref}`ch08`.

## The cost

- **On hardware without a real second tier it is a straight loss.** Measured above. A tier whose
  transfers cannot overlap with compute is competing for the resource it exists to save.
- **The tier is a cache with a cache's failure mode.** It can thrash too, one level down, and when
  it does the symptom is a promotion that misses and turns into the prefill you were avoiding. The
  hit rate of the tier is a metric ({ref}`ch27`), not a detail.
- **Cloning on demotion costs bandwidth and a copy** at exactly the moment the engine is under
  memory pressure. Zero-copy demotion is possible and is a considerably harder allocator.
- **Promotion happens on the critical path** of a request that is already waiting. It is cheaper
  than a prefill, which is the whole point, but it is not free and it lands on time to first token.
- **It is another thing that can be stale.** A block promoted from a tier was computed by some
  version of the weights; nothing in this implementation records which. A fleet that offloads to a
  *shared* store across replicas has a cache-invalidation problem the moment two replicas run
  different model versions, and this chapter does not solve it.
- **Host memory is not free either.** It is competing with the page cache, the allocator's own
  pinned buffers, and whatever else shares the box. A tier sized without regard to that turns a
  device-memory problem into a host-memory problem.

## Key takeaways

- Eviction does not have to mean destruction. A block moved down a tier costs a transfer to get
  back; a block dropped costs a prefill.
- **Reuse is restored and, on this rig, latency gets worse anyway** — because the premise of the
  technique is a second device, and there isn't one. The mechanism and its economics are separate
  claims and are measured separately.
- **Key the tier by content, not by block number.** The physical slot is the one thing about a
  demoted block that will not survive.
- Clone on the way out. A view into a block the allocator is about to reuse is silent corruption.
- Restore before the lookup, not after it, so the existing prefill path needs no changes and there
  is only one code path to test.
- **Break-even bandwidth is low enough that every real tier clears it** — host memory, NVMe, even a
  network. The question is how many tiers, not whether.
- The economics depend on the link and not on the engine, so this is a chapter whose decisive table
  is arithmetic and whose measurement is about the mechanism working at all.

## Looking ahead

Part III is done: the engine batches continuously, pages its memory, reuses what it has seen, chunks
its prefills, splits its phases and now spills to a second tier. All of it has taken the model's
arithmetic as given.

Part IV stops doing that. {ref}`ch13` starts with attention itself — how it is computed, how much
KV it produces, and how both of those can be made cheaper before the scheduler ever sees them.

## Further reading

The paging analogy is not a metaphor and the operating-systems literature on page replacement
transfers directly — including the parts about thrashing, which is the failure mode here too. On the
LLM side, the systems built around this idea (LMCache, Mooncake and vLLM's own CPU offloading) are
worth reading for how they handle the two things this chapter does not: sharing a tier across
replicas, and invalidating it when the weights change.
