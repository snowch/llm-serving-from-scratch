---
title: "Multi-Replica: Routing, Autoscaling and Cold Starts"
short_title: "ch18 Multi-Replica"
---

(ch18)=
# ch18 · Multi-Replica: Routing, Autoscaling and Cold Starts

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch09](#ch09), [ch10](#ch10) |
| **Scorecard** | Routing policy, not replica count, decides the tail. The policy that balances load worst wins. |
:::

## The problem

One engine eventually runs out of device. The standard answer is to run several and put a load
balancer in front of them, and the standard load balancer is round-robin.

Round-robin rests on an assumption that is true of web requests and false of ours: that requests
are interchangeable units of work. {ref}`ch03` already showed they are not — a request's cost
depends on its prompt length, its output length and which phase it is in. {ref}`ch09` added a
second, sharper problem: a replica that has already seen a prompt's prefix can serve it far more
cheaply than one that has not.

Those two facts put round-robin in direct opposition to the cache. Spreading requests evenly is
exactly the way to ensure that every replica has to compute every prefix, and that no replica
benefits from having seen one before. **The load balancer undoes the optimisation.**

## The idea

Three policies, in the order that people reach for them.

**Round-robin** balances request *counts*. Wrong quantity: requests differ in cost by orders of
magnitude, and counting them tells you nothing about how busy a replica is.

```{literalinclude} ../llmserve/router.py
:language: python
:start-at: class RoundRobin
:end-before: class LeastOutstandingTokens
```

**Least outstanding tokens** is the fix people reach for next, and it is a real improvement over
the least-*connections* policy a generic balancer offers. A connection is not a unit of work; a
token nearly is. Summing prompt length plus remaining output budget over everything a replica has
in flight gives a usable estimate of queued work:

```{literalinclude} ../llmserve/router.py
:language: python
:start-at: def _outstanding_tokens
:end-before: class Router
```

**Prefix affinity** stops trying to balance. Hash a fixed-length prompt prefix and send every
request with that prefix to the same replica, so that replica's prefix cache is warm for it and
the others never pay for it at all:

```{literalinclude} ../llmserve/router.py
:language: python
:start-at:     name = "prefix-affinity"
:end-before: def _outstanding_tokens
```

This is a deliberate bet: cache hits are worth more than an even queue. It is also the policy with
a failure mode. If one prefix dominates traffic, every request lands on one replica and the rest
idle — so the policy carries a guard that falls back to the least-loaded replica once the preferred
one is too far above the mean. Every production cache-aware router has some version of that guard,
and it is the part people leave out.

### What the router is, structurally

```{literalinclude} ../llmserve/router.py
:language: python
:start-at: class Router:
:end-before:     @property
```

The router implements the same `Engine` interface as everything else in this book, so {ref}`ch02`'s
harness measures a fleet with no changes at all. That is worth more than it sounds: the fleet is
measured by the same code, against the same SLO, as the single engine in {ref}`ch01`.

:::{warning} What this rig cannot tell you
Our four replicas share one CPU. A real fleet has four devices, so a fleet's *aggregate throughput*
is a property of the hardware, not of the software, and measuring it here would measure the test
rig. So this chapter never compares a fleet against a single engine. It compares policies against
each other on identical hardware and an identical trace, which is the comparison the chapter is
actually about — and the one that survives being run on real hardware.
:::

## The build

Routing only matters when requests actually share prefixes, and share *more than one* of them. With
a single shared system prompt every replica warms its own copy within the first few requests and
every policy looks identical. The trace therefore carries several tenants, each with its own system
prompt:

```{literalinclude} ../bench/traces.py
:language: python
:start-at: def make_multi_tenant_trace
:end-before:     tenants = tenants or TENANT_PROMPTS
```

Six system prompts, four replicas, Poisson arrivals, and each replica is a {ref}`ch09` prefix-caching
engine. The only thing that changes between runs is the policy.

## The measurement

```{include} _generated/ch18-routing.md
```

Read the last two columns first, because they explain the first three.

Round-robin scatters: each of the six prefixes is spread across all four replicas, so every replica
ends up computing every prefix and reuse is low. Least-outstanding-tokens improves on it a little —
its tie-breaking happens to cluster requests, so reuse rises — and the improvement is a fraction of
what the next row gets. Both policies end up perfectly balanced and both serve **no** requests inside
the objective. **Balancing better was not the answer.** That is the result worth sitting with: almost
the whole gain in this chapter comes from the policy that stops balancing.

Prefix affinity is a different regime. Reuse more than doubles against round-robin, median TTFT
falls to a fraction of its value, and goodput goes from nothing to real traffic served inside the
objective. Nothing about the hardware changed. The same four replicas, the same trace, the same
engine; only the choice of where each request went.

And it paid for that with the imbalance column. The busiest replica carries meaningfully more than
an even share, which is not a flaw in the policy — it is arithmetic. Six prefixes hashed onto four
replicas cannot be even; some replica holds two. The policy trades exactly that unevenness for the
cache hits, and the trade is overwhelmingly worth it here.

Two honest caveats about the reuse figure. It is well short of 100% because each replica must still
compute each prefix it owns once, cold, and because the user's actual question is unique to the
request and never reusable. And this is a four-replica fleet on one CPU: a policy that concentrates
work concentrates it onto the same processor everything else is using, so the absolute throughput
figures understate what prefix affinity buys on separate devices, where the idle replicas would
genuinely be idle.

## Autoscaling, and why the obvious signal is wrong

Scaling on CPU utilisation is the default in most orchestrators and is close to useless here.
{ref}`ch03` explains why: decode is memory-bandwidth-bound, so a replica can be fully saturated —
unable to accept another sequence without hurting everyone already on it — while its compute
utilisation looks unremarkable. Utilisation is not the constraint, so it is not the signal.

The two signals that are the constraint:

- **Queue depth**, or its derivative. Requests waiting to be admitted is the most direct statement
  of "we do not have enough replicas". It is also *leading*: the queue grows before latency does.
- **KV-cache utilisation**, from {ref}`ch08`. When the block allocator is near full, the scheduler
  starts preempting, and preemption is the mechanism by which one replica's tail latency falls off a
  cliff. A replica at high block occupancy is out of capacity whatever its CPU says.

Both are already instrumented in this engine — {ref}`ch08` records peak KV utilisation and
preemption counts precisely because they are the operational signals, not just chapter material.

## Cold starts

Autoscaling has an upper bound on how useful it can be, and the bound is how long a new replica
takes to become useful. Almost all of that is the weights:

```{include} _generated/ch18-cold-start.md
```

Those are floors — pure transfer time, at the stated bandwidth, with nothing else happening. On top
sits process start, framework import, allocator warmup, and on a GPU the CUDA-graph capture that
makes decode fast in the first place. The first request after a deploy is always terrible, and this
table is most of the reason.

Three consequences follow directly, and all three are arithmetic rather than opinion:

1. **A reactive autoscaler is late by a cold start.** If load spikes and a replica takes a minute to
   arrive, you served a minute of overload. Scaling on a leading indicator buys back only some of
   that; the rest has to come from headroom you are already paying for.
2. **Scale-to-zero and latency SLOs are close to incompatible** for large models. The table is the
   argument: there is no way to hide tens of seconds inside a request.
3. **Quantisation ({ref}`ch14`) is an availability feature.** The INT4 rows are roughly a quarter
   the fp16 rows, which is a quarter of the time to recover from losing a replica. That is a
   different reason to quantise than the memory saving, and often a better one.

The engineering answers are all the same shape: make the bytes smaller, move them from closer, or
have them already there. Local NVMe cache over object store, `mmap`-ed safetensors so pages load on
demand rather than in one blocking read, a warm pool that absorbs the spike while a cold replica
starts, and — for planned changes — starting the new replica before draining the old one.

## The cost

- **A router is a new failure domain.** Everything now depends on a component that did not exist
  before, and one that has to hold state (which prefix is where, how loaded each replica is) to do
  its job well. A stateless round-robin proxy is genuinely more robust; it is just worse.
- **Prefix affinity can hot-spot.** The measurement shows it mildly. A production trace with one
  dominant tenant shows it severely, and the guard that prevents it is another tunable with its own
  failure mode: set it too tight and you are back to round-robin, too loose and one replica melts.
- **Routing decisions need request content.** A balancer that hashes prompt prefixes has to see the
  prompt, which means it cannot be a dumb L4 load balancer and it is now handling user data.
- **Replica state makes autoscaling harder, not easier.** Removing a replica throws away a warm
  prefix cache, so scaling down has a cost that does not appear on any dashboard: the next requests
  for those prefixes are cold again, on a different replica.
- **Per-replica metrics stop being enough.** The interesting quantities are now fleet-wide —
  reuse across replicas, imbalance, per-tenant tail latency — and none of them are visible from
  inside a single engine. {ref}`ch25` picks this up.

## Key takeaways

- Round-robin is the wrong LLM load balancer. It balances request counts, which correlate with
  neither cost nor cache locality, and it actively destroys the prefix cache.
- Least-outstanding-*tokens* is the right generic policy — a token is nearly a unit of work, a
  connection is not — but on this trace it bought a fraction of what abandoning balance did.
  Balancing better is not the win.
- **Cache-aware routing is the win, and it works by refusing to balance.** Concentrating a prefix
  on one replica beats spreading it, by enough to turn zero goodput into real goodput.
- Cache-aware routing must carry an imbalance guard, or one hot prefix pins the whole fleet to one
  replica.
- Autoscale on queue depth and KV utilisation. CPU utilisation is not the constraint, so it is not
  the signal.
- Cold start is dominated by weight transfer, which makes quantisation an availability feature and
  makes scale-to-zero incompatible with a tight TTFT objective.

## Looking ahead

This chapter routed by prefix and treated every request as equally entitled to service. {ref}`ch19`
removes that assumption: many tenants sharing one base model, with per-tenant adapters and
per-tenant fairness, where "which replica" becomes "which adapter, and whose turn".

## Further reading

The cache-aware routing idea is most clearly stated in SGLang's RadixAttention work (Appendix E),
which pairs a prefix-tree cache with a router that knows about it — the two halves of {ref}`ch09` and
this chapter. Production routers worth reading for their policy choices include vLLM's production
stack and AIBrix. For the autoscaling side, the useful literature is mostly about serverless cold
starts rather than about LLMs, and it transfers better than you would expect, because the problem
is the same one: the unit of scaling carries too much state to start quickly.
