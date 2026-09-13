---
title: "Disaggregating Prefill and Decode"
short_title: "ch11 Disaggregation"
---

(ch11)=
# ch11 · Disaggregating Prefill and Decode

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch10](#ch10) |
| **Scorecard** | Roughly a wash here — and the measurement cannot show the benefit. Read the caveat. |
:::

## The problem

{ref}`ch10` spent an entire chapter refereeing between prefill and decode: which gets the budget,
in what order, at whose expense. That work exists only because both jobs run on the same hardware,
in the same loop, competing for the same step.

They are not similar jobs. {ref}`ch03` established that prefill is compute-bound and decode is
memory-bandwidth-bound. We have been buying one machine to do well at two things that want
different machines.

## The idea

Run them in separate pools. Prefill a request in one, ship its keys and values to the other,
decode there. The pools can then be sized independently — and, in a real deployment, provisioned
on different hardware entirely.

The consequences are appealing:

- **No interference.** A long prefill cannot stall a streaming user, because it is not on the same
  device. The whole of {ref}`ch10` becomes unnecessary.
- **Independent scaling.** Prefill-heavy traffic ({ref}`ch21`'s RAG) and decode-heavy traffic can
  be scaled separately instead of by one batch-size dial.
- **Hardware specialisation.** Buy compute for the prefill pool and memory bandwidth for the
  decode pool, rather than compromising on both.

And one cost: **every byte of KV cache has to cross between them.**

```{literalinclude} ../llmserve/engines/disaggregated.py
:language: python
:start-at:     def _handoff
:end-before:         assert state.past is not None
```

## The build

The engine is a prefill pool, a handoff, and a decode pool:

```{literalinclude} ../llmserve/engines/disaggregated.py
:language: python
:start-at:         # -- prefill pool
:end-before:     def _emit
```

Two pool sizes rather than one — `prefill_batch_size` and `max_batch_size` — which is the entire
architectural point, even though a single process cannot exploit it.

## The measurement

:::{warning} What this measurement can and cannot show
In one process the two pools take turns. The concurrency that makes disaggregation worthwhile —
prefill and decode progressing *simultaneously* on separate hardware — is not available, so no
benefit can appear here no matter how good the idea is.

What is measurable is the cost. Treat the numbers below as an honest lower bound on overhead, and
the arithmetic that follows as the real argument.
:::

```{include} _generated/ch11-disaggregation.md
```

Roughly a wash, which is the expected result: the handoff is a memcpy within one process, and at
this model size a request's whole KV cache is well under a megabyte. Nothing here argues for or
against the architecture.

Now the number that decides it. The handoff is not a design detail — it is the design. Here is
what one request's cache costs to move, across realistic interconnects:

```{include} _generated/ch11-handoff.md
```

Read the bottom row. An 8B model serving a 32k-token context has a KV cache of several gigabytes
*per request*. Over ordinary datacentre Ethernet, moving it takes **seconds** — far longer than
the prefill that produced it, and vastly longer than any latency budget. Over NVLink it takes
about ten milliseconds, which is affordable.

That single comparison contains the whole chapter:

> **Disaggregation is an interconnect decision, not a scheduling one.** It is an excellent
> architecture when prefill and decode sit on the same fabric, and unusable when they do not. No
> amount of scheduler cleverness compensates for a slow link.

It also explains why the technique appeared when it did. Disaggregated serving became practical
once high-bandwidth interconnects between accelerators became ordinary; on commodity networking it
would have been obviously absurd, and the papers would never have been written.

## The cost

- **Substantially more moving parts.** Two pools, a transfer path, and failure modes that belong
  to neither — a prefill completing into a decode pool that has no room, a transfer failing
  midway, the pools disagreeing about who owns a sequence.
- **A hard dependency on interconnect bandwidth**, which is a procurement decision, not a
  software one.
- **It is wrong below a certain scale.** With one accelerator there is nothing to disaggregate,
  and the handoff is pure loss. Do not reach for this because it is modern.
- **Prefix caching gets harder.** {ref}`ch09`'s cache lives in the prefill pool, but the blocks it
  saves are consumed in the decode pool. Sharing across a handoff is real work that our
  implementation does not attempt.

## Key takeaways

- Prefill and decode are bound by different resources, so making one machine serve both is a
  compromise on both.
- Separating them removes interference by construction and allows the pools to be sized and
  specialised independently.
- The price is moving the entire KV cache between pools, once per request.
- That cost scales with model size and context length, and at production scale it is measured in
  gigabytes. Interconnect bandwidth, not scheduling, determines whether the architecture works.
- Below the scale where you have a fast fabric and enough traffic to justify two pools,
  disaggregation is pure overhead.
- A measurement that structurally cannot show a benefit is not evidence against it. Say so, and
  make the argument with arithmetic instead.

## Looking ahead

Part III has taken the engine from one request at a time to a scheduler with paged memory, prefix
reuse and an explicit prefill policy. Every remaining inefficiency is now in the mathematics
itself rather than in how work is organised. {ref}`ch12` turns to attention: how to compute it
with far less memory traffic, and how the architectures that shrink the KV cache change the
arithmetic this entire part has been fighting.

## Further reading

DistServe (Zhong et al., Appendix E) is the primary source and introduces the goodput framing this
book has used since {ref}`ch02`; Splitwise covers the same ground with a stronger emphasis on
hardware heterogeneity. Both are worth reading with this chapter's handoff table beside them.
