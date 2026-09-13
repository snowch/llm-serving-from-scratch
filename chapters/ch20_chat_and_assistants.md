---
title: "Chat and Assistants"
short_title: "ch20 Chat and Assistants"
---

(ch20)=
# ch20 · Chat and Assistants

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch09](#ch09), [ch10](#ch10) |
| **Scorecard** | The same engine, tuned for this workload. Everything here is configuration, not code. |
:::

## Part VI starts with a table

Four workloads. One engine, one configuration, one arrival rate, one machine. Nothing differs but
the shape of the traffic:

```{include} _generated/ch20-workloads.md
```

This table is the argument for the whole of Part VI, and the row to look at first is the agent one.
It has the **longest prompts in the table** and the **best time to first token**. Retrieval, with
prompts of almost exactly the same length, is two orders of magnitude worse.

So prompt length does not predict cost. **Reusable prompt length does.** The agent workload replays
a transcript, so nearly every token has been seen before and prefill barely happens; retrieval
assembles a fresh set of passages each time, so almost every token is new and prefill is the entire
story. Same engine, same settings, same prompt sizes, opposite outcomes.

The rest of Part VI is four chapters working out what follows from each row. This one takes the
first.

## What makes chat chat

A chat client resends the whole conversation every turn. The prompt is therefore *monotonically
growing*, and all of the growth except the newest message is text the engine has already processed:

```{literalinclude} ../bench/traces.py
:language: python
:start-at: def make_session_turns
:end-before:     rng = np.random.default_rng(seed)
```

That makes it the best case in the book for {ref}`ch09`, and the effect compounds with depth:

```{include} _generated/ch20-turns.md
```

The prompt grows by most of its own length over six turns. The cost per request barely moves. Reuse
climbs because the *shared* portion grows while the new portion stays the size of one message, so
each turn is a slightly better cache hit than the last.

**A long conversation is cheaper than it looks, and a fresh one is more expensive than it looks.**
That is the opposite of the intuition that longer prompts cost more, and it is worth stating to
anyone sizing capacity from average prompt length: turn one is the expensive one.

It also means the cache is doing most of the work in a chat deployment, which makes
{ref}`ch18`'s session affinity worth more here than anywhere else — routing a returning user to a
different replica throws away exactly the reuse this table is measuring.

## The dial that matters: batch size

Every chapter up to here has treated a larger batch as straightforwardly better. Chat is where that
stops being true, because a token that arrives late is *visible*: a human is reading the stream, and
inter-token latency is how fast the text appears.

```{include} _generated/ch20-batch.md
```

Read the ITL column and the SLO column together, because either one alone gives the wrong answer.

**The ITL-optimal setting fails.** Serving one sequence at a time gives the best inter-token latency
in the table by a clear margin — and meets the objective for well under half of requests. Nothing is
wrong with its ITL; the requests that fail are queued behind the ones being served, and their time
to first token is hopeless. Optimising the metric a user notices, in isolation, produced a
configuration most users would hate.

**Throughput saturates long before the tail does.** Past a batch of a few, throughput gains are
small and the ITL cost keeps accruing. The right setting for chat is the smallest batch that
saturates throughput, not the largest the memory allows — and that is a genuinely different answer
from the one {ref}`ch23`'s offline workload gets from the same table.

**The gap between first and last row is not large.** On this model, at this scale, the whole ITL
range is a few milliseconds. On a production model it is tens of milliseconds and the argument is
the same, which is why the shape of the curve matters more than the values.

## What else chat changes

- **Streaming is the API contract**, not an optimisation. {ref}`ch24` covers the mechanics; what
  matters here is that a client which cannot see tokens until the request completes gets no benefit
  from anything in this chapter.
- **Session affinity is worth more than load balance** ({ref}`ch18`), for exactly the reuse reason
  above.
- **Cache eviction policy becomes user-visible.** Evicting a conversation's prefix means the next
  turn of that conversation is a full prefill — the user perceives a specific, repeatable pause
  after a gap in the conversation. LRU is the right default here precisely because recency predicts
  the next turn.

## The cost

- **Tuning for ITL costs throughput**, and the table above prices it. A chat deployment deliberately
  runs at lower utilisation than a batch one, and that shows up in {ref}`ch27`'s cost model as a
  higher cost per token. It is the right trade and it is not free.
- **Growing prompts mean growing KV footprint.** Reuse makes each turn cheap in *compute* and does
  nothing about memory: a long conversation still occupies blocks proportional to its length. A
  deployment with many long-lived conversations runs out of blocks before it runs out of compute.
- **The cache becomes load-bearing.** With three quarters of prompt tokens served from cache, a
  cold cache after a deploy is not a small regression — it is the workload running at
  {ref}`ch08`'s numbers instead of {ref}`ch09`'s until it warms.
- **Session affinity is a scheduling constraint**, and it fights every other routing goal. The
  imbalance guard from {ref}`ch18` is doing real work here.

## Key takeaways

- **Prompt length does not predict cost; reusable prompt length does.** The framing table's agent
  row has the longest prompts and the best latency.
- A conversation's prompt grows monotonically and almost all the growth is already cached, so later
  turns are cheaper than earlier ones. Turn one is the expensive one.
- Inter-token latency is user-visible, so batch size stops being a free dial. But the ITL-optimal
  batch fails the objective outright, because it destroys time to first token.
- Pick the smallest batch that saturates throughput. {ref}`ch23` picks the largest that fits, from
  the same table, for a workload with no human in it.
- Chat leans on the prefix cache harder than any other workload except agents, which makes session
  affinity and cache warmth operational concerns rather than optimisations.

## Looking ahead

The framing table's worst row is retrieval, and it is worst by two orders of magnitude. {ref}`ch21`
takes it apart: why a workload with the same prompt length as agents behaves so differently, why the
prefix cache cannot save it, and what can.

## Further reading

SGLang's RadixAttention work (Appendix E) is the most relevant here: a radix tree handles branching
conversations — regenerations, edits, multiple replies to one turn — that this book's flat block cache
handles poorly, and branching is common in real assistant traffic.
