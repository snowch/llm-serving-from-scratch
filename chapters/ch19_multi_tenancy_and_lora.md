---
title: "Multi-Tenancy and LoRA at Serving Time"
short_title: "ch19 Multi-Tenancy and LoRA"
---

(ch19)=
# ch19 · Multi-Tenancy and LoRA at Serving Time

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch07](#ch07), [ch08](#ch08) |
| **Scorecard** | Many tenants on one base model. Adapter overhead is real and scales with diversity; fairness is a scheduler change, not a bigger machine. |
:::

## The problem

Fifty customers each want the model fine-tuned on their own data. The obvious implementation gives
each of them a copy of the weights, and it fails immediately on arithmetic: fifty copies of an 8B
model in fp16 is eight hundred gigabytes, to serve weights that are about ninety-nine percent
identical to each other.

There is a second problem hiding behind the first, and it survives even if the memory problem is
solved. {ref}`ch18` put several tenants' traffic through one engine and treated the waiting queue
as a single line. A single line is exactly right for one caller and exactly wrong for several: one
tenant submitting a burst is served before everyone who arrived later, and every other customer
waits behind work they did not cause.

## The idea, part one: the weights

A fine-tune is a change to the weights, and empirically it is a *low-rank* change. LoRA expresses
that change as two thin matrices per targeted projection — `W + (alpha/r) · B·A`, with `r` far
smaller than either dimension — and freezes the base entirely. What each tenant owns shrinks from a
full weight matrix to a pair of slivers.

```{literalinclude} ../llmserve/lora.py
:language: python
:start-at: class LoRAConfig
:end-before:     @property
```

The arithmetic is the whole argument:

```{include} _generated/ch19-adapter-memory.md
```

Three orders of magnitude, from one copy per tenant to thousands of tenants per accelerator. And
the rank column is a genuine dial rather than a free lunch: rank buys capacity, and it buys it
linearly.

One implementation detail that is easy to get wrong and expensive to debug:

```{literalinclude} ../llmserve/lora.py
:language: python
:start-at: def make_adapter
:end-before:     config = config or LoRAConfig()
```

`B` starts at zero, so a fresh adapter is *exactly* the identity — not approximately. An adapter
that perturbed the model at initialisation would make the first training steps repair damage they
caused themselves, and would make an untrained adapter indistinguishable from a broken one. When
serving several adapters, that distinction is the difference between a debuggable system and a
mysterious one.

### Does an adapter actually do anything?

A serving chapter can happily measure the cost of applying adapters that do nothing, which is a
good way to build a fast implementation of the wrong thing. So each tenant here gets a corpus of
its own — a different seed of the {ref}`ch14` word process, which yields a different vocabulary
over the same syllables — and a briefly-trained adapter of its own.

```{include} _generated/ch19-quality.md
```

Read the diagonal against the row. Each tenant's own adapter is the best model of that tenant's
text; the other tenant's adapter is worse; the base model is worst of all. Both adapters beat the
base on both corpora, which is what you would expect from any extra training — the claim of
specialisation rests on the *gap between them*, not on the improvement over the base.

## The idea, part two: serving them together

Here is where multi-tenancy stops being a memory question. A batch drawn from several tenants wants
a different `B·A` for each row, and there is no single matrix multiply that does that.

```{literalinclude} ../llmserve/lora.py
:language: python
:start-at:     def forward(self, x: torch.Tensor) -> torch.Tensor:
:end-before: def attach
```

The loop is written plainly on purpose. It groups rows by adapter and takes one pass per distinct
adapter present, which is the naive implementation and exactly the thing S-LoRA and Punica exist to
replace with a batched kernel. Before reaching for one, measure what it costs:

```{include} _generated/ch19-batch.md
```

Four things in one table.

**One adapter across the whole batch is nearly free.** Two skinny matrix multiplies against a rank
of eight is a rounding error next to the projections themselves. If your deployment has one
fine-tune, LoRA costs you essentially nothing at serving time.

**Cost scales with adapter *diversity*, not adapter count.** Eight adapters in the batch means eight
passes over a slice of it. Holding a thousand adapters in memory is free; putting eight of them in
one step is not.

**Merging removes the overhead entirely** — and removes multi-tenancy with it:

```{literalinclude} ../llmserve/lora.py
:language: python
:start-at: def merge
:end-before:     for layer in layers.values():
```

Back to baseline, because after the merge there is no adapter left to apply. It is the right answer
for one tenant and unavailable for many, since the merge destroys the shared base every other
tenant was borrowing.

**The batched-kernel work has a clear target.** The gap between the one-adapter row and the
many-adapter rows is exactly what a grouped GEMM recovers. That is a worthwhile optimisation with a
known ceiling, which is a much better position than optimising on instinct.

## The idea, part three: fairness

Memory solved and adapters applied, the queue is still one line. Chapters 7 through 10 built an
increasingly clever scheduler that is entirely blind to *who* each request belongs to — and the
engine cannot schedule fairly between callers it cannot tell apart, so the request grows a field:

```{literalinclude} ../llmserve/request.py
:language: python
:start-at: class Request:
:end-before:     @property
```

Then admission stops taking the head of the queue and starts taking the head of whichever tenant is
furthest behind its entitled share:

```{literalinclude} ../llmserve/engines/tenant.py
:language: python
:start-at:     def _next_waiting
:end-before:     def _prefill
```

This is deficit round robin, which comes from packet scheduling and is about thirty years old. The
problem has the same shape there: a shared resource, flows of wildly differing size, and no way to
be fair without accounting per flow.

When to charge a tenant is a real decision, and the intuitive answer is the wrong one:

```{literalinclude} ../llmserve/engines/tenant.py
:language: python
:start-at:     def _prefill
:end-before:         for state in states:
```

Charging for work already completed is more accurate and useless: by the time a burst shows up in
the accounting, it has been served. Charging the whole expected cost at admission over-charges
requests that stop early, and that error is in the safe direction.

### The measurement

A trace with two well-behaved tenants and one submitting a burst of larger requests — a backfill
job, a retry storm, one customer's batch export. The same trace, served twice, changing nothing but
the scheduler:

```{include} _generated/ch19-fairness.md
```

Under first-come-first-served the quiet tenants' tail is as bad as the noisy one's. They did nothing
to deserve it; they simply queued behind eighteen large requests. Under fair queueing their tails
collapse and they meet the objective, while the noisy tenant's tail stays roughly where it was.

That last part is the important one, and it is why this is isolation rather than a speedup. **The
total work is unchanged.** The scheduler moved the waiting onto the tenant that caused it. Nobody
got a faster machine; one tenant stopped being able to spend everyone else's latency budget.

## The cost

- **Isolation is best-effort, not a guarantee.** Fair *admission* does not make the KV cache fair.
  A tenant whose requests are individually enormous still occupies blocks that everyone else needs,
  and can still cause the preemptions of {ref}`ch08` for sequences that belong to someone else.
  Genuine isolation needs per-tenant block quotas, which is a considerably larger change.
- **Adapter diversity costs throughput**, per the table above, and diversity is driven by traffic
  rather than by anything you control. A scheduler that batched by adapter would reduce the cost
  and increase queueing delay — a trade this chapter does not make, and a real one.
- **Adapters are one more thing to store, version, load and evict.** They are small enough to hold
  many and not small enough to hold all of them, which means an adapter cache, which means adapter
  cache misses on the critical path of a request. {ref}`ch08`'s allocator is the natural home for
  that and this chapter does not build it.
- **Accounting is a policy surface.** Charging by expected tokens is a choice; charging by prompt
  length, by GPU time or by blocks held are all defensible and all give different answers about who
  is being greedy. Whatever you pick, some tenant is structurally advantaged by it.
- **Tenancy leaks into everything downstream.** Metrics, logs, traces and alerts all now need a
  tenant dimension or they will average a starved customer into a healthy fleet. {ref}`ch25`
  inherits this.
- **A shared base means shared failures.** One base model serving fifty tenants is fifty customers
  behind one bad deploy, and the blast radius of a weight-loading bug is now the whole fleet.

## Key takeaways

- A fine-tune is a low-rank change, so a tenant's weights can be slivers instead of a copy. The
  memory argument is three orders of magnitude and is not close.
- Initialise `B` to zero. A new adapter must be exactly the identity, or an untrained adapter and a
  broken one look the same.
- Serving cost scales with how many *distinct* adapters are in a batch, not how many exist. One
  adapter is nearly free; eight is not.
- Merging an adapter removes all serving overhead and all multi-tenancy. It is the right answer for
  exactly one tenant.
- Fairness needs the engine to know who submitted what. One field on the request and one method on
  the scheduler turns a single queue into per-tenant queues.
- Charge on admission, not completion — a scheduler that waits for completion cannot see a burst
  until after it has served it.
- Fair queueing does not create capacity. It decides who waits, and that is the whole point.

## Looking ahead

Part V is done: the engine now spans devices, replicas and tenants. Everything so far has treated
the workload as a single abstract stream of requests, tuned for the average of it.

Part VI stops doing that. The same engine, tuned four different ways for four real workloads —
chat, retrieval, agents and code completion — because the length distribution of the traffic decides
almost every configuration choice, and no single setting is right for all four. {ref}`ch20` starts
with the one whose shape this book has quietly assumed throughout.

## Further reading

S-LoRA and Punica (Appendix E) are the two papers on serving many adapters concurrently, and both
are largely about replacing the loop in this chapter with a batched kernel. The original LoRA paper
is worth reading for its ablation over which projections to target — the convention of adapting
queries and values is an empirical result, not an arbitrary default. For the scheduling half,
the packet-scheduling literature on deficit round robin and weighted fair queueing transfers almost
unchanged, and is clearer than anything written about it in an LLM context.
