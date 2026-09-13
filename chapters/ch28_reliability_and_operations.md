---
title: "Reliability and Operations"
short_title: "ch28 Reliability and Operations"
---

(ch28)=
# ch28 · Reliability and Operations

:::{note} Chapter header
:class: dropdown

| | |
|---|---|
| **Tier** | Tier 1 — CPU (any laptop) |
| **Prerequisites** | [ch02](#ch02), [ch08](#ch08), [ch27](#ch27) |
| **Scorecard** | Refusing a quarter of the requests doubles goodput and cuts the tail fourfold. |
:::

## The problem

An overloaded engine that accepts everything serves **nothing** well. The queue grows without bound,
every request misses its objective, and the only signal the caller gets is that the entire service
became slow at once — which is indistinguishable, from outside, from the service being down.

{ref}`ch02` has been measuring this since the beginning. Goodput collapses under overload while
throughput looks fine, because the engine is producing tokens for requests nobody will wait for.

## Refusing work, on purpose

The alternative feels worse and measures better. Refuse some requests quickly, so the rest are served
properly:

```{include} _generated/ch28-shedding.md
```

Refusing half the offered requests **doubles goodput and cuts the tail latency to a quarter of its
value**. Throughput is essentially unchanged, which is the point: the engine was always producing
tokens at the same rate, and shedding changed how many of them belonged to a request that still had
a future.

This is the most reliable single intervention in Part VII, and it is also the one people resist
hardest, because the failure it prevents is diffuse and the cost it imposes is specific. Nobody files
a ticket saying "every request was 20% slower"; the rejected customer files immediately.

## When to refuse

Not when the CPU is busy — {ref}`ch27` covered why that number does not track the constraint. The two
signals that do:

```{literalinclude} ../llmserve/engines/shedding.py
:language: python
:start-at:     def should_admit
:end-before:         if self.draining:
```

Queue depth is the direct statement that there is already more work here than can be served soon. KV
utilisation catches what a queue length cannot: a few very long sequences can exhaust the block budget
while the queue looks short, and admitting into that produces {ref}`ch08`'s preemption thrash rather
than an honest refusal.

**A rejection must reach the caller.** The implementation reuses {ref}`ch24`'s cancellation path for
exactly this reason:

```{literalinclude} ../llmserve/engines/shedding.py
:language: python
:start-at:     def add_request
:end-before:         if not self.should_admit():
```

A server that refuses work by dropping connections looks, from the client's side, precisely like a
server that has hung — and the client's retry then makes the overload worse. That is the mechanism by
which a load spike becomes an outage, and an explicit, fast rejection with a retry-after is what
breaks it.

## Draining

The other half of operations is changing the software without dropping what is in flight:

```{literalinclude} ../llmserve/engines/shedding.py
:language: python
:start-at:     def drain
:end-before:         self.draining = True
```

```{include} _generated/ch28-drain.md
```

Those tokens are the cost of getting this wrong. Terminating a process with sequences still decoding
drops every one of those streams, and the client sees a **truncated response rather than an error** —
which is worse, because nothing retries it and nothing alerts. A user receives half an answer and
believes the model produced it.

The drain is short because generation is short. That is the useful shape: the wait to drain scales
with `max_tokens`, not with traffic, so it is bounded and knowable. A rolling upgrade should start
the new replica, wait for it to be ready, drain the old one, and only then stop it — and the drain
step is the one that gets skipped.

## The failure modes worth rehearsing

Four, in rough order of how often they happen.

**Memory exhaustion under a length spike.** Not an error condition — a scheduling event. {ref}`ch08`
preempts, which recovers, but preemption thrash looks like a hang. The signals are KV utilisation and
preemption rate, and the response is to shed rather than to restart.

**NaN or inf in the output.** {ref}`ch06` and {ref}`ch18` both found the same cause: masking with
negative infinity, a row that is masked everywhere, a softmax that produces NaN, and corruption that
spreads across the whole batch because one padded row poisoned the arithmetic. Use a finite floor.
The reason this is in a reliability chapter is that it presents as *other requests* returning
nonsense, which is the hardest kind of bug to attribute.

**A device fault.** The only honest answer is to fail the affected requests fast and take the replica
out, because a partially-working accelerator produces wrong numbers rather than errors.

**Silent quality regression after a config change.** The worst one, and the reason it is worst is that
nothing in {ref}`ch27`'s dashboard moves. A quantisation setting, a sampling default, a chat template
({ref}`ch26`) — all change what the model says while leaving every latency and throughput signal
untouched. The defence is not monitoring; it is the equivalence and distribution tests from
{ref}`ch17`, run in CI, on every change that touches the numerical path.

## A runbook, in six lines

1. **Tail latency rising, queue growing** → overload. Shed, then add capacity. Check the arrival rate
   before assuming a regression.
2. **Tail latency rising, queue flat** → look at preemption rate and KV utilisation. Something is
   holding memory; a length spike or a leak.
3. **Throughput down, batch size down** → the scheduler is starved. Check admission, check whether
   the prefix cache is holding blocks it should be giving up ({ref}`ch09`).
4. **Cache hit rate dropped** → somebody changed a prompt or a template ({ref}`ch26`), or routing
   changed ({ref}`ch20`). Almost never the cache.
5. **One tenant complaining, aggregate healthy** → {ref}`ch21`. The aggregate is hiding them.
6. **Output looks wrong, metrics look fine** → a config change. Diff the sampling parameters, the
   quantisation settings and the template against the last known-good deploy.

## The cost

- **Shedding rejects real customers**, and the decision of *whose* request to drop is a policy nobody
  wants to own. Per-tenant weights ({ref}`ch21`) at least make it explicit rather than arbitrary.
- **Thresholds need tuning**, and they are workload-dependent: a queue depth that is healthy for
  {ref}`ch25`'s batch traffic is an emergency for {ref}`ch22`'s chat.
- **Draining slows deploys** by the length of the longest generation, which is a real constraint on
  how fast you can roll out a fix — including a fix for the incident you are currently in.
- **Load shedding masks capacity problems.** A fleet that sheds smoothly is a fleet that looks
  healthy while turning away a fifth of its traffic. The rejection rate has to be a first-class
  metric or the shed becomes permanent by accident.
- **Every failure mode above needs its own test**, and the silent one needs tests that check output
  distributions rather than latency — which is a different kind of test suite than most serving teams
  have.

## Key takeaways

- **An overloaded engine that accepts everything serves nothing well.** Refusing half the offered
  requests doubled goodput here and cut the tail fourfold.
- Shed on queue depth and KV utilisation. Never on CPU utilisation.
- **Reject explicitly and fast.** A dropped connection is indistinguishable from a hang, and the
  client's retry is what turns a spike into an outage.
- Drain before stopping. A killed process produces truncated responses, which nothing retries and
  nothing alerts on.
- The worst failure is the silent one: a config change that moves quality while every latency and
  throughput signal stays flat. Only tests catch it.
- Make the rejection rate a first-class metric, or a fleet that sheds well will look healthy while
  turning away traffic indefinitely.

## Looking ahead

{ref}`ch29` asks what all of this costs. The answer depends far less on the hardware than anyone
expects, and far more on a number most deployments never measure.

## Further reading

The load-shedding and graceful-degradation literature from large-scale web serving applies almost
unchanged, and is more mature than anything written about it for LLMs. The one LLM-specific twist is
that a request's cost is unknown at admission time — you do not know how many tokens it will generate —
which makes admission control genuinely harder than it is for requests of predictable size, and is
why the engine here charges an *estimate* ({ref}`ch21`) rather than a measurement.
