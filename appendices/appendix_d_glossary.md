---
title: "Glossary"
short_title: "Appendix D"
---

(appendix-d)=
# Appendix D · Glossary

Terms as this book uses them, with the chapter that earns each one. Where the industry uses a word
loosely, the entry says so — several of the most common terms mean different things in different
papers, and that ambiguity causes real arguments.

## The two phases

**Prefill** — processing the prompt. All tokens are known, so they are computed in parallel; the
phase is compute-bound. One forward pass over the whole prompt. ({ref}`ch03`)

**Decode** — generating output, one token at a time. Each token depends on the last, so there is no
parallelism within a sequence; the phase is memory-bandwidth-bound, because it reads the whole
weight matrix to produce one token. ({ref}`ch03`)

**Compute-bound / memory-bound** — which resource sets the ceiling. Almost every surprising result
in serving comes from these two phases having opposite bottlenecks, and from optimisations that help
one hurting the other. ({ref}`ch03`)

**Arithmetic intensity** — FLOPs performed per byte read from memory. Prefill has high intensity,
decode has very low intensity, and batching is mostly a way to raise decode's. ({ref}`ch03`)

## Latency and throughput

**TTFT (time to first token)** — from request arrival to the first token reaching the caller.
Dominated by queueing plus prefill. The number a user perceives as "did it respond". ({ref}`ch02`)

**ITL (inter-token latency)** — the gap between successive output tokens. What a streaming reader
perceives as speed. Also called TPOT (time per output token); some sources define TPOT as the *mean*
over a request and ITL as the per-gap distribution, and this book reports percentiles of the
distribution. ({ref}`ch02`)

**E2E latency** — arrival to completion. TTFT plus the sum of every ITL. The number that matters for
an agent step, where nothing is streamed to a human. ({ref}`ch22`)

**Throughput** — tokens per second across all requests. A server-side measure; no individual user
experiences it.

**Goodput** — requests per second that met a stated service objective. The only headline number in
this book, because throughput and latency can both look fine while the service is failing. Quoting
goodput without quoting the SLO it is measured against is meaningless. ({ref}`ch02`)

**SLO (service level objective)** — the latency thresholds a request must meet to count. Here, a
TTFT bound and an ITL bound. ({ref}`ch02`)

**Coordinated omission** — the measurement error where a load generator waits for the server, so its
own offered load falls when the server slows down. The queue never grows and the tail disappears.
The reason this book's harness is open-loop. ({ref}`ch02`, {ref}`ch28`)

**Open loop / closed loop** — an open-loop generator submits on a schedule fixed before the run; a
closed-loop one keeps N requests in flight and submits when one finishes. Closed-loop measures a
concurrency level, not an arrival rate, and reports far better tails for the same work.
({ref}`ch28`)

## Batching and scheduling

**Static batching** — collect N requests, run them together to completion. Every sequence occupies a
slot until the longest one finishes, so a third of the slots do nothing. ({ref}`ch06`)

**Continuous batching / iteration-level scheduling** — decide what is in the batch at every step
rather than every request. A finished sequence leaves immediately and a waiting one takes its slot.
The single largest improvement in this book. ({ref}`ch07`)

**Step** — one iteration of the engine loop: sweep, admit, prefill, decode. The unit of scheduling.
({ref}`ch07`)

**Admission** — deciding which queued request enters the running batch. Bounded by batch size and by
free KV blocks. ({ref}`ch08`)

**Preemption** — evicting a running sequence to free its blocks, then re-admitting it later.
Recompute (throw the cache away) or swap (move it to host memory); this book recomputes.
({ref}`ch08`)

**Chunked prefill** — splitting a long prompt's prefill across several steps under a per-step token
budget, so one large prompt cannot stall everyone else's decoding. ({ref}`ch10`)

**Token budget** — the per-step cap on tokens processed, which is the dial chunked prefill turns.
({ref}`ch10`)

**Disaggregation** — running prefill and decode on separate pools of hardware, transferring the KV
cache between them. Isolates the two phases' interference at the cost of a real network payload.
({ref}`ch11`)

## Memory

**KV cache** — stored keys and values for every token processed so far, so each decode step attends
over history without recomputing it. The reason decode is fast and the reason memory runs out.
({ref}`ch05`)

**KV bytes per token** — `2 × n_layers × n_kv_heads × head_dim × bytes_per_element`. The most useful
formula in the book; it decides concurrency. ({ref}`ch03`)

**Paged attention** — storing the KV cache in fixed-size blocks with a per-sequence block table,
instead of one contiguous region per sequence. Removes the fragmentation that forces reserving for
the maximum length. ({ref}`ch08`)

**Block / block table** — the fixed-size unit of KV allocation, and the per-sequence list of which
physical blocks hold its tokens. ({ref}`ch08`)

**Prefix caching** — reusing KV blocks for prompt prefixes already computed, found by hashing block
contents. Nearly free throughput on workloads with shared system prompts. ({ref}`ch09`)

**Reference counting** — how shared prefix blocks know when they can be freed. ({ref}`ch08`,
{ref}`ch09`)

## Attention

**MHA / GQA / MQA** — multi-head attention gives every query head its own KV head; grouped-query
attention shares one KV head across a group; multi-query attention shares one across all. The KV
cache shrinks proportionally, which is usually the point. ({ref}`ch12`)

**Online softmax** — computing softmax in one pass with a running maximum and a correction factor,
rather than materialising all scores first. The mathematical core of FlashAttention. ({ref}`ch12`)

**FlashAttention** — an IO-aware attention implementation that tiles the computation to keep
intermediates in SRAM. It performs slightly *more* arithmetic than the naive version and is much
faster, because it avoids writing the score matrix to HBM. ({ref}`ch12`)

## Making it smaller and faster

**Quantisation** — storing weights or the KV cache in fewer bits. Usually a memory win rather than an
arithmetic one, since the values are often widened again before the multiply. ({ref}`ch14`)

**Per-tensor / per-channel / grouped** — the granularity of the scale factor. Finer granularity
costs more metadata and handles outlier channels far better; the difference is large enough to be
the whole result. ({ref}`ch14`)

**Speculative decoding** — a cheap drafter proposes several tokens and the target model verifies
them in one pass. With the right acceptance rule it samples from exactly the target's distribution.
({ref}`ch15`)

**Acceptance rate** — the fraction of drafted tokens the target accepts. Decides whether speculation
pays for itself. ({ref}`ch15`)

**Residual correction** — on rejection, sampling from `max(0, p_target − p_draft)` renormalised. The
step that makes speculative decoding exact rather than approximate, and the one implementations get
wrong. ({ref}`ch15`)

**Constrained decoding** — masking tokens that would make the output ungrammatical, so invalid
output is unreachable rather than unlikely. Constrains shape, not length. ({ref}`ch16`)

**LoRA adapter** — a fine-tune expressed as a low-rank update to selected projections, so a tenant's
weights are megabytes rather than gigabytes. Serving cost scales with how many *distinct* adapters
appear in one batch. ({ref}`ch19`)

**Rank / alpha** — an adapter's capacity, and the scaling that keeps its effective magnitude
independent of that capacity. ({ref}`ch19`)

## Scaling out

**Tensor parallelism** — splitting each layer's matrices across devices, with a collective per layer.
Bandwidth-hungry; wants NVLink. ({ref}`ch17`)

**Pipeline parallelism** — splitting layers across devices, with activations flowing between them.
Cheaper on bandwidth, introduces bubbles. ({ref}`ch17`)

**Cache-aware / prefix-affinity routing** — sending requests that share a prompt prefix to the same
replica, so its prefix cache is warm for them. Deliberately unbalances load, and wins anyway.
({ref}`ch18`)

**Least outstanding tokens** — routing by queued *work* rather than queued requests. The right
generic policy, and on cache-heavy traffic still much worse than prefix affinity. ({ref}`ch18`)

**Cold start** — the time from starting a replica to serving a request. Dominated by weight
transfer, which makes quantisation an availability feature. ({ref}`ch18`)

**Deficit round robin** — the fair-queueing scheme this book uses to share an engine between tenants.
From packet scheduling, where the problem has the same shape. ({ref}`ch19`)

**Noisy neighbour** — a tenant whose load degrades others sharing the same engine. Fair admission
fixes the queue; it does not fix the shared KV cache. ({ref}`ch19`)

## Operating it

**Load shedding** — refusing requests quickly when the engine is already beyond what it can serve
inside the SLO. Feels worse, measures better. ({ref}`ch26`)

**Draining** — stopping new admissions while finishing in-flight work, so a deploy does not cut live
streams. ({ref}`ch26`)

**Backpressure** — signalling upstream that the server is saturated, rather than silently queueing.

**Leading / lagging indicator** — queue depth rises before latency does, so it is the signal to
autoscale and alert on. Utilisation is neither: it does not track the constraint at all.
({ref}`ch25`)

**Chat template** — the exact rendering of messages into tokens. Changing it changes the model's
behaviour *and* the prefix cache's hit rate, and a per-request value inside it destroys the latter
silently. ({ref}`ch24`)
