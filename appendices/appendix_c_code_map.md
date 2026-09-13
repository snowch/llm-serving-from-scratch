---
title: "Code Map of the Reference Engine"
short_title: "Appendix C"
---

(appendix-c)=
# Appendix C · Code Map of the Reference Engine

Where everything lives, and which chapter put it there. The repository is small on purpose: if a
module is hard to find, that is a bug in the layout rather than something to document around.

## `llmserve/` — the engine

### The things every chapter touches

| Module | Chapter | What it is |
|---|---|---|
| `config.py` | {ref}`ch01` | `ModelConfig`, `REFERENCE_MODEL`, dtype sizes. The single description of the model everything else derives from. |
| `model.py` | {ref}`ch01` | `TinyGPT`: RoPE, grouped-query attention, RMSNorm, SwiGLU. Built from seeded random weights so it runs anywhere. |
| `tokenizer.py` | {ref}`ch04` | `ByteTokenizer` and `IncrementalDetokenizer`. Byte-level, which simplifies {ref}`ch16` enormously and is called out there. |
| `sampling.py` | {ref}`ch04` | `SamplingParams`, repetition penalty, top-k/top-p/min-p, the sampler. |
| `request.py` | {ref}`ch01` | `Request` (what a caller submits), `RequestState` (the engine's bookkeeping), `StepOutput`. The split is what keeps the public interface stable while the engine is rebuilt underneath it. |
| `arithmetic.py` | {ref}`ch03` | The predictions: KV bytes per token, decode ceiling, prefill FLOPs. Takes a `ModelConfig`, so a prediction cannot drift from the model it describes. |

### The engines, in the order they were built

Each is a chapter. Each keeps the same interface, so {ref}`ch02`'s harness measures all of them
without modification.

| Module | Chapter | What changed |
|---|---|---|
| `engines/base.py` | {ref}`ch01` | The `Engine` protocol: `add_request`, `has_work`, `step`. Three methods, and nothing else has ever been needed. |
| `engines/naive.py` | {ref}`ch01`, {ref}`ch05` | `NaiveEngine` (recomputes everything), then `CachedEngine` (keeps a KV cache). |
| `engines/batched.py` | {ref}`ch06`, {ref}`ch07` | `StaticBatchEngine`, then `ContinuousBatchEngine` — the largest single jump in the book. |
| `engines/paged.py` | {ref}`ch08` | Block tables, the allocator, preemption. The admission loop lives here and every later engine inherits it. |
| `engines/prefix.py` | {ref}`ch09` | Content-addressed reuse of prompt prefixes. |
| `engines/chunked.py` | {ref}`ch10` | Per-step token budgets, so a long prefill cannot monopolise a step. |
| `engines/disaggregated.py` | {ref}`ch11` | Separate prefill and decode pools with a KV handoff. |
| `engines/tenant.py` | {ref}`ch19` | Weighted fair admission across tenants. Overrides one method of `paged.py` and nothing else. |
| `engines/cancel.py` | {ref}`ch22` | Dropping a request the caller abandoned, and telling them so. |
| `engines/shedding.py` | {ref}`ch26` | Admission control and draining. |

### Everything else

| Module | Chapter | What it is |
|---|---|---|
| `cache/blocks.py` | {ref}`ch08` | `BlockAllocator` with reference counting, `PagedKVCache`. |
| `cache/prefix.py` | {ref}`ch09` | The prefix cache: cumulative block hashing, lookup, publish, LRU eviction. |
| `attention.py` | {ref}`ch12` | Standard attention and the online-softmax formulation FlashAttention is built on. |
| `quant.py` | {ref}`ch14` | INT8 per-tensor and per-channel, grouped INT4, quantised KV. |
| `speculative.py` | {ref}`ch15` | Drafters, the acceptance rules, and the residual correction that makes speculation exact. |
| `constrain.py` | {ref}`ch16` | A JSON grammar as an FSM, and logit masking against it. |
| `router.py` | {ref}`ch18` | A fleet behind one interface, with round-robin, least-outstanding-tokens and prefix-affinity policies. |
| `lora.py` | {ref}`ch19` | Low-rank adapters, per-row application in a mixed batch, and merging. |
| `api.py` | {ref}`ch24` | Chat templates, request translation, SSE framing, usage accounting. No web framework, deliberately. |
| `metrics.py` | {ref}`ch25` | Bucketed histograms and per-step engine sampling. |
| `cost.py` | {ref}`ch27` | Cost per million tokens, break-even volume. Pure arithmetic over stated inputs. |

## `step()`, the function the whole book is about

Every engine from {ref}`ch07` onward has the same shape, and it is worth reading once as a whole:

```{literalinclude} ../llmserve/engines/paged.py
:language: python
:start-at:     def step(self) -> list[StepOutput]:
:end-before:     # -- the two shapes of work
```

Four things happen, in this order, and each is a chapter:

1. **Sweep finished sequences.** They free their blocks ({ref}`ch08`).
2. **Admit what fits.** Bounded by batch size *and* by free blocks, with a reservation counter so
   the scheduler cannot promise memory it does not have ({ref}`ch08`). Which request is considered
   next is one overridable method, and that is the whole of {ref}`ch19`'s fairness change.
3. **Prefill the admitted ones**, reusing any cached prefix ({ref}`ch09`) and respecting a token
   budget ({ref}`ch10`).
4. **Decode everyone else**, one token each, in one batched pass ({ref}`ch07`).

That interleaving of prefill and decode inside a single step is *iteration-level scheduling*, and it
is the idea the rest of the engine is organised around.

## `bench/` — how the engine is measured

| Module | What it is |
|---|---|
| `harness.py` | `run_benchmark`, the open-loop generator ({ref}`ch02`), plus the result stamping and the code fingerprint that keeps figures honest. |
| `closed_loop.py` | The wrong generator, implemented faithfully so {ref}`ch28` can show what it hides. |
| `traces.py` | Every workload shape: chat, multi-tenant, retrieval, agent, completion, offline batch, noisy neighbour. |
| `train_tiny.py` | Trains the reference model on a synthetic corpus, for the chapters where quality has to be measurable. |
| `train_lora.py` | Per-tenant adapters, for the same reason ({ref}`ch19`). |
| `scorecard.py` | Every table in the book, rendered from committed JSON. |
| `scorecards.py` | Which fragment contains which results — one declaration per table, in one place. |
| `run_*.py` | One runner per chapter's measurements. Each writes stamped JSON into `bench/results/`. |

### Pointing the harness at something else

`run_benchmark` only needs three methods, so anything implementing `add_request`, `has_work` and
`step` can be measured by it — including a thin client wrapping a real server. That is the intended
path out of this book and into a production engine, and {ref}`ch28` is about doing it fairly.

## `tests/` — what is actually guarded

The suite is not organised by module but by *claim*. Each file corresponds to a chapter's central
assertion, and most tests are named as the sentence they defend.

Three kinds, in rough order of value:

- **Equivalence tests.** An optimisation must not change the output. {ref}`ch05`'s cached engine
  produces the same tokens as {ref}`ch01`'s naive one; {ref}`ch07`'s batching produces the same
  tokens as serving one at a time.
- **Distributional tests.** For {ref}`ch15`, where "the same output" is the wrong standard —
  speculative decoding is only correct if it samples from the *same distribution*, which is a claim
  about many samples rather than one.
- **Property tests.** Invariants that a scalar result would hide: no block is leaked, no padded row
  leaks into another sequence's attention, a trace's declared prompt length matches its tokens.

## Checkpoint tags

Every chapter that changes the engine has a tag, so you can read from anywhere:

```bash
git checkout ch07-continuous-batching   # the engine exactly as it stood at the end of ch07
```

`CHECKPOINTS.md` lists them with what the engine looked like at each one.
