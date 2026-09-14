# LLM Serving from Scratch — Book Plan

**Working title:** LLM Serving from Scratch
**Subtitle:** *Build a production inference engine, one measurement at a time*
**Author:** Chris Snow
**Published at:** `https://snowch.github.io/llm-serving-from-scratch/`
**Linked from:** [snowch.github.io](https://snowch.github.io) — new landing page + `myst.yml` TOC entry (see [§10](#10-linking-from-snowchgithubio))
**Status:** plan — nothing authored yet

---

## 1. What this book is

Most material on LLM inference falls into one of two camps: papers that describe a single
technique in isolation, or documentation that tells you which flags to pass to vLLM. Neither
teaches you how a serving engine actually works, and neither lets you feel *why* each
optimisation exists.

This book builds one. Starting from a ~100-line FastAPI server that wraps
`model.generate()` and serves roughly one request per second, the reader incrementally builds
a real inference engine: KV cache, continuous batching, a paged block manager, prefix caching,
chunked prefill, speculative decoding, quantisation, tensor parallelism, an OpenAI-compatible
API, and the observability needed to run it. Every chapter ends by re-running the same
benchmark, so progress is *measured*, never asserted.

By the end the reader has a working engine of their own, understands the design of vLLM /
SGLang / TensorRT-LLM well enough to read their source, and can size, price and operate a
serving deployment.

### 1.1 Who it is for

| Reader | What they get |
|---|---|
| ML / platform engineer told to "self-host a model" | A defensible architecture, a capacity model, and the vocabulary to evaluate vLLM vs TGI vs a managed API |
| Backend engineer new to GPUs | The two-bound mental model ([§3](#3-the-pedagogical-spine)) that makes every other decision obvious |
| Practitioner who finished a *build-a-GPT* course | The missing half: the model trains, now make it serve 1,000 concurrent users |
| SRE / observability engineer | What to actually put on a dashboard for a serving engine, and what each signal means |

**Assumed background:** comfortable Python, basic PyTorch, knows what a transformer is
(self-attention, layers, logits). Does *not* assume CUDA, distributed systems, or queueing
theory — each is introduced where needed.

### 1.2 What makes it different

1. **It builds the engine, not a tour of flags.** Every core mechanism is implemented in
   readable Python before any library is reached for.
2. **Every claim is measured.** The running scorecard ([§3.2](#32-the-running-scorecard)) is
   re-run at the end of every chapter. No "up to 20× faster" without a reproducible number
   and the conditions that produced it.
3. **Correctness is a first-class concern.** Each optimisation ships with an equivalence test
   against a reference implementation. This is the thing real teams get wrong: it is very easy
   to make a serving stack fast and subtly wrong.
4. **One mental model, applied everywhere.** Prefill is compute-bound, decode is
   memory-bandwidth-bound. Every technique in the book is reduced to "which bound does this
   attack, and what does it cost?"
5. **It runs on a laptop.** The default path is CPU-only with a small model, so no reader is
   blocked by hardware. GPU numbers are provided and reproducible for those who have one.
6. **Operations and economics are in scope.** Observability, load shedding, rolling upgrades,
   and $/million-tokens — not just kernels.

### 1.3 Non-goals

- Not a CUDA book. One optional chapter writes a Triton kernel; otherwise we compose PyTorch
  and call FlashAttention rather than reimplementing it.
- Not about training, fine-tuning, or RLHF — that ground is already covered by the
  *LLM From Scratch* series on the site.
- Not a vLLM competitor. The engine we build is a teaching artefact: clear, tested, and
  deliberately missing features, with the omissions named explicitly.
- Not a framework comparison shoot-out, though Chapter 31 shows how to run one honestly.

---

## 2. Relationship to existing snowch.github.io content

The site already has a three-part *LLM From Scratch* series. The overlap is real and needs to
be handled deliberately, not ignored:

| Existing page | Current form | Relationship to this book |
|---|---|---|
| `ai-eng/llmfs/L10_Inference_and_Sampling.md` | Lesson, ~350 lines | Prerequisite. Book ch04 goes deeper (numerical stability, stop strings, incremental detokenisation) and cites it. |
| `ai-eng/llmfs-scaling/L17_Attention_Optimizations.md` | Lesson, ~420 lines, `[DRAFT]` | Surveys FlashAttention / KV cache / GQA. Book ch05, ch13, ch14 **build** them. |
| `ai-eng/llmfs-scaling/L20_Quantization_Inference.md` | Lesson, ~390 lines, `[DRAFT]` | Shows how to *use* bitsandbytes / AutoGPTQ. Book ch15 adds KV-cache quantisation, FP8, and quality/speed/memory measurement. |
| `ai-eng/llmfs-scaling/L21_Deployment_Serving.md` | Lesson, ~495 lines, `[DRAFT]` | Shows how to *call* vLLM. This book is the long-form version of that lesson — it is the single largest source of overlap. |

**Recommended resolution:** keep the lessons as short, finished *summaries* and let them point
into the book for depth. Concretely:

- Finish L17/L20/L21 as ~200-line overviews (drop the `[DRAFT]` marker) and add a callout at
  the top of each: *"This lesson surveys the technique. To build it yourself, see
  [LLM Serving from Scratch](https://snowch.github.io/llm-serving-from-scratch/)."*
- Do **not** delete them: they carry existing inbound links and rank for search terms the
  book will also want.
- The book's Chapter 1 links *back* to the `llmfs` core series as the prerequisite for readers
  who want to understand the model itself.

This reciprocal linking is also the main discoverability mechanism for the new book
([§10](#10-linking-from-snowchgithubio)).

---

## 3. The pedagogical spine

Two devices hold the book together. They are decided up front because every chapter depends
on them.

### 3.1 The two-bound model

Introduced in Chapter 3 and referenced in every subsequent chapter:

- **Prefill** processes the whole prompt at once. It is **compute-bound**: roughly
  `2 × params` FLOPs per prompt token.
- **Decode** produces one token at a time. It is **memory-bandwidth-bound**: each step reads
  ~all weights plus the KV cache from HBM to do very little arithmetic.

Two consequences the reader should be able to derive by the end of Chapter 3:

- A single-stream decode ceiling of roughly `HBM bandwidth / bytes read per forward pass`
  tokens/second — which explains why a bigger GPU helps decode more than a faster one.
- KV cache per token = `2 × n_layers × n_kv_heads × head_dim × bytes_per_element`, which is
  the budget that every Part III chapter is fighting over.

Every technique in the book is then labelled with the bound it attacks:

| Technique | Attacks | Cost |
|---|---|---|
| Continuous batching (ch07) | Decode bandwidth (amortises weight reads) | Scheduler complexity, TTFT variance |
| Paged attention (ch08) | KV memory fragmentation | Indirection, kernel complexity |
| Prefix caching (ch09) | Redundant prefill compute | Cache memory, eviction policy |
| Chunked prefill (ch10) | Prefill/decode interference | Slightly slower prefill |
| GQA / MQA (ch13) | KV cache size | Model must be trained for it |
| Quantisation (ch15) | Weight bytes read, KV bytes | Quality loss, calibration |
| Speculative decoding (ch17) | Decode serialisation | Extra compute; only wins at low batch |
| Tensor parallelism (ch19) | Weights per GPU, aggregate bandwidth | Collective communication per layer |

### 3.2 The running scorecard

One benchmark harness, built in Chapter 2, re-run at the end of every chapter that changes the
engine. It reports, at fixed request rates against a fixed trace:

- TTFT p50 / p95 / p99
- Inter-token latency (ITL) p50 / p95
- Output tokens/second (aggregate) and requests/second completed
- **Goodput**: requests/second that met a stated SLO — the number that actually matters
- Peak KV-cache utilisation, preemption rate, mean batch size

Each chapter ends with the same table, one new row appended, plus one sentence on *why* the
number moved. The book's final chapter shows the whole table, from Chapter 1's baseline to the
finished engine. This is the book's single strongest structural idea and should be built
before any optimisation chapter is written.

**Discipline required:** benchmark numbers are generated by a committed script, stored as JSON
in the repo, and stamped with hardware, model, dates and library versions. Never hand-typed
into prose. See [§6.3](#63-how-numbers-get-into-the-book).

---

## 4. Outline

Seven parts, 32 chapters, 5 appendices. Chapter lengths target 2,500–5,000 words plus code;
the optional/advanced chapters may run longer.

### Part I — The Serving Problem (ch01–ch03)

| # | Chapter | What the reader does |
|---|---|---|
| 1 | **What an Inference Server Actually Does** | Trace the full request lifecycle: HTTP → chat template → tokenise → prefill → decode loop → detokenise → stream → disconnect. Build the naive server (~100 lines, FastAPI + HF `generate`). Establish the baseline everything is measured against. |
| 2 | **Measuring What Matters** | Define TTFT, ITL/TPOT, e2e latency, throughput, goodput. Why means lie and percentiles don't. Build the load generator — open-loop Poisson arrivals, and why closed-loop harnesses hide overload (coordinated omission). This chapter's output is the tool used for the rest of the book. |
| 3 | **The Arithmetic of Inference** | Derive FLOPs and bytes-moved per token. Roofline and arithmetic intensity. Compute the decode ceiling and the KV-cache budget for a real model, then verify the prediction against the naive server. The conceptual spine ([§3.1](#31-the-two-bound-model)). |

### Part II — The Decode Loop (ch04–ch06)

| # | Chapter | What the reader does |
|---|---|---|
| 4 | **Generating Tokens Correctly** | Implement sampling from scratch: greedy, temperature, top-k, top-p, min-p, repetition/presence penalties — numerically stable, seeded, reproducible. Stop conditions, EOS, `max_tokens`, stop strings. Incremental detokenisation: partial UTF-8 and BPE boundaries, a genuine and under-documented bug source. |
| 5 | **The KV Cache** | Build it. Show the O(n²)→O(n) recompute saving, measure the speedup, then compute the memory cost per token and per concurrent request. Hit the memory wall on purpose — it motivates all of Part III. |
| 6 | **Static Batching and Its Limits** | Batch multiple requests: padding, attention masks, position IDs, left vs right padding. Measure the throughput win, then measure the waste: the ragged-completion problem, where the whole batch is held hostage by its longest generation. Quantify GPU idle time. |

### Part III — Building the Engine (ch07–ch12)

The core of the book. Each chapter is a significant restructuring of the engine.

| # | Chapter | What the reader does |
|---|---|---|
| 7 | **Continuous Batching** | Invert the loop: instead of batch-in/batch-out, `step()` over a running set, admitting and retiring sequences *per iteration* (iteration-level scheduling, after Orca). Implement the scheduler. This produces the single largest jump on the scorecard. |
| 8 | **Paged Attention and the Block Manager** | Why a contiguous per-sequence cache fragments and over-reserves. Implement fixed-size KV blocks, block tables, a logical→physical allocator, copy-on-write for forked sequences, and preemption (recompute vs swap-to-CPU) when memory runs out. Naive PyTorch gather first, then a real kernel. |
| 9 | **Prefix Caching** | Share KV blocks across requests via content hashing. Build up to a radix-tree cache (à la SGLang RadixAttention) with LRU eviction. Show the effect on a chat trace with a long shared system prompt — often the single biggest real-world win, and almost free. |
| 10 | **Chunked Prefill and Scheduling Policy** | Prefill and decode fight each other; a long prompt stalls every streaming response. Implement chunked prefill and a per-step token budget. Then scheduling policy: FCFS vs priority vs fair-share, admission control, and enough queueing theory to explain why p99 explodes as utilisation approaches 1. |
| 11 | **Disaggregating Prefill and Decode** | Run prefill and decode as separate pools with a KV handoff (DistServe / Splitwise pattern). Implement a simplified version, measure the latency/complexity trade-off, and be honest about when it does *not* pay. |
| 12 | **Offloading the KV Cache** | Eviction becomes demotion: a second, larger tier (host memory here; NVMe or a shared store in production) keyed by content rather than by block number. Measures that reuse survives a block pool too small to hold it, and computes the fetch-versus-recompute crossover that decides whether a tier pays. |

### Part IV — Making the Math Cheaper (ch13–ch18)

| # | Chapter | What the reader does |
|---|---|---|
| 13 | **Attention at Speed** | Online softmax and tiling from first principles — derive why FlashAttention is IO-aware rather than fewer-FLOPs. Then KV-shrinking architectures: MQA, GQA, sliding-window, and MLA (DeepSeek-style latent compression). What to write yourself vs what to call. |
| 14 | **Writing a Paged Attention Kernel in Triton** *(optional)* | For readers with a GPU: implement the paged-attention decode kernel in Triton, benchmark against the PyTorch gather from ch08 and against FlashAttention. Clearly marked skippable; nothing later depends on it. |
| 15 | **Quantisation for Serving** | Weight-only INT8/INT4 (GPTQ, AWQ), FP8 on recent hardware, and **KV-cache quantisation** — usually the bigger win for long context, and usually the one people forget. Calibration, per-channel/group scales. Measure all three axes: quality (perplexity *and* a task eval), speed, memory. |
| 16 | **Bounding the Context** | The cache stops growing with the conversation: sliding windows, attention sinks, and the position question that comes with them. The first chapter whose optimisation can change what the model says, so it is measured on perplexity — on both sides of the trained context length, where the answer reverses. |
| 17 | **Speculative Decoding** | Draft-model speculation, self-speculation (Medusa/EAGLE), and prompt-lookup n-gram. Derive the acceptance-rate maths and show *why* rejection sampling preserves the target distribution exactly — this is the chapter's real payload. Implement one variant. Show it winning at low batch and losing at high batch. |
| 18 | **Constrained and Structured Decoding** | JSON-schema and grammar-constrained generation via FSM-driven logit masking (Outlines/XGrammar approach). Token healing. Where the masking cost lands, why naive implementations destroy throughput, and how constraints interact with speculation and prefix caching. |

### Part V — Scaling Out (ch19–ch21)

| # | Chapter | What the reader does |
|---|---|---|
| 19 | **Multi-GPU: Tensor, Pipeline and Expert Parallelism** | Where the collectives go and what they cost. Tensor parallelism for a single layer (all-reduce per block), pipeline parallelism and its bubbles, expert parallelism for MoE. NCCL basics; the things that break (head counts not divisible, uneven splits, one slow rank). When quantisation is the better answer than another GPU. |
| 20 | **Multi-Replica: Routing, Autoscaling and Cold Starts** | Why round-robin is the wrong LLM load balancer. Cache-aware/prefix-affinity routing, least-outstanding-tokens. Autoscaling on queue depth rather than CPU. Cold starts: weight loading (safetensors mmap), warmup and CUDA-graph capture, and why the first request after a deploy is always terrible. |
| 21 | **Multi-Tenancy and LoRA at Serving Time** | Serve many adapters on one base model with batched adapter application (S-LoRA pattern). Per-tenant fairness, quotas, rate limits, and noisy-neighbour isolation in a shared KV cache. |

### Part VI — Serving Patterns by Workload (ch22–ch25)

The same engine, tuned four different ways. The point of the part: there is no single optimal
configuration, and the workload's length distribution decides almost everything.

| # | Chapter | Workload characteristics and what changes |
|---|---|---|
| 22 | **Chat and Assistants** | Long shared system prompts, multi-turn growth, human-perceptible streaming. Prefix caching and session affinity dominate; ITL matters more than throughput. |
| 23 | **RAG and Long Context** | Prefill-heavy, enormous prompts, KV pressure. Chunked prefill + prefix cache + KV quantisation together. Co-serving embedding and reranker models on the same hardware. |
| 24 | **Agents and Tool Use** | Many short, highly repetitive calls; cancellation is constant; tail latency amplifies across a chain of N calls. Prefix reuse and cheap cancellation are worth more than raw throughput. |
| 25 | **Code Completion and Offline Batch** | The two extremes: fill-in-the-middle completion needing single-digit-millisecond TTFT and aggressive speculation, versus offline batch inference where latency is irrelevant and only tokens-per-dollar counts. |

### Part VII — Running It in Production (ch26–ch31)

| # | Chapter | What the reader does |
|---|---|---|
| 26 | **The API Surface** | OpenAI-compatible `/v1/chat/completions`, SSE streaming, usage accounting, tool-call plumbing. Client disconnect and cancellation — wasted GPU nobody notices. Backpressure, timeouts, request size limits. Chat templates and the many ways they are silently wrong. |
| 27 | **Observability for Serving Engines** | The signals that explain the engine, not just the box: queue depth, KV utilisation, preemption rate, batch-size histogram, prefix-cache hit rate, speculation acceptance rate, TTFT/ITL histograms. OpenTelemetry traces across the request lifecycle. What a good dashboard looks like, and SLO burn-rate alerting. |
| 28 | **Reliability and Operations** | Load shedding and graceful degradation. Draining and rolling upgrades without dropping in-flight streams. Failure modes: OOM under a length spike, NaN/inf, GPU fault, and the worst one — silent quality regression after a config change. Runbook-shaped. |
| 29 | **Cost and Capacity Planning** | Derive $/million tokens from hardware cost, utilisation and token mix. The batch-size vs SLO frontier and where to sit on it. Hardware selection, spot/preemptible economics, and an honest build-vs-buy comparison against hosted APIs. Ends with a capacity model the reader can reuse. |
| 30 | **Choosing a Serving Framework** | A capability rubric built from this book's own mechanisms — one row per chapter — plus how to fill it in without trusting a README. Deliberately contains no vendor comparison table: those cannot be verified here and are wrong within months, so the deliverable is the sheet and the method. |
| 31 | **Benchmarking and Verifying a Serving Stack** | How to run a comparison nobody can dismiss: trace-driven load with realistic length distributions, warmup, statistical reporting, version pinning. Correctness verification — output-distribution equivalence tests, not eyeballing. Then a fair run of our engine against vLLM, SGLang, TGI and TensorRT-LLM. |

### Capstone

| # | Chapter | What the reader does |
|---|---|---|
| 32 | **The Finished Engine** | The full scorecard from ch01 to now, in one table. A design retrospective: what we built, what we deliberately did not (multi-node, MoE routing at scale, custom CUDA), and what each omission would cost. Then a guided map into vLLM, SGLang and TensorRT-LLM source, showing where each chapter's concept lives in each codebase — so the reader can read the real thing fluently. |

### Appendices

| # | Appendix | Contents |
|---|---|---|
| A | **GPU and Hardware Primer for Serving** | HBM vs SRAM, SMs, tensor cores, NVLink/PCIe, MIG, and the handful of spec-sheet numbers that actually predict serving performance. |
| B | **Environment Setup** | Three supported paths: CPU-only laptop, single consumer GPU, rented cloud GPU. Exact versions, install scripts, and how to verify the setup before Chapter 1. |
| C | **Code Map of the Reference Engine** | Module-by-module tour of `llmserve/`, with the chapter that introduced each piece. |
| D | **Glossary** | TTFT, ITL, TPOT, goodput, prefill, decode, paged attention, block table, continuous batching, speculation acceptance rate, and ~40 more. |
| E | **Reading List** | The primary sources, grouped by chapter: Orca (iteration-level scheduling), vLLM/PagedAttention, FlashAttention 1–3, SGLang/RadixAttention, Sarathi-Serve (chunked prefill), DistServe and Splitwise (disaggregation), speculative decoding (Leviathan et al., Chen et al.), Medusa, EAGLE, GPTQ, AWQ, SmoothQuant, LLM.int8(), S-LoRA, Outlines/XGrammar. |

---

## 5. Hardware and execution strategy

This is the hardest practical problem for this book and the one most likely to sink it. A book
about GPU serving whose examples nobody can run is a blog post with extra steps.

**Three-tier approach:**

| Tier | Hardware | Model | What works |
|---|---|---|---|
| **Default** | Any laptop, CPU-only | `TinyGPT` — a ~5.8M-parameter transformer **defined in `llmserve/model.py` with seeded random weights**, no download | Every mechanism in the book runs: KV cache, continuous batching, paged blocks, prefix caching, scheduling, the full API server. Absolute numbers are not GPU numbers, but the *ratios and mechanisms* hold. |
| **Recommended** | One 24 GB consumer GPU (RTX 3090/4090) or cloud L4/A10G | Qwen2.5-1.5B / Llama-3.2-1B-Instruct, plus a 7–8B model for realism | All published scorecard numbers, quantisation, Triton kernel, speculation. |
| **Advanced** | 2–4× A100/H100, rented hourly | 7–8B and one 70B quantised | Parts V chapters 17–18 only. Each of these chapters states the hourly cost of reproducing it. |

**Rules that follow from this:**

- Every chapter states its tier in the header. No chapter above Tier 1 is a prerequisite for
  a later Tier 1 chapter, so a laptop-only reader can complete the main arc.
- The engine is written so the device is a parameter, not an assumption. CPU is a supported
  backend, not a degraded mode.
- CI executes the Tier 1 path only ([§9](#9-build-and-publishing-pipeline)); no chapter may
  have an executable cell needing a GPU. GPU results are generated manually by a committed
  script and checked into `bench/results/`.
- **The Tier 1 model is built from code, not downloaded.** A seeded random-weight `TinyGPT`
  makes the default path work with no network access at all, keeps results bit-reproducible
  across machines, and cannot be invalidated by a model being pulled from the Hub. Its output is
  meaningless text, which is fine: every quantity this book measures depends on the *shape* of
  the computation, not the values in the weights. A byte-level tokenizer comes with it, which
  also makes ch04's incremental-detokenisation bug concrete rather than hypothetical.
- **Where output quality genuinely matters — ch15's quantisation chapter above all — a trained
  model is required, and that chapter says so.** Those sections need Hugging Face access.
- Model weights are never committed.

---

## 6. Companion code

A book called *from scratch* lives or dies on its code being real, runnable, and readable.

### 6.1 Layout

The code lives in the **same repository** as the book (not a separate repo), so a chapter and
its code can never drift out of sync:

```
llm-serving-from-scratch/
├── llmserve/                    # the engine, built up across the book
│   ├── config.py
│   ├── tokenizer.py             # ch04: incremental detokenisation
│   ├── sampling.py              # ch04
│   ├── cache/                   # ch05, ch08: KV cache, blocks, allocator
│   ├── scheduler/               # ch07, ch10: continuous batching, policy
│   ├── prefix/                  # ch09: hash + radix prefix cache
│   ├── attention/               # ch13, ch14: paged attention backends
│   ├── quant/                   # ch15
│   ├── speculative/             # ch17
│   ├── constrain/               # ch18
│   ├── parallel/                # ch19
│   ├── engine.py                # the step() loop
│   └── server/                  # ch26: OpenAI-compatible API, SSE
├── bench/                       # ch02: load generator, traces, scorecard
│   ├── harness.py
│   ├── traces/
│   └── results/*.json           # committed, hardware-stamped
├── tests/                       # equivalence + unit tests (CPU, run in CI)
├── chapters/                    # ch01.md … ch32.md
├── appendices/
└── ...                          # see §8
```

### 6.2 Per-chapter checkpoints

Readers must be able to start at any chapter. Two mechanisms:

- **Git tags** `ch07-continuous-batching`, `ch08-paged-attention`, … one per engine-changing
  chapter. `git checkout ch07-continuous-batching` gives the engine exactly as it stood at the
  end of that chapter.
- **A `CHECKPOINTS.md`** table mapping chapter → tag → one-line description of the engine's
  state → the scorecard row it produced.

Chapter text quotes code from the working tree with MyST's `{literalinclude}` directive,
anchored on `:start-at:` / `:end-at:` (a function signature, say) rather than line numbers,
which rot on the first edit. Code is never duplicated inline. **This is non-negotiable**:
copy-pasted code in prose goes stale within two chapters.

### 6.3 How numbers get into the book

1. `bench/harness.py` writes `bench/results/<engine>-rate<N>-<tier>.json`, stamping model,
   hardware, library versions, trace, and request rate.
2. Each table is declared once in `bench/scorecards.py` and rendered to a markdown fragment in
   `chapters/_generated/` by `scripts/render-scorecards.py`. Chapters pull it in with
   `{include}`.
3. Two CI guards: `scripts/verify-numbers.py` (every cited result exists, carries its stamps, and
   does not predate the engine code it measures) and `scripts/render-scorecards.py --check`
   (every committed fragment still matches the results).

**Chapters contain no executable cells.** Rendering figures by executing Python during the book
build was the original plan; it makes every deploy depend on a Jupyter kernel and a torch install,
which fail for reasons unrelated to the book. Pre-rendering keeps the build pure markdown and lets
CI *diff* the regenerated output — a stronger staleness guarantee than execution provides.

No number is ever typed into prose by hand. This is what lets the book make performance
claims credibly and keeps it honest when a library upgrade changes the answer.

### 6.4 Correctness testing

Per [§1.2](#12-what-makes-it-different), each optimisation gets an equivalence test:

- **Greedy equivalence:** optimised path must produce token-identical output to the
  HuggingFace reference for a fixed set of prompts.
- **Distributional equivalence:** for sampling paths (notably speculative decoding), compare
  empirical token distributions over many seeds with a statistical test.
- **Invariant checks:** prefix-cache hits must not change output; paging must not change
  output; quantisation gets a *bounded* quality-delta assertion rather than exact equality.

These run on CPU in CI, so every push proves the engine is still correct.

---

## 7. Toolchain

**Decided: Jupyter Book 2 / MyST** (`mystmd`), matching `snowch.github.io` itself and
`learn_probability`.

This is the same toolchain the site already builds in CI, which is the deciding practical
advantage: the GitHub Actions workflow, the `BASE_URL` handling for project sites, the execute
cache, and the sitemap script are all already solved and proven in
`snowch/snowch.github.io/.github/workflows/deploy.yml`. This book copies that, rather than
introducing a second publishing stack to maintain.

**What this changes versus the Quarto alternative** — worth naming so nothing surprises us at
chapter 13:

| Need | How it is met under MyST |
|---|---|
| Chapter source | `.md` with MyST frontmatter (`jupytext`/`kernelspec` when a chapter executes), or `.ipynb` — matching the existing site's chapter files |
| Config / TOC | `myst.yml` with `project.toc`, exactly like the site |
| Quoting code from `llmserve/` | `{literalinclude}` directive with `:start-at:` / `:end-at:` anchors ([§6.2](#62-per-chapter-checkpoints)) |
| Not executing GPU code in CI | Nothing executes at build time at all — figures are pre-rendered fragments ([§6.3](#63-how-numbers-get-into-the-book)), so the book build needs no kernel and no torch |
| Execution caching | `_build/execute` and `_build/templates`, cached in Actions by a key hashing `requirements.txt` + `myst.yml` |
| Cross-references | `(label)=` targets with `[](#label)`; citations via `@citekey` against `references.bib` |
| PDF | `myst build --pdf` (LaTeX/Typst). Lower fidelity than Quarto's book PDF — acceptable; HTML is the primary format |
| EPUB | **Not a native MyST export.** Either drop EPUB, or add a pandoc post-build step. Recommend dropping it for v1.0 and revisiting only if readers ask |

The one real loss is book-quality PDF/EPUB. That is a fair trade for a single toolchain across
the whole site, and HTML is where the readers are.

**Stack:** `mystmd` (Node 20) + `jupyter-book` · Python 3.11 · PyTorch · `transformers`/
`tokenizers` · FastAPI + uvicorn · `ruff` (pinned) · `pytest` · `pre-commit` · GitHub Actions →
GitHub Pages.

## 8. Repository layout

Mirrors the MyST conventions already used by `snowch.github.io`, with the code-package addition
from [§6](#6-companion-code):

```
llm-serving-from-scratch/
├── myst.yml                  # project config: TOC, site template, bibliography
├── index.md                  # preface: why this book, how to read it, tiers
├── chapters/ch01_*.md … ch29_*.md   # descriptive slugs, for URLs and search
├── appendices/appendix_a_*.md … appendix_e_*.md
├── llmserve/                 # the engine (§6.1)
├── bench/                    # harness + committed results
├── tests/
├── scripts/
│   ├── ci-check.sh           # the exact checks CI runs, runnable locally
│   ├── run-benchmarks.sh     # GPU-tier regeneration (manual)
│   ├── verify-numbers.py     # §6.3 guard
│   └── generate_sitemap.py   # copy-forward from snowch.github.io
├── references.bib            # primary sources (appendix E)
├── custom.css                # site.options.style
├── robots.txt
├── favicon.ico
├── pyproject.toml            # ruff config + package metadata
├── requirements.txt          # pinned engine + book deps
├── requirements-dev.txt      # ruff/pytest/pre-commit, installed by the quality workflow
├── package.json              # pins mystmd — the single source of truth for its version
├── .pre-commit-config.yaml
├── .github/workflows/{quality.yml,deploy.yml}
├── .claude/SessionStart      # bootstrap deps so web sessions can run tests
├── AUTHORING_GUIDE.md
├── CHECKPOINTS.md            # §6.2
├── CLAUDE.md
├── PLAN.md                   # this file
├── LICENSE                   # CC-BY-NC-4.0 (prose)
├── LICENSE-CODE              # Apache-2.0 (llmserve/, bench/, tests/)
└── README.md
```

Built output goes to `_build/html`, which is gitignored.

**Licensing note:** the embeddings book uses CC-BY-NC-4.0 throughout. That is wrong for this
book's code — a non-commercial licence makes `llmserve/` useless as a reference implementation
people can borrow from. Split the licence: CC-BY-NC-4.0 for prose, Apache-2.0 for code, stated
clearly in the README.

## 9. Build and publishing pipeline

Two workflows. `deploy.yml` is adapted from `snowch.github.io/.github/workflows/deploy.yml`,
which already solves the parts that are easy to get wrong.

**`quality.yml`** — on every push and PR:
- `ruff check` + `ruff format --check` on `llmserve/`, `bench/`, `tests/`, `scripts/`
- `pytest tests/` — CPU tier only, including the equivalence tests from [§6.4](#64-correctness-testing)
- `python scripts/verify-numbers.py`
- link check on the built HTML

**`deploy.yml`** — on push to `main` and `workflow_dispatch`:
- `actions/configure-pages@v5`, then Node 20 + `npm install -g mystmd`, then Python 3.11 +
  `pip install -r requirements.txt`
- Restore the `_build/execute` + `_build/templates` cache, keyed on `requirements.txt`,
  `myst.yml`, `package-lock.json` and `bench/results/*.json`
- `myst build --html --execute` with
  `BASE_URL: ${{ steps.pages.outputs.base_path }}` — **this is the critical line for a project
  site**; without it every asset and link 404s under `/llm-serving-from-scratch/`. The site's
  workflow already has the correct ternary that maps `/` to an empty string, so copy it verbatim
- `python scripts/generate_sitemap.py`; copy `robots.txt`
- `actions/upload-pages-artifact@v3` on `_build/html`, then `actions/deploy-pages@v4` with the
  retry-and-backoff pattern the site's workflow uses

Published to `https://snowch.github.io/llm-serving-from-scratch/` with Pages source set to
**GitHub Actions**.

### CI gotchas to avoid

Three from `embeddings-at-scale-book`, two specific to MyST. All are cheap to get right at
scaffolding time and annoying to debug later.

1. **A `workflow_run` trigger naming a workflow that does not exist.** That repo's
   `publish.yml` waits on `workflows: ["Code Quality"]`, but `.github/workflows/` contains only
   `publish.yml` — so the book publishes on manual dispatch only. Here, either commit
   `quality.yml` under exactly the name `deploy.yml` waits for, or trigger deploy directly on
   push to `main`. *(Also worth fixing in that repo — it is almost certainly not intentional.)*
2. **Lint paths that do not match real directories.** That repo's `scripts/ci-check.sh` lints
   `code_examples/`, which is not in the repo. Make `ci-check.sh` the single source of truth
   that both CI and `pre-commit` invoke, so drift is impossible.
3. **Unpinned `mystmd`.** `npm install -g mystmd` installs whatever is latest, so a build that
   worked yesterday can break with no commit. Pin the version.
4. **Execute-cache staleness.** A cached `_build/execute` can keep publishing stale output
   indefinitely. Include `bench/results/*.json` in the cache key, and let
   `verify-numbers.py` ([§6.3](#63-how-numbers-get-into-the-book)) fail the build when a chapter
   cites a result file older than the code it describes.
5. **`--execute` reaching for a GPU.** No chapter may have an executable cell that needs a GPU
   or downloads large weights — CI has neither. GPU work is a static code block plus a
   committed result file; only cheap cells execute.

## 10. Linking from snowch.github.io

The ask is that this be linkable from the site. Four concrete edits in
`snowch/snowch.github.io`, all modelled on how *Embeddings at Scale* is already wired in:

**1. Register the project site** in `myst.yml` so it joins the sitemap index:

```yaml
  project_sites:
    - learn_probability
    - embeddings-at-scale-book
    - learn_linear_algebra
    - llm-serving-from-scratch      # add
```

**2. Add a landing page** `llm-serving.md` at the site root, mirroring `embeddings.md`:
title, a paragraph on what the book covers, a *Topics Covered* list, and a prominent
**[LLM Serving from Scratch →](https://snowch.github.io/llm-serving-from-scratch/)** link.
This page — not the book — is what ranks on the site and what gets linked from LinkedIn/talks.

**3. Add it to the site TOC** in `myst.yml`, alongside the other book entries:

```yaml
    - file: embeddings.md
      title: Embeddings at Scale Book
    - file: llm-serving.md
      title: LLM Serving from Scratch Book    # add here — before the non-book pages
```

**4. Cross-link from the existing LLM series** (the highest-intent traffic the site already
has, per [§2](#2-relationship-to-existing-snowchgithubio-content)):
- `ai-eng/llmfs-scaling/index.md` — add a line to *The Series* table and a closing pointer:
  serving is where this series ends and the book begins.
- `ai-eng/llmfs-scaling/L21_Deployment_Serving.md` — callout at the top linking to the book.
- `ai-eng/llmfs-scaling/L20_Quantization_Inference.md` and `L17_Attention_Optimizations.md` —
  same callout, pointing at ch15 and ch13/ch14 respectively.

**Not needed:** an entry in `books.md`. That table is for external/older guides; the major
books each get their own landing page and TOC entry.

---

## 11. Delivery roadmap

The book should be *linkable and useful* long before it is finished — the site already marks
in-progress work `[DRAFT]`, so shipping incrementally is consistent with existing practice.

**Status: all 32 chapters and all 5 appendices are written**, so the releases below are now a record
of the order things were built in rather than a plan. Two things promised here were not delivered as
promised, and both are stated in the chapters themselves rather than quietly dropped:

- **ch14 contains no Triton kernel.** It contains the paged-decode algorithm in PyTorch, verified
  against the ch13 reference on every boundary case, the arithmetic that says the ch08 gather halves
  the decode ceiling, and a precise account of the Triton translation. A kernel cannot be compiled or
  run without a GPU, and shipping an unverified kernel would contradict §6.3 — the whole point of
  which is that unverified performance claims are worthless.
- **ch31 contains no framework comparison.** It contains the methodology, the coordinated-omission
  demonstration, and an engine-agnostic harness that measures anything implementing three methods.
  A fair run against vLLM, SGLang, TGI and TensorRT-LLM requires each of them tuned on the same GPU;
  run here it would compare four engines' CPU fallback paths, which measures nothing.

ch19 is delivered in a third form: the tensor-parallel split is built and proved *exactly correct* on
one device, and the collective cost is computed from the model's shape and the link's bandwidth and
latency rather than timed. That is a stronger result than a timing on the wrong hardware, and the
chapter says which half is which.

| Release | Contents | Why this is the cut |
|---|---|---|
| **v0.1 — Foundations** ✅ | Repo scaffolding, CI, Pages deploy · index/preface · ch01–ch03 written with measured figures · `bench/` harness · `llmserve` model/tokenizer/sampling/engines · 42 tests | Establishes the baseline *and* the scorecard. Nothing later can be written credibly without the harness. |
| **v0.2 — The Decode Loop** | ch04–ch06 · `llmserve` KV cache + static batching · equivalence tests · first three scorecard rows | Proves the measure-every-chapter format works end to end at small scale. |
| **v0.3 — The Engine** | ch07–ch10 · continuous batching, paged blocks, prefix cache, chunked prefill · checkpoint tags | The centre of gravity. At this point the book is already the most useful thing on the site about serving. |
| **v0.4 — Cheaper Math** | ch13, ch15, ch17 · quantisation + speculation · GPU-tier results published | First release with meaningful GPU numbers; needs the Tier 2 machine. |
| **v0.5 — API and Operations** | ch26–ch28 · OpenAI-compatible server, observability, reliability | Pulled forward ahead of Parts V–VI: a reader with the engine plus an API plus a dashboard can actually deploy something. Highest practical value per page. |
| **v0.6 — Scale and Patterns** | ch11, ch14, ch18–ch25 · disaggregation, Triton, constrained decoding, multi-GPU, workload patterns | The advanced and specialist material, once the core arc is solid. |
| **v1.0 — Complete** | ch29–ch32 · appendices A–E · full scorecard · PDF export if it earns its keep | Costing, honest benchmarking, and the capstone retrospective land last because they summarise everything before them. The framework comparison moved to a reader exercise; see the status note above. |

**Rationale for the ordering:** Parts are written in dependency order except that Part VII's
API/observability chapters are pulled ahead of Parts V–VI. That is deliberate — an engine with
an API and a dashboard is deployable; an engine with tensor parallelism but no API is not.

---

## 12. Conventions and quality bar

### 12.1 Chapter template

Every chapter follows the same shape, so the book reads as one system:

1. **Header block** — tier (CPU / single GPU / multi GPU), prerequisites, the scorecard row
   this chapter will move.
2. **The problem** — a measurement from the previous chapter that is unacceptable, shown, not
   asserted.
3. **The idea** — the mechanism, from first principles, with the arithmetic.
4. **The build** — incremental implementation, code quoted from `llmserve/`.
5. **The measurement** — scorecard re-run, new row, one paragraph on why it moved.
6. **The cost** — what this technique made worse (complexity, latency variance, quality,
   memory). Never skipped; every chapter has one.
7. **Key takeaways** — 3–5 bullets.
8. **Looking ahead** — the transition that sets up the next chapter's problem.
9. **Further reading** — primary sources, cited via `references.bib`.

### 12.2 Style

- British English, second person, active voice. Short sentences.
- Mathematics only where it predicts something the reader will then verify.
- Every diagram must show a mechanism (block table indirection, the scheduler state machine,
  prefill/decode interleaving on a timeline) — no decorative figures.
- Numbers always carry their conditions: hardware, model, dtype, request rate, trace, date.
- `[DRAFT]` in the chapter title while incomplete, matching the site's existing convention;
  removed only when the chapter meets [§12.3](#123-definition-of-done).

### 12.3 Definition of done (per chapter)

- [ ] Code merged into `llmserve/` and passing `ruff` + `pytest` on CPU
- [ ] Equivalence test added for any behaviour-affecting change
- [ ] Scorecard row regenerated from a committed result file (not hand-typed)
- [ ] Checkpoint tag pushed, `CHECKPOINTS.md` updated
- [ ] "The cost" section written — the chapter is not done while the trade-off is missing
- [ ] Cross-references and `references.bib` entries resolve; `myst build --html` clean
- [ ] `[DRAFT]` removed

### 12.4 Version pinning and staleness

This is the fastest-moving topic the author has written about; a book on it is stale in months
unless designed against that.

- Pin exact versions in `requirements.txt` **and `package.json`** (`mystmd` included); state
  them in Appendix B and in each chapter that depends on library behaviour.
- Prefer mechanisms over APIs: the paged-attention *idea* will outlive any vLLM API. Where a
  library call is shown, show the mechanism first.
- Date-stamp every benchmark and put a "last verified" line in the preface.
- Keep a short `CHANGELOG.md` so returning readers can see what moved.

---

## 13. Decisions — settled

All four opening decisions are now made.

| # | Decision | Outcome |
|---|---|---|
| 1 | Toolchain | **Jupyter Book 2 / MyST** (`mystmd`) — same stack as the rest of the site, and its Actions workflow is already proven ([§7](#7-toolchain), [§9](#9-build-and-publishing-pipeline)). The cost is book-quality PDF and native EPUB. |
| 2 | Hardware floor | **CPU-first, GPU optional** ([§5](#5-hardware-and-execution-strategy)). Only ch14 and ch19 require a GPU, and nothing depends on them. |
| 3 | Fate of the L17/L20/L21 drafts | **Finish them as ~200-line summaries that link into the book** ([§2](#2-relationship-to-existing-snowchgithubio-content)). Preserves inbound links and avoids maintaining the same material at two depths. |
| 4 | Triton kernel chapter (ch14) | **Included, clearly optional.** The most genuinely *from scratch* chapter in the book; nothing later depends on it. |

Two smaller ones remain, and neither blocks authoring:

- **Downloadable notebooks per chapter.** MyST can serve `.ipynb` alongside `.md`; worth doing
  once the chapter format has settled.
- **PDF.** `myst build --pdf` needs a LaTeX toolchain in CI (the site's workflow already
  installs `texlive-latex-*`). Defer until v1.0 and decide then whether the fidelity is worth it.
- **Audio chapter intros.** Considered and **deferred**. Google's licensing very likely permits
  publishing NotebookLM Audio Overviews, but Google itself labels them as possibly inaccurate and
  "not a citable record" — which sits badly against a book whose whole claim is that every number
  is measured. Revisit only with narration over reviewed prose. NotebookLM is still useful here as
  a *drafting* tool: where its hosts garble a chapter is where readers will too.

## 14. Immediate next steps

1. ~~Scaffold the repo per [§8](#8-repository-layout).~~ **Done.** All 29 chapter stubs and 5
   appendices exist, carrying the [§12.1](#121-chapter-template) template with per-chapter
   guidance; `myst.yml`, both workflows, pinned dependencies, `pyproject.toml`, licences,
   `AUTHORING_GUIDE.md`, `CHECKPOINTS.md` and `.claude/SessionStart` are in place. Verified
   locally: `ruff` clean, tests pass, `myst build --strict` builds all 35 pages with zero
   content warnings.
2. **Enable GitHub Pages** on this repo with source = **GitHub Actions** (a repository setting,
   so it cannot be done from a commit), then confirm the stub book deploys correctly under
   `/llm-serving-from-scratch/`. `BASE_URL` is the one thing that will silently break every link
   if it is wrong, and it is far easier to verify now than after 29 chapters.
3. Make the four `snowch.github.io` edits in [§10](#10-linking-from-snowchgithubio) so the link
   exists from day one, with the landing page marked *in progress*.
4. ~~Build `bench/harness.py`, then ch01–ch03, as v0.1.~~ **Done**, and then everything after it.
   All 32 chapters and 5 appendices are written; `./scripts/ci-check.sh` is clean, 273 tests pass,
   and every figure traces to a stamped result in `bench/results/`.
5. Revisit the three `llmfs-scaling` lessons per decision 3 now that ch13/ch15 exist to link to.
6. **On Tier 2/3 hardware**, in priority order: write ch14's Triton kernel against the reference and
   tests already in the repo; time ch19's tensor-parallel split rather than computing it; and run
   ch31's methodology against vLLM, SGLang, TGI and TensorRT-LLM. Each of the three has its
   specification, its harness and its verification suite committed; what is missing is only the
   hardware.
7. Re-measure Parts II–III on a GPU. Every committed figure is CPU-only, and while the *shape* of
   each result is structural, the magnitudes are not — a GPU-tier column beside the Tier 1 one would
   make the book considerably more useful and would cost a few hours of compute.
