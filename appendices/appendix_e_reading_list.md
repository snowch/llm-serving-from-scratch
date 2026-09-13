---
title: "Reading List"
short_title: "Appendix E"
---

(appendix-e)=
# Appendix E · Reading List

The primary sources, grouped by the chapter that uses them. Full bibliographic entries are in
`references.bib`.

This is a short list on purpose. Serving is a fast-moving field with a very large literature and a
fairly small number of load-bearing ideas, and almost every paper below is one of those ideas rather
than an incremental improvement on one. If you read six of them you will understand most production
engines.

**A note on reading them.** Serving papers report speedups, and a speedup is a claim about a
workload, a baseline and a service objective. Before believing a number, find those three. Several
famous figures are true and describe a situation you are not in.

## Start here

If you read only three:

- **Orca** (Yu et al., OSDI 2022) — iteration-level scheduling. The idea behind every modern engine's
  main loop, and the largest single improvement in this book ({ref}`ch07`).
- **PagedAttention / vLLM** (Kwon et al., SOSP 2023) — KV cache as paged memory. Explains why the
  memory management in {ref}`ch08` looks like an operating system's.
- **DistServe** (Zhong et al., 2024) — prefill/decode disaggregation, and the clearest statement of
  *goodput* as the metric that matters ({ref}`ch02`, {ref}`ch11`).

## Scheduling and batching

- **Orca** (Yu et al., OSDI 2022). Iteration-level scheduling. ({ref}`ch07`)
- **Sarathi-Serve** (Agrawal et al., 2024). Chunked prefill and stall-free batching — the source of
  {ref}`ch10`'s token budget, and honest about the throughput it costs.

## Memory

- **PagedAttention / vLLM** (Kwon et al., SOSP 2023). Blocks, block tables, copy-on-write sharing.
  ({ref}`ch08`)
- **SGLang / RadixAttention** (Zheng et al., 2024). Automatic prefix reuse via a radix tree, plus a
  router that knows about it — which is {ref}`ch09` and {ref}`ch20` as one system.

## Memory hierarchy

- **LMCache** and **Mooncake** are the systems built around {ref}`ch12`'s idea — a KV tier shared
  across a fleet rather than one process's spillover. Read them for the two problems this book's
  version does not have: invalidating a shared tier when the weights change, and deciding which
  replica's cache is worth keeping.
- The operating-systems literature on **page replacement** transfers directly, thrashing included.

## Attention

- **Multi-Query Attention** (Shazeer, 2019). Four pages, and the origin of the observation the whole
  field now runs on: decode is bottlenecked by the KV cache, not by the arithmetic. ({ref}`ch13`)
- **GQA** (Ainslie et al., 2023). The interpolation between MHA and MQA that everything now uses,
  and the recipe for converting an existing checkpoint. ({ref}`ch13`)
- **DeepSeek-V2** (2024). Multi-head latent attention: compress the cache in the architecture
  rather than by sharing heads. The {ref}`ch13` section on it is arithmetic only, because the
  quality half of the trade belongs to whoever trains the model.
- **FlashAttention** (Dao et al., NeurIPS 2022) and **FlashAttention-2** (Dao, 2023). IO-awareness:
  more arithmetic, less memory traffic, much faster. Read the first for the idea and the second for
  what it takes to actually saturate the hardware. ({ref}`ch13`)

## Disaggregation

- **DistServe** (Zhong et al., 2024) and **Splitwise** (Patel et al., ISCA 2024). The same idea from
  two directions; Splitwise is more explicit about the hardware heterogeneity that makes it pay.
  ({ref}`ch11`)

## Bounding the context

- **StreamingLLM** (Xiao et al., 2024). The attention-sink result {ref}`ch16` measures. Worth
  reading for the figure showing where attention actually goes — the technique rests on an
  observation someone made by looking, not by theorising.
- **H2O** (Zhang et al., 2023) and **SnapKV** (Li et al., 2024) attack the same problem by scoring
  which tokens to keep rather than taking the most recent, trading compute for a better choice.

## Quantisation

- **LLM.int8()** (Dettmers et al., NeurIPS 2022). The outlier-channel problem, which is the reason
  per-channel scaling exists and {ref}`ch15` measures directly.
- **SmoothQuant** (Xiao et al., ICML 2023). Migrating activation outliers into the weights, where
  they are easier to quantise.
- **GPTQ** (Frantar et al., 2023) and **AWQ** (Lin et al., MLSys 2024). The two post-training methods
  you will actually meet in a model card. ({ref}`ch15`)

## Speculation

- **Leviathan et al.** (ICML 2023) and **Chen et al.** (2023). Two independent derivations of the
  same acceptance rule. Read one of them for the proof that it preserves the target distribution —
  it is the part implementations get wrong, and {ref}`ch17` measures what happens when they do.
- **Medusa** (Cai et al., 2024) and **EAGLE** (Li et al., 2024). Drafting without a separate draft
  model: extra heads, and feature-level prediction respectively. ({ref}`ch17`)

## Structured output

- **Outlines** (Willard and Louf, 2023). Guided generation as FSM-indexed logit masking, and the
  token-lifting problem this book's byte-level tokenizer lets it skip. ({ref}`ch18`)
- **XGrammar** (Dong et al., 2024). The same problem attacked with a much faster precomputation.

## Parallelism

- **Megatron-LM** (Shoeybi et al., 2019). The tensor-parallel layer split, written for training and
  used unchanged for serving. ({ref}`ch19`)

## Multi-tenancy

- **LoRA** (Hu et al., 2021). Worth reading for the ablation over which projections to adapt — the
  convention of adapting queries and values is an empirical result, not a default someone picked.
  ({ref}`ch21`)
- **S-LoRA** (Sheng et al., 2024) and **Punica** (Chen et al., MLSys 2024). Serving many adapters
  concurrently. Both are largely about replacing {ref}`ch21`'s naive per-adapter loop with a batched
  kernel, and {ref}`ch21` measures the gap they are closing.
- **Deficit Round-Robin** (Shreedhar and Varghese, 1996). Thirty years old, from packet scheduling,
  and still the clearest treatment of the fairness problem {ref}`ch21` runs into.

## Measurement

- **"How NOT to Measure Latency"** (Tene, 2015). A conference talk, not a paper, and the best
  explanation of coordinated omission there is. {ref}`ch02`'s harness design is downstream of it,
  and {ref}`ch31` measures what it warns about.

## Reading the engines themselves

At some point the source is better than any paper. All three are readable, and {ref}`ch32` maps this
book's chapters onto each of them:

- **vLLM** — the most widely deployed, and the closest in structure to this book's engine.
- **SGLang** — the place to look for prefix caching and structured output done seriously.
- **TensorRT-LLM** — the least like this book and the most instructive for it: a compiler-oriented
  design where much of what is runtime scheduling here is a build-time decision there.
