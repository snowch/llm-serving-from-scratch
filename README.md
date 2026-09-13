# LLM Serving from Scratch

> **Build a production inference engine, one measurement at a time.**

An online book. Start from a ~100-line server that wraps `model.generate()`, and incrementally
build a real LLM inference engine: KV cache, continuous batching, a paged block manager, prefix
caching, chunked prefill, speculative decoding, quantisation, tensor parallelism, multi-tenant LoRA,
an OpenAI-compatible API, and the observability and admission control needed to operate it.

Every chapter re-runs the same benchmark, so progress is measured rather than asserted.

## Status

✅ **All 29 chapters and 5 appendices written**, with every figure measured on a real run.

- **The engine runs**: a code-defined transformer, byte-level tokenizer with streaming-safe
  detokenisation, samplers, KV cache, and eleven engines behind one interface — naive, cached,
  static batching, continuous batching, paged attention, prefix caching, chunked prefill,
  disaggregated prefill/decode, a multi-replica router, tenant-fair admission, and load shedding.
  Plus quantisation, speculative decoding, grammar-constrained decoding, LoRA adapters and a
  tensor-parallel split.
- **The harness runs**: open-loop Poisson load generation, TTFT/ITL percentiles, goodput against a
  stated SLO, per-tenant breakdowns, and traces for chat, retrieval, agents, code completion,
  offline batch, multi-tenant and noisy-neighbour workloads. **229 tests**, including equivalence
  tests proving every optimisation leaves output token-identical and a distributional test proving
  speculative decoding samples from the target distribution.
- **Every number is stamped.** Each figure traces to a JSON result recording the model, hardware,
  library versions and a content hash of the code that produced it; CI fails if any of them drifts.
- **Two chapters are explicit about hardware this repository does not have.** ch13 gives the
  paged-decode algorithm, its tests and its arithmetic but no Triton kernel — a kernel cannot be
  verified without a GPU, and shipping an unverified one would contradict the book's own standard.
  ch17 proves the tensor-parallel split exactly correct on one device and computes the collective
  cost rather than timing it. Both say so in the chapter.

No model download is needed: the reference model is built from code with seeded random weights,
so the whole Tier 1 path runs on any laptop with no network access. Run
`python3 scripts/verify-setup.py` to see which tiers your machine can complete.

- **[PLAN.md](PLAN.md)** — the full book plan: outline, companion-code design, hardware strategy,
  publishing pipeline, delivery roadmap, and the decisions behind them
- **[AUTHORING_GUIDE.md](AUTHORING_GUIDE.md)** — how to write a chapter
- **[CHECKPOINTS.md](CHECKPOINTS.md)** — the per-chapter git tag scheme

## Building it

```bash
npm install -g "mystmd@$(node -p "require('./package.json').devDependencies.mystmd")"
pip install -r requirements.txt -r requirements-dev.txt

myst start              # live preview
./scripts/ci-check.sh   # exactly what CI runs
```

Built with **Jupyter Book 2 / MyST** (`mystmd`), matching the rest of
[snowch.github.io](https://snowch.github.io). Published to
**https://snowch.github.io/llm-serving-from-scratch/**.

## Relationship to the LLM From Scratch series

This book is the long-form successor to the serving material in the
[LLM From Scratch: Scaling & Optimization](https://snowch.github.io/ai-eng/llmfs-scaling/)
series. Those lessons survey the techniques; this book builds them.

Recommended prerequisite: the
[LLM From Scratch core series](https://snowch.github.io/ai-eng/llmfs/).

## Licence

A split licence — CC-BY-NC-4.0 for the prose, Apache-2.0 for the companion engine code, so the
reference implementation is actually reusable. See [PLAN.md §8](PLAN.md).

---

Author: [Chris Snow](https://snowch.github.io)
