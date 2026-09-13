# LLM Serving from Scratch

> **Build a production inference engine, one measurement at a time.**

An online book, in planning. Start from a ~100-line server that wraps `model.generate()`,
and incrementally build a real LLM inference engine: KV cache, continuous batching, a paged
block manager, prefix caching, chunked prefill, speculative decoding, quantisation, tensor
parallelism, an OpenAI-compatible API, and the observability needed to operate it.

Every chapter re-runs the same benchmark, so progress is measured rather than asserted.

## Status

🚧 **Scaffolded, authoring not started.** All 29 chapters and 5 appendices exist as stubs —
each carrying its hardware tier, prerequisites, the benchmark row it is expected to move, and an
outline of all nine template sections. No prose is written yet; every chapter is marked
`[DRAFT]`.

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
[snowch.github.io](https://snowch.github.io). Once authoring begins, the book will be published
to **https://snowch.github.io/llm-serving-from-scratch/** and linked from the site.

## Relationship to the LLM From Scratch series

This book is the long-form successor to the serving material in the
[LLM From Scratch: Scaling & Optimization](https://snowch.github.io/ai-eng/llmfs-scaling/)
series. Those lessons survey the techniques; this book builds them.

Recommended prerequisite: the
[LLM From Scratch core series](https://snowch.github.io/ai-eng/llmfs/).

## Licence

Planned as a split licence — CC-BY-NC-4.0 for the prose, Apache-2.0 for the companion engine
code, so the reference implementation is actually reusable. See [PLAN.md §8](PLAN.md).

---

Author: [Chris Snow](https://snowch.github.io)
