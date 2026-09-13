# LLM Serving from Scratch

> **Build a production inference engine, one measurement at a time.**

An online book, in planning. Start from a ~100-line server that wraps `model.generate()`,
and incrementally build a real LLM inference engine: KV cache, continuous batching, a paged
block manager, prefix caching, chunked prefill, speculative decoding, quantisation, tensor
parallelism, an OpenAI-compatible API, and the observability needed to operate it.

Every chapter re-runs the same benchmark, so progress is measured rather than asserted.

## Status

📋 **Planning.** No chapters written yet. See **[PLAN.md](PLAN.md)** for the full book plan:
outline (7 parts, 29 chapters), companion-code design, toolchain, publishing pipeline, and
delivery roadmap.

Once authoring begins, the book will be published to
**https://snowch.github.io/llm-serving-from-scratch/** and linked from
[snowch.github.io](https://snowch.github.io).

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
