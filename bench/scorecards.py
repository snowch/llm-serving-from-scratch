"""Which scorecard fragments the book contains, and what goes in each.

One declaration per table, in one place, so a chapter can never quietly cite a different set of
runs than the one its prose discusses. ``scripts/render-scorecards.py`` turns these into markdown
fragments under ``chapters/_generated/``, and ``scripts/verify-numbers.py`` fails the build if a
committed fragment no longer matches what the results say.

This replaces executing code inside chapters. It keeps the invariant that matters — no number is
ever typed into prose (PLAN.md §6.3) — while keeping the book build free of a Jupyter kernel,
torch, and everything else that can fail for reasons unrelated to the book.
"""

from __future__ import annotations

#: fragment name -> rows of (label, result file stem)
SCORECARDS: dict[str, list[tuple[str, str]]] = {
    "ch01-baseline": [
        ("Naive, 1 req/s", "naive-rate1-tier1"),
        ("Naive, 4 req/s", "naive-rate4-tier1"),
        ("Naive, 8 req/s", "naive-rate8-tier1"),
        ("Naive, 16 req/s", "naive-rate16-tier1"),
    ],
    "ch02-goodput-collapse": [
        ("Naive, 4 req/s", "naive-rate4-tier1"),
        ("Naive, 8 req/s", "naive-rate8-tier1"),
        ("Naive, 16 req/s", "naive-rate16-tier1"),
    ],
    "ch05-kv-cache": [
        ("ch01 Naive, 1 req/s", "naive-rate1-tier1"),
        ("ch05 KV cache, 1 req/s", "cached-rate1-tier1"),
        ("ch01 Naive, 8 req/s", "naive-rate8-tier1"),
        ("ch05 KV cache, 8 req/s", "cached-rate8-tier1"),
        ("ch01 Naive, 16 req/s", "naive-rate16-tier1"),
        ("ch05 KV cache, 16 req/s", "cached-rate16-tier1"),
    ],
    "ch06-static-batching": [
        ("ch05 One at a time, 4 req/s", "cached-rate4-tier1"),
        ("ch06 Static batch, 4 req/s", "static-rate4-tier1"),
        ("ch05 One at a time, 8 req/s", "cached-rate8-tier1"),
        ("ch06 Static batch, 8 req/s", "static-rate8-tier1"),
        ("ch05 One at a time, 16 req/s", "cached-rate16-tier1"),
        ("ch06 Static batch, 16 req/s", "static-rate16-tier1"),
    ],
    "ch07-continuous-batching": [
        ("ch06 Static batch, 4 req/s", "static-rate4-tier1"),
        ("ch07 Continuous, 4 req/s", "continuous-rate4-tier1"),
        ("ch06 Static batch, 8 req/s", "static-rate8-tier1"),
        ("ch07 Continuous, 8 req/s", "continuous-rate8-tier1"),
        ("ch06 Static batch, 16 req/s", "static-rate16-tier1"),
        ("ch07 Continuous, 16 req/s", "continuous-rate16-tier1"),
    ],
    "ch08-paged": [
        ("ch07 Continuous, 4 req/s", "continuous-rate4-tier1"),
        ("ch08 Paged, 4 req/s", "paged-rate4-tier1"),
        ("ch07 Continuous, 8 req/s", "continuous-rate8-tier1"),
        ("ch08 Paged, 8 req/s", "paged-rate8-tier1"),
        ("ch07 Continuous, 16 req/s", "continuous-rate16-tier1"),
        ("ch08 Paged, 16 req/s", "paged-rate16-tier1"),
    ],
    "ch09-prefix-caching": [
        ("ch08 Paged, chat 4 req/s", "paged-chat-rate4-tier1"),
        ("ch09 Prefix cache, chat 4 req/s", "prefix-chat-rate4-tier1"),
        ("ch08 Paged, chat 8 req/s", "paged-chat-rate8-tier1"),
        ("ch09 Prefix cache, chat 8 req/s", "prefix-chat-rate8-tier1"),
        ("ch08 Paged, chat 16 req/s", "paged-chat-rate16-tier1"),
        ("ch09 Prefix cache, chat 16 req/s", "prefix-chat-rate16-tier1"),
    ],
    "ch10-token-budget": [
        ("Budget 64 tokens/step", "chunked-budget64-tier1"),
        ("Budget 512 tokens/step", "chunked-budget512-tier1"),
        ("Budget 8192 (no chunking)", "chunked-budget8192-tier1"),
    ],
    # The running scorecard: the whole journey so far, at one saturating rate.
    "ch07-running-scorecard": [
        ("ch01 Naive", "naive-rate16-tier1"),
        ("ch05 KV cache", "cached-rate16-tier1"),
        ("ch06 Static batching", "static-rate16-tier1"),
        ("ch07 Continuous batching", "continuous-rate16-tier1"),
    ],
}

#: fragment name -> result file whose conditions line to print
CONDITIONS: dict[str, str] = {
    "ch01-baseline": "naive-rate1-tier1",
    "ch02-goodput-collapse": "naive-rate4-tier1",
    "ch03-arithmetic": "cached-rate1-tier1",
    "ch05-kv-cache": "cached-rate8-tier1",
    "ch06-static-batching": "static-rate8-tier1",
    "ch07-continuous-batching": "continuous-rate8-tier1",
    "ch07-running-scorecard": "continuous-rate16-tier1",
    "ch08-paged": "paged-rate16-tier1",
    "ch09-prefix-caching": "prefix-chat-rate8-tier1",
    "ch10-token-budget": "chunked-budget512-tier1",
}
