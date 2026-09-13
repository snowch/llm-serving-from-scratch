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
        ("Naive, 0.5 req/s", "naive-rate0.5-tier1"),
        ("Naive, 1 req/s", "naive-rate1-tier1"),
        ("Naive, 2 req/s", "naive-rate2-tier1"),
        ("Naive, 4 req/s", "naive-rate4-tier1"),
        ("Naive, 8 req/s", "naive-rate8-tier1"),
    ],
    "ch02-goodput-collapse": [
        ("Naive, 2 req/s", "naive-rate2-tier1"),
        ("Naive, 4 req/s", "naive-rate4-tier1"),
        ("Naive, 8 req/s", "naive-rate8-tier1"),
    ],
    "ch05-kv-cache": [
        ("ch01 Naive, 1 req/s", "naive-rate1-tier1"),
        ("ch05 KV cache, 1 req/s", "cached-rate1-tier1"),
        ("ch01 Naive, 4 req/s", "naive-rate4-tier1"),
        ("ch05 KV cache, 4 req/s", "cached-rate4-tier1"),
        ("ch01 Naive, 8 req/s", "naive-rate8-tier1"),
        ("ch05 KV cache, 8 req/s", "cached-rate8-tier1"),
    ],
}

#: fragment name -> result file whose conditions line to print
CONDITIONS: dict[str, str] = {
    "ch01-baseline": "naive-rate1-tier1",
    "ch03-arithmetic": "cached-rate1-tier1",
    "ch02-goodput-collapse": "naive-rate4-tier1",
    "ch05-kv-cache": "cached-rate4-tier1",
}
