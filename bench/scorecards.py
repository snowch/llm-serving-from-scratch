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
    "ch11-disaggregation": [
        ("ch07 One pool, 8 req/s", "onepool-rate8-tier1"),
        ("ch11 Two pools, 8 req/s", "disagg-rate8-tier1"),
        ("ch07 One pool, 16 req/s", "onepool-rate16-tier1"),
        ("ch11 Two pools, 16 req/s", "disagg-rate16-tier1"),
    ],
    "ch13-gqa": [
        ("MHA, 8 KV heads", "attn-kv8-tier1"),
        ("GQA 2:1, 4 KV heads", "attn-kv4-tier1"),
        ("GQA 4:1, 2 KV heads", "attn-kv2-tier1"),
        ("MQA, 1 KV head", "attn-kv1-tier1"),
    ],
    # The running scorecard: the whole journey so far, at one saturating rate.
    "ch07-running-scorecard": [
        ("ch01 Naive", "naive-rate16-tier1"),
        ("ch05 KV cache", "cached-rate16-tier1"),
        ("ch06 Static batching", "static-rate16-tier1"),
        ("ch07 Continuous batching", "continuous-rate16-tier1"),
    ],
}

#: fragment name -> result files a *derived* fragment reads.
#:
#: Derived fragments compute their rows rather than tabulating a result set, so they do not appear
#: in ``SCORECARDS``. They still rest on stamped results, and without this declaration those
#: results would escape ``verify-numbers.py`` entirely: nothing would notice that a quantisation
#: or speculation figure was produced by code that has since changed. Fragments computed purely
#: from the model config — chapter 8's memory table, chapter 11's handoff table, chapter 13's
#: footprint and score-matrix tables — read no results and are correctly absent.
DERIVED_SOURCES: dict[str, list[str]] = {
    "ch03-arithmetic": ["cached-rate1-tier1", "naive-rate1-tier1"],
    "ch10-chunk-cost": ["chunk-cost-tier1"],
    "ch12-offload": ["offload-tier1"],
    "ch12-tiers": ["offload-tier1"],
    "ch15-quantisation": ["quant-tier1"],
    "ch15-kv-quantisation": ["quant-tier1"],
    "ch15-outliers": ["quant-tier1"],
    "ch16-window": ["window-tier1"],
    "ch16-sinks": ["window-tier1"],
    "ch16-window-memory": ["window-tier1"],
    "ch17-acceptance": ["speculative-tier1"],
    "ch17-distribution": ["speculative-tier1"],
    "ch18-validity": ["constrained-tier1"],
    "ch18-mask-cost": ["constrained-tier1"],
    "ch20-routing": [
        "router-round-robin-tier1",
        "router-least-tokens-tier1",
        "router-prefix-affinity-tier1",
    ],
    "ch21-quality": ["lora-tier1"],
    "ch21-batch": ["lora-tier1"],
    "ch21-fairness": ["tenants-fifo-tier1", "tenants-fair-tier1"],
    "ch22-workloads": [
        "workload-chat-tier1",
        "workload-rag-tier1",
        "workload-agent-tier1",
        "workload-completion-tier1",
    ],
    "ch22-turns": ["chat-patterns-tier1"],
    "ch22-batch": ["chat-patterns-tier1"],
    "ch23-interference": ["rag-tier1"],
    "ch24-chains": ["agents-tier1"],
    "ch24-cancellation": ["agents-tier1"],
    "ch25-completion": ["offline-tier1"],
    "ch25-offline": ["offline-tier1"],
    "ch26-templates": ["api-tier1"],
    "ch27-signals": ["observability-tier1"],
    "ch27-overhead": ["observability-tier1"],
    "ch28-shedding": ["reliability-tier1"],
    "ch28-drain": ["reliability-tier1"],
    "ch31-generators": ["benchmarking-tier1"],
    "ch32-journey": [
        "naive-rate16-tier1",
        "cached-rate16-tier1",
        "static-rate16-tier1",
        "continuous-rate16-tier1",
        "paged-rate16-tier1",
        "paged-chat-rate8-tier1",
        "prefix-chat-rate8-tier1",
        "chunked-budget512-tier1",
        "offload-tier1",
        "disagg-rate16-tier1",
        "router-prefix-affinity-tier1",
        "tenants-fair-tier1",
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
    "ch11-disaggregation": "disagg-rate16-tier1",
    "ch13-gqa": "attn-kv2-tier1",
    "ch20-routing": "router-prefix-affinity-tier1",
    "ch21-quality": "lora-tier1",
    "ch21-fairness": "tenants-fair-tier1",
    "ch22-workloads": "workload-chat-tier1",
    "ch22-batch": "chat-patterns-tier1",
    "ch23-interference": "rag-tier1",
    "ch24-chains": "agents-tier1",
    "ch25-offline": "offline-tier1",
    "ch26-templates": "api-tier1",
    "ch27-signals": "observability-tier1",
    "ch28-shedding": "reliability-tier1",
    "ch31-generators": "benchmarking-tier1",
}
