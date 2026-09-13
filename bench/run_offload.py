"""Generate the chapter 12 results: what a second KV tier is worth.

Run with ``python -m bench.run_offload``. Two halves, and they answer different questions.

The measured half asks whether the mechanism *works*: on a workload that overflows the block pool,
does demoting blocks instead of dropping them keep the reuse chapter 9 built? That question has
nothing to do with how fast the link is, so it is answerable here.

The arithmetic half asks whether it *pays*, which is entirely a question about the link — and on one
machine "host memory" is the same memory, so timing a promotion here would measure a memcpy rather
than a transfer. Chapter 19 has the same shape and handles it the same way.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, SLO, run_benchmark, stamped_payload
from bench.traces import make_multi_tenant_trace
from llmserve.arithmetic import kv_bytes_per_token
from llmserve.cache.offload import break_even_bandwidth, fetch_vs_recompute
from llmserve.config import REFERENCE_MODEL, ModelConfig
from llmserve.engines.offload import OffloadEngine
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import build_model

#: Production-scale model for the arithmetic half.
LLAMA8B = ModelConfig(
    vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, dtype="fp16"
)
LINKS = [
    ("PCIe 5.0 x16", 6.4e10),
    ("PCIe 4.0 x16", 3.2e10),
    ("Local NVMe", 7.0e9),
    ("100 GbE", 1.25e10),
]


def _arithmetic(prefill_tokens_per_second: float) -> dict:
    per_token = kv_bytes_per_token(LLAMA8B)
    rows = []
    for label, bandwidth in LINKS:
        result = fetch_vs_recompute(
            per_token,
            16,
            128,
            link_bytes_per_second=bandwidth,
            prefill_tokens_per_second=prefill_tokens_per_second,
        )
        rows.append({"link": label, "bandwidth": bandwidth, **result})
    return {
        "tiers": rows,
        "kv_bytes_per_token": per_token,
        "prefill_tokens_per_second": prefill_tokens_per_second,
        "break_even_bytes_per_second": break_even_bandwidth(per_token, prefill_tokens_per_second),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--blocks", type=int, default=192, help="deliberately too few")
    parser.add_argument("--prefill-rate", type=float, default=10_000.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }
    # Several prefixes and a block pool too small to hold them: without a second tier the cache
    # spends the run evicting blocks it is about to want again.
    trace = make_multi_tenant_trace(args.requests, rate_per_second=args.rate, seed=12)

    rows = []
    engines = {
        "ch09 Evict and recompute": lambda: PrefixCachedEngine(
            model, REFERENCE_MODEL, n_blocks=args.blocks, block_size=16
        ),
        "ch12 Demote to a second tier": lambda: OffloadEngine(
            model, REFERENCE_MODEL, n_blocks=args.blocks, block_size=16
        ),
    }
    for label, factory in engines.items():
        engine = factory()
        result = run_benchmark(
            engine,
            trace,
            rate_per_second=args.rate,
            slo=SLO(ttft_seconds=1.0, itl_seconds=0.05),
            meta={"model": model_meta, "trace": f"multi-tenant, {args.blocks} blocks"},
        )
        summary = result.summary()
        prefix = engine.prefix
        tier = getattr(engine, "tier", None)
        rows.append(
            {
                "policy": label,
                "token_reuse": round(
                    prefix.hit_tokens / prefix.total_prompt_tokens
                    if prefix.total_prompt_tokens
                    else 0.0,
                    4,
                ),
                "ttft_p95": summary["ttft_p95"],
                "output_tokens_per_second": summary["output_tokens_per_second"],
                "goodput_per_second": summary["goodput_per_second"],
                "promotions": tier.fetches if tier else 0,
                "demotions": tier.stores if tier else 0,
                "tier_hit_rate": round(tier.hit_rate, 4) if tier else None,
            }
        )
        print(
            f"{label:<30} reuse={rows[-1]['token_reuse']:.0%} ttft_p95={summary['ttft_p95']:<8} "
            f"tok/s={summary['output_tokens_per_second']:<7} "
            f"promotions={rows[-1]['promotions']}"
        )

    arithmetic = _arithmetic(args.prefill_rate)
    print(
        f"\nbreak-even link bandwidth: {arithmetic['break_even_bytes_per_second'] / 1e9:.2f} GB/s"
    )
    for row in arithmetic["tiers"]:
        print(
            f"  {row['link']:<14} 2048 tokens: fetch {row['fetch_seconds'] * 1000:6.1f} ms "
            f"vs recompute {row['recompute_seconds'] * 1000:6.1f} ms  ({row['speedup']:.1f}x)"
        )

    payload = stamped_payload(
        engine="offload",
        model=model_meta,
        code_sources=[
            "llmserve/cache/offload.py",
            "llmserve/engines/offload.py",
            "llmserve/cache/prefix.py",
        ],
        conditions={
            "trace": f"multi-tenant, {args.requests} requests, {args.blocks} blocks",
            "rate_per_second": args.rate,
            "slo": {"ttft_seconds": 1.0, "itl_seconds": 0.05},
        },
        summary={"policies": rows, "arithmetic": arithmetic},
    )
    (RESULTS_DIR / "offload-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote bench/results/offload-tier1.json")


if __name__ == "__main__":
    main()
