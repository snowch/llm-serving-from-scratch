"""Generate the chapter 19 fairness results: one noisy tenant, two quiet ones.

Run with ``python -m bench.run_fairness``. The same trace served twice — once by the chapter 9
engine, whose queue is one line, and once by the chapter 19 engine, whose queue is one line per
tenant. Nothing else differs, so any change in the quiet tenants' latency is the scheduler.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, SLO, run_benchmark
from bench.traces import make_noisy_neighbour_trace
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.engines.tenant import TenantFairEngine
from llmserve.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }
    trace = make_noisy_neighbour_trace(args.rate, seed=11)
    tenants = sorted({spec.tenant for spec in trace if spec.tenant})

    engines = {
        "fifo": lambda: PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=512, block_size=16),
        "fair": lambda: TenantFairEngine(model, REFERENCE_MODEL, n_blocks=512, block_size=16),
    }

    for label, factory in engines.items():
        engine = factory()
        result = run_benchmark(
            engine,
            trace,
            rate_per_second=args.rate,
            slo=SLO(ttft_seconds=1.0, itl_seconds=0.05),
            meta={
                "model": model_meta,
                "trace": f"noisy neighbour: {len(trace)} requests across {len(tenants)} tenants",
                "scheduler": label,
                "torch_threads": args.threads,
            },
        )
        if hasattr(engine, "service_share"):
            result.meta["service_share"] = engine.service_share
        result.to_json(RESULTS_DIR / f"tenants-{label}-tier1.json")

        summary = result.summary()
        print(f"{label:<5} overall ttft_p95={summary['ttft_p95']}")
        for tenant, row in sorted(summary.get("by_tenant", {}).items()):
            print(
                f"      {tenant:<10} n={row['requests']:<3} ttft_p50={row['ttft_p50']:<8} "
                f"ttft_p95={row['ttft_p95']:<8} goodput_fraction={row['goodput_fraction']}"
            )


if __name__ == "__main__":
    main()
