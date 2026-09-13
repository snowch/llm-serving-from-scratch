"""Generate the chapter 11 results: one pool versus two.

Run with ``python -m bench.run_disagg``. The comparison is deliberately unflattering to
disaggregation, because a single process cannot run the two pools concurrently — which is the
entire reason the architecture exists. What it can measure honestly is the handoff cost, and that
is what this reports.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, SLO, make_poisson_trace, run_benchmark
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.batched import ContinuousBatchEngine
from llmserve.engines.disaggregated import DisaggregatedEngine
from llmserve.model import build_model

ENGINES = {
    "onepool": lambda model: ContinuousBatchEngine(model),
    "disagg": lambda model: DisaggregatedEngine(model),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--rates", type=float, nargs="+", default=[8, 16])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }

    for rate in args.rates:
        trace = make_poisson_trace(
            args.requests, rate_per_second=rate, prompt_len=(64, 256), output_len=(16, 48), seed=7
        )
        for name, factory in ENGINES.items():
            engine = factory(model)
            result = run_benchmark(
                engine,
                trace,
                rate_per_second=rate,
                slo=SLO(ttft_seconds=1.0, itl_seconds=0.05),
                meta={
                    "model": model_meta,
                    "trace": "poisson, prompt 64-256, output 16-48, seed 7",
                    "n_requests": args.requests,
                    "torch_threads": args.threads,
                },
            )
            if hasattr(engine, "transfer_overhead"):
                result.meta["transfer"] = engine.transfer_overhead
            path = RESULTS_DIR / f"{name}-rate{rate:g}-tier1.json"
            result.to_json(path)
            s = result.summary()
            extra = ""
            if hasattr(engine, "transfer_overhead"):
                o = engine.transfer_overhead
                extra = f" transfer={o['megabytes']}MB in {o['seconds']}s ({o['mb_per_transfer']}MB each)"
            print(
                f"{name:<8} rate={rate:<4g} ttft_p50={s['ttft_p50']:<8} "
                f"tok/s={s['output_tokens_per_second']:<7} goodput={s['goodput_per_second']:<6}{extra}"
            )


if __name__ == "__main__":
    main()
