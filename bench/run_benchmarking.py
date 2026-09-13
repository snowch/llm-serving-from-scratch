"""Generate the chapter 28 results: the same engine, measured two ways, reported differently.

Run with ``python -m bench.run_benchmarking``. One engine, one workload, two load generators — the
open loop this book has used since chapter 2, and the closed loop most load tests are written with.
The gap between the two numbers is coordinated omission, and it is not small.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from bench.closed_loop import run_closed_loop
from bench.harness import RESULTS_DIR, SLO, make_poisson_trace, run_benchmark, stamped_payload
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import build_model

SERVING_SLO = SLO(ttft_seconds=0.5, itl_seconds=0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--rate", type=float, default=24.0)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }
    trace = make_poisson_trace(args.requests, args.rate, seed=37)

    rows = []
    engine = PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=512, block_size=16)
    open_loop = run_benchmark(
        engine,
        trace,
        rate_per_second=args.rate,
        slo=SERVING_SLO,
        meta={"model": model_meta, "trace": "open loop at a saturating rate"},
    )
    s = open_loop.summary()
    rows.append(
        {
            "generator": "Open loop",
            "detail": f"{args.rate:.0f} req/s offered",
            "ttft_p50": s["ttft_p50"],
            "ttft_p95": s["ttft_p95"],
            "ttft_p99": s["ttft_p99"],
            "output_tokens_per_second": s["output_tokens_per_second"],
            "requests_per_second": s["requests_per_second"],
        }
    )
    print(
        f"open loop   offered={args.rate:<5} ttft_p95={s['ttft_p95']:<8} "
        f"achieved={s['requests_per_second']} req/s"
    )

    for workers in args.workers:
        engine = PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=512, block_size=16)
        closed = run_closed_loop(
            engine, trace, n_workers=workers, slo=SERVING_SLO, meta={"model": model_meta}
        )
        s = closed.summary()
        rows.append(
            {
                "generator": "Closed loop",
                "detail": f"{workers} workers",
                "ttft_p50": s["ttft_p50"],
                "ttft_p95": s["ttft_p95"],
                "ttft_p99": s["ttft_p99"],
                "output_tokens_per_second": s["output_tokens_per_second"],
                "requests_per_second": s["requests_per_second"],
            }
        )
        print(
            f"closed loop workers={workers:<4} ttft_p95={s['ttft_p95']:<8} "
            f"achieved={s['requests_per_second']} req/s"
        )

    payload = stamped_payload(
        engine="benchmarking",
        model=model_meta,
        code_sources=[
            "bench/closed_loop.py",
            "llmserve/engines/prefix.py",
            "llmserve/engines/paged.py",
        ],
        conditions={
            "trace": f"{args.requests} requests, same workload under both generators",
            "rate_per_second": args.rate,
            "slo": asdict(SERVING_SLO),
        },
        summary={"generators": rows},
    )
    (RESULTS_DIR / "benchmarking-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote bench/results/benchmarking-tier1.json")


if __name__ == "__main__":
    main()
