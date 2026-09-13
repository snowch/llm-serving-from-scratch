"""Generate the result files chapters 1-5 cite.

Run with ``python -m bench.run_v01``. Every figure those chapters quote comes from the JSON this
writes, never from prose (PLAN.md §6.3), so re-running this is how the book stays honest when
the code underneath it changes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, SLO, make_poisson_trace, run_benchmark
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.naive import CachedEngine, NaiveEngine
from llmserve.model import build_model

ENGINES = {"naive": NaiveEngine, "cached": CachedEngine}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tag", default="tier1")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model = build_model(REFERENCE_MODEL)
    n_params = sum(p.numel() for p in model.parameters())
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": n_params,
        **asdict(REFERENCE_MODEL),
    }

    for rate in args.rates:
        trace = make_poisson_trace(
            args.requests,
            rate_per_second=rate,
            prompt_len=(32, 96),
            output_len=(16, 48),
            seed=7,
        )
        for name, cls in ENGINES.items():
            result = run_benchmark(
                cls(model),
                trace,
                rate_per_second=rate,
                slo=SLO(ttft_seconds=1.0, itl_seconds=0.05),
                meta={
                    "model": model_meta,
                    "trace": "poisson, prompt 32-96, output 16-48, seed 7",
                    "n_requests": args.requests,
                    "torch_threads": args.threads,
                },
            )
            path = RESULTS_DIR / f"{name}-rate{rate:g}-{args.tag}.json"
            result.to_json(path)
            s = result.summary()
            print(
                f"{name:<7} rate={rate:<4g} "
                f"ttft_p50={s['ttft_p50']:<7} ttft_p95={s['ttft_p95']:<7} "
                f"itl_p50={s['itl_p50']:<7} tok/s={s['output_tokens_per_second']:<7} "
                f"goodput={s['goodput_per_second']}"
            )


if __name__ == "__main__":
    main()
