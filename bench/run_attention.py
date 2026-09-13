"""Generate the chapter 12 results: what grouped-query attention costs and saves.

Run with ``python -m bench.run_attention``. Varies only ``n_kv_heads`` — from full multi-head
attention down to multi-query — and measures both the KV cache footprint and what it does to
serving. The model is otherwise identical in every run, so the difference is attributable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace

import torch

from bench.harness import RESULTS_DIR, SLO, make_poisson_trace, run_benchmark
from llmserve.arithmetic import kv_bytes_per_token
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.batched import ContinuousBatchEngine
from llmserve.model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--kv-heads", type=int, nargs="+", default=[8, 4, 2, 1])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    trace = make_poisson_trace(
        args.requests, rate_per_second=args.rate, prompt_len=(32, 96), output_len=(16, 48), seed=7
    )

    footprint = []
    for n_kv in args.kv_heads:
        config = replace(REFERENCE_MODEL, n_kv_heads=n_kv)
        model = build_model(config)
        label = {8: "MHA", 1: "MQA"}.get(n_kv, f"GQA {config.group_size}:1")
        model_meta = {
            "name": f"TinyGPT, {n_kv} KV heads ({label})",
            "params": sum(p.numel() for p in model.parameters()),
            **asdict(config),
        }
        result = run_benchmark(
            ContinuousBatchEngine(model),
            trace,
            rate_per_second=args.rate,
            slo=SLO(ttft_seconds=1.0, itl_seconds=0.05),
            meta={
                "model": model_meta,
                "trace": "poisson, prompt 32-96, output 16-48, seed 7",
                "n_requests": args.requests,
                "n_kv_heads": n_kv,
                "kv_bytes_per_token": kv_bytes_per_token(config),
                "torch_threads": args.threads,
            },
        )
        result.to_json(RESULTS_DIR / f"attn-kv{n_kv}-tier1.json")
        s = result.summary()
        footprint.append(
            {
                "label": label,
                "n_kv_heads": n_kv,
                "params": model_meta["params"],
                "kv_bytes_per_token": kv_bytes_per_token(config),
            }
        )
        print(
            f"{label:<8} kv_heads={n_kv}  kv/token={kv_bytes_per_token(config):>6.0f} B  "
            f"params={model_meta['params']:>9,}  tok/s={s['output_tokens_per_second']:<7} "
            f"goodput={s['goodput_per_second']}"
        )


if __name__ == "__main__":
    main()
