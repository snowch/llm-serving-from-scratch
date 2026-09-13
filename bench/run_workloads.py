"""Part VI's framing measurement: one engine, one configuration, four workloads.

Run with ``python -m bench.run_workloads``. Nothing about the engine changes between these runs —
same model, same block budget, same batch size, same arrival rate. Only the shape of the traffic
differs, and that is the point of the part: the workload decides what the engine is good at, and no
single configuration is right for all four.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, SLO, run_benchmark
from bench.traces import (
    make_agent_trace,
    make_chat_trace,
    make_completion_trace,
    make_rag_trace,
)
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import build_model

WORKLOADS = {
    "chat": (make_chat_trace, "multi-turn chat: shared system prompt, growing transcript"),
    "rag": (make_rag_trace, "retrieval: shared instruction, then divergent passages"),
    "agent": (make_agent_trace, "agent loop: shared tool preamble, replayed transcript"),
    "completion": (make_completion_trace, "code completion: small prompt, tiny output"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }

    for name, (make_trace, description) in WORKLOADS.items():
        trace = make_trace(args.requests, args.rate, seed=7)
        engine = PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=768, block_size=16)
        result = run_benchmark(
            engine,
            trace,
            rate_per_second=args.rate,
            slo=SLO(ttft_seconds=1.0, itl_seconds=0.05),
            meta={
                "model": model_meta,
                "trace": description,
                "workload": name,
                "n_requests": args.requests,
                "mean_prompt_len": round(sum(s.prompt_len for s in trace) / len(trace), 1),
                "mean_output_len": round(sum(s.max_tokens for s in trace) / len(trace), 1),
                "torch_threads": args.threads,
            },
        )
        prefix = engine.prefix
        result.meta["token_reuse"] = round(
            prefix.hit_tokens / prefix.total_prompt_tokens if prefix.total_prompt_tokens else 0.0, 4
        )
        result.to_json(RESULTS_DIR / f"workload-{name}-tier1.json")

        s = result.summary()
        print(
            f"{name:<11} prompt~{result.meta['mean_prompt_len']:<6} out~"
            f"{result.meta['mean_output_len']:<5} ttft_p95={s['ttft_p95']:<8} "
            f"itl_p95={s['itl_p95']:<8} tok/s={s['output_tokens_per_second']:<7} "
            f"reuse={result.meta['token_reuse']:.0%}"
        )


if __name__ == "__main__":
    main()
