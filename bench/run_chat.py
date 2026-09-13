"""Generate the chapter 9 results: paged serving with and without prefix reuse.

Run with ``python -m bench.run_chat``. Uses the chat trace, where every request shares one system
prompt — the workload prefix caching exists for. On the uniform-random trace used by chapters 1-8
there is nothing to reuse and the two engines are identical, which is itself worth knowing.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, SLO, run_benchmark
from bench.traces import make_chat_trace
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.paged import PagedEngine
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import build_model

ENGINES = {
    "paged": lambda model, cfg: PagedEngine(model, cfg, n_blocks=512, block_size=16),
    "prefix": lambda model, cfg: PrefixCachedEngine(model, cfg, n_blocks=512, block_size=16),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--rates", type=float, nargs="+", default=[4, 8, 16])
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
        trace = make_chat_trace(args.requests, rate_per_second=rate, seed=7)
        for name, factory in ENGINES.items():
            engine = factory(model, REFERENCE_MODEL)
            result = run_benchmark(
                engine,
                trace,
                rate_per_second=rate,
                slo=SLO(ttft_seconds=1.0, itl_seconds=0.05),
                meta={
                    "model": model_meta,
                    "trace": "chat: shared system prompt, unique question, output 16-48, seed 7",
                    "n_requests": args.requests,
                    "torch_threads": args.threads,
                },
            )
            if hasattr(engine, "prefix"):
                result.meta["prefix_hit_rate"] = round(engine.prefix.hit_rate, 3)
                result.meta["prefix_token_reuse"] = round(engine.prefix.token_reuse_rate, 3)
            path = RESULTS_DIR / f"{name}-chat-rate{rate:g}-tier1.json"
            result.to_json(path)
            s = result.summary()
            reuse = result.meta.get("prefix_token_reuse", 0.0)
            print(
                f"{name:<7} rate={rate:<4g} ttft_p50={s['ttft_p50']:<8} "
                f"tok/s={s['output_tokens_per_second']:<8} goodput={s['goodput_per_second']:<7} "
                f"token_reuse={reuse:.0%}"
            )


if __name__ == "__main__":
    main()
