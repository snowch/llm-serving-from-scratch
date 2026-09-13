"""Generate the chapter 21 results: what retrieval does to an engine tuned for chat.

Run with ``python -m bench.run_rag``. Two measurements.

The first is about the prefix cache. Chapter 9 measured it on chat, where it is close to free
throughput; retrieval shares far less, and the number says how much less.

The second is about interference. A retrieval prompt is long enough that prefilling it in one pass
stalls every short request behind it, which is exactly the problem chapter 10 built chunked prefill
for. It is measured here at two arrival rates, because the answer is different on each side of
saturation and reporting only one of them would be the more flattering half of the truth.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

from bench.harness import RESULTS_DIR, SLO, run_benchmark, stamped_payload
from bench.traces import make_completion_trace, make_rag_trace
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.chunked import ChunkedPrefillEngine
from llmserve.model import build_model

#: Prompts at or below this are the interactive traffic sharing the engine with retrieval.
SHORT_PROMPT = 300


def _mixed_trace(n_requests: int, rate: float):
    """Long retrieval prompts and short interactive ones, arriving together."""
    heavy = make_rag_trace(n_requests, rate, seed=13)
    light = make_completion_trace(n_requests, rate, seed=13)
    return sorted([*heavy, *light], key=lambda s: s.arrival)


def _split_latencies(result) -> dict:
    """TTFT percentiles for the short requests and the long ones, separately.

    The aggregate hides the entire effect: chunking makes the long prompts slightly slower and the
    short ones dramatically faster, and a single percentile over both averages that away.
    """
    out = {}
    for label, keep in (
        ("short", lambda r: r.prompt_len <= SHORT_PROMPT),
        ("long", lambda r: r.prompt_len > SHORT_PROMPT),
    ):
        mine = [r for r in result.records if keep(r)]
        ttfts = np.array([r.ttft for r in mine if r.ttft is not None])
        # Inter-token latency is the quantity chunking is *for*: it protects the smoothness of
        # sequences already streaming from being stalled by somebody else's prefill. Reporting only
        # TTFT would miss the effect entirely and then conclude it does not exist.
        itls = np.array([r.itl for r in mine if r.itl is not None])
        out[label] = {
            "requests": len(mine),
            "ttft_p50": round(float(np.percentile(ttfts, 50)), 4) if ttfts.size else None,
            "ttft_p95": round(float(np.percentile(ttfts, 95)), 4) if ttfts.size else None,
            "itl_p95": round(float(np.percentile(itls, 95)), 4) if itls.size else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--rates", type=float, nargs="+", default=[2.0, 8.0])
    parser.add_argument("--budgets", type=int, nargs="+", default=[512, 8192])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }

    rows = []
    for rate in args.rates:
        trace = _mixed_trace(args.requests, rate)
        for budget in args.budgets:
            engine = ChunkedPrefillEngine(
                model, REFERENCE_MODEL, token_budget=budget, n_blocks=2048, block_size=16
            )
            result = run_benchmark(
                engine,
                trace,
                rate_per_second=rate,
                slo=SLO(ttft_seconds=0.5, itl_seconds=0.05),
                meta={"model": model_meta, "trace": "retrieval mixed with completion"},
            )
            s = result.summary()
            rows.append(
                {
                    "rate_per_second": rate,
                    "token_budget": budget,
                    "chunked": budget < 1024,
                    "by_length": _split_latencies(result),
                    "output_tokens_per_second": s["output_tokens_per_second"],
                    "goodput_per_second": s["goodput_per_second"],
                }
            )
            short, long = rows[-1]["by_length"]["short"], rows[-1]["by_length"]["long"]
            print(
                f"rate {rate:<5} budget {budget:<5} short ttft_p95={short['ttft_p95']:<8} "
                f"long ttft_p95={long['ttft_p95']:<8} tok/s={s['output_tokens_per_second']}"
            )

    payload = stamped_payload(
        engine="rag",
        model=model_meta,
        code_sources=["llmserve/engines/chunked.py", "llmserve/engines/paged.py"],
        conditions={
            "trace": f"{2 * args.requests} requests: retrieval prompts mixed with short completions",
            "rates": args.rates,
            "short_prompt_threshold": SHORT_PROMPT,
            "slo": {"ttft_seconds": 0.5, "itl_seconds": 0.05},
        },
        summary={"budgets": rows},
    )
    (RESULTS_DIR / "rag-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote bench/results/rag-tier1.json")


if __name__ == "__main__":
    main()
