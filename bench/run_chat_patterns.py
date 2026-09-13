"""Generate the chapter 20 results: what changes when the workload is a conversation.

Run with ``python -m bench.run_chat_patterns``. Two measurements.

The turn sweep shows how prefix reuse behaves as a conversation deepens — the transcript grows,
and almost all of the growth is text the engine has already seen.

The batch sweep shows the trade chat actually cares about. Every other chapter has treated a larger
batch as straightforwardly better; a chat client streams tokens to a human, so inter-token latency
is a user-visible property and the batch size that maximises throughput is not the one that serves
a conversation well.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, SLO, run_benchmark, stamped_payload
from bench.traces import make_chat_trace, make_session_turns
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams


def _turn_sweep(model, args) -> list[dict]:
    """Serve every session's turn 1, then turn 2, and so on, measuring reuse at each depth."""
    turns = make_session_turns(args.sessions, args.turns, seed=3)
    engine = PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=2048, block_size=16)
    rows = []

    for depth, prompts in enumerate(turns, start=1):
        before_hits = engine.prefix.hit_tokens
        before_total = engine.prefix.total_prompt_tokens
        start = time.perf_counter()
        for prompt in prompts:
            engine.add_request(
                Request(prompt_token_ids=list(prompt), params=SamplingParams(max_tokens=8))
            )
        while engine.has_work():
            engine.step()
        elapsed = time.perf_counter() - start

        served = engine.prefix.total_prompt_tokens - before_total
        reused = engine.prefix.hit_tokens - before_hits
        rows.append(
            {
                "turn": depth,
                "prompt_tokens": round(served / len(prompts), 1),
                "reuse": round(reused / served if served else 0.0, 4),
                "ms_per_request": round(elapsed * 1000 / len(prompts), 1),
            }
        )
    return rows


def _batch_sweep(model, args, model_meta) -> list[dict]:
    """The same chat trace at several batch-size caps: throughput against inter-token latency."""
    rows = []
    for cap in args.batch_sizes:
        trace = make_chat_trace(args.requests, args.rate, seed=3)
        engine = PrefixCachedEngine(
            model, REFERENCE_MODEL, max_batch_size=cap, n_blocks=1024, block_size=16
        )
        result = run_benchmark(
            engine,
            trace,
            rate_per_second=args.rate,
            slo=SLO(ttft_seconds=1.0, itl_seconds=0.02),
            meta={"model": model_meta, "trace": "chat", "max_batch_size": cap},
        )
        s = result.summary()
        rows.append(
            {
                "max_batch_size": cap,
                "itl_p50": s["itl_p50"],
                "itl_p95": s["itl_p95"],
                "output_tokens_per_second": s["output_tokens_per_second"],
                "goodput_fraction": s["goodput_fraction"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }

    turns = _turn_sweep(model, args)
    for row in turns:
        print(
            f"turn {row['turn']}: prompt~{row['prompt_tokens']:>7.1f} tokens  "
            f"reuse={row['reuse']:.0%}  {row['ms_per_request']:>7.1f} ms/request"
        )

    print()
    batch = _batch_sweep(model, args, model_meta)
    for row in batch:
        print(
            f"batch<={row['max_batch_size']:<3} itl_p95={row['itl_p95']:<8} "
            f"tok/s={row['output_tokens_per_second']:<7} "
            f"goodput_fraction={row['goodput_fraction']}"
        )

    payload = stamped_payload(
        engine="chat-patterns",
        model=model_meta,
        code_sources=["llmserve/engines/prefix.py", "llmserve/engines/paged.py"],
        conditions={
            "trace": f"{args.sessions} sessions x {args.turns} turns; chat trace for the batch sweep",
            "rate_per_second": args.rate,
            "slo": {"ttft_seconds": 1.0, "itl_seconds": 0.02},
        },
        summary={"turns": turns, "batch": batch},
    )
    (RESULTS_DIR / "chat-patterns-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote bench/results/chat-patterns-tier1.json")


if __name__ == "__main__":
    main()
