"""Generate the chapter 23 results: the two extremes of the latency/throughput axis.

Run with ``python -m bench.run_offline``. Code completion and offline batch are the same engine
serving requests of similar size, and almost every configuration choice comes out opposite.

Completion is measured against a deliberately brutal objective, because a suggestion that arrives
after the developer typed the next character is worth nothing. Offline batch is measured with no
objective at all, because there is nobody waiting — only tokens per unit of hardware time.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, SLO, run_benchmark, stamped_payload
from bench.traces import make_completion_trace, make_offline_batch_trace
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams

#: What a completion has to beat to be useful at all.
COMPLETION_SLO = SLO(ttft_seconds=0.05, itl_seconds=0.05)


def _completion_sweep(model, args, model_meta) -> list[dict]:
    """Completion traffic at several batch caps, against a tight time-to-first-token objective."""
    rows = []
    for cap in args.batch_sizes:
        trace = make_completion_trace(args.requests, args.rate, seed=23)
        engine = PrefixCachedEngine(
            model, REFERENCE_MODEL, max_batch_size=cap, n_blocks=1024, block_size=16
        )
        result = run_benchmark(
            engine,
            trace,
            rate_per_second=args.rate,
            slo=COMPLETION_SLO,
            meta={"model": model_meta, "trace": "code completion"},
        )
        s = result.summary()
        rows.append(
            {
                "max_batch_size": cap,
                "ttft_p50": s["ttft_p50"],
                "ttft_p95": s["ttft_p95"],
                "output_tokens_per_second": s["output_tokens_per_second"],
                "goodput_fraction": s["goodput_fraction"],
            }
        )
    return rows


def _offline_sweep(model, args) -> list[dict]:
    """Offline batch at several batch caps. No arrivals, no SLO, one number that matters.

    Driven directly rather than through the harness: the harness is open-loop because a real client
    does not wait, and offline batch has no client at all. Feeding it a schedule of arrivals would
    be measuring a fiction.
    """
    rows = []
    for cap in args.batch_sizes:
        trace = make_offline_batch_trace(args.batch_requests, seed=23)
        engine = PrefixCachedEngine(
            model, REFERENCE_MODEL, max_batch_size=cap, n_blocks=4096, block_size=16
        )
        for spec in trace:
            engine.add_request(
                Request(
                    prompt_token_ids=[(i % 250) + 1 for i in range(spec.prompt_len)],
                    params=SamplingParams(max_tokens=spec.max_tokens),
                )
            )
        produced = 0
        start = time.perf_counter()
        while engine.has_work():
            for out in engine.step():
                produced += len(out.token_ids)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "max_batch_size": cap,
                "wall_time": round(elapsed, 3),
                "output_tokens_per_second": round(produced / elapsed, 2),
                "peak_kv_utilisation": round(engine.peak_kv_utilisation, 3),
                "preemptions": engine.preemptions,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--batch-requests", type=int, default=32)
    parser.add_argument("--rate", type=float, default=16.0)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }

    completion = _completion_sweep(model, args, model_meta)
    for row in completion:
        print(
            f"completion batch<={row['max_batch_size']:<3} ttft_p95={row['ttft_p95']:<8} "
            f"tok/s={row['output_tokens_per_second']:<7} "
            f"met_slo={row['goodput_fraction']}"
        )

    print()
    offline = _offline_sweep(model, args)
    for row in offline:
        print(
            f"offline    batch<={row['max_batch_size']:<3} "
            f"tok/s={row['output_tokens_per_second']:<7} "
            f"kv_peak={row['peak_kv_utilisation']:<6} preemptions={row['preemptions']}"
        )

    payload = stamped_payload(
        engine="offline",
        model=model_meta,
        code_sources=["llmserve/engines/prefix.py", "llmserve/engines/paged.py"],
        conditions={
            "trace": "code completion at a fixed rate; offline batch with every request at t=0",
            "rate_per_second": args.rate,
            "batch_requests": args.batch_requests,
            "completion_slo": asdict(COMPLETION_SLO),
        },
        summary={"completion": completion, "offline": offline},
    )
    (RESULTS_DIR / "offline-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote bench/results/offline-tier1.json")


if __name__ == "__main__":
    main()
