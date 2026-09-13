"""Generate the chapter 25 results: which signal moves first, and what instrumentation costs.

Run with ``python -m bench.run_observability``. The engine is driven into overload while sampling
itself every step, so the ordering of the signals is visible: the claim is that queue depth rises
before latency does, and a claim about ordering needs both series on one axis.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import torch

from bench.harness import RESULTS_DIR, make_poisson_trace, stamped_payload
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.metrics import Metrics, leading_indicator_windows
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams


def _serve_with_metrics(model, trace, rate, *, instrumented: bool):
    """Serve an open-loop trace, optionally sampling the engine on every step."""
    engine = PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=512, block_size=16)
    metrics = Metrics() if instrumented else None
    records: dict[int, dict] = {}
    index = 0
    start = time.perf_counter()

    while index < len(trace) or engine.has_work():
        now = time.perf_counter() - start
        while index < len(trace) and trace[index].arrival <= now:
            spec = trace[index]
            request = Request(
                prompt_token_ids=[(i % 250) + 1 for i in range(spec.prompt_len)],
                params=SamplingParams(max_tokens=spec.max_tokens, seed=0),
            )
            engine.add_request(request)
            records[request.request_id] = {"arrival": now, "first": None}
            index += 1

        if not engine.has_work():
            time.sleep(0.001)
            continue

        outputs = engine.step()
        stamp = time.perf_counter() - start
        for out in outputs:
            record = records[out.request_id]
            if record["first"] is None and out.token_ids:
                record["first"] = stamp
                if metrics:
                    metrics.observe_request(stamp - record["arrival"], None)
        if metrics:
            metrics.observe_step(engine, outputs)

    return engine, metrics, time.perf_counter() - start, records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--rate", type=float, default=24.0)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }
    trace = make_poisson_trace(args.requests, args.rate, seed=29)

    engine, metrics, wall, records = _serve_with_metrics(model, trace, args.rate, instrumented=True)
    windows = leading_indicator_windows(metrics.snapshots, [], window=args.window)

    # TTFT for the requests whose first token landed in each window, on the same step axis.
    step_of = {}
    for snapshot in metrics.snapshots:
        step_of[snapshot.step] = snapshot
    ttfts = sorted(
        (r["first"], r["first"] - r["arrival"]) for r in records.values() if r["first"] is not None
    )
    per_window = np.array_split(np.array([t for _, t in ttfts]), max(len(windows), 1))
    for row, chunk in zip(windows, per_window, strict=False):
        row["ttft_p95"] = round(float(np.percentile(chunk, 95)), 4) if chunk.size else None

    for row in windows:
        print(
            f"steps {row['step']:>4}+  queue~{row['mean_queue_depth']:<6} "
            f"peak={row['peak_queue_depth']:<3} kv={row['kv_utilisation']:<6} "
            f"ttft_p95={row['ttft_p95']}"
        )

    # What the instrumentation costs, measured by running the same trace without it.
    _, _, bare_wall, _ = _serve_with_metrics(model, trace, args.rate, instrumented=False)
    overhead = {
        "instrumented_wall": round(wall, 3),
        "bare_wall": round(bare_wall, 3),
        "overhead_fraction": round(wall / bare_wall - 1, 4),
    }
    print(f"\ninstrumentation overhead: {overhead['overhead_fraction']:+.2%}")

    payload = stamped_payload(
        engine="observability",
        model=model_meta,
        code_sources=["llmserve/metrics.py", "llmserve/engines/prefix.py"],
        conditions={
            "trace": f"{args.requests} requests at {args.rate}/s, deliberately beyond capacity",
            "rate_per_second": args.rate,
            "window_steps": args.window,
        },
        summary={"report": metrics.report(), "windows": windows, "overhead": overhead},
    )
    (RESULTS_DIR / "observability-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote bench/results/observability-tier1.json")


if __name__ == "__main__":
    main()
