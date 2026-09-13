"""Generate the chapter 22 results: agent traffic, chained latency, and abandoned work.

Run with ``python -m bench.run_agents``. Three measurements.

Chain latency is the one that changes how you read every other chapter in this book. An agent run is
a *sequence* of requests and the user waits for all of them, so what matters is the distribution of
the sum rather than of one call — and sums behave very differently from what "optimise the tail"
would lead you to expect.

Cancellation measures the work an engine does for callers who have already left.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np
import torch

from bench.harness import RESULTS_DIR, SLO, run_benchmark, stamped_payload
from bench.traces import make_agent_trace
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.cancel import CancellableEngine
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams


def _chain_amplification(result, lengths, samples=20_000, seed=0) -> list[dict]:
    """Resample the measured per-call latencies into chains of N sequential calls.

    Sampling with replacement from the observed distribution, rather than assuming one, because the
    conclusion is about the shape of the real distribution and not about a model of it.
    """
    observed = np.array([r.e2e for r in result.records if r.e2e is not None])
    rng = np.random.default_rng(seed)
    rows = []
    for n in lengths:
        totals = rng.choice(observed, size=(samples, n), replace=True).sum(axis=1)
        p50 = float(np.percentile(totals, 50))
        rows.append(
            {
                "calls": n,
                "p50": round(p50, 4),
                "p95": round(float(np.percentile(totals, 95)), 4),
                "p99": round(float(np.percentile(totals, 99)), 4),
                # How far the tail sits above the middle. It *falls* as the chain lengthens.
                "p95_over_p50": round(float(np.percentile(totals, 95)) / p50, 3),
            }
        )
    return rows


def _cancellation(model, args) -> dict:
    """Serve an agent trace, abandoning a fraction of requests part-way through.

    Driven directly rather than through the harness, because the harness models a client that waits
    and this measurement is about one that does not.
    """
    trace = make_agent_trace(args.requests, args.rate, seed=17)
    rng = np.random.default_rng(19)
    doomed = set(rng.choice(len(trace), size=len(trace) // 3, replace=False).tolist())

    results = {}
    for honour_cancellation in (False, True):
        engine = CancellableEngine(model, REFERENCE_MODEL, n_blocks=2048, block_size=16)
        pending = {}
        for index, spec in enumerate(trace):
            request = Request(
                prompt_token_ids=list(spec.tokens),
                params=SamplingParams(max_tokens=spec.max_tokens),
            )
            engine.add_request(request)
            if index in doomed:
                pending[request.request_id] = spec.max_tokens // 2

        steps = 0
        produced = 0
        while engine.has_work():
            outputs = engine.step()
            steps += 1
            for out in outputs:
                produced += len(out.token_ids)
            if honour_cancellation:
                for request_id in list(pending):
                    pending[request_id] -= 1
                    if pending[request_id] <= 0:
                        engine.abort(request_id)
                        pending.pop(request_id)

        results["with_cancellation" if honour_cancellation else "without"] = {
            "steps": steps,
            "tokens_produced": produced,
            "aborted": engine.aborted,
        }

    without, with_cancel = results["without"], results["with_cancellation"]
    results["steps_saved"] = round(1 - with_cancel["steps"] / without["steps"], 4)
    results["abandoned_fraction"] = round(len(doomed) / len(trace), 4)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--chain-lengths", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }

    trace = make_agent_trace(args.requests, args.rate, seed=17)
    engine = CancellableEngine(model, REFERENCE_MODEL, n_blocks=2048, block_size=16)
    result = run_benchmark(
        engine,
        trace,
        rate_per_second=args.rate,
        slo=SLO(ttft_seconds=0.5, itl_seconds=0.05),
        meta={"model": model_meta, "trace": "agent loop"},
    )
    summary = result.summary()
    prefix = engine.prefix
    reuse = prefix.hit_tokens / prefix.total_prompt_tokens if prefix.total_prompt_tokens else 0.0
    print(f"per call: ttft_p95={summary['ttft_p95']}  e2e reuse={reuse:.0%}")

    chains = _chain_amplification(result, args.chain_lengths)
    for row in chains:
        print(f"  chain of {row['calls']:>2}: p50={row['p50']:<8} p95={row['p95']:<8}")

    cancellation = _cancellation(model, args)
    print(
        f"\ncancellation: {cancellation['steps_saved']:.0%} fewer steps when "
        f"{cancellation['abandoned_fraction']:.0%} of requests are abandoned"
    )

    payload = stamped_payload(
        engine="agents",
        model=model_meta,
        code_sources=["llmserve/engines/cancel.py", "llmserve/engines/prefix.py"],
        conditions={
            "trace": f"agent loop, {args.requests} calls",
            "rate_per_second": args.rate,
            "slo": {"ttft_seconds": 0.5, "itl_seconds": 0.05},
        },
        summary={
            "per_call": summary,
            "token_reuse": round(reuse, 4),
            "chains": chains,
            "cancellation": cancellation,
        },
    )
    (RESULTS_DIR / "agents-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote bench/results/agents-tier1.json")


if __name__ == "__main__":
    main()
