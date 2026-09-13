"""Generate the chapter 26 results: what refusing work buys, and what draining costs.

Run with ``python -m bench.run_reliability``. The same overload served twice — once accepting
everything, once with admission control — and then a drain, to show that finishing in-flight work
is cheap and dropping it is not.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, SLO, make_poisson_trace, run_benchmark, stamped_payload
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.engines.shedding import SheddingEngine
from llmserve.model import build_model
from llmserve.request import Request
from llmserve.sampling import SamplingParams

SERVING_SLO = SLO(ttft_seconds=0.5, itl_seconds=0.05)


def _drain_cost(model, args) -> dict:
    """How much work is in flight when a drain starts, and what dropping it would throw away."""
    engine = SheddingEngine(model, REFERENCE_MODEL, n_blocks=512, block_size=16)
    for _ in range(args.drain_requests):
        engine.add_request(
            Request(
                prompt_token_ids=[(i % 250) + 1 for i in range(96)],
                params=SamplingParams(max_tokens=32),
            )
        )
    for _ in range(8):
        engine.step()

    in_flight = len(engine.running)
    tokens_at_risk = sum(len(s.output_token_ids) for s in engine.running)
    engine.drain()
    rejected_after_drain = 0
    before = engine.rejected
    engine.add_request(Request(prompt_token_ids=[1, 2, 3], params=SamplingParams(max_tokens=4)))
    rejected_after_drain = engine.rejected - before

    steps = 0
    while not engine.drained:
        engine.step()
        steps += 1
    return {
        "in_flight_at_drain": in_flight,
        "tokens_at_risk": tokens_at_risk,
        "steps_to_drain": steps,
        "new_requests_rejected": rejected_after_drain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--rate", type=float, default=24.0)
    parser.add_argument("--queue-depth", type=int, default=6)
    parser.add_argument("--drain-requests", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }
    trace = make_poisson_trace(args.requests, args.rate, seed=31)

    rows = []
    engines = {
        "accept everything": lambda: PrefixCachedEngine(
            model, REFERENCE_MODEL, n_blocks=512, block_size=16
        ),
        "shed on queue depth": lambda: SheddingEngine(
            model,
            REFERENCE_MODEL,
            n_blocks=512,
            block_size=16,
            max_queue_depth=args.queue_depth,
        ),
    }
    for label, factory in engines.items():
        engine = factory()
        result = run_benchmark(
            engine,
            trace,
            rate_per_second=args.rate,
            slo=SERVING_SLO,
            meta={"model": model_meta, "trace": "overload", "policy": label},
        )
        s = result.summary()
        rejected = getattr(engine, "rejected", 0)
        rows.append(
            {
                "policy": label,
                "rejected": rejected,
                # A rejection ends the request, so the harness counts it as completed. Reporting
                # that number as "completed" would claim the shedding engine served everything.
                "served": s["completed"] - rejected,
                "ttft_p95": s["ttft_p95"],
                "goodput_per_second": s["goodput_per_second"],
                "output_tokens_per_second": s["output_tokens_per_second"],
            }
        )
        print(
            f"{label:<20} rejected={rows[-1]['rejected']:<3} ttft_p95={s['ttft_p95']:<8} "
            f"goodput={s['goodput_per_second']:<6} tok/s={s['output_tokens_per_second']}"
        )

    drain = _drain_cost(model, args)
    print(
        f"\ndrain: {drain['in_flight_at_drain']} in flight, {drain['tokens_at_risk']} tokens at risk"
    )

    payload = stamped_payload(
        engine="reliability",
        model=model_meta,
        code_sources=["llmserve/engines/shedding.py", "llmserve/engines/cancel.py"],
        conditions={
            "trace": f"{args.requests} requests at {args.rate}/s, beyond capacity",
            "rate_per_second": args.rate,
            "max_queue_depth": args.queue_depth,
            "slo": asdict(SERVING_SLO),
        },
        summary={"policies": rows, "drain": drain},
    )
    (RESULTS_DIR / "reliability-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote bench/results/reliability-tier1.json")


if __name__ == "__main__":
    main()
