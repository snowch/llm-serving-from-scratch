"""Generate the chapter 20 results: routing policy across a fleet of replicas.

Run with ``python -m bench.run_routing``. Policies are compared against each other, never against
a single engine: our replicas share one CPU, so a fleet is not faster than one engine here and any
such comparison would measure the test rig rather than the architecture.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, SLO, run_benchmark
from bench.traces import make_multi_tenant_trace
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import build_model
from llmserve.router import LeastOutstandingTokens, PrefixAffinity, RoundRobin, Router

POLICIES = [RoundRobin, LeastOutstandingTokens, PrefixAffinity]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }
    trace = make_multi_tenant_trace(args.requests, rate_per_second=args.rate, seed=5)
    prefixes = len({spec.tokens[:64] for spec in trace})

    for policy_cls in POLICIES:
        policy = policy_cls()
        router = Router(
            lambda: PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=384, block_size=16),
            args.replicas,
            policy,
        )
        result = run_benchmark(
            router,
            trace,
            rate_per_second=args.rate,
            slo=SLO(ttft_seconds=1.0, itl_seconds=0.05),
            meta={
                "model": model_meta,
                "trace": f"multi-tenant: {prefixes} system prompts, {args.replicas} replicas",
                "n_requests": args.requests,
                "replicas": args.replicas,
                "policy": policy.name,
                "torch_threads": args.threads,
            },
        )
        # Router counters are only meaningful once the trace has been served, so they are
        # recorded after the run rather than in the ``meta`` literal above.
        result.meta["token_reuse"] = round(router.token_reuse_rate, 4)
        result.meta["imbalance"] = round(router.imbalance, 3)
        result.meta["assignments"] = router.assignments
        result.to_json(RESULTS_DIR / f"router-{policy.name}-tier1.json")
        s = result.summary()
        print(
            f"{policy.name:<16} ttft_p50={s['ttft_p50']:<8} tok/s={s['output_tokens_per_second']:<7} "
            f"goodput={s['goodput_per_second']:<6} reuse={router.token_reuse_rate:.0%} "
            f"imbalance={router.imbalance:.2f}"
        )


if __name__ == "__main__":
    main()
