"""Generate the chapter 19 results: what an adapter buys, and what serving many of them costs.

Run with ``python -m bench.run_lora``. Two measurements that answer different questions.

The quality matrix answers "does an adapter do anything?" — each tenant's adapter against each
tenant's held-out text, with the base model as the control. Without it the serving measurements
would be timing no-ops.

The batch sweep answers "what does multi-tenancy cost?" — the same decode step with one adapter
across the batch, and with several, which is the case a shared fleet is actually in.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, stamped_payload
from bench.train_lora import TENANT_SEEDS, perplexity_matrix, train_tenant_adapters
from bench.train_tiny import train_reference_model
from llmserve.config import REFERENCE_MODEL
from llmserve.lora import LoRAConfig, attach, make_adapter, merge, set_active, set_row_adapters


def _batch_sweep(model, args) -> list[dict]:
    """Time one decode step under increasing adapter diversity in the same batch.

    The adapters here are untrained, and deliberately so: this measures the *shape* of the work,
    which depends only on how many distinct adapters the batch contains, not on what they learned.
    """
    config = LoRAConfig(rank=args.rank)
    pool = [make_adapter(model, f"t{i}", config, seed=100 + i) for i in range(args.batch)]
    layers = attach(model, pool)
    ids = torch.randint(1, 250, (args.batch, 1))
    _, past = model(torch.randint(1, 250, (args.batch, args.context)))

    # One warmup before the sweep, not one per case. Timing the first case cold and the rest warm
    # made the baseline look slower than the adapter path, which is the sort of result that looks
    # like a discovery and is actually a measurement bug.
    set_active(layers, None)
    for _ in range(20):
        model(ids, past)

    def timed(setup) -> float:
        """Fastest of several repeats: the minimum is the one not polluted by other work."""
        setup()
        for _ in range(5):
            model(ids, past)
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(args.trials):
                model(ids, past)
            best = min(best, (time.perf_counter() - start) * 1000 / args.trials)
        return round(best, 3)

    rows = [
        {
            "case": "No adapter (base model)",
            "distinct": 0,
            "ms": timed(lambda: set_active(layers, None)),
        },
        {
            "case": "One adapter, whole batch",
            "distinct": 1,
            "ms": timed(lambda: set_active(layers, "t0")),
        },
    ]
    for k in (2, 4, args.batch):
        if k > args.batch or any(row["distinct"] == k for row in rows):
            continue
        names = [f"t{i % k}" for i in range(args.batch)]
        rows.append(
            {
                "case": f"{k} adapters in one batch",
                "distinct": k,
                "ms": timed(lambda names=names: set_row_adapters(layers, names)),
            }
        )

    # Merging removes the adapter entirely: the right answer for one tenant, unavailable for many.
    merge(layers, "t0")
    rows.append({"case": "Merged into the weights", "distinct": 1, "ms": timed(lambda: None)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400, help="base model training steps")
    parser.add_argument("--adapter-steps", type=int, default=300)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    print(f"training the base model for {args.steps} steps...")
    model, base_ppl = train_reference_model(steps=args.steps, log_every=200)
    print(f"base model held-out perplexity {base_ppl:.4f}\n")

    print(f"training one adapter per tenant ({', '.join(TENANT_SEEDS)})...")
    adapters, layers = train_tenant_adapters(model, steps=args.adapter_steps, rank=args.rank)
    quality = perplexity_matrix(model, layers, adapters)
    for tenant, row in quality.items():
        parts = "  ".join(f"{name}={value:.4f}" for name, value in row.items())
        print(f"  {tenant}'s text: {parts}")

    one = next(iter(adapters.values()))
    memory = {
        "rank": args.rank,
        "targets": list(one.config.targets),
        "base_params": sum(p.numel() for p in model.parameters()),
        "adapter_params": one.n_parameters,
    }
    print(
        f"\nadapter is {one.n_parameters:,} parameters against a "
        f"{memory['base_params']:,}-parameter base"
    )

    # A fresh model for the timing, so the trained adapters above are not what is being measured.
    timing_model, _ = train_reference_model(steps=1, log_every=0)
    batch = _batch_sweep(timing_model, args)
    print()
    for row in batch:
        print(f"  {row['case']:<28} {row['ms']:>7.3f} ms/step")

    payload = stamped_payload(
        engine="lora",
        model={
            "name": "TinyGPT, trained on the synthetic corpus",
            "params": memory["base_params"],
            **asdict(REFERENCE_MODEL),
        },
        code_sources=["llmserve/lora.py", "bench/train_lora.py", "bench/train_tiny.py"],
        conditions={
            "trace": "held-out synthetic corpus per tenant; decode-step timing",
            "train_steps": args.steps,
            "adapter_steps": args.adapter_steps,
            "batch_size": args.batch,
            "context": args.context,
            "base_perplexity": round(base_ppl, 4),
        },
        summary={"quality": quality, "memory": memory, "batch": batch},
    )
    (RESULTS_DIR / "lora-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote bench/results/lora-tier1.json")


if __name__ == "__main__":
    main()
