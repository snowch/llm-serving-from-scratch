"""Generate the chapter 10 results: the same engine at several per-step token budgets.

Run with ``python -m bench.run_chunked``. One engine, one trace, one dial. A very large budget is
equivalent to not chunking at all, which makes it the honest baseline for what chunking buys and
what it costs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, SLO, run_benchmark
from bench.traces import make_mixed_prompt_trace
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.chunked import ChunkedPrefillEngine
from llmserve.model import build_model


def _measure_chunk_cost(model, model_meta) -> None:
    """How long one long prefill takes, split into different numbers of passes.

    Separate from the serving benchmark because it answers a different question: not "does
    chunking schedule better?" but "does splitting a prefill cost anything at all?" The answer is
    not the obvious one.
    """
    import json
    import time

    from bench.harness import stamped_payload

    prompt_len = 1536
    rows = []
    for n_passes in (1, 6, 24):
        size = prompt_len // n_passes
        torch.manual_seed(0)
        start = time.perf_counter()
        past = None
        for i in range(n_passes):
            ids = torch.randint(1, 250, (1, size))
            positions = torch.arange(i * size, (i + 1) * size).unsqueeze(0)
            _, past = model(ids, past, positions)
        rows.append(
            {
                "passes": n_passes,
                "chunk": size,
                "ms": round((time.perf_counter() - start) * 1000, 1),
            }
        )

    path = RESULTS_DIR / "chunk-cost-tier1.json"
    # No engine involved: this times the model directly, so the core sources are the whole
    # dependency. It still gets a real fingerprint, because a change to the model would change
    # the answer.
    payload = stamped_payload(
        engine="microbenchmark",
        model=model_meta,
        conditions={"prompt_len": prompt_len, "trace": "single prefill, no serving"},
        summary={"passes": rows},
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        print(
            f"prefill {prompt_len} tokens in {row['passes']:>2} pass(es) of {row['chunk']:>4}: {row['ms']:>7.1f} ms"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--budgets", type=int, nargs="+", default=[64, 512, 8192])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }
    trace = make_mixed_prompt_trace(
        args.requests,
        rate_per_second=args.rate,
        long_len=1536,
        long_fraction=0.35,
        short_len=(16, 32),
        output_len=(32, 64),
        seed=3,
    )

    _measure_chunk_cost(model, model_meta)

    for budget in args.budgets:
        engine = ChunkedPrefillEngine(
            model, REFERENCE_MODEL, n_blocks=1024, block_size=16, token_budget=budget
        )
        result = run_benchmark(
            engine,
            trace,
            rate_per_second=args.rate,
            slo=SLO(ttft_seconds=1.0, itl_seconds=0.05),
            meta={
                "model": model_meta,
                "trace": "mixed: 35% of prompts 1536 tokens, rest 16-32, output 32-64, seed 3",
                "n_requests": args.requests,
                "token_budget": budget,
                "torch_threads": args.threads,
            },
        )
        result.meta["chunked_prefills"] = engine.chunked_prefills
        path = RESULTS_DIR / f"chunked-budget{budget}-tier1.json"
        result.to_json(path)
        s = result.summary()
        print(
            f"budget={budget:<6} ttft_p50={s['ttft_p50']:<8} ttft_p95={s['ttft_p95']:<8} "
            f"itl_p50={s['itl_p50']:<8} itl_p95={s['itl_p95']:<8} "
            f"tok/s={s['output_tokens_per_second']:<7} goodput={s['goodput_per_second']:<6} "
            f"chunks={engine.chunked_prefills}"
        )


if __name__ == "__main__":
    main()
