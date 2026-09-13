"""Generate the chapter 14 results: quality, speed and memory, together.

Run with ``python -m bench.run_quant``. Trains the reference model first (a couple of minutes), because
a quantisation measurement on random weights is meaningless: their perplexity is already at chance,
so damage cannot show up.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict

import torch
from torch import nn

from bench.harness import RESULTS_DIR, stamped_payload
from bench.train_tiny import evaluate_perplexity, held_out_data, train_reference_model
from llmserve.arithmetic import kv_bytes_per_token
from llmserve.config import REFERENCE_MODEL
from llmserve.quant import (
    QuantizedLinear,
    dequantize,
    quantize_int8_per_tensor,
    quantize_int8_symmetric,
    quantize_kv,
    quantize_model,
)


def _linear_bytes(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, QuantizedLinear):
            total += module.stored_bytes()
        elif isinstance(module, nn.Linear):
            total += module.weight.numel() * module.weight.element_size()
    return total


def _decode_speed(model, steps: int = 40) -> float:
    """Tokens per second for single-stream decode — the operation quantisation targets."""
    ids = torch.tensor([[7]])
    _, past = model(torch.tensor([[(i % 250) + 1 for i in range(64)]]))
    start = time.perf_counter()
    for i in range(steps):
        _, past = model(ids, past, torch.tensor([[64 + i]]))
    return steps / (time.perf_counter() - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    print(f"training the reference model for {args.steps} steps so quality can degrade...")
    model, baseline_ppl = train_reference_model(steps=args.steps, log_every=0)
    eval_data = held_out_data()
    rows = [
        {
            "label": "fp32 (baseline)",
            "bits": 32,
            "perplexity": round(baseline_ppl, 4),
            "linear_bytes": _linear_bytes(model),
            "decode_tok_per_s": round(_decode_speed(model), 1),
        }
    ]
    print(f"fp32 perplexity {baseline_ppl:.4f}")

    schemes = [
        ("INT8 per-tensor (naive)", 8, {"per_tensor": True}),
        ("INT8 per-channel", 8, {}),
        ("INT4 grouped (g=64)", 4, {}),
    ]
    for label, bits, options in schemes:
        quantised = quantize_model(copy.deepcopy(model), bits=bits, **options)
        ppl = evaluate_perplexity(quantised, eval_data)
        rows.append(
            {
                "label": label,
                "bits": bits,
                "perplexity": round(ppl, 4),
                "linear_bytes": _linear_bytes(quantised),
                "decode_tok_per_s": round(_decode_speed(quantised), 1),
            }
        )
        print(
            f"{label:<24} perplexity {ppl:.4f} ({100 * (ppl - baseline_ppl) / baseline_ppl:+.2f}%)"
        )

    # KV-cache quantisation is measured separately: it costs nothing in weights and everything in
    # how many conversations fit.
    kv_rows = []
    for bits in (8, 4):
        damaged = _perplexity_with_quantised_kv(model, eval_data, bits)
        kv_rows.append(
            {
                "bits": bits,
                "perplexity": round(damaged, 4),
                "kv_bytes_per_token": kv_bytes_per_token(REFERENCE_MODEL) * bits // 32,
            }
        )
        print(f"KV INT{bits} perplexity {damaged:.4f}")

    outliers = _outlier_demonstration()
    print(
        f"\noutlier demo: per-tensor error {outliers['per_tensor_error']:.5f} vs "
        f"per-channel {outliers['per_channel_error']:.5f} "
        f"({outliers['ratio']:.0f}x worse)"
    )

    payload = stamped_payload(
        engine="quantisation",
        model={
            "name": "TinyGPT, trained on the synthetic corpus",
            "params": sum(p.numel() for p in model.parameters()),
            **asdict(REFERENCE_MODEL),
        },
        # Both the quantisers and the training run decide these numbers.
        code_sources=["llmserve/quant.py", "bench/train_tiny.py"],
        conditions={
            "trace": "held-out synthetic corpus, non-overlapping 128-token windows",
            "train_steps": args.steps,
            "baseline_perplexity": round(baseline_ppl, 4),
        },
        summary={"weights": rows, "kv_cache": kv_rows, "outliers": outliers},
    )
    (RESULTS_DIR / "quant-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote bench/results/quant-tier1.json")


def _outlier_demonstration(scale: float = 200.0) -> dict:
    """Why per-channel scaling exists, measured directly on a matrix with one outlier channel.

    The end-to-end perplexity measurement does not show this, because a small well-trained model
    has no channel hundreds of times larger than its neighbours. Large models do — it is the whole
    reason LLM.int8() and SmoothQuant exist — so the mechanism is demonstrated here on a matrix
    that has the property, rather than implied from a measurement that does not.
    """
    torch.manual_seed(0)
    weight = torch.randn(8, 64)
    weight[0] *= scale  # one channel dwarfing the rest

    per_channel = dequantize(*quantize_int8_symmetric(weight))
    per_tensor = dequantize(*quantize_int8_per_tensor(weight))

    # Error on the *other* channels: the ones the outlier's scale ruins.
    per_tensor_error = float((weight[1:] - per_tensor[1:]).abs().mean())
    per_channel_error = float((weight[1:] - per_channel[1:]).abs().mean())
    return {
        "outlier_scale": scale,
        "per_tensor_error": round(per_tensor_error, 5),
        "per_channel_error": round(per_channel_error, 5),
        "ratio": round(per_tensor_error / per_channel_error, 1),
    }


@torch.inference_mode()
def _perplexity_with_quantised_kv(
    model, data: torch.Tensor, bits: int, seq_len: int = 128
) -> float:
    """Perplexity when the cache is stored at reduced precision.

    Approximated by quantising the keys and values produced during a normal forward pass, which
    measures the precision loss without needing a quantised cache implementation in the engine.
    """
    losses = []
    for start in range(0, len(data) - seq_len - 1, seq_len):
        x = data[start : start + seq_len].unsqueeze(0)
        y = data[start + 1 : start + seq_len + 1].unsqueeze(0)
        prefix = x[:, :1]
        logits, past = model(prefix)
        past = [(quantize_kv(k, bits), quantize_kv(v, bits)) for k, v in past]
        rest = x[:, 1:]
        positions = torch.arange(1, x.shape[1]).unsqueeze(0)
        logits_rest, _ = model(rest, past, positions)
        combined = torch.cat([logits, logits_rest], dim=1)
        losses.append(
            nn.functional.cross_entropy(
                combined.reshape(-1, model.cfg.vocab_size), y.reshape(-1)
            ).item()
        )
    return float(torch.exp(torch.tensor(losses).mean()))


if __name__ == "__main__":
    main()
