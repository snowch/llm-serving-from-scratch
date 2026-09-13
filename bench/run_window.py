"""Generate the chapter 16 results: what a bounded KV cache costs in quality.

Run with ``python -m bench.run_window``. This is the first chapter whose optimisation can change
what the model says, so the measurement has to be a quality one — perplexity on held-out text, with
the cache bounded in several different ways and the full-context number as the control.

Three questions, and the three of them are the chapter:

* does a plain sliding window work?
* do a handful of retained first tokens rescue it?
* does it matter which position indices the surviving tokens are given?

The last one sounds like an implementation detail and is worth as much as the other two.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch
from torch import nn

from bench.harness import RESULTS_DIR, stamped_payload
from bench.train_tiny import held_out_data, train_reference_model
from llmserve.arithmetic import kv_bytes_per_token
from llmserve.cache.window import WindowPolicy, kv_bytes_with_window
from llmserve.config import REFERENCE_MODEL, ModelConfig


@torch.inference_mode()
def windowed_perplexity(model, data: torch.Tensor, policy: WindowPolicy | None) -> float:
    """Perplexity over ``data``, decoded one token at a time through a bounded cache.

    Token by token rather than in one pass because that is what the technique actually does: the
    cache is evicted as generation proceeds, and a batched evaluation would quietly give every token
    the full context the policy is supposed to have taken away.

    Positions are the original ones, and that is not a choice — it is a constraint. A key is rotated
    when it is computed, so its position is baked in; renumbering the survivors would mean
    re-rotating them, which means caching keys *before* rotation. This model does not, so the
    renumbering half of the technique is not measurable here and the chapter says so rather than
    reporting a number from an implementation that only looks like it.
    """
    losses = []
    past = None
    #: original positions currently held in the cache, in cache order
    held: list[int] = []

    for index in range(len(data) - 1):
        position = index
        logits, past = model(
            data[index].view(1, 1),
            past,
            torch.tensor([[position]], dtype=torch.long),
        )
        losses.append(nn.functional.cross_entropy(logits[:, -1, :], data[index + 1].view(1)).item())
        held.append(index)

        if policy is not None and len(held) > policy.budget:
            keep = policy.keep(index + 1)
            location = {original: slot for slot, original in enumerate(held)}
            slots = [location[original] for original in keep if original in location]
            past = [(k[:, :, slots, :], v[:, :, slots, :]) for k, v in past]
            held = [held[slot] for slot in slots]

    return float(torch.exp(torch.tensor(losses).mean()))


@torch.inference_mode()
def attention_mass_on_first_tokens(model, data: torch.Tensor, first: int = 4) -> dict:
    """How much of the model's attention lands on the opening tokens, per layer.

    The case for keeping sinks is that softmax has to put its mass somewhere, and in a trained model
    the opening positions become where it goes — so evicting them removes the outlet and quality
    falls apart. That is a claim about the model, not about the eviction policy, and it is directly
    measurable: run the text and look at where attention actually goes.

    Measured with a forward hook rather than by changing the model. The hook sees each attention
    layer's input, which is all that is needed to recompute the queries from weights we already
    have; the keys come back in the cache the model returns anyway.
    """
    captured: list[torch.Tensor] = []
    handles = []

    def hook(module, inputs, output):
        hidden = inputs[0]
        queries = module.q_proj(hidden)
        batch, length, _ = queries.shape
        queries = queries.view(batch, length, model.cfg.n_heads, model.cfg.head_dim).transpose(1, 2)
        keys = output[1][0] if isinstance(output, tuple) else None
        if keys is None:
            return
        # Grouped-query attention: every group of query heads shares one KV head.
        repeats = model.cfg.n_heads // model.cfg.n_kv_heads
        keys = keys.repeat_interleave(repeats, dim=1)
        scores = queries @ keys.transpose(-2, -1) / (model.cfg.head_dim**0.5)
        mask = torch.ones(length, keys.shape[2], dtype=torch.bool).tril(
            diagonal=keys.shape[2] - length
        )
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        captured.append(scores.softmax(dim=-1)[..., :first].sum(dim=-1).mean())

    for block in model.blocks:
        handles.append(block.attn.register_forward_hook(hook))
    try:
        model(data.view(1, -1))
    finally:
        for handle in handles:
            handle.remove()

    per_layer = [round(float(value), 4) for value in captured]
    return {
        "first_tokens": first,
        "per_layer_mass": per_layer,
        "mean_mass": round(sum(per_layer) / len(per_layer), 4) if per_layer else None,
        "uniform_share": round(first / len(data), 4),
    }


def _memory_rows(policy: WindowPolicy) -> list[dict]:
    """What the bound is worth at production scale, where the cache is the binding constraint."""
    llama8b = ModelConfig(
        vocab_size=128_256, n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128, dtype="fp16"
    )
    per_token = kv_bytes_per_token(llama8b)
    rows = []
    for context in (2048, 8192, 32768, 131072):
        full = per_token * context
        bounded = kv_bytes_with_window(per_token, policy, context)
        rows.append(
            {
                "context": context,
                "full_bytes": full,
                "bounded_bytes": bounded,
                "sequences_in_40gb_full": int(40e9 // full),
                "sequences_in_40gb_bounded": int(40e9 // bounded),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument(
        "--short-tokens", type=int, default=128, help="held-out tokens, inside the training window"
    )
    parser.add_argument("--tokens", type=int, default=768, help="held-out tokens, inside context")
    parser.add_argument(
        "--long-tokens", type=int, default=3072, help="held-out tokens, past the trained context"
    )
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--sinks", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    print(f"training the reference model for {args.steps} steps...")
    model, _ = train_reference_model(steps=args.steps, log_every=200)

    # Both sides of the trained context length, because the position question has a different
    # answer on each side and reporting only one of them would be the more convenient half.
    # Three regimes, and the first one exists because leaving it out would have let this chapter
    # report that bounding the cache *improves* quality. It does, at 768 tokens — but only because
    # the model was trained on 128-token sequences, so the full-context baseline there is itself
    # extrapolating. Comparing against a baseline that is out of its depth measures the baseline.
    lengths = [
        ("inside the training sequence length", args.short_tokens),
        ("past it, inside the rotary table", args.tokens),
        ("past the rotary table", args.long_tokens),
    ]
    configurations = [
        ("Full context", None),
        (f"Sliding window {args.window}, no sinks", WindowPolicy(args.window, 0)),
        (f"Window {args.window} + {args.sinks} sinks", WindowPolicy(args.window, args.sinks)),
    ]

    rows = []
    for regime, length in lengths:
        data = held_out_data()[:length]
        print(f"\n{length} tokens ({regime}, model trained to {REFERENCE_MODEL.max_seq_len}):")
        baseline = None
        for label, policy in configurations:
            try:
                perplexity = windowed_perplexity(model, data, policy)
            except IndexError:
                # Not a bug to work around: a model's rotary table is sized to the context it was
                # trained on, so a position past the end has no embedding to look up. Configurations
                # that feed original positions simply cannot run out here, and recording that is
                # more useful than engineering a number out of it.
                rows.append(
                    {
                        "regime": regime,
                        "tokens": length,
                        "configuration": label,
                        "window": policy.window if policy else None,
                        "sinks": policy.sinks if policy else None,
                        "kv_tokens": policy.budget if policy else length,
                        "perplexity": None,
                        "change": None,
                        "failed": "position past the trained context has no rotary embedding",
                    }
                )
                print(f"  {label:<52} unavailable (position past trained context)")
                continue
            if baseline is None:
                baseline = perplexity
            rows.append(
                {
                    "regime": regime,
                    "tokens": length,
                    "configuration": label,
                    "window": policy.window if policy else None,
                    "sinks": policy.sinks if policy else None,
                    "kv_tokens": policy.budget if policy else length,
                    "perplexity": round(perplexity, 4),
                    "change": round(perplexity / baseline - 1, 4),
                }
            )
            print(f"  {label:<52} perplexity {perplexity:8.4f}")

    sinks_report = attention_mass_on_first_tokens(model, held_out_data()[: args.tokens])
    print(
        f"\nattention on the first {sinks_report['first_tokens']} tokens: "
        f"{sinks_report['mean_mass']:.1%} of the mass "
        f"(an even spread would be {sinks_report['uniform_share']:.1%})"
    )

    payload = stamped_payload(
        engine="window",
        model={
            "name": "TinyGPT, trained on the synthetic corpus",
            "params": sum(p.numel() for p in model.parameters()),
            **asdict(REFERENCE_MODEL),
        },
        code_sources=["llmserve/cache/window.py", "bench/train_tiny.py"],
        conditions={
            "trace": f"held-out text decoded one token at a time, at {args.tokens} and "
            f"{args.long_tokens} tokens against a {REFERENCE_MODEL.max_seq_len}-token context",
            "train_steps": args.steps,
            "max_seq_len": REFERENCE_MODEL.max_seq_len,
            "train_seq_len": 128,
            "window": args.window,
            "sinks": args.sinks,
        },
        summary={
            "quality": rows,
            "sinks": sinks_report,
            "memory": _memory_rows(WindowPolicy(4096, args.sinks)),
        },
    )
    (RESULTS_DIR / "window-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote bench/results/window-tier1.json")


if __name__ == "__main__":
    main()
