"""Generate the chapter 18 results: does constraining work, and what does it cost?

Run with ``python -m bench.run_constrained``. Three questions, measured separately: how often
unconstrained generation produces valid JSON, how often constrained generation does, and what
enforcing the grammar costs per step with and without mask caching.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, stamped_payload
from bench.train_tiny import train_reference_model
from llmserve.config import REFERENCE_MODEL
from llmserve.constrain import JSONGrammar, State, is_valid_json_object
from llmserve.sampling import SamplingParams, sample
from llmserve.tokenizer import ByteTokenizer

PROMPT = b'Return JSON. {"name": "server", "role": "cache"} '


def _generate(
    model, prompt: list[int], n: int, grammar=None, *, cached=True, seed=0
) -> tuple[str, bool]:
    """Generate up to n tokens, optionally under a grammar.

    Returns the text and whether the grammar reached an accepting state. That second value matters
    more than it looks: a grammar guarantees no *illegal* token is ever emitted, but it cannot make
    the model finish. Running out of budget mid-document produces output that is unparseable
    despite every token having been legal.
    """
    generator = torch.Generator().manual_seed(seed)
    params = SamplingParams(max_tokens=n, temperature=1.0)
    tokens = list(prompt)
    produced: list[int] = []
    state = State.START

    _, past = model(torch.tensor([tokens]))
    logits = None
    for _ in range(n):
        if logits is None:
            out, past = model(torch.tensor([tokens]))
            logits = out[:, -1, :]
        if grammar is not None:
            logits = grammar.apply(logits[0], state, cached=cached).unsqueeze(0)
        token = int(sample(logits, params, generator=generator).item())
        produced.append(token)
        if grammar is not None:
            state = grammar.step(state, token)
            if state is State.DONE:
                return ByteTokenizer().decode(produced), True
        tokens.append(token)
        out, past = model(torch.tensor([[token]]), past, torch.tensor([[len(tokens) - 1]]))
        logits = out[:, -1, :]

    # Fell out of the loop: the budget ran out before the grammar accepted.
    return ByteTokenizer().decode(produced), False


def _mask_cost() -> dict:
    """Cost of building one mask, at this book's vocabulary and at a realistic one.

    The end-to-end timing shows caching making no difference, because a 260-entry mask is free
    next to a forward pass. Production vocabularies are hundreds of times larger, and that is the
    regime the caching exists for — so measure the mask alone rather than conclude from a
    comparison that cannot show it.
    """
    rows = []
    for vocab_size in (260, 32_000, 128_256):
        grammar = JSONGrammar(vocab_size=vocab_size, eos_id=vocab_size - 1)
        start = time.perf_counter()
        for _ in range(200):
            grammar.build_mask(State.IN_VALUE)
        per_mask_us = (time.perf_counter() - start) / 200 * 1e6
        rows.append({"vocab_size": vocab_size, "microseconds_per_mask": round(per_mask_us, 1)})
    return {"per_mask": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--tokens", type=int, default=40)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    print("training the reference model...")
    model, _ = train_reference_model(steps=400, log_every=0)
    prompt = list(PROMPT)

    free_valid = sum(
        is_valid_json_object(_generate(model, prompt, args.tokens, seed=s)[0])
        for s in range(args.trials)
    )
    constrained_valid = completed = 0
    for s in range(args.trials):
        grammar = JSONGrammar(vocab_size=REFERENCE_MODEL.vocab_size, eos_id=257)
        text, finished = _generate(model, prompt, args.tokens, grammar, seed=s)
        completed += finished
        constrained_valid += is_valid_json_object(text)

    print(f"unconstrained valid JSON:        {free_valid}/{args.trials}")
    print(f"constrained, reached end state:  {completed}/{args.trials}")
    print(f"constrained valid JSON:          {constrained_valid}/{args.trials}")
    print(f"constrained AND completed valid: {constrained_valid}/{max(completed, 1)}")

    # -- cost of enforcement, cached against naive ---------------------------------------
    timings = {}
    for label, cached in (("cached masks", True), ("rebuilt every step", False)):
        grammar = JSONGrammar(vocab_size=REFERENCE_MODEL.vocab_size, eos_id=257)
        start = time.perf_counter()
        for s in range(args.trials):
            _generate(model, prompt, args.tokens, grammar, cached=cached, seed=s)
        elapsed = time.perf_counter() - start
        timings[label] = {
            "seconds": round(elapsed, 3),
            "masks_built": grammar.masks_built,
        }
        print(f"{label:<20} {elapsed:.3f}s, masks built: {grammar.masks_built}")

    payload = stamped_payload(
        engine="constrained",
        model={
            "name": "TinyGPT, trained on the synthetic corpus",
            "params": sum(p.numel() for p in model.parameters()),
            **asdict(REFERENCE_MODEL),
        },
        code_sources=["llmserve/constrain.py", "bench/train_tiny.py"],
        conditions={
            "trace": f"{args.trials} samples of up to {args.tokens} tokens, temperature 1.0",
            "grammar": "JSON object subset over bytes",
        },
        summary={
            "validity": {
                "trials": args.trials,
                "unconstrained_valid": free_valid,
                "constrained_valid": constrained_valid,
                "constrained_completed": completed,
            },
            "mask_cost": _mask_cost(),
            "cost": timings,
        },
    )
    (RESULTS_DIR / "constrained-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote bench/results/constrained-tier1.json")


if __name__ == "__main__":
    main()
