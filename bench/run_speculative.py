"""Generate the chapter 15 results: acceptance economics, and the correctness claim.

Run with ``python -m bench.run_speculative``. Two measurements that answer different questions.

The sweep answers "is it worth it?" — how acceptance falls as you propose more, and how many
tokens each target forward pass actually yields.

The distribution test answers "is it the same model?" — and it is the more important of the two.
A speculative engine that is *almost* the target model is a different model, and no user could
tell which one answered.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, stamped_payload
from bench.train_tiny import make_corpus, train_reference_model
from llmserve.config import REFERENCE_MODEL
from llmserve.sampling import SamplingParams
from llmserve.speculative import (
    NgramDrafter,
    SpeculationStats,
    accept_sampled,
    speculative_generate,
)


def _prompt_from_corpus(length: int = 96) -> list[int]:
    """A prompt drawn from the text the model was trained on.

    Speculation is only meaningful against a model that produces structured output. With random
    weights the target's next token is arbitrary, so an n-gram drafter copying earlier text almost
    never matches and acceptance sits at zero — which measures the drafter against noise rather
    than measuring speculation.
    """
    return list(make_corpus().encode("utf-8")[:length])


def _greedy_reference(model, prompt: list[int], n: int) -> list[int]:
    tokens, output = list(prompt), []
    for _ in range(n):
        logits, _ = model(torch.tensor([tokens]))
        token = int(logits[0, -1].argmax())
        output.append(token)
        tokens.append(token)
    return output


def _biased_accept(target_logits, draft_probs, proposed, params, generator=None):
    """The plausible-looking bug: on rejection, resample from p_target instead of the residual.

    Kept in the benchmark rather than only described, because a correctness argument you cannot
    falsify is not worth much. This is the negative control that shows the test has power.
    """
    accepted = []
    temperature = max(params.temperature, 1e-6)
    for i, token in enumerate(proposed):
        probs = (target_logits[i] / temperature).softmax(dim=-1)
        ratio = (probs[token] / draft_probs[i][token].clamp(min=1e-10)).clamp(max=1.0)
        if torch.rand(1, generator=generator).item() < ratio.item():
            accepted.append(token)
            continue
        return accepted, int(torch.multinomial(probs, 1, generator=generator).item())
    bonus = (target_logits[len(proposed)] / temperature).softmax(dim=-1)
    return accepted, int(torch.multinomial(bonus, 1, generator=generator).item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    print("training the reference model so its output has structure to predict...")
    model, perplexity = train_reference_model(steps=400, log_every=0)
    print(f"held-out perplexity {perplexity:.3f}")
    prompt = _prompt_from_corpus()

    # -- acceptance economics -----------------------------------------------------------
    reference = _greedy_reference(model, prompt, 16)
    sweep = []
    for k in (1, 2, 4, 8):
        stats = SpeculationStats()
        output = speculative_generate(
            model, NgramDrafter(3), prompt, SamplingParams(max_tokens=16), k=k, stats=stats
        )
        sweep.append(
            {
                "k": k,
                "acceptance_rate": round(stats.acceptance_rate, 4),
                "tokens_per_round": round(stats.tokens_per_round, 3),
                "target_passes": stats.rounds,
                "identical_to_greedy": output == reference,
            }
        )
        print(
            f"k={k}: accept {stats.acceptance_rate:.0%}  tokens/pass "
            f"{stats.tokens_per_round:.2f}  identical={output == reference}"
        )

    # -- does it sample from the target distribution? ------------------------------------
    #
    # Deliberately measured on the UNTRAINED model. The trained one is so confident that the
    # drafted token has probability ~0.9998, and a distribution that is nearly a point mass cannot
    # reveal a sampling bias: correct and incorrect rules both return that token essentially
    # always. A broad distribution is what gives this test power, so each measurement here runs
    # under the conditions where it can actually discriminate.
    from llmserve.model import build_model

    model = build_model(REFERENCE_MODEL)
    drafted = NgramDrafter(3).propose(prompt, 4)[0]
    logits, _ = model(torch.tensor([prompt]))
    target_prob = float(logits[0, -1].softmax(dim=-1)[drafted])
    params = SamplingParams(max_tokens=1, temperature=1.0)

    def frequency(rule) -> float:
        import llmserve.speculative as module

        original, module.accept_sampled = module.accept_sampled, rule
        try:
            hits = 0
            for seed in range(args.samples):
                generator = torch.Generator().manual_seed(seed)
                token = speculative_generate(
                    model, NgramDrafter(3), prompt, params, k=4, generator=generator
                )[0]
                hits += token == drafted
            return hits / args.samples
        finally:
            module.accept_sampled = original

    correct = frequency(accept_sampled)
    biased = frequency(_biased_accept)
    standard_error = (target_prob * (1 - target_prob) / args.samples) ** 0.5

    distribution = {
        "drafted_token": drafted,
        "target_probability": round(target_prob, 5),
        "predicted_if_biased": round(target_prob * (2 - target_prob), 5),
        "measured_correct_rule": round(correct, 5),
        "measured_biased_rule": round(biased, 5),
        "standard_error": round(standard_error, 5),
        "correct_rule_sigma": round(abs(correct - target_prob) / standard_error, 2),
        "biased_rule_sigma": round(abs(biased - target_prob) / standard_error, 2),
        "samples": args.samples,
    }
    print(
        f"\nP(drafted token): target {target_prob:.5f} | correct rule {correct:.5f} "
        f"({distribution['correct_rule_sigma']} SE) | biased rule {biased:.5f} "
        f"({distribution['biased_rule_sigma']} SE)"
    )

    payload = stamped_payload(
        engine="speculative",
        model={
            "name": "TinyGPT (reference, random weights)",
            "params": sum(p.numel() for p in model.parameters()),
            **asdict(REFERENCE_MODEL),
        },
        # The acceptance sweep runs on a trained model, so the corpus is part of the measurement.
        code_sources=["llmserve/speculative.py", "bench/train_tiny.py"],
        conditions={
            "trace": "acceptance on the trained model; distribution test on the untrained one",
            "drafter": "ngram(3)",
            "held_out_perplexity": round(perplexity, 4),
        },
        summary={"sweep": sweep, "distribution": distribution},
    )
    (RESULTS_DIR / "speculative-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote bench/results/speculative-tier1.json")


if __name__ == "__main__":
    main()
