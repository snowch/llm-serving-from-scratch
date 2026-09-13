"""Generate the chapter 26 results: what the API layer does to the engine underneath it.

Run with ``python -m bench.run_api``. The measurement is about chat templates, because that is the
part of the API surface with a silent, expensive failure mode: a template that varies per request —
a timestamp, a request id, a session id in the system prompt — is syntactically fine, semantically
harmless, and destroys the prefix cache chapter 9 built.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from bench.harness import RESULTS_DIR, SLO, stamped_payload
from bench.traces import make_session_turns
from llmserve.api import DEFAULT_TEMPLATE, ChatMessage, ChatTemplate, CompletionRequest
from llmserve.config import REFERENCE_MODEL
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import build_model
from llmserve.tokenizer import ByteTokenizer

SYSTEM = (
    "You are a helpful assistant. Answer concisely and accurately. "
    "Cite your sources where possible. Do not speculate beyond the evidence."
)


def _templates() -> dict[str, tuple[ChatTemplate, str]]:
    """Three templates: one stable, one with a per-request value, one with the system prompt last."""
    trailing = ChatTemplate(generation_prompt="<|assistant|> ")
    return {
        "stable": (DEFAULT_TEMPLATE, "system prompt identical on every request"),
        "per-request value": (
            DEFAULT_TEMPLATE,
            "a timestamp injected at the top of the system prompt",
        ),
        "system prompt last": (trailing, "conversation first, system prompt at the end"),
    }


def _messages(variant: str, turn_index: int, transcript: str) -> list[ChatMessage]:
    system = SYSTEM
    if variant == "per-request value":
        # The bug, exactly as it appears in real systems: harmless-looking, and different every time.
        system = f"Current time: 2026-09-13T12:{turn_index:02d}:00Z. {SYSTEM}"
    conversation = [ChatMessage("user", transcript)]
    if variant == "system prompt last":
        return [*conversation, ChatMessage("system", system)]
    return [ChatMessage("system", system), *conversation]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=6)
    parser.add_argument("--turns", type=int, default=4)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    model = build_model(REFERENCE_MODEL)
    model_meta = {
        "name": "TinyGPT (reference, random weights)",
        "params": sum(p.numel() for p in model.parameters()),
        **asdict(REFERENCE_MODEL),
    }
    tokenizer = ByteTokenizer()
    turns = make_session_turns(args.sessions, args.turns, seed=41)

    rows = []
    for variant, (template, description) in _templates().items():
        engine = PrefixCachedEngine(model, REFERENCE_MODEL, n_blocks=4096, block_size=16)
        requests = []
        for turn_index, prompts in enumerate(turns):
            for prompt in prompts:
                transcript = bytes(t % 128 for t in prompt).decode("ascii", "replace")
                completion = CompletionRequest(
                    messages=_messages(variant, turn_index, transcript), max_tokens=8
                )
                requests.append(completion.to_engine_request(tokenizer, template, seed=0))

        prompt_tokens = sum(r.prompt_len for r in requests)
        for request in requests:
            engine.add_request(request)
        while engine.has_work():
            engine.step()

        prefix = engine.prefix
        rows.append(
            {
                "variant": variant,
                "description": description,
                "requests": len(requests),
                "prompt_tokens": prompt_tokens,
                "token_reuse": round(
                    prefix.hit_tokens / prefix.total_prompt_tokens
                    if prefix.total_prompt_tokens
                    else 0.0,
                    4,
                ),
            }
        )
        print(f"{variant:<20} reuse={rows[-1]['token_reuse']:.0%}  ({description})")

    payload = stamped_payload(
        engine="api",
        model=model_meta,
        code_sources=["llmserve/api.py", "llmserve/engines/prefix.py"],
        conditions={
            "trace": f"{args.sessions} sessions x {args.turns} turns, rendered three ways",
            "slo": asdict(SLO(ttft_seconds=1.0, itl_seconds=0.05)),
        },
        summary={"templates": rows},
    )
    (RESULTS_DIR / "api-tier1.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nwrote bench/results/api-tier1.json")


if __name__ == "__main__":
    main()
