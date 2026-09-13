"""Trace shapes beyond the uniform-random default.

The trace is part of the measurement, not a detail of it. Chapter 21 shows that changing the
input/output length distribution can reverse which engine looks faster, and chapter 9 needs a
workload where requests genuinely share text — on the default random trace, prefix caching has
nothing to reuse and correctly does nothing.
"""

from __future__ import annotations

import numpy as np

from bench.harness import RequestSpec

#: A system prompt of the kind that sits in front of every request in a real assistant.
SYSTEM_PROMPT = (
    b"You are a helpful assistant. Answer concisely and accurately. "
    b"Cite your sources where possible. Do not speculate beyond the evidence. "
)


def make_chat_trace(
    n_requests: int,
    rate_per_second: float,
    *,
    system_prompt: bytes = SYSTEM_PROMPT,
    user_len: tuple[int, int] = (8, 48),
    output_len: tuple[int, int] = (16, 48),
    seed: int = 0,
) -> list[RequestSpec]:
    """Poisson arrivals where every request shares one long system prompt.

    This is the defining shape of assistant traffic: a fixed preamble that every request pays for,
    followed by a short unique question. It is also why prefix caching is usually the largest
    single win available on a real workload — most of what arrives has already been computed.
    """
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / rate_per_second, size=n_requests)
    arrivals = np.cumsum(gaps)
    user_lengths = rng.integers(user_len[0], user_len[1] + 1, size=n_requests)
    outputs = rng.integers(output_len[0], output_len[1] + 1, size=n_requests)

    prefix = list(system_prompt)
    specs = []
    for arrival, user, out in zip(arrivals, user_lengths, outputs, strict=True):
        # A distinct question per request, so only the system prompt is shareable.
        question = [int(x) for x in rng.integers(33, 126, size=int(user))]
        tokens = tuple(prefix + question)
        specs.append(
            RequestSpec(
                arrival=float(arrival),
                prompt_len=len(tokens),
                max_tokens=int(out),
                tokens=tokens,
            )
        )
    return specs


def make_mixed_prompt_trace(
    n_requests: int,
    rate_per_second: float,
    *,
    short_len: tuple[int, int] = (16, 48),
    long_len: int = 512,
    long_fraction: float = 0.2,
    output_len: tuple[int, int] = (24, 48),
    seed: int = 0,
) -> list[RequestSpec]:
    """Mostly short prompts with an occasional very long one.

    This is the shape chapter 10 exists for. A long prompt is not itself a problem; a long prompt
    processed in a single indivisible step is, because every sequence currently streaming stops
    until it finishes. Traces with uniform prompt lengths hide the effect entirely, which is why
    the earlier chapters' numbers never showed it.
    """
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / rate_per_second, size=n_requests)
    arrivals = np.cumsum(gaps)
    is_long = rng.random(n_requests) < long_fraction
    shorts = rng.integers(short_len[0], short_len[1] + 1, size=n_requests)
    outputs = rng.integers(output_len[0], output_len[1] + 1, size=n_requests)

    specs = []
    for arrival, long_flag, short, out in zip(arrivals, is_long, shorts, outputs, strict=True):
        length = long_len if long_flag else int(short)
        specs.append(RequestSpec(arrival=float(arrival), prompt_len=length, max_tokens=int(out)))
    return specs
