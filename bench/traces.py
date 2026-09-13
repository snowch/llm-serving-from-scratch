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


#: Several distinct system prompts, as a multi-tenant deployment would have.
TENANT_PROMPTS = [
    b"You are a helpful assistant. Answer concisely and cite sources. ",
    b"You are a code reviewer. Point out bugs and suggest improvements clearly. ",
    b"You are a translator. Preserve tone and register in the target language. ",
    b"You are a summariser. Produce three bullet points and no preamble. ",
    b"You are a SQL assistant. Return only valid queries against the schema. ",
    b"You are a support agent. Be polite, specific, and never speculate. ",
]


def make_multi_tenant_trace(
    n_requests: int,
    rate_per_second: float,
    *,
    tenants: list[bytes] | None = None,
    repeat: int = 6,
    user_len: tuple[int, int] = (8, 32),
    output_len: tuple[int, int] = (16, 32),
    seed: int = 0,
) -> list[RequestSpec]:
    """Poisson arrivals across several tenants, each with its own system prompt.

    This is the shape that makes routing policy matter (chapter 18). With a single shared prefix
    every replica warms its own copy and the policy is irrelevant. With several prefixes, a router
    that scatters them makes every replica cache every prefix, while one that keeps a prefix on a
    replica lets each cache only what it serves.
    """
    tenants = tenants or TENANT_PROMPTS
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / rate_per_second, size=n_requests)
    arrivals = np.cumsum(gaps)
    # A long preamble per tenant, so the cacheable part dominates the unique part.
    prefixes = [list(prompt * repeat) for prompt in tenants]
    choice = rng.integers(0, len(prefixes), size=n_requests)
    user_lengths = rng.integers(user_len[0], user_len[1] + 1, size=n_requests)
    outputs = rng.integers(output_len[0], output_len[1] + 1, size=n_requests)

    specs = []
    for arrival, tenant, user, out in zip(arrivals, choice, user_lengths, outputs, strict=True):
        question = [int(x) for x in rng.integers(33, 126, size=int(user))]
        tokens = tuple(prefixes[int(tenant)] + question)
        specs.append(
            RequestSpec(
                arrival=float(arrival),
                prompt_len=len(tokens),
                max_tokens=int(out),
                tokens=tokens,
            )
        )
    return specs


def make_noisy_neighbour_trace(
    rate_per_second: float,
    *,
    quiet_tenants: tuple[str, ...] = ("alice", "bob"),
    noisy_tenant: str = "hog",
    quiet_requests: int = 6,
    burst: int = 18,
    burst_at: float = 0.2,
    quiet_prompt: int = 64,
    quiet_output: int = 16,
    burst_prompt: int = 256,
    burst_output: int = 64,
    seed: int = 0,
) -> list[RequestSpec]:
    """Two well-behaved tenants and one that submits a large burst of expensive requests.

    This is the shape that exposes head-of-line blocking between customers, and it is deliberately
    not subtle: the noisy tenant's requests are both more numerous and individually larger, so a
    first-come-first-served queue serves almost nothing else until the burst drains. Real noisy
    neighbours look exactly like this — a backfill job, a retry storm, one customer's batch export.
    """
    rng = np.random.default_rng(seed)
    specs: list[RequestSpec] = []

    for tenant in quiet_tenants:
        gaps = rng.exponential(1.0 / rate_per_second, size=quiet_requests)
        for arrival in np.cumsum(gaps):
            specs.append(
                RequestSpec(
                    arrival=float(arrival),
                    prompt_len=quiet_prompt,
                    max_tokens=quiet_output,
                    tenant=tenant,
                )
            )

    # The burst arrives close together, as a burst does.
    for i in range(burst):
        specs.append(
            RequestSpec(
                arrival=burst_at + i * 0.01,
                prompt_len=burst_prompt,
                max_tokens=burst_output,
                tenant=noisy_tenant,
            )
        )

    return sorted(specs, key=lambda s: s.arrival)


#: Retrieved passages, as a RAG pipeline's corpus chunks. Each is long enough to dominate a prompt.
CORPUS_CHUNKS = 12
CHUNK_TOKENS = 192

RAG_SYSTEM = b"Answer only from the passages below. Cite the passage number. "


def make_rag_trace(
    n_requests: int,
    rate_per_second: float,
    *,
    top_k: int = 3,
    chunks: int = CORPUS_CHUNKS,
    chunk_tokens: int = CHUNK_TOKENS,
    output_len: tuple[int, int] = (16, 48),
    seed: int = 0,
) -> list[RequestSpec]:
    """Retrieval-augmented prompts: a shared instruction, then ``top_k`` retrieved passages.

    The shape that matters for chapter 21 is where requests stop being identical. Every request
    shares the system prompt, so the first block or two are reusable. After that each request
    carries a different set of passages in a different order, so the prompt diverges — and a prefix
    cache can only reuse up to the first difference. That is why prefix caching does far less for
    retrieval than it does for chat, and it is a property of the workload, not of the cache.
    """
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / rate_per_second, size=n_requests)
    arrivals = np.cumsum(gaps)
    # A deterministic corpus: chunk i is a repeated byte pattern unique to i.
    corpus = [[(i * 7 + j) % 200 + 33 for j in range(chunk_tokens)] for i in range(chunks)]
    outputs = rng.integers(output_len[0], output_len[1] + 1, size=n_requests)

    specs = []
    for arrival, out in zip(arrivals, outputs, strict=True):
        retrieved = rng.choice(chunks, size=top_k, replace=False)
        tokens = list(RAG_SYSTEM)
        for index in retrieved:
            tokens += corpus[int(index)]
        tokens += [int(x) for x in rng.integers(33, 126, size=24)]  # the user's question
        specs.append(
            RequestSpec(
                arrival=float(arrival),
                prompt_len=len(tokens),
                max_tokens=int(out),
                tokens=tuple(tokens),
            )
        )
    return specs


AGENT_PREAMBLE = (
    b"You are an agent. Tools: search(query), read(path), write(path, text), run(cmd). "
    b"Think step by step and emit exactly one tool call per turn. "
)


def make_agent_trace(
    n_requests: int,
    rate_per_second: float,
    *,
    preamble_repeat: int = 4,
    history_turns: tuple[int, int] = (0, 6),
    turn_tokens: int = 48,
    output_len: tuple[int, int] = (8, 24),
    seed: int = 0,
) -> list[RequestSpec]:
    """An agent loop: the same tool preamble every time, a growing transcript, a short reply.

    Agent traffic is the most repetitive workload in this book — every step of every run replays
    the entire transcript so far, so request *n* of a run is request *n-1* plus one turn. That is
    the best case for prefix reuse and the worst case for anything that recomputes prompts.
    """
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / rate_per_second, size=n_requests)
    arrivals = np.cumsum(gaps)
    preamble = list(AGENT_PREAMBLE * preamble_repeat)
    turns = rng.integers(history_turns[0], history_turns[1] + 1, size=n_requests)
    outputs = rng.integers(output_len[0], output_len[1] + 1, size=n_requests)

    specs = []
    for arrival, n_turns, out in zip(arrivals, turns, outputs, strict=True):
        tokens = list(preamble)
        # The transcript is deterministic given its length, so step n really does extend step n-1.
        for turn in range(int(n_turns)):
            tokens += [(turn * 31 + j) % 200 + 33 for j in range(turn_tokens)]
        specs.append(
            RequestSpec(
                arrival=float(arrival),
                prompt_len=len(tokens),
                max_tokens=int(out),
                tokens=tuple(tokens),
            )
        )
    return specs


def make_completion_trace(
    n_requests: int,
    rate_per_second: float,
    *,
    prefix_len: int = 160,
    suffix_len: int = 48,
    output_len: tuple[int, int] = (6, 20),
    seed: int = 0,
) -> list[RequestSpec]:
    """Fill-in-the-middle code completion: a small prompt, a tiny output, a brutal latency target.

    Nothing here is large. The whole difficulty is that a completion is useless if it arrives after
    the developer has typed the next character, so this workload is the one place in the book where
    time to first token is the only measure that matters and throughput is nearly irrelevant.
    """
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / rate_per_second, size=n_requests)
    arrivals = np.cumsum(gaps)
    outputs = rng.integers(output_len[0], output_len[1] + 1, size=n_requests)

    specs = []
    for arrival, out in zip(arrivals, outputs, strict=True):
        # The file under the cursor: mostly stable across keystrokes, which is the point.
        buffer_id = int(rng.integers(0, 4))
        prefix = [(buffer_id * 13 + j) % 200 + 33 for j in range(prefix_len)]
        suffix = [int(x) for x in rng.integers(33, 126, size=suffix_len)]
        tokens = prefix + suffix
        specs.append(
            RequestSpec(
                arrival=float(arrival),
                prompt_len=len(tokens),
                max_tokens=int(out),
                tokens=tuple(tokens),
            )
        )
    return specs


def make_offline_batch_trace(
    n_requests: int,
    *,
    prompt_len: tuple[int, int] = (128, 512),
    output_len: tuple[int, int] = (32, 96),
    seed: int = 0,
) -> list[RequestSpec]:
    """Every request available at once, because there is no user waiting for any of them.

    This is the only workload in the book without arrivals, and removing them removes the entire
    reason the chapter 2 harness is open-loop. Offline batch has no time to first token worth
    reporting and no service objective: the only number is tokens per unit of hardware time.
    """
    rng = np.random.default_rng(seed)
    prompts = rng.integers(prompt_len[0], prompt_len[1] + 1, size=n_requests)
    outputs = rng.integers(output_len[0], output_len[1] + 1, size=n_requests)
    return [
        RequestSpec(arrival=0.0, prompt_len=int(p), max_tokens=int(o))
        for p, o in zip(prompts, outputs, strict=True)
    ]


def make_session_turns(
    n_sessions: int,
    turns: int,
    *,
    system_prompt: bytes = SYSTEM_PROMPT,
    system_repeat: int = 4,
    user_len: int = 40,
    reply_len: int = 40,
    seed: int = 0,
) -> list[list[tuple[int, ...]]]:
    """Prompts for ``n_sessions`` conversations, ``turns`` deep, indexed ``[turn][session]``.

    Turn *t* of a session is turn *t-1* plus the assistant's reply plus the next user message, which
    is what a chat API actually receives: the client resends the whole transcript every time. The
    growth is the defining property of the workload — each turn's prompt is longer than the last and
    every byte of it except the newest message has been seen before.

    Returned per turn rather than as a flat trace because chapter 20 measures how reuse changes
    *with depth*, which means serving all of turn 1, then all of turn 2, and so on.
    """
    rng = np.random.default_rng(seed)
    preamble = list(system_prompt * system_repeat)
    transcripts = [list(preamble) for _ in range(n_sessions)]
    out: list[list[tuple[int, ...]]] = []

    for _ in range(turns):
        this_turn = []
        for session in range(n_sessions):
            transcripts[session] += [int(x) for x in rng.integers(33, 126, size=user_len)]
            this_turn.append(tuple(transcripts[session]))
            # The assistant's reply becomes part of the next turn's prompt.
            transcripts[session] += [int(x) for x in rng.integers(33, 126, size=reply_len)]
        out.append(this_turn)
    return out
