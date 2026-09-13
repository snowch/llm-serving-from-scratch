"""The workload shapes Part VI is about. Each test states a property a chapter relies on."""

import pytest

from bench.traces import (
    make_agent_trace,
    make_completion_trace,
    make_multi_tenant_trace,
    make_noisy_neighbour_trace,
    make_offline_batch_trace,
    make_rag_trace,
    make_session_turns,
)


def _shared_prefix(a, b) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def test_traces_are_deterministic():
    """A benchmark whose trace changes between runs measures nothing."""
    assert make_rag_trace(8, 4.0, seed=1) == make_rag_trace(8, 4.0, seed=1)
    assert make_agent_trace(8, 4.0, seed=1) == make_agent_trace(8, 4.0, seed=1)


def test_arrivals_are_ordered():
    for trace in (
        make_rag_trace(12, 4.0),
        make_agent_trace(12, 4.0),
        make_completion_trace(12, 4.0),
    ):
        arrivals = [s.arrival for s in trace]
        assert arrivals == sorted(arrivals)


def test_retrieval_prompts_diverge_after_the_instruction():
    """Chapter 21's claim: RAG shares the instruction and then stops sharing.

    Every request opens with the same system prompt, so there is something to reuse; the retrieved
    passages differ, so there is much less of it than chat enjoys.
    """
    trace = make_rag_trace(12, 8.0, seed=2)
    shared = [
        _shared_prefix(a.tokens, b.tokens)
        for a, b in zip(trace, trace[1:], strict=False)
        if a.tokens != b.tokens
    ]
    assert shared, "expected distinct retrieval prompts"
    instruction = len(b"Answer only from the passages below. Cite the passage number. ")
    assert min(shared) >= instruction - 1
    assert max(shared) < min(s.prompt_len for s in trace) / 2


def test_agent_prompts_extend_one_another():
    """Chapter 22's claim: an agent replays the transcript, so step n contains step n-1."""
    trace = make_agent_trace(24, 8.0, seed=4)
    by_length = sorted(trace, key=lambda s: s.prompt_len)
    short, long = by_length[0], by_length[-1]
    assert short.prompt_len < long.prompt_len
    assert long.tokens[: short.prompt_len] == short.tokens


def test_session_turns_grow_and_contain_their_history():
    turns = make_session_turns(3, 4, seed=5)
    assert len(turns) == 4
    for session in range(3):
        lengths = [len(turns[t][session]) for t in range(4)]
        assert lengths == sorted(lengths)
        for t in range(1, 4):
            previous = turns[t - 1][session]
            assert turns[t][session][: len(previous)] == previous


def test_completion_prompts_are_small_and_outputs_smaller():
    trace = make_completion_trace(16, 20.0)
    assert max(s.prompt_len for s in trace) < 256
    assert max(s.max_tokens for s in trace) <= 20


def test_offline_batch_has_no_arrivals():
    """Chapter 23: there is nobody waiting, so there is no arrival process to model."""
    trace = make_offline_batch_trace(16)
    assert {s.arrival for s in trace} == {0.0}


def test_noisy_neighbour_burst_dominates_the_trace():
    trace = make_noisy_neighbour_trace(4.0)
    counts: dict[str, int] = {}
    for spec in trace:
        counts[spec.tenant] = counts.get(spec.tenant, 0) + 1
    assert counts["hog"] > sum(v for k, v in counts.items() if k != "hog")


def test_multi_tenant_trace_has_several_distinct_prefixes():
    """Chapter 18 needs more than one prefix, or every routing policy looks identical."""
    trace = make_multi_tenant_trace(24, 8.0, seed=6)
    assert len({spec.tokens[:64] for spec in trace}) > 1


@pytest.mark.parametrize(
    "make",
    [
        lambda: make_rag_trace(8, 4.0),
        lambda: make_agent_trace(8, 4.0),
        lambda: make_completion_trace(8, 4.0),
        lambda: make_multi_tenant_trace(8, 4.0),
    ],
)
def test_declared_prompt_length_matches_the_tokens(make):
    """A spec whose prompt_len disagrees with its tokens silently corrupts every measurement."""
    for spec in make():
        assert spec.prompt_len == len(spec.tokens)
