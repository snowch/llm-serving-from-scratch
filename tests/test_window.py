"""Chapter 16: bounding the KV cache, and the position rule that goes with it."""

import pytest

from llmserve.arithmetic import kv_bytes_per_token
from llmserve.cache.window import (
    WindowPolicy,
    blocks_to_keep,
    kv_bytes_with_window,
    positions_for,
)
from llmserve.config import REFERENCE_MODEL


def test_a_short_sequence_is_kept_whole():
    assert WindowPolicy(window=8, sinks=2).keep(6) == [0, 1, 2, 3, 4, 5]


def test_a_long_sequence_keeps_the_sinks_and_the_recent_window():
    kept = WindowPolicy(window=4, sinks=2).keep(20)
    assert kept == [0, 1, 16, 17, 18, 19]


def test_a_plain_sliding_window_drops_the_opening_tokens():
    """The configuration chapter 16 measures to show why sinks were proposed at all."""
    assert WindowPolicy(window=4, sinks=0).keep(20) == [16, 17, 18, 19]


def test_the_budget_is_constant_however_long_the_sequence():
    """The whole point: the cache stops being linear in context."""
    policy = WindowPolicy(window=64, sinks=4)
    assert len(policy.keep(1_000)) == len(policy.keep(1_000_000)) == policy.budget


def test_a_policy_that_keeps_nothing_is_rejected():
    with pytest.raises(ValueError, match="nothing to attend to"):
        WindowPolicy(window=0)
    with pytest.raises(ValueError, match="negative"):
        WindowPolicy(window=8, sinks=-1)


def test_renumbering_compacts_positions_and_not_renumbering_preserves_them():
    kept = WindowPolicy(window=4, sinks=2).keep(20)
    assert positions_for(kept) == [0, 1, 2, 3, 4, 5]
    assert positions_for(kept, renumber=False) == kept


def test_renumbered_positions_stay_inside_the_trained_context():
    """Why renumbering exists: the original indices are what a rotary table cannot look up."""
    policy = WindowPolicy(window=64, sinks=4)
    kept = policy.keep(1_000_000)
    assert max(positions_for(kept)) < policy.budget
    assert max(positions_for(kept, renumber=False)) == 999_999


def test_blocks_are_rounded_outwards_not_inwards():
    """Rounding inwards would drop tokens the policy promised to keep."""
    policy = WindowPolicy(window=4, sinks=2)
    blocks = blocks_to_keep(policy, 20, block_size=4)
    kept = set(policy.keep(20))
    covered = {position for block in blocks for position in range(block * 4, block * 4 + 4)}
    assert kept <= covered


def test_a_sequence_inside_the_budget_keeps_every_block():
    assert blocks_to_keep(WindowPolicy(window=64, sinks=4), 20, block_size=4) == [0, 1, 2, 3, 4]


def test_bounded_memory_stops_growing_with_context():
    per_token = kv_bytes_per_token(REFERENCE_MODEL)
    policy = WindowPolicy(window=512, sinks=4)
    short = kv_bytes_with_window(per_token, policy, 256)
    long = kv_bytes_with_window(per_token, policy, 1_000_000)
    assert short < long
    assert long == per_token * policy.budget
