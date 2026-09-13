"""Constrained decoding: making invalid output impossible rather than unlikely.

Chapter 16. Callers who want JSON currently get JSON *usually*. Prompting improves the odds and
retrying covers some of the rest, and both are ways of paying for a guarantee you can simply have:
at every step, mask the logits of tokens that cannot legally come next.

The grammar is a finite-state machine over the vocabulary. In state *s*, only some tokens keep the
output parseable; every other logit is set to negative infinity before sampling. Invalid output
stops being unlikely and becomes unreachable.

A byte-level tokenizer makes this much simpler than it is in production. With BPE a single token
can span several grammar symbols, so the FSM must be lifted from characters to token sequences —
that lifting, not the grammar, is where real implementations spend their complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import torch


class State(Enum):
    """States of a JSON-object grammar: ``{"key": "value", "key2": "value2"}``.

    Deliberately a subset. Nesting, numbers, arrays and escapes add states without adding
    understanding, and the mechanism is identical for all of them.
    """

    START = auto()
    EXPECT_KEY_OR_END = auto()
    IN_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_VALUE_START = auto()
    IN_VALUE = auto()
    EXPECT_COMMA_OR_END = auto()
    DONE = auto()


QUOTE = ord('"')
COLON = ord(":")
COMMA = ord(",")
OPEN_BRACE = ord("{")
CLOSE_BRACE = ord("}")
SPACE = ord(" ")

#: Characters allowed inside a string. Excludes the quote (which ends it) and the backslash
#: (which would open an escape sequence this subset does not model).
STRING_BYTES = frozenset(b for b in range(0x20, 0x7F) if b not in (QUOTE, ord("\\")))


@dataclass
class JSONGrammar:
    """A finite-state machine over bytes, and the token masks each state implies.

    Masks are cached per state. Building one costs a pass over the vocabulary; doing that on every
    decode step is the naive implementation chapter 16 measures against, and the reason
    constrained decoding has a reputation for being slow.
    """

    vocab_size: int
    eos_id: int | None = None
    _cache: dict[State, torch.Tensor] = field(default_factory=dict, repr=False)
    #: how many masks were built rather than reused — the number caching drives to almost nothing
    masks_built: int = 0

    def allowed_bytes(self, state: State) -> frozenset[int]:
        """Which byte values keep the output parseable from this state."""
        if state is State.START:
            return frozenset({OPEN_BRACE})
        if state is State.EXPECT_KEY_OR_END:
            return frozenset({QUOTE, CLOSE_BRACE, SPACE})
        if state is State.IN_KEY:
            return frozenset(STRING_BYTES | {QUOTE})
        if state is State.EXPECT_COLON:
            return frozenset({COLON, SPACE})
        if state is State.EXPECT_VALUE_START:
            return frozenset({QUOTE, SPACE})
        if state is State.IN_VALUE:
            return frozenset(STRING_BYTES | {QUOTE})
        if state is State.EXPECT_COMMA_OR_END:
            return frozenset({COMMA, CLOSE_BRACE, SPACE})
        return frozenset()  # DONE: only the end-of-sequence token remains

    def step(self, state: State, byte: int) -> State:
        """Advance the machine. Callers must only pass bytes the mask allowed."""
        if state is State.START:
            return State.EXPECT_KEY_OR_END
        if state is State.EXPECT_KEY_OR_END:
            if byte == QUOTE:
                return State.IN_KEY
            return State.DONE if byte == CLOSE_BRACE else state
        if state is State.IN_KEY:
            return State.EXPECT_COLON if byte == QUOTE else state
        if state is State.EXPECT_COLON:
            return State.EXPECT_VALUE_START if byte == COLON else state
        if state is State.EXPECT_VALUE_START:
            return State.IN_VALUE if byte == QUOTE else state
        if state is State.IN_VALUE:
            return State.EXPECT_COMMA_OR_END if byte == QUOTE else state
        if state is State.EXPECT_COMMA_OR_END:
            if byte == COMMA:
                return State.EXPECT_KEY_OR_END
            return State.DONE if byte == CLOSE_BRACE else state
        return State.DONE

    def build_mask(self, state: State) -> torch.Tensor:
        """A boolean mask over the vocabulary: True where the token is legal here."""
        self.masks_built += 1
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for byte in self.allowed_bytes(state):
            mask[byte] = True
        if state is State.DONE and self.eos_id is not None:
            mask[self.eos_id] = True
        return mask

    def mask(self, state: State, *, cached: bool = True) -> torch.Tensor:
        """The mask for a state, built once and reused.

        There are a handful of states and thousands of decode steps, so caching turns a per-step
        vocabulary scan into a dictionary lookup. Passing ``cached=False`` reproduces the naive
        implementation for comparison.
        """
        if not cached:
            return self.build_mask(state)
        if state not in self._cache:
            self._cache[state] = self.build_mask(state)
        return self._cache[state]

    def apply(self, logits: torch.Tensor, state: State, *, cached: bool = True) -> torch.Tensor:
        """Mask illegal tokens.

        Uses the dtype's most negative finite value rather than -inf, for the reason chapter 6
        found the hard way: a row that ends up entirely -inf softmaxes to NaN, and a grammar that
        permits nothing is a bug you want to see as a strange token rather than as corruption
        spreading through a batch.
        """
        allowed = self.mask(state, cached=cached)
        return logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)


def is_valid_json_object(text: str) -> bool:
    """Whether the grammar's output actually parses. The claim, checked independently."""
    import json

    try:
        return isinstance(json.loads(text), dict)
    except (ValueError, TypeError):
        return False
