"""Chapter 16: keeping the KV cache bounded, whatever the context length.

Every chapter so far has made the cache cheaper per token — fewer KV heads (chapter 13), fewer bytes
per entry (chapter 15), fewer copies (chapter 14). All of them leave one thing untouched: the cache
still grows linearly with the conversation, so a long enough sequence exhausts any budget. Chapter
23's retrieval workload runs into this, and so does any agent whose transcript keeps growing.

The other lever is to stop keeping all of it. That is a different kind of trade from everything
before it, and an uncomfortable one: this is the first chapter where the optimisation can change what
the model says. Dropping a token's keys and values does not slow the model down, it removes
information the model would otherwise have attended to.

What makes it work at all is a result that looks like a bug when you first meet it. The first few
tokens of a sequence matter enormously, out of all proportion to their content, because softmax must
put its attention mass *somewhere* and those positions have become the place it goes. Drop them and
quality collapses; keep a handful and a window of recent tokens is enough.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowPolicy:
    """Which of a sequence's tokens to keep once it outgrows its budget.

    ``window`` is how many recent tokens are always retained. ``sinks`` is how many tokens from the
    very beginning are retained regardless of how far back they fall — the "attention sinks" of the
    StreamingLLM result. Setting ``sinks`` to zero gives a plain sliding window, which is the
    obvious implementation and the one chapter 16 measures to show why it is not enough.
    """

    window: int = 256
    sinks: int = 4

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError("a window of no tokens leaves nothing to attend to")
        if self.sinks < 0:
            raise ValueError("sinks cannot be negative")

    @property
    def budget(self) -> int:
        """The most tokens this policy will ever hold for one sequence. Constant, not linear."""
        return self.window + self.sinks

    def keep(self, length: int) -> list[int]:
        """Positions to retain for a sequence of ``length`` tokens, in order.

        Returns original positions, not renumbered ones. What the model is then told about those
        positions is a separate decision, and the one implementations most often get wrong — see
        :func:`positions_for`.
        """
        if length <= self.budget:
            return list(range(length))
        recent = range(length - self.window, length)
        return [*range(min(self.sinks, length)), *recent]


def positions_for(kept: list[int], renumber: bool = True) -> list[int]:
    """The position indices the model should be given for the tokens that survived.

    This is the subtle half of the technique, and the half where the obvious rule is wrong half the
    time. After eviction the cache holds tokens 0-3 and 9000-9255. Renumbering by position *within
    the cache* keeps every distance inside the range the model was trained on; keeping the original
    indices tells the model the truth about how far apart those groups are.

    Which is better depends entirely on whether the original positions are still inside the trained
    context, and chapter 16 measures both sides of that line rather than asserting either:

    * **Inside it**, renumbering is a lie the model can detect. It says four tokens from the distant
      past are adjacent to the most recent ones, and quality is worse than simply telling the truth.
    * **Beyond it**, the truth is unusable. Position 9000 on a model trained to 4096 is a rotation
      it has never seen, and renumbering is what turns a bounded window from broken into workable.

    So the answer is not "always renumber" — it is "renumber once you are past what the model was
    trained on", which is a narrower and more useful rule than the one usually given.
    """
    return list(range(len(kept))) if renumber else list(kept)


def blocks_to_keep(policy: WindowPolicy, length: int, block_size: int) -> list[int]:
    """Which *blocks* of a sequence survive, for an engine that allocates by block (chapter 8).

    Eviction happens a block at a time because that is the unit the allocator hands out, so the
    retained set is rounded outwards to whole blocks and the real budget is slightly larger than the
    policy asks for. Rounding inwards instead would silently drop tokens the policy promised to
    keep, which is the wrong direction to be wrong in.
    """
    if length <= policy.budget:
        return list(range((length + block_size - 1) // block_size))
    kept = policy.keep(length)
    return sorted({position // block_size for position in kept})


def kv_bytes_with_window(
    bytes_per_token: float, policy: WindowPolicy, context_length: int
) -> float:
    """KV bytes one sequence occupies under this policy — capped, rather than linear in context."""
    return bytes_per_token * min(context_length, policy.budget)
