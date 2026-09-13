"""Prefix caching: stop recomputing prompts the engine has already seen.

Chapter 8 made a sequence's cache a list of block numbers, and nothing about a block ties it to
one sequence. That is the whole opening. If two requests begin with the same tokens, the blocks
holding that prefix are identical, so the second request can point at the first one's blocks
instead of recomputing them.

The prefix is keyed by content, not by request. A block's key is a hash of *every token up to and
including it*, not just the tokens inside it — otherwise two different prefixes that happen to
share a middle block would collide, and a request would attend to somebody else's context.
"""

from __future__ import annotations

from collections import OrderedDict


class PrefixCache:
    """Maps prompt prefixes to the physical blocks holding their keys and values.

    Only *full* blocks are shareable. A partially filled block is still being written, so its
    contents depend on tokens that have not arrived; publishing it would let another sequence read
    whatever happens to be in the unused slots.

    Eviction is LRU over entries that no live sequence is using. Reference counts come from the
    chapter 8 allocator, so a cached block is never reclaimed while somebody is reading it.
    """

    def __init__(self, block_size: int, capacity: int = 4096) -> None:
        self.block_size = block_size
        self.capacity = capacity
        self._entries: OrderedDict[int, int] = OrderedDict()  # prefix hash -> physical block
        self.hits = 0
        self.misses = 0
        self.hit_tokens = 0
        self.total_prompt_tokens = 0

    @staticmethod
    def _key(tokens: tuple[int, ...]) -> int:
        return hash(tokens)

    def block_keys(self, token_ids: list[int]) -> list[int]:
        """Cumulative hashes, one per full block of ``token_ids``."""
        n_full = len(token_ids) // self.block_size
        return [self._key(tuple(token_ids[: (i + 1) * self.block_size])) for i in range(n_full)]

    def lookup(self, token_ids: list[int]) -> list[int]:
        """Return the physical blocks covering the longest cached prefix of ``token_ids``.

        Stops at the first miss: a prefix is only usable if every block before it is too, so a
        later hit after a gap is worthless and must not be used.
        """
        blocks: list[int] = []
        for key in self.block_keys(token_ids):
            block = self._entries.get(key)
            if block is None:
                break
            self._entries.move_to_end(key)
            blocks.append(block)

        self.total_prompt_tokens += len(token_ids)
        if blocks:
            self.hits += 1
            self.hit_tokens += len(blocks) * self.block_size
        else:
            self.misses += 1
        return blocks

    def publish(self, token_ids: list[int], block_table: list[int]) -> list[int]:
        """Register this sequence's full blocks for reuse. Returns blocks newly published.

        The caller adds a reference for each, so the cache's own hold keeps them alive after the
        sequence that created them has finished.
        """
        published = []
        for i, key in enumerate(self.block_keys(token_ids)):
            if key in self._entries or i >= len(block_table):
                continue
            self._entries[key] = block_table[i]
            published.append(block_table[i])
        return published

    def evict_oldest(self) -> int | None:
        """Drop the least recently used entry and return its block, or None if empty."""
        if not self._entries:
            return None
        _, block = self._entries.popitem(last=False)
        return block

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    @property
    def token_reuse_rate(self) -> float:
        """Fraction of prompt tokens served from cache rather than recomputed.

        The number that actually matters: a hit on one block of a long prompt saves little, and
        this reports what was really avoided.
        """
        return self.hit_tokens / self.total_prompt_tokens if self.total_prompt_tokens else 0.0
