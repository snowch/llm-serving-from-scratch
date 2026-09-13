"""Chapter 12: a second tier for the KV cache.

Chapter 8's allocator has exactly one answer when it runs out of blocks: throw something away.
Chapter 9 softened it — give up a cached prefix before preempting a live sequence — but the shape is
the same. Whatever is evicted has to be *recomputed* if it is wanted again, and recomputing a long
prefix is a full prefill nobody asked for.

There is a third option the engine has not had: move it somewhere slower and larger. Device memory
is not the only memory in the machine, and a block sitting in host RAM is still far cheaper to fetch
than to recompute. That turns eviction from a loss into a demotion, and it is the same trade an
operating system makes when it pages rather than discards.

Whether it pays is arithmetic, and the arithmetic is chapter 11's handoff calculation wearing a
different hat: fetching costs bytes over a link, recomputing costs a prefill. Below a crossover
point one wins and above it the other does, and the crossover moves with the link.
"""

from __future__ import annotations

import torch


class OffloadTier:
    """A larger, slower place to keep KV blocks that device memory cannot hold.

    Host RAM here, because that is the tier every machine has. The interesting production tiers —
    local NVMe, a shared KV store across a fleet — differ only in bandwidth and latency, which is
    exactly what :func:`fetch_vs_recompute` takes as inputs.

    Blocks are stored by content key rather than by physical block number. A physical block is a
    slot the allocator hands out and reuses; a key identifies *what was in it*, which is what makes
    a demoted block findable after the slot it came from has been given to somebody else. Chapter 9
    already computes exactly such a key for every full block.
    """

    def __init__(self, n_layers: int, capacity_blocks: int = 4096) -> None:
        if capacity_blocks < 1:
            raise ValueError("an offload tier with no capacity is not a tier")
        self.n_layers = n_layers
        self.capacity_blocks = capacity_blocks
        #: key -> per-layer (keys, values), held in host memory
        self._store: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        #: insertion order, for LRU demotion out of the tier itself
        self._order: list[int] = []
        self.stores = 0
        self.fetches = 0
        self.misses = 0
        self.evicted = 0

    def __len__(self) -> int:
        return len(self._store)

    @property
    def utilisation(self) -> float:
        return len(self._store) / self.capacity_blocks

    def store(self, key: int, layers: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        """Demote one block's KV out of device memory.

        The tensors are cloned. Keeping a view would leave the tier pointing into a block the
        allocator is about to hand to another sequence, which is the quiet kind of corruption that
        shows up as one request reading another's context.
        """
        if key in self._store:
            self._touch(key)
            return
        while len(self._store) >= self.capacity_blocks:
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)
            self.evicted += 1
        self._store[key] = [(k.clone(), v.clone()) for k, v in layers]
        self._order.append(key)
        self.stores += 1

    def fetch(self, key: int) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Promote a block back, or report that it is gone and must be recomputed."""
        found = self._store.get(key)
        if found is None:
            self.misses += 1
            return None
        self._touch(key)
        self.fetches += 1
        return found

    def _touch(self, key: int) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)

    @property
    def hit_rate(self) -> float:
        total = self.fetches + self.misses
        return self.fetches / total if total else 0.0


def fetch_bytes(bytes_per_token: float, block_size: int, n_blocks: int = 1) -> float:
    """Bytes a promotion moves. One block's keys and values, per layer, already counted in
    ``bytes_per_token``."""
    return bytes_per_token * block_size * n_blocks


def fetch_vs_recompute(
    bytes_per_token: float,
    block_size: int,
    n_blocks: int,
    *,
    link_bytes_per_second: float,
    prefill_tokens_per_second: float,
) -> dict:
    """Which is cheaper for one sequence's worth of evicted prefix: fetching it or recomputing it.

    The comparison that decides whether a second tier is worth having, and it is decided by the link
    rather than by anything in the engine. Recompute time is tokens divided by prefill throughput;
    fetch time is bytes divided by bandwidth. Both are honest floors — they ignore latency, which
    hurts the fetch, and ignore the scheduling damage a surprise prefill does to everyone else,
    which hurts the recompute considerably more.
    """
    tokens = block_size * n_blocks
    fetch_seconds = fetch_bytes(bytes_per_token, block_size, n_blocks) / link_bytes_per_second
    recompute_seconds = tokens / prefill_tokens_per_second
    return {
        "tokens": tokens,
        "bytes": fetch_bytes(bytes_per_token, block_size, n_blocks),
        "fetch_seconds": fetch_seconds,
        "recompute_seconds": recompute_seconds,
        "speedup": recompute_seconds / fetch_seconds if fetch_seconds else float("inf"),
    }


def break_even_bandwidth(bytes_per_token: float, prefill_tokens_per_second: float) -> float:
    """Link bandwidth at which fetching a block costs exactly what recomputing it does.

    Above this, a second tier pays; below it, recomputing is cheaper and the tier is a liability.
    Note what cancels: block size drops out entirely, so the answer is a property of the model and
    the hardware rather than of the allocator's geometry.
    """
    return bytes_per_token * prefill_tokens_per_second
