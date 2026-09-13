"""Chapter 12's engine: eviction becomes demotion.

Chapter 9's engine gives up a cached prefix when it needs blocks, and whatever it gives up has to be
recomputed if it is wanted again. This one hands the block to a slower, larger tier on the way out,
so wanting it again costs a transfer instead of a prefill.

Everything else is chapter 9's engine. That is the point worth noticing: the change is entirely in
what happens at the eviction boundary, which is why an engine that got chapter 8's block accounting
right can grow a memory hierarchy without touching its scheduler.
"""

from __future__ import annotations

import torch

from llmserve.cache.offload import OffloadTier
from llmserve.config import ModelConfig
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import TinyGPT
from llmserve.request import RequestState, StepOutput


class OffloadEngine(PrefixCachedEngine):
    """Prefix-cached serving with a second KV tier behind it."""

    name = "offload"

    def __init__(
        self,
        model: TinyGPT,
        config: ModelConfig,
        *,
        max_batch_size: int = 8,
        n_blocks: int = 256,
        block_size: int = 16,
        tier_blocks: int = 4096,
    ) -> None:
        super().__init__(
            model, config, max_batch_size=max_batch_size, n_blocks=n_blocks, block_size=block_size
        )
        self.tier = OffloadTier(self.cache.n_layers, capacity_blocks=tier_blocks)

    # -- demotion ----------------------------------------------------------------------

    def _demote(self) -> bool:
        """Move the least recently used cached block down a tier. True if there was one."""
        evicted = self.prefix.evict_oldest_with_key()
        if evicted is None:
            return False
        key, block = evicted
        self.tier.store(
            key,
            [
                (self.cache.keys[layer][block], self.cache.values[layer][block])
                for layer in range(self.cache.n_layers)
            ],
        )
        self.cache.allocator.free([block])
        return True

    def _make_room(self, n: int) -> bool:
        while self.cache.allocator.n_free < n:
            if not self._demote():
                return False
        return True

    def _free_blocks_for(self, n: int) -> bool:
        while self.cache.allocator.n_free < n:
            if self._demote():
                continue
            if not self._preempt_newest():
                return False
        return True

    # -- promotion ---------------------------------------------------------------------

    @torch.inference_mode()
    def _promote(self, token_ids: list[int]) -> int:
        """Bring back whatever of this prompt the tier still holds, before the cache is consulted.

        Restoring *before* the lookup rather than patching the lookup's result is what keeps this
        chapter's change small: chapter 9's prefill then finds the blocks resident and proceeds
        exactly as it always has, with no second code path to keep in step.

        Stops at the first key the tier does not have, for chapter 9's reason — a hit across a gap
        is unusable, because the blocks after the gap were computed against context that is missing.
        """
        promoted = 0
        for key in self.prefix.block_keys(token_ids):
            if key in self.prefix._entries:
                continue
            layers = self.tier.fetch(key)
            if layers is None:
                break
            if self.cache.allocator.n_free < 1 and not self._make_room(1):
                break
            block = self.cache.allocator.allocate(1)[0]
            for layer, (keys, values) in enumerate(layers):
                self.cache.keys[layer][block] = keys
                self.cache.values[layer][block] = values
            self.prefix.insert(key, block)
            promoted += 1
        return promoted

    def _prefill(self, states: list[RequestState]) -> list[StepOutput]:
        for state in states:
            self._promote(state.all_token_ids)
        return super()._prefill(states)
