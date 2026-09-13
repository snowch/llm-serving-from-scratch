"""Chapter 9's engine: paged serving that reuses prompt prefixes it has already computed.

Chapter 8 put every sequence's keys and values in shared, content-agnostic blocks and added
reference counting that nothing used. This chapter uses it. When a request's prompt starts with
tokens the engine has seen before, it points at the existing blocks instead of recomputing them,
and prefills only the part that is genuinely new.

Unlike chapter 8, this costs nothing. It is the rare optimisation with no trade in it, which is
why every production engine turns it on by default.
"""

from __future__ import annotations

import torch

from llmserve.cache.blocks import OutOfBlocksError
from llmserve.cache.prefix import PrefixCache
from llmserve.config import ModelConfig
from llmserve.engines.batched import _sample_batch
from llmserve.engines.paged import PagedEngine
from llmserve.model import TinyGPT
from llmserve.request import RequestState, StepOutput


class PrefixCachedEngine(PagedEngine):
    """Paged serving plus content-addressed reuse of prompt prefixes."""

    name = "prefix"

    def __init__(
        self,
        model: TinyGPT,
        config: ModelConfig,
        *,
        max_batch_size: int = 8,
        n_blocks: int = 256,
        block_size: int = 16,
    ) -> None:
        super().__init__(
            model, config, max_batch_size=max_batch_size, n_blocks=n_blocks, block_size=block_size
        )
        self.prefix = PrefixCache(block_size=block_size)

    # -- memory ------------------------------------------------------------------------

    def _free_blocks_for(self, n: int) -> bool:
        """Make ``n`` blocks available, evicting cached prefixes before preempting anyone.

        Order matters. A cached prefix costs only recomputation if it is wanted again; a preempted
        sequence costs recomputation *and* a latency spike for a caller who is already waiting.
        Evict the cheap thing first.
        """
        while self.cache.allocator.n_free < n:
            block = self.prefix.evict_oldest()
            if block is None:
                return self._preempt_newest()
            self.cache.allocator.free([block])
        return True

    # -- prefill -----------------------------------------------------------------------

    def _prefill(self, states: list[RequestState]) -> list[StepOutput]:
        """Prefill each admitted request, skipping whatever prefix is already cached.

        Requests are handled one at a time rather than as a batch, because each may share a
        different number of blocks and so has a different amount of work left. Batching sequences
        with unequal cached prefixes is possible but fiddly, and a request that hits the cache has
        so little left to do that batching it saves little.
        """
        outputs: list[StepOutput] = []
        for state in states:
            tokens = state.all_token_ids
            shared = self.prefix.lookup(tokens)
            # Never reuse the whole prompt: the model must run on at least one token to produce a
            # distribution for the next one.
            max_shared = max(0, (len(tokens) - 1) // self.cache.block_size)
            shared = shared[:max_shared]

            n_cached = len(shared) * self.cache.block_size
            for block in shared:
                self.cache.allocator.share(block)
            state.block_table = list(shared)

            suffix = tokens[n_cached:]
            needed = self.cache.allocator.blocks_needed(len(tokens)) - len(state.block_table)
            if needed > 0:
                if not self._free_blocks_for(needed):
                    raise OutOfBlocksError(f"cannot allocate {needed} blocks for prefill")
                state.block_table += self.cache.allocator.allocate(needed)

            past = None
            if n_cached:
                past = [
                    self.cache.gather(layer, state.block_table, n_cached)
                    for layer in range(self.cache.n_layers)
                ]

            input_ids = torch.tensor([suffix], dtype=torch.long)
            positions = torch.arange(n_cached, len(tokens), dtype=torch.long).unsqueeze(0)
            logits, present = self.model(input_ids, past, positions)

            for layer in range(self.cache.n_layers):
                k = present[layer][0][0, :, n_cached:, :]
                v = present[layer][1][0, :, n_cached:, :]
                self.cache.write(layer, state.block_table, n_cached, k, v)
            state.stored = len(tokens)
            self._publish(state)

            outputs += self._emit([state], _sample_batch(logits[:, -1, :], [state]))
        return outputs

    def _publish(self, state: RequestState) -> None:
        """Make this sequence's completed blocks available to other requests, immediately.

        Publishing on completion of the *request* would be far too late: requests that arrive
        together all prefill before any of them finishes, so every one of them would recompute the
        same shared prompt. A block is shareable the moment it is full, not the moment its author
        is done with it.
        """
        published = self.prefix.publish(state.all_token_ids[: state.stored], state.block_table)
        for block in published:
            self.cache.allocator.share(block)

    # -- completion --------------------------------------------------------------------

    def _release(self, state: RequestState) -> None:
        """Publish this sequence's full blocks before returning the rest to the pool.

        Publishing adds a reference, so the cache keeps those blocks alive after the sequence that
        produced them is gone — which is the entire point.
        """
        if state.block_table and state.stored:
            self._publish(state)  # any blocks that filled during decode
        super()._release(state)
