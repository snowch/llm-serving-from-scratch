"""Chapter 10's engine: a token budget per step, and prefill broken into chunks.

Every engine so far has treated a prefill as indivisible. Admitting a request means running its
entire prompt in one step, and every sequence already streaming waits for that step to finish. A
caller mid-sentence experiences somebody else's long prompt as a pause.

The fix is to stop letting a step be as large as the work that arrives. Give each step a token
budget, spend it on decode first because those tokens are latency-critical, and fill whatever is
left with a *piece* of a pending prefill. A long prompt then arrives over several steps instead of
stopping the world for one.

This is Sarathi-Serve's scheduling, and it makes the trade between time-to-first-token and smooth
streaming explicit and adjustable rather than accidental.
"""

from __future__ import annotations

import torch

from llmserve.cache.blocks import OutOfBlocksError
from llmserve.config import ModelConfig
from llmserve.engines.batched import _sample_batch
from llmserve.engines.prefix import PrefixCachedEngine
from llmserve.model import TinyGPT
from llmserve.request import RequestState, StepOutput


class ChunkedPrefillEngine(PrefixCachedEngine):
    """Prefix-cached paged serving under a per-step token budget.

    ``token_budget`` is the dial. Small keeps streaming smooth at the cost of slower prefill, so
    time-to-first-token rises. Large does the reverse. There is no universally correct setting;
    Part VI tunes it per workload.
    """

    name = "chunked"

    def __init__(
        self,
        model: TinyGPT,
        config: ModelConfig,
        *,
        max_batch_size: int = 8,
        n_blocks: int = 512,
        block_size: int = 16,
        token_budget: int = 64,
    ) -> None:
        super().__init__(
            model, config, max_batch_size=max_batch_size, n_blocks=n_blocks, block_size=block_size
        )
        self.token_budget = token_budget
        #: prefills that were split rather than run whole — the mechanism, counted
        self.chunked_prefills = 0

    # -- admission ----------------------------------------------------------------------

    def _begin(self, state: RequestState) -> None:
        """Reserve blocks and consume any cached prefix, without computing anything yet."""
        tokens = state.all_token_ids
        shared = self.prefix.lookup(tokens)
        max_shared = max(0, (len(tokens) - 1) // self.cache.block_size)
        shared = shared[:max_shared]

        for block in shared:
            self.cache.allocator.share(block)
        state.block_table = list(shared)
        state.stored = len(shared) * self.cache.block_size
        # Fixed at admission. For a resumed sequence this is everything generated before it was
        # preempted, so the cache is rebuilt over exactly what the caller has already received.
        state.prefill_target = len(tokens)

        needed = self.cache.allocator.blocks_needed(len(tokens)) - len(state.block_table)
        if needed > 0:
            if not self._free_blocks_for(needed):
                raise OutOfBlocksError(f"cannot allocate {needed} blocks")
            state.block_table += self.cache.allocator.allocate(needed)

    def _prefill_chunk(self, state: RequestState, budget: int) -> StepOutput | None:
        """Prefill up to ``budget`` more tokens. Returns a token only once the prompt is done.

        A partially prefilled sequence produces nothing — there is no distribution for the next
        token until the whole prompt has been through the model. That is the cost of chunking, and
        it is why the budget cannot be made arbitrarily small.
        """
        tokens = state.all_token_ids[: state.prefill_target]
        start = state.stored
        end = min(start + budget, len(tokens))
        if end <= start:
            return None

        # Carry the cache forward between chunks rather than re-reading it from the blocks.
        #
        # Re-gathering per chunk makes a chunked prefill quadratic in prompt length — a 1536-token
        # prompt at a 64-token budget would copy the whole cache twenty-four times — and that cost
        # swamps every scheduling benefit chunking is meant to deliver. A production engine avoids
        # it by having attention read the blocks in place; until chapter 14 writes that kernel, an
        # incremental contiguous cache during prefill gets the same asymptotics.
        if state.past is None:
            state.past = (
                [
                    self.cache.gather(layer, state.block_table, start)
                    for layer in range(self.cache.n_layers)
                ]
                if start
                else None
            )

        input_ids = torch.tensor([tokens[start:end]], dtype=torch.long)
        positions = torch.arange(start, end, dtype=torch.long).unsqueeze(0)
        logits, present = self.model(input_ids, state.past, positions)
        state.past = present
        state.stored = end

        if end < len(tokens):
            return None  # more prompt to come; this sequence stays silent

        # Prompt complete: commit the whole cache to blocks and drop the temporary.
        for layer in range(self.cache.n_layers):
            self.cache.write(
                layer, state.block_table, 0, present[layer][0][0], present[layer][1][0]
            )
        state.past = None
        self._publish(state)

        return self._emit([state], _sample_batch(logits[:, -1, :], [state]))[0]

    # -- the step -----------------------------------------------------------------------

    def step(self) -> list[StepOutput]:
        self.running = [s for s in self.running if not s.finished]
        outputs: list[StepOutput] = []

        # 1. Decode first. These tokens are what a waiting caller is watching, so they get the
        #    budget before any prefill does.
        decoding = [s for s in self.running if s.stored >= s.prefill_target]
        if decoding:
            outputs += self._decode(decoding)
        # Sequences that finished on this step have already had their blocks released, so they
        # must leave the running set before anything else looks at it. Leaving them in makes a
        # finished sequence look like one that still needs prefilling, against an empty block
        # table.
        self.running = [s for s in self.running if not s.finished]
        budget = max(0, self.token_budget - len(decoding))

        # 2. Continue prefills already in progress, oldest first, so nothing starves.
        pending = [s for s in self.running if s.stored < s.prefill_target]
        for state in pending:
            if budget <= 0:
                break
            before = state.stored
            out = self._prefill_chunk(state, budget)
            spent = state.stored - before
            budget -= spent
            if state.stored < state.prefill_target:
                self.chunked_prefills += 1
            if out is not None:
                outputs.append(out)

        # 3. Admit new work with whatever budget survives.
        while self.waiting and len(self.running) < self.max_batch_size and budget > 0:
            candidate = self.waiting[0]
            need = self.cache.allocator.blocks_needed(len(candidate.all_token_ids) + 1)
            if need > self.cache.allocator.n_free:
                break
            state = self.waiting.pop(0)
            self._begin(state)
            self.running.append(state)
            before = state.stored
            out = self._prefill_chunk(state, budget)
            budget -= state.stored - before
            if state.stored < state.prefill_target:
                self.chunked_prefills += 1
            if out is not None:
                outputs.append(out)

        self.peak_running = max(self.peak_running, len(self.running))
        self.peak_kv_utilisation = max(self.peak_kv_utilisation, self.cache.allocator.utilisation)
        return outputs
