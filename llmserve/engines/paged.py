"""Chapter 8's engine: continuous batching over a paged KV cache.

Chapter 7 left two problems, both about memory rather than arithmetic. Every sequence owned a
contiguous cache that had to be reserved before its final length was known, and every step copied
each sequence's whole cache to assemble a padded batch.

This engine keeps chapter 7's scheduling exactly and changes where the keys and values live: a
fixed pool of blocks, allocated one at a time as a sequence grows. Memory becomes the admission
criterion, and running out of it becomes a scheduling event — preemption — rather than a crash.
"""

from __future__ import annotations

import torch

from llmserve.cache.blocks import OutOfBlocksError, PagedKVCache
from llmserve.config import ModelConfig
from llmserve.engines.batched import _prefill_batch, _sample_batch
from llmserve.model import TinyGPT
from llmserve.request import Request, RequestState, StepOutput


class PagedEngine:
    """Continuous batching with block-structured KV storage and memory-aware admission."""

    name = "paged"

    def __init__(
        self,
        model: TinyGPT,
        config: ModelConfig,
        *,
        max_batch_size: int = 8,
        n_blocks: int = 512,
        block_size: int = 16,
    ) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self.cache = PagedKVCache(
            n_layers=config.n_layers,
            n_kv_heads=config.n_kv_heads,
            head_dim=config.head_dim,
            n_blocks=n_blocks,
            block_size=block_size,
        )
        self.waiting: list[RequestState] = []
        self.running: list[RequestState] = []
        #: how often a running sequence had to be evicted to make room for another
        self.preemptions = 0
        #: the highest number of sequences ever running at once, in a fixed memory budget
        self.peak_running = 0
        self.peak_kv_utilisation = 0.0

    # -- memory -------------------------------------------------------------------------

    def _grow(self, state: RequestState, n_tokens: int) -> None:
        """Make room for ``n_tokens`` more tokens, allocating blocks only as needed."""
        needed = self.cache.allocator.blocks_needed(state.stored + n_tokens)
        if needed > len(state.block_table):
            state.block_table += self.cache.allocator.allocate(needed - len(state.block_table))

    def _release(self, state: RequestState) -> None:
        self.cache.allocator.free(state.block_table)
        state.block_table = []
        state.stored = 0

    def _preempt_newest(self) -> bool:
        """Evict the most recently admitted running sequence and requeue it.

        Newest-first because it has generated least, so recomputing it wastes the least work.
        The alternative — swapping its cache to host memory — trades bandwidth for compute and
        is the right call at larger scale; here recompute is simpler and cheaper.
        """
        if not self.running:
            return False
        victim = self.running.pop()
        self._release(victim)
        # Its generated tokens are kept. Only the cache is discarded, and re-admission recomputes
        # it from all_token_ids. Dropping the tokens instead would make a preempted caller receive
        # output twice, which is a correctness bug wearing a scheduling costume.
        self.waiting.insert(0, victim)
        self.preemptions += 1
        return True

    # -- engine interface ---------------------------------------------------------------

    def add_request(self, request: Request) -> None:
        self.waiting.append(RequestState(request=request))

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def step(self) -> list[StepOutput]:
        self.running = [s for s in self.running if not s.finished]

        admitted: list[RequestState] = []
        # Blocks are not actually taken until prefill, so the admission loop must subtract what
        # it has already promised. Checking each candidate against the same free count admits a
        # batch that cannot fit, and prefill then fails on a request the scheduler said yes to.
        reserved = 0
        while self.waiting and len(self.running) + len(admitted) < self.max_batch_size:
            candidate = self.waiting[0]
            need = self.cache.allocator.blocks_needed(len(candidate.all_token_ids) + 1)
            if need > self.cache.allocator.n_free - reserved:
                break  # not enough memory; leave it queued rather than thrash
            reserved += need
            admitted.append(self.waiting.pop(0))

        outputs: list[StepOutput] = []
        if admitted:
            outputs += self._prefill(admitted)
            self.running += admitted

        decoding = [s for s in self.running if s not in admitted and not s.finished]
        if decoding:
            outputs += self._decode(decoding)

        self.peak_running = max(self.peak_running, len(self.running))
        self.peak_kv_utilisation = max(self.peak_kv_utilisation, self.cache.allocator.utilisation)
        return outputs

    # -- the two shapes of work ----------------------------------------------------------

    def _prefill(self, states: list[RequestState]) -> list[StepOutput]:
        logits, caches = _prefill_batch(self.model, states)
        for state, cache in zip(states, caches, strict=True):
            length = len(state.all_token_ids)
            self._grow(state, length)
            for layer, (k, v) in enumerate(cache):
                self.cache.write(layer, state.block_table, 0, k[0], v[0])
            state.stored = length
        return self._emit(states, _sample_batch(logits, states))

    def _decode(self, states: list[RequestState]) -> list[StepOutput]:
        """One token for every running sequence, reading and writing blocks.

        Gathering each sequence into a padded batch is the same shape as chapter 7, but the
        memory underneath is no longer reserved per sequence. The gather itself is still a copy —
        chapter 13 removes it with a kernel that reads blocks in place.
        """
        while True:
            try:
                for state in states:
                    self._grow(state, 1)
                break
            except OutOfBlocksError:
                if not self._preempt_newest():
                    raise
                states = [s for s in states if s in self.running]
                if not states:
                    return []

        lengths = [s.stored for s in states]
        width = max(lengths)
        n_layers = self.cache.n_layers

        past = []
        for layer in range(n_layers):
            keys, values = [], []
            for state, length in zip(states, lengths, strict=True):
                k, v = self.cache.gather(layer, state.block_table, length)
                pad = width - length
                if pad:
                    k = torch.nn.functional.pad(k, (0, 0, 0, pad))
                    v = torch.nn.functional.pad(v, (0, 0, 0, pad))
                keys.append(k)
                values.append(v)
            past.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

        valid = torch.zeros((len(states), width + 1), dtype=torch.bool)
        for i, length in enumerate(lengths):
            valid[i, :length] = True
            valid[i, width] = True

        input_ids = torch.tensor([[s.output_token_ids[-1]] for s in states], dtype=torch.long)
        positions = torch.tensor([[length] for length in lengths], dtype=torch.long)
        logits, present = self.model(input_ids, past, positions, valid)

        for i, state in enumerate(states):
            for layer in range(n_layers):
                k = present[layer][0][i : i + 1, :, width:, :]
                v = present[layer][1][i : i + 1, :, width:, :]
                self.cache.write(layer, state.block_table, state.stored, k[0], v[0])
            state.stored += 1

        return self._emit(
            states, _sample_batch(logits[:, -1, :] if logits.dim() == 3 else logits, states)
        )

    def _emit(self, states: list[RequestState], tokens: list[int]) -> list[StepOutput]:
        outputs = []
        for state, token in zip(states, tokens, strict=True):
            state.output_token_ids.append(token)
            state.check_finished()
            if state.finished:
                self._release(state)
            outputs.append(
                StepOutput(
                    request_id=state.request_id,
                    token_ids=[token],
                    finished=state.finished,
                    finish_reason=state.finish_reason,
                )
            )
        return outputs
