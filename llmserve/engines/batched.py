"""Batched engines: chapter 6 batches badly, chapter 7 batches well.

Both exist to exploit the fact chapter 3 established — a decode step reads every weight once
regardless of how many sequences are in flight, so the second sequence in a batch is nearly free.
The difference between them is not arithmetic. It is *when* a sequence is allowed to join and
leave, and that scheduling decision turns out to be worth more than any kernel in this book.
"""

from __future__ import annotations

import torch

from llmserve.model import TinyGPT
from llmserve.request import Request, RequestState, StepOutput
from llmserve.sampling import sample


def _prefill_batch(
    model: TinyGPT, states: list[RequestState]
) -> tuple[torch.Tensor, list[list[tuple[torch.Tensor, torch.Tensor]]]]:
    """Prefill several sequences of different lengths in one pass.

    Prefills ``all_token_ids`` rather than the prompt alone. For a fresh request those are the
    same thing; for one resumed after preemption (ch08) it rebuilds the cache over the tokens
    already generated, so preemption costs compute but never costs the caller output it has
    already been sent.

    Prompts are **left**-padded so that every sequence's final token sits at the same index, which
    is what lets the next decode step read one column. Two things must then be corrected for, and
    both fail silently rather than loudly if they are not:

    * **Positions.** A left-padded sequence starts at index ``pad``, but its first real token is at
      position 0. Feeding the padded index as the position tells the model the prompt begins
      somewhere it does not.
    * **Masking.** Padded slots hold arbitrary values. Without a mask they are attended to, and one
      request's padding leaks into another's output.
    """
    lengths = [len(state.all_token_ids) for state in states]
    width = max(lengths)
    device = next(model.parameters()).device

    input_ids = torch.full((len(states), width), 0, dtype=torch.long, device=device)
    positions = torch.zeros((len(states), width), dtype=torch.long, device=device)
    valid = torch.zeros((len(states), width), dtype=torch.bool, device=device)

    for i, (state, length) in enumerate(zip(states, lengths, strict=True)):
        pad = width - length
        input_ids[i, pad:] = torch.tensor(state.all_token_ids, dtype=torch.long)
        positions[i, pad:] = torch.arange(length, dtype=torch.long)
        valid[i, pad:] = True

    logits, present = model(input_ids, None, positions, valid)
    # Split the shared cache back into per-sequence caches, dropping the padding.
    per_request = []
    for i, length in enumerate(lengths):
        pad = width - length
        per_request.append(
            [(k[i : i + 1, :, pad:, :], v[i : i + 1, :, pad:, :]) for k, v in present]
        )
    return logits[:, -1, :], per_request


def _decode_batch(model: TinyGPT, states: list[RequestState]) -> torch.Tensor:
    """Advance every running sequence by exactly one token.

    Each sequence keeps its own cache, so the batch is assembled here and taken apart again
    afterwards. That gather-pad-scatter on every step is real work proportional to the longest
    sequence in the batch, and it is the cost chapter 8 removes by giving every sequence the same
    block-structured memory instead.
    """
    lengths = [state.cache_len for state in states]
    width = max(lengths)
    device = next(model.parameters()).device
    n_layers = len(states[0].past)

    past: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer in range(n_layers):
        keys, values = [], []
        for state, length in zip(states, lengths, strict=True):
            k, v = state.past[layer]
            pad = width - length
            if pad:
                k = torch.nn.functional.pad(k, (0, 0, 0, pad))
                v = torch.nn.functional.pad(v, (0, 0, 0, pad))
            keys.append(k)
            values.append(v)
        past.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))

    # Valid over [cache padded to width] + [the new token at index width].
    valid = torch.zeros((len(states), width + 1), dtype=torch.bool, device=device)
    for i, length in enumerate(lengths):
        valid[i, :length] = True
        valid[i, width] = True

    input_ids = torch.tensor([[s.output_token_ids[-1]] for s in states], dtype=torch.long)
    positions = torch.tensor([[length] for length in lengths], dtype=torch.long)

    logits, present = model(input_ids, past, positions, valid)

    # Scatter the new key/value back onto each sequence's own cache, at its own length.
    for i, state in enumerate(states):
        state.past = [
            (
                torch.cat(
                    [state.past[layer][0], present[layer][0][i : i + 1, :, width:, :]], dim=2
                ),
                torch.cat(
                    [state.past[layer][1], present[layer][1][i : i + 1, :, width:, :]], dim=2
                ),
            )
            for layer in range(n_layers)
        ]
    return logits[:, -1, :]


def _sample_batch(logits: torch.Tensor, states: list[RequestState]) -> list[int]:
    tokens = []
    for i, state in enumerate(states):
        prev = torch.tensor([state.all_token_ids], dtype=torch.long)
        token = sample(logits[i : i + 1], state.request.params, prev_ids=prev)
        tokens.append(int(token.item()))
    return tokens


class StaticBatchEngine:
    """Chapter 6: form a batch, run it to completion, then form the next one.

    The batch is the unit of work. A sequence that finishes early keeps its slot until every other
    sequence in the batch has finished too, and a request that arrives one moment after the batch
    forms waits for all of them. Both are pure waste, and the chapter measures how much.
    """

    name = "static-batch"

    def __init__(self, model: TinyGPT, *, max_batch_size: int = 8) -> None:
        self.model = model
        self.max_batch_size = max_batch_size
        self.waiting: list[RequestState] = []
        self.running: list[RequestState] = []
        #: slot-steps spent on already-finished sequences — the waste this engine cannot avoid
        self.wasted_slot_steps = 0
        self.total_slot_steps = 0

    def add_request(self, request: Request) -> None:
        self.waiting.append(RequestState(request=request))

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def step(self) -> list[StepOutput]:
        if not self.running:
            if not self.waiting:
                return []
            self.running = self.waiting[: self.max_batch_size]
            self.waiting = self.waiting[self.max_batch_size :]
            logits, caches = _prefill_batch(self.model, self.running)
            for state, cache in zip(self.running, caches, strict=True):
                state.past = cache
        else:
            logits = _decode_batch(self.model, self.running)

        tokens = _sample_batch(logits, self.running)
        outputs = []
        for state, token in zip(self.running, tokens, strict=True):
            self.total_slot_steps += 1
            if state.finished:
                # Still occupying a slot, still being computed, output discarded.
                self.wasted_slot_steps += 1
                continue
            state.output_token_ids.append(token)
            state.check_finished()
            outputs.append(
                StepOutput(
                    request_id=state.request_id,
                    token_ids=[token],
                    finished=state.finished,
                    finish_reason=state.finish_reason,
                )
            )

        # The batch is released only when every member of it has finished.
        if all(state.finished for state in self.running):
            self.running = []
        return outputs


class ContinuousBatchEngine(StaticBatchEngine):
    """Chapter 7: admit and retire per iteration rather than per batch.

    One change: the running set is re-evaluated every step. A finished sequence leaves immediately
    and a waiting one takes its place in the same step, so no slot is ever spent on a request that
    has already finished. No new mathematics, no new kernel — only when a sequence is allowed to
    join and leave.
    """

    name = "continuous-batch"

    def step(self) -> list[StepOutput]:
        # Retire anything that finished on the previous step, then refill the freed slots.
        self.running = [state for state in self.running if not state.finished]
        admitted: list[RequestState] = []
        while self.waiting and len(self.running) + len(admitted) < self.max_batch_size:
            admitted.append(self.waiting.pop(0))

        outputs: list[StepOutput] = []

        if admitted:
            logits, caches = _prefill_batch(self.model, admitted)
            for state, cache in zip(admitted, caches, strict=True):
                state.past = cache
            outputs += self._emit(admitted, _sample_batch(logits, admitted))
            self.running += admitted

        decoding = [state for state in self.running if state not in admitted and not state.finished]
        if decoding:
            outputs += self._emit(
                decoding, _sample_batch(_decode_batch(self.model, decoding), decoding)
            )

        return outputs

    def _emit(self, states: list[RequestState], tokens: list[int]) -> list[StepOutput]:
        outputs = []
        for state, token in zip(states, tokens, strict=True):
            self.total_slot_steps += 1
            state.output_token_ids.append(token)
            state.check_finished()
            outputs.append(
                StepOutput(
                    request_id=state.request_id,
                    token_ids=[token],
                    finished=state.finished,
                    finish_reason=state.finish_reason,
                )
            )
        return outputs
