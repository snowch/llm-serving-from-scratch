"""Inference arithmetic: the predictions chapter 3 is built on.

Two quantities drive almost every serving decision:

* how many bytes the KV cache costs per token, which bounds concurrency; and
* how many tokens per second decode can possibly reach, which is set by memory bandwidth rather
  than by compute.

Both are cheap to compute and land surprisingly close to reality, so we compute them before
optimising anything. They take a :class:`~llmserve.config.ModelConfig` — the same object the
model itself is built from — so a prediction can never drift from the model it describes.
"""

from __future__ import annotations

from llmserve.config import BYTES_PER_DTYPE, ModelConfig


def kv_bytes_per_token(model: ModelConfig, kv_dtype: str | None = None) -> float:
    """Bytes of KV cache a single token occupies.

    The factor of 2 is because we cache both K and V. ``kv_dtype`` is separate from the model's
    dtype because quantising the cache independently of the weights is a real technique
    (chapter 15) and usually the bigger win at long context.
    """
    element = BYTES_PER_DTYPE[kv_dtype] if kv_dtype else model.bytes_per_element
    return 2 * model.n_layers * model.n_kv_heads * model.head_dim * element


def kv_bytes_for(model: ModelConfig, context_length: int, batch_size: int = 1) -> float:
    """Total KV cache for ``batch_size`` sequences of ``context_length`` tokens."""
    return kv_bytes_per_token(model) * context_length * batch_size


def max_concurrent_sequences(model: ModelConfig, memory_bytes: float, context_length: int) -> int:
    """How many sequences of a given length fit in a KV-cache budget.

    This is the concurrency ceiling chapter 5 runs into, and the one chapter 8 raises by removing
    the fragmentation that a contiguous per-sequence cache forces on you.
    """
    per_sequence = kv_bytes_per_token(model) * context_length
    return 0 if per_sequence <= 0 else int(memory_bytes // per_sequence)


def prefill_flops_per_token(model: ModelConfig) -> float:
    """Roughly ``2 * params`` FLOPs per prompt token: one multiply and one add per parameter.

    It ignores attention's quadratic term, which is negligible at short context and is not at
    long context — chapter 23 is where that stops being a safe approximation.
    """
    return 2.0 * model.n_params


def decode_bytes_per_step(model: ModelConfig, context_length: int, batch_size: int = 1) -> float:
    """Bytes that must be read from memory to produce one token per sequence in a batch.

    Every weight is read once regardless of batch size — which is the entire reason batching
    works — plus each sequence's KV cache.
    """
    weights = model.n_params * model.bytes_per_element
    return weights + kv_bytes_for(model, context_length, batch_size)


def decode_ceiling_tokens_per_second(
    model: ModelConfig,
    bandwidth_bytes_per_second: float,
    context_length: int = 0,
    batch_size: int = 1,
) -> float:
    """Upper bound on decode throughput from memory bandwidth alone.

    Divide available bandwidth by the bytes a step must read. No kernel can beat this, so treat
    it as a ceiling rather than a forecast: it ignores compute, kernel launch overhead and cache
    effects. Chapter 3 measures the gap and explains it.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    steps_per_second = bandwidth_bytes_per_second / decode_bytes_per_step(
        model, context_length, batch_size
    )
    return steps_per_second * batch_size


def naive_decode_flops(model: ModelConfig, prompt_len: int, n_output: int) -> float:
    """Total prefill+decode FLOPs *without* a KV cache, where every step redoes the whole prefix.

    Chapter 1's engine pays this. Comparing it to :func:`cached_decode_flops` is the arithmetic
    behind chapter 5's speedup, and it is quadratic in the sequence length.
    """
    per_token = prefill_flops_per_token(model)
    return per_token * sum(prompt_len + i for i in range(n_output + 1))


def cached_decode_flops(model: ModelConfig, prompt_len: int, n_output: int) -> float:
    """Total prefill+decode FLOPs *with* a KV cache: the prompt once, then one token at a time."""
    per_token = prefill_flops_per_token(model)
    return per_token * (prompt_len + n_output)
