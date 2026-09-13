"""Inference arithmetic: the predictions Chapter 3 is built on.

Two quantities drive every decision in the book:

* how many bytes the KV cache costs per token, which bounds concurrency; and
* how many tokens per second decode can possibly reach, which is set by memory bandwidth
  rather than by compute.

Both are cheap to compute and surprisingly accurate, so we compute them before optimising
anything.
"""

from dataclasses import dataclass

BYTES_PER_DTYPE = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
    "int8": 1,
    "int4": 0.5,
}


@dataclass(frozen=True)
class ModelSpec:
    """The handful of model dimensions that predict serving behaviour.

    ``n_kv_heads`` is the number of *key/value* heads, which equals ``n_heads`` for vanilla
    multi-head attention but is smaller under GQA and is 1 under MQA (see ch12).
    """

    name: str
    n_params: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    dtype: str = "fp16"

    @property
    def bytes_per_element(self) -> float:
        return BYTES_PER_DTYPE[self.dtype]

    @property
    def weight_bytes(self) -> float:
        return self.n_params * self.bytes_per_element


def kv_bytes_per_token(model: ModelSpec, kv_dtype: str | None = None) -> float:
    """Bytes of KV cache a single token occupies.

    ``2`` because we cache both K and V. Quantising the cache independently of the weights is a
    ch14 technique, hence the separate ``kv_dtype``.
    """
    element = BYTES_PER_DTYPE[kv_dtype] if kv_dtype else model.bytes_per_element
    return 2 * model.n_layers * model.n_kv_heads * model.head_dim * element


def decode_ceiling_tokens_per_second(
    model: ModelSpec,
    hbm_bandwidth_bytes_per_second: float,
    context_length: int = 0,
    batch_size: int = 1,
) -> float:
    """Upper bound on decode throughput, from memory bandwidth alone.

    Each decode step must read every weight once, plus the KV cache for every sequence in the
    batch. Dividing bandwidth by that gives a ceiling no kernel can beat. It ignores compute,
    kernel launch overhead and cache effects, so treat it as a ceiling rather than a forecast —
    ch03 measures the gap.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    kv_bytes = kv_bytes_per_token(model) * context_length * batch_size
    bytes_per_step = model.weight_bytes + kv_bytes
    steps_per_second = hbm_bandwidth_bytes_per_second / bytes_per_step
    return steps_per_second * batch_size
