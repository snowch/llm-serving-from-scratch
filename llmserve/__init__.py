"""llmserve — the reference inference engine built across *LLM Serving from Scratch*.

Modules arrive chapter by chapter; see ``CHECKPOINTS.md`` for the tag that corresponds to each
chapter's state of the engine, and Appendix C for a module-by-module map.
"""

__version__ = "0.0.0"

__all__ = ["ModelSpec", "kv_bytes_per_token", "decode_ceiling_tokens_per_second"]

from llmserve.arithmetic import (
    ModelSpec,
    decode_ceiling_tokens_per_second,
    kv_bytes_per_token,
)
