"""A byte-level tokenizer, and the incremental detokenisation that streaming actually needs.

Byte-level keeps the book self-contained: no vocabulary to download, and the mapping is obvious.
It also makes chapter 4's hardest problem concrete rather than theoretical. A multi-byte UTF-8
character spans several tokens, so a server that decodes each token as it is generated will emit
broken text partway through any non-ASCII character. Real BPE tokenizers have exactly this
problem; byte-level just makes it impossible to ignore.
"""

from __future__ import annotations

import codecs

# Byte values occupy ids 0-255; specials live above them.
BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
VOCAB_SIZE = 260


class ByteTokenizer:
    """Maps text to bytes to token ids, one byte per token."""

    vocab_size = VOCAB_SIZE
    bos_id = BOS_ID
    eos_id = EOS_ID
    pad_id = PAD_ID

    def encode(self, text: str, *, add_bos: bool = False) -> list[int]:
        ids = list(text.encode("utf-8"))
        return [BOS_ID, *ids] if add_bos else ids

    def decode(self, ids: list[int]) -> str:
        """Decode a complete sequence. Invalid trailing bytes are replaced, not raised."""
        payload = bytes(i for i in ids if i < 256)
        return payload.decode("utf-8", errors="replace")


class IncrementalDetokenizer:
    """Turns a stream of token ids into a stream of *safe to emit* text fragments.

    The contract: feed tokens one at a time, and receive only text that will never be revised.
    When a token completes a multi-byte character, the whole character is emitted at once; until
    then the partial bytes are held back.

    The subtlety that makes a hand-rolled version wrong is telling a *truncated* sequence apart
    from an *invalid* one. Both raise ``UnicodeDecodeError``, but the first should be held back
    and the second emitted as a replacement character. Python's incremental codec already draws
    that line correctly, so we use it rather than re-deriving the UTF-8 state machine.

    Without this, a streaming endpoint emits replacement characters mid-word for any non-ASCII
    text, which looks like model corruption and is actually a serving bug.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def append(self, token_id: int) -> str:
        """Add one token; return the text that is now safe to emit (possibly empty)."""
        if token_id >= 256:  # specials carry no text
            return ""
        return self._decoder.decode(bytes([token_id]))

    def finalize(self) -> str:
        """Flush whatever remains at end of stream, replacing any genuinely truncated bytes."""
        text = self._decoder.decode(b"", final=True)
        self._decoder.reset()
        return text
