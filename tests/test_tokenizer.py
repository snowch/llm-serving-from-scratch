"""The streaming detokenisation contract from chapter 4.

The bug these guard against is subtle and common: a server that decodes each token as it arrives
emits replacement characters in the middle of any non-ASCII character. It looks like the model
is broken when it is the serving layer that is.
"""

import pytest

from llmserve.tokenizer import ByteTokenizer, IncrementalDetokenizer

ROUNDTRIP_CASES = [
    "ascii only",
    "Hello, wörld!",
    "日本語のテキスト",
    "😀 emoji 👍 mixed with text",
    "naïve café 中文测试",
    "",
]


@pytest.fixture
def tok():
    return ByteTokenizer()


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_encode_decode_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


@pytest.mark.parametrize("text", ROUNDTRIP_CASES)
def test_incremental_matches_batch_decode(tok, text):
    """Streaming one token at a time must produce exactly the same text as decoding once."""
    det = IncrementalDetokenizer()
    streamed = "".join(det.append(i) for i in tok.encode(text)) + det.finalize()
    assert streamed == text


def test_partial_multibyte_is_held_back_not_emitted(tok):
    """The heart of it: no output until a character is complete."""
    ids = tok.encode("日")  # three bytes
    det = IncrementalDetokenizer()
    assert det.append(ids[0]) == ""
    assert det.append(ids[1]) == ""
    assert det.append(ids[2]) == "日"


def test_invalid_byte_becomes_a_replacement_character():
    """An invalid byte is not the same as an incomplete one, and must not stall the stream."""
    det = IncrementalDetokenizer()
    assert det.append(0xFF) + det.append(0x41) + det.finalize() == "�A"


def test_truncated_tail_is_flushed_at_end_of_stream():
    """A stream that stops mid-character must still terminate, with a replacement character."""
    det = IncrementalDetokenizer()
    emitted = det.append(0xE6) + det.append(0x97)
    assert emitted == ""
    assert det.finalize() == "�"


def test_special_tokens_produce_no_text(tok):
    det = IncrementalDetokenizer()
    assert det.append(tok.eos_id) == ""
