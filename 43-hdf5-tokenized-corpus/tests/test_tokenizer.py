from __future__ import annotations

import pytest

from main import BOUNDARY_TOKEN_ID, BYTE_VOCAB_SIZE, ByteTokenizer


@pytest.mark.parametrize(
    "text",
    [
        "plain ASCII text",
        "Olá, mundo!",
        "你好，世界",
        "emoji 🚀🔥",
        "",
    ],
)
def test_byte_tokenizer_round_trips_utf8(text: str) -> None:
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(text)
    assert len(ids) == len(text.encode("utf-8"))
    assert all(0 <= token <= 0xFF for token in ids)
    assert tokenizer.decode(ids) == text


def test_byte_tokenizer_reserves_the_boundary_id() -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.vocab_size == BYTE_VOCAB_SIZE == 257
    assert tokenizer.boundary_token_id == BOUNDARY_TOKEN_ID == 256
    literal = tokenizer.encode("<|endoftext|>")
    assert BOUNDARY_TOKEN_ID not in literal
    assert max(literal) <= 0xFF


def test_decode_rejects_boundary_and_out_of_range_ids() -> None:
    tokenizer = ByteTokenizer()
    with pytest.raises(ValueError, match="byte vocabulary"):
        tokenizer.decode([65, BOUNDARY_TOKEN_ID])
    with pytest.raises(ValueError, match="byte vocabulary"):
        tokenizer.decode([-1])
    with pytest.raises(ValueError, match="must be an int"):
        tokenizer.decode([1.5])  # type: ignore[list-item]
