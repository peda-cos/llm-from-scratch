from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from main import (
    BASE_VOCAB_SIZE,
    BYTE_VOCAB_SIZE,
    HELD_OUT_TEXT,
    SPECIAL_IDS,
    BPETokenizer,
    load_corpus,
)

TINY_CORPUS = "ab ab ab bc bc bc"
ROUND_TRIP_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Emoji 🚀 and CJK 漢字 survive.",
    "<|endoftext|>",
    "",
    "\x00\x01\t\r\n",
    "e\u0301 café — “quotes”",
    bytes(range(256)).decode("latin-1"),
]


@pytest.fixture()
def tiny() -> BPETokenizer:
    return BPETokenizer().train(TINY_CORPUS, 260)


@pytest.fixture(scope="module")
def curve_tokenizers() -> tuple[BPETokenizer, BPETokenizer]:
    corpus = load_corpus()
    return (
        BPETokenizer().train(corpus, 300),
        BPETokenizer().train(corpus, 600),
    )


@pytest.mark.parametrize(
    "text",
    ["", "Hello, world!\n", "don't stop", "漢字と🚀", "e\u0301", "\x00\x01\t\r\n"],
)
def test_pretokenizer_preserves_text(text: str) -> None:
    assert "".join(BPETokenizer.pretokenize(text)) == text


def test_pretokenizer_splits_words_punctuation_and_whitespace() -> None:
    assert BPETokenizer.pretokenize("Hello, world!") == [
        "Hello",
        ",",
        " ",
        "world",
        "!",
    ]


def test_byte_alphabet_floor() -> None:
    tokenizer = BPETokenizer()
    for byte_id in range(BYTE_VOCAB_SIZE):
        assert tokenizer.token_bytes[byte_id] == bytes([byte_id])
    assert tokenizer.token_bytes[256] == b"<|endoftext|>"
    assert tokenizer.token_bytes[257] == b"<|pad|>"


def test_special_ids_are_reserved_before_merges(tiny: BPETokenizer) -> None:
    assert tiny.vocab_size == 260
    assert tiny.merges[0][2] == BASE_VOCAB_SIZE
    reserved = set(SPECIAL_IDS.values())
    for left, right, new_id in tiny.merges:
        assert left not in reserved
        assert right not in reserved
        assert new_id not in reserved


def test_train_learns_expected_merges(tiny: BPETokenizer) -> None:
    assert tiny.merges == [(97, 98, 258), (98, 99, 259)]


def test_training_is_deterministic() -> None:
    first = BPETokenizer().train(TINY_CORPUS, 260)
    second = BPETokenizer().train(TINY_CORPUS, 260)
    assert first.merges == second.merges


def test_encode_applies_lowest_rank_first(tiny: BPETokenizer) -> None:
    assert tiny.encode("abc") == [258, 99]
    assert tiny.encode("abab") == [258, 258]
    assert tiny.decode([258, 99]) == "abc"


@pytest.mark.parametrize("text", ROUND_TRIP_TEXTS)
def test_round_trip_unseen_text(tiny: BPETokenizer, text: str) -> None:
    assert tiny.decode(tiny.encode(text)) == text


def test_special_tokens_are_atomic_with_flag_on(tiny: BPETokenizer) -> None:
    assert tiny.encode("<|endoftext|>", allow_special=True) == [256]
    assert tiny.encode("<|pad|>", allow_special=True) == [257]
    encoded = tiny.encode("A<|pad|>B", allow_special=True)
    assert encoded.count(257) == 1
    plain = tiny.encode("<|endoftext|>", allow_special=False)
    assert 256 not in plain
    assert tiny.decode(plain) == "<|endoftext|>"


def test_special_tokens_round_trip_with_flag_on(tiny: BPETokenizer) -> None:
    text = "Before<|endoftext|>middle<|pad|>after"
    ids = tiny.encode(text, allow_special=True)
    assert ids.count(256) == 1
    assert ids.count(257) == 1
    assert tiny.decode(ids) == text
    assert tiny.encode("<|endoftext|><|pad|>", allow_special=True) == [256, 257]
    assert tiny.decode(tiny.encode(text)) == text


def test_decode_rejects_unknown_id(tiny: BPETokenizer) -> None:
    for token_id in (99999, -1):
        with pytest.raises(ValueError, match="unknown token id"):
            tiny.decode([token_id])


def test_invalid_vocab_size_rejected(tiny: BPETokenizer) -> None:
    with pytest.raises(ValueError, match="at least 258"):
        tiny.train("x", 257)
    with pytest.raises(TypeError, match="must be an int"):
        tiny.train("x", cast(Any, "300"))


def test_train_stops_when_no_pairs_remain() -> None:
    empty = BPETokenizer().train("", 300)
    assert empty.vocab_size == BASE_VOCAB_SIZE
    assert empty.encode("") == []

    singles = BPETokenizer().train("a a a", 300)
    assert singles.vocab_size == BASE_VOCAB_SIZE
    assert singles.decode(singles.encode("a a")) == "a a"


def test_save_load_preserves_merges_and_encoding(
    tiny: BPETokenizer, tmp_path: Path
) -> None:
    model_path = tmp_path / "tiny.json"
    tiny.save(model_path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    assert payload["format"] == "bpe-tokenizer"
    assert payload["version"] == 1

    loaded = BPETokenizer.load(model_path)
    assert loaded.merges == tiny.merges
    assert loaded.specials == tiny.specials
    assert loaded.vocab_size == tiny.vocab_size

    text = "Fresh Unicode: café 🚀 and <|pad|>."
    assert loaded.encode(text, allow_special=True) == tiny.encode(
        text, allow_special=True
    )
    assert loaded.decode(loaded.encode(text, allow_special=True)) == text


def test_load_rejects_malformed_model(tmp_path: Path) -> None:
    malformed_path = tmp_path / "bad.json"
    malformed_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid model file"):
        BPETokenizer.load(malformed_path)

    wrong_new: dict[str, object] = {
        "byte_vocab_size": 256,
        "format": "bpe-tokenizer",
        "merges": [[97, 98, 999]],
        "specials": SPECIAL_IDS,
        "version": 1,
        "vocab_size": 259,
    }
    malformed_path.write_text(json.dumps(wrong_new), encoding="utf-8")
    with pytest.raises(ValueError, match="expected id 258"):
        BPETokenizer.load(malformed_path)

    wrong_format = dict(wrong_new)
    wrong_format["format"] = "other"
    malformed_path.write_text(json.dumps(wrong_format), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected format"):
        BPETokenizer.load(malformed_path)


def test_larger_vocab_compresses_held_out_text(
    curve_tokenizers: tuple[BPETokenizer, BPETokenizer],
) -> None:
    small, large = curve_tokenizers
    small_ids = small.encode(HELD_OUT_TEXT)
    large_ids = large.encode(HELD_OUT_TEXT)

    assert small.vocab_size == 300
    assert large.vocab_size == 600
    assert len(large.merges) > len(small.merges)
    assert large.merges[: len(small.merges)] == small.merges
    assert small.decode(small_ids) == HELD_OUT_TEXT
    assert large.decode(large_ids) == HELD_OUT_TEXT
    assert len(large_ids) < len(small_ids)
