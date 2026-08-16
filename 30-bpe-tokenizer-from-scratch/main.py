from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

BYTE_VOCAB_SIZE = 256
SPECIAL_IDS: dict[str, int] = {"<|endoftext|>": 256, "<|pad|>": 257}
BASE_VOCAB_SIZE = 258
DEFAULT_VOCAB_SIZE = 300
DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent / "data" / "corpus.txt"
HELD_OUT_TEXT = "Tokenizers turn text into integers; the model never sees a single character. 🚀"
PRETOKEN_RE = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)
Merge = tuple[int, int, int]


def load_corpus(path: str | Path | None = None) -> str:
    corpus_path = DEFAULT_CORPUS_PATH if path is None else Path(path)
    return corpus_path.read_text(encoding="utf-8")


class BPETokenizer:
    def __init__(self) -> None:
        self.merges: list[Merge] = []
        self.specials: dict[str, int] = dict(SPECIAL_IDS)
        self.token_bytes: dict[int, bytes] = {}
        self._merge_ranks: dict[tuple[int, int], int] = {}
        self._encode_cache: dict[str, tuple[int, ...]] = {}
        self._reset()

    @property
    def vocab_size(self) -> int:
        return BASE_VOCAB_SIZE + len(self.merges)

    @staticmethod
    def pretokenize(text: str) -> list[str]:
        return PRETOKEN_RE.findall(text)

    @staticmethod
    def _merge_symbols(
        symbols: Sequence[int], pair: tuple[int, int], new_id: int
    ) -> list[int]:
        merged: list[int] = []
        index = 0
        while index < len(symbols):
            if (
                index + 1 < len(symbols)
                and symbols[index] == pair[0]
                and symbols[index + 1] == pair[1]
            ):
                merged.append(new_id)
                index += 2
            else:
                merged.append(symbols[index])
                index += 1
        return merged

    def train(self, text: str, vocab_size: int = DEFAULT_VOCAB_SIZE) -> BPETokenizer:
        if type(vocab_size) is not int:
            raise TypeError("vocab_size must be an int")
        if vocab_size < BASE_VOCAB_SIZE:
            raise ValueError(f"vocab_size must be at least {BASE_VOCAB_SIZE}")

        self._reset()
        word_counts = Counter(self.pretokenize(text))
        words: list[tuple[tuple[int, ...], int]] = [
            (tuple(word.encode("utf-8")), frequency)
            for word, frequency in word_counts.items()
        ]

        while self.vocab_size < vocab_size:
            pair_counts: Counter[tuple[int, int]] = Counter()
            for symbols, frequency in words:
                for left, right in zip(symbols, symbols[1:]):
                    pair_counts[(left, right)] += frequency
            if not pair_counts:
                break

            pair = min(
                pair_counts,
                key=lambda candidate: (-pair_counts[candidate], candidate),
            )
            new_id = BASE_VOCAB_SIZE + len(self.merges)
            self.token_bytes[new_id] = (
                self.token_bytes[pair[0]] + self.token_bytes[pair[1]]
            )
            self.merges.append((pair[0], pair[1], new_id))
            self._merge_ranks[pair] = len(self.merges) - 1
            words = [
                (tuple(self._merge_symbols(symbols, pair, new_id)), frequency)
                for symbols, frequency in words
            ]

        return self

    def _reset(self) -> None:
        self.merges = []
        self._merge_ranks = {}
        self._encode_cache = {}
        self.token_bytes = {byte_id: bytes([byte_id]) for byte_id in range(BYTE_VOCAB_SIZE)}
        for token, token_id in self.specials.items():
            self.token_bytes[token_id] = token.encode("utf-8")
