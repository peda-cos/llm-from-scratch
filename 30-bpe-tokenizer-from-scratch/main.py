from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import cast

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

    def _encode_pretoken(self, pretoken: str) -> tuple[int, ...]:
        cached = self._encode_cache.get(pretoken)
        if cached is not None:
            return cached

        ids = list(pretoken.encode("utf-8"))
        while len(ids) >= 2:
            pair: tuple[int, int] | None = None
            best_rank = len(self.merges)
            for index in range(len(ids) - 1):
                candidate = (ids[index], ids[index + 1])
                rank = self._merge_ranks.get(candidate)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    pair = candidate
            if pair is None:
                break
            ids = self._merge_symbols(ids, pair, self.merges[best_rank][2])

        encoded = tuple(ids)
        self._encode_cache[pretoken] = encoded
        return encoded

    def _encode_plain(self, text: str) -> list[int]:
        ids: list[int] = []
        for pretoken in self.pretokenize(text):
            ids.extend(self._encode_pretoken(pretoken))
        return ids

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        if not allow_special:
            return self._encode_plain(text)

        pattern = "(" + "|".join(re.escape(token) for token in self.specials) + ")"
        ids: list[int] = []
        for part in re.split(pattern, text):
            if not part:
                continue
            special_id = self.specials.get(part)
            if special_id is not None:
                ids.append(special_id)
            else:
                ids.extend(self._encode_plain(part))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        raw = bytearray()
        for token_id in ids:
            chunk = self.token_bytes.get(token_id)
            if chunk is None:
                raise ValueError(f"unknown token id: {token_id}")
            raw.extend(chunk)
        return raw.decode("utf-8")

    def save(self, path: str | Path) -> None:
        payload = {
            "byte_vocab_size": BYTE_VOCAB_SIZE,
            "format": "bpe-tokenizer",
            "merges": [list(merge) for merge in self.merges],
            "specials": self.specials,
            "version": 1,
            "vocab_size": self.vocab_size,
        }
        Path(path).write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        try:
            raw: object = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid model file: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("invalid model file: expected a JSON object")
        payload = cast(dict[str, object], raw)
        if payload.get("format") != "bpe-tokenizer":
            raise ValueError("invalid model file: unexpected format")
        if payload.get("version") != 1:
            raise ValueError("invalid model file: unsupported version")
        if payload.get("byte_vocab_size") != BYTE_VOCAB_SIZE:
            raise ValueError("invalid model file: byte_vocab_size must be 256")
        if payload.get("specials") != SPECIAL_IDS:
            raise ValueError("invalid model file: specials do not match reserved tokens")
        merges_raw = payload.get("merges")
        if not isinstance(merges_raw, list):
            raise ValueError("invalid model file: merges must be a list")
        if payload.get("vocab_size") != BASE_VOCAB_SIZE + len(merges_raw):
            raise ValueError("invalid model file: vocab_size does not match merges")

        tokenizer = cls()
        special_ids = set(tokenizer.specials.values())
        for index, item in enumerate(merges_raw):
            if not isinstance(item, list) or len(item) != 3:
                raise ValueError(f"invalid merge at index {index}: expected three ids")
            left_raw, right_raw, new_raw = item
            if (
                type(left_raw) is not int
                or type(right_raw) is not int
                or type(new_raw) is not int
            ):
                raise ValueError(f"invalid merge at index {index}: ids must be ints")
            left = left_raw
            right = right_raw
            new_id = new_raw
            expected_new = BASE_VOCAB_SIZE + index
            if new_id != expected_new:
                raise ValueError(
                    f"invalid merge at index {index}: expected id {expected_new}"
                )
            if left not in tokenizer.token_bytes or right not in tokenizer.token_bytes:
                raise ValueError(f"invalid merge at index {index}: unknown input id")
            if left in special_ids or right in special_ids:
                raise ValueError(
                    f"invalid merge at index {index}: special ids cannot be merged"
                )
            pair = (left, right)
            if pair in tokenizer._merge_ranks:
                raise ValueError(f"invalid merge at index {index}: duplicate pair")
            tokenizer.token_bytes[new_id] = (
                tokenizer.token_bytes[left] + tokenizer.token_bytes[right]
            )
            tokenizer._merge_ranks[pair] = index
            tokenizer.merges.append((left, right, new_id))
        return tokenizer

    def _reset(self) -> None:
        self.merges = []
        self._merge_ranks = {}
        self._encode_cache = {}
        self.token_bytes = {byte_id: bytes([byte_id]) for byte_id in range(BYTE_VOCAB_SIZE)}
        for token, token_id in self.specials.items():
            self.token_bytes[token_id] = token.encode("utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and run a byte-level BPE tokenizer from scratch."
    )
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="train and print a held-out round-trip")
    demo.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    demo.add_argument("--corpus", type=Path, default=None)
    demo.add_argument(
        "--curve",
        action="store_true",
        help="train --vocab-size and double it, then print the compression curve",
    )

    train_parser = subparsers.add_parser("train", help="train and save a model")
    train_parser.add_argument("--out", type=Path, required=True)
    train_parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    train_parser.add_argument("--corpus", type=Path, default=None)

    encode_parser = subparsers.add_parser(
        "encode", help="encode text with a saved model"
    )
    encode_parser.add_argument("--model", type=Path, required=True)
    encode_parser.add_argument("--allow-special", action="store_true")
    encode_parser.add_argument("text")

    decode_parser = subparsers.add_parser(
        "decode", help="decode ids with a saved model"
    )
    decode_parser.add_argument("--model", type=Path, required=True)
    decode_parser.add_argument("ids", type=int, nargs="+")

    return parser


def _run_demo(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.corpus)
    target: int = args.vocab_size
    sizes = [target, target * 2] if args.curve else [target]
    pretokens = BPETokenizer.pretokenize(corpus)
    print(
        f"corpus: {len(corpus)} chars, {len(pretokens)} pretokens, "
        f"{len(set(pretokens))} unique"
    )

    results: list[tuple[BPETokenizer, list[int], float]] = []
    for size in sizes:
        started = time.perf_counter()
        tokenizer = BPETokenizer().train(corpus, size)
        elapsed = time.perf_counter() - started
        ids = tokenizer.encode(HELD_OUT_TEXT)
        results.append((tokenizer, ids, elapsed))
        if args.curve:
            continue
        decoded = tokenizer.decode(ids)
        print(
            f"vocab: requested {size}, actual {tokenizer.vocab_size} "
            f"({len(tokenizer.merges)} merges) in {elapsed:.3f}s"
        )
        print(f"held-out: {HELD_OUT_TEXT}")
        print(f"ids ({len(ids)}): {ids}")
        print(f"decoded: {decoded}")
        print(f"round-trip: {'ok' if decoded == HELD_OUT_TEXT else 'FAILED'}")
        if ids:
            print(f"chars/token: {len(HELD_OUT_TEXT) / len(ids):.3f}")
        else:
            print("chars/token: n/a")

    if not args.curve:
        return

    print(f"held-out: {HELD_OUT_TEXT}")
    print("compression curve")
    print(f"{'vocab':>7} {'merges':>7} {'ids':>5} {'chars/token':>12} {'train_s':>8}")
    for tokenizer, ids, elapsed in results:
        ratio = len(HELD_OUT_TEXT) / len(ids) if ids else 0.0
        print(
            f"{tokenizer.vocab_size:>7} {len(tokenizer.merges):>7} "
            f"{len(ids):>5} {ratio:>12.3f} {elapsed:>8.3f}"
        )
    first_ids = results[0][1]
    last_ids = results[-1][1]
    delta = len(last_ids) - len(first_ids)
    reduction = 100.0 * (1 - len(last_ids) / len(first_ids)) if first_ids else 0.0
    print(f"held-out length: {len(first_ids)} -> {len(last_ids)} ids ({delta:+d})")
    print(f"compression: {reduction:.1f}% fewer ids")
    for tokenizer, ids, _ in results:
        if tokenizer.decode(ids) != HELD_OUT_TEXT:
            raise ValueError("round-trip failed")
    print("round-trip: ok")


def _run_train(args: argparse.Namespace) -> None:
    corpus = load_corpus(args.corpus)
    started = time.perf_counter()
    tokenizer = BPETokenizer().train(corpus, args.vocab_size)
    elapsed = time.perf_counter() - started
    tokenizer.save(args.out)
    print(
        f"trained vocab={tokenizer.vocab_size} merges={len(tokenizer.merges)} "
        f"in {elapsed:.3f}s"
    )
    print(f"saved {args.out}")


def _run_encode(args: argparse.Namespace) -> None:
    tokenizer = BPETokenizer.load(args.model)
    ids = tokenizer.encode(args.text, allow_special=args.allow_special)
    print(" ".join(str(token_id) for token_id in ids))


def _run_decode(args: argparse.Namespace) -> None:
    tokenizer = BPETokenizer.load(args.model)
    print(tokenizer.decode(args.ids))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list:
        args_list = ["demo"]
    args = parser.parse_args(args_list)
    try:
        if args.command == "demo":
            _run_demo(args)
        elif args.command == "train":
            _run_train(args)
        elif args.command == "encode":
            _run_encode(args)
        elif args.command == "decode":
            _run_decode(args)
        else:
            parser.print_help()
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

