"""Sliding-window token dataset and deterministic batches for causal LM pretraining."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

CORPUS_PATH = Path(__file__).resolve().parent / "data" / "corpus.txt"


class Tokenizer(Protocol):
    """Minimal tokenizer contract used by the one-shot corpus encoder."""

    def encode(self, text: str) -> list[int]:
        ...


class ByteTokenizer:
    """UTF-8 byte tokenizer: 256 ids, no training, exact round-trip.

    This module must not import the BPE tokenizer from the previous lesson.
    A byte tokenizer keeps this lesson focused on the data conveyor; any object
    with the same ``encode`` signature, including the BPE tokenizer, plugs in.
    """

    vocab_size = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(token_ids).decode("utf-8")


def _encode_corpus_to_ids(tokenizer: Tokenizer, text: str) -> Tensor:
    """Encode the whole corpus exactly once and return a flat long tensor."""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def _require_int(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


class SlidingWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Map fixed-length windows of one id stream to next-token pairs.

    Every example is ``T + 1`` consecutive ids: the first ``T`` are the input
    and the last ``T`` are the target. A partial tail window is dropped.
    Indexing slices the stored stream; it allocates no per-example table.
    """

    def __init__(
        self,
        token_ids: Sequence[int] | Tensor,
        context_length: int,
        stride: int | None = None,
    ) -> None:
        context_length = _require_int("context_length", context_length, 1)
        if stride is None:
            stride = context_length
        stride = _require_int("stride", stride, 1)

        if isinstance(token_ids, Tensor):
            ids: Tensor = token_ids
        else:
            ids = torch.tensor(list(token_ids), dtype=torch.long)

        if ids.ndim != 1:
            raise ValueError("token_ids must be one-dimensional")
        if ids.dtype == torch.bool or torch.is_floating_point(ids) or torch.is_complex(ids):
            raise TypeError("token_ids must have an integer dtype")
        ids = ids.to(dtype=torch.long)
        if ids.numel() > 0 and bool((ids < 0).any()):
            raise ValueError("token_ids must be non-negative")

        self.token_ids = ids
        self.context_length = context_length
        self.stride = stride

    @staticmethod
    def num_examples(num_ids: int, context_length: int, stride: int) -> int:
        """Number of complete windows: ``max(0, 1 + (N - (T + 1)) // S)``."""
        num_ids = _require_int("num_ids", num_ids, 0)
        context_length = _require_int("context_length", context_length, 1)
        stride = _require_int("stride", stride, 1)
        return max(0, 1 + (num_ids - (context_length + 1)) // stride)

    def __len__(self) -> int:
        return self.num_examples(self.token_ids.numel(), self.context_length, self.stride)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        length = len(self)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(f"index {index} out of range for dataset of length {length}")

        start = index * self.stride
        window = self.token_ids[start : start + self.context_length + 1]
        return window[:-1], window[1:]


def epoch_seed(base_seed: int, epoch_index: int) -> int:
    """The lesson's seed contract: ``base_seed + epoch_index``."""
    base_seed = _require_int("base_seed", base_seed, 0)
    epoch_index = _require_int("epoch_index", epoch_index, 0)
    return base_seed + epoch_index


def make_dataloader(
    dataset: SlidingWindowDataset,
    batch_size: int,
    *,
    base_seed: int,
    epoch_index: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Build a single-epoch loader with a freshly seeded local generator.

    Rebuild the loader at the top of every epoch with the incremented
    ``epoch_index``. Iterating one loader twice advances its generator and is
    not the per-epoch contract. An empty dataset ignores ``shuffle`` because
    PyTorch's RandomSampler rejects zero samples.
    """
    batch_size = _require_int("batch_size", batch_size, 1)
    generator = torch.Generator()
    generator.manual_seed(epoch_seed(base_seed, epoch_index))
    # PyTorch's RandomSampler rejects a zero-length dataset; there is no order to shuffle.
    shuffle = shuffle and len(dataset) > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=drop_last,
        num_workers=0,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sliding-window token dataset demo for causal LM pretraining."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="encode the corpus and stream one batch per epoch")
    demo.add_argument("--corpus", type=Path, default=CORPUS_PATH, help="UTF-8 corpus path")
    demo.add_argument("--context-length", type=int, default=16, help="tokens per example (T)")
    demo.add_argument("--stride", type=int, default=None, help="window advance (default: T)")
    demo.add_argument("--batch-size", type=int, default=8, help="examples per batch (B)")
    demo.add_argument("--epochs", type=int, default=2, help="number of seeded epochs")
    demo.add_argument("--seed", type=int, default=1234, help="base seed")
    return parser


def _run_epoch(
    dataset: SlidingWindowDataset,
    batch_size: int,
    base_seed: int,
    epoch_index: int,
    *,
    report_first: bool,
) -> tuple[list[int], float]:
    loader = make_dataloader(
        dataset,
        batch_size,
        base_seed=base_seed,
        epoch_index=epoch_index,
        drop_last=True,
    )
    expected_shape = (batch_size, dataset.context_length)
    starts: list[int] = []
    first = True
    started = time.perf_counter()

    for input_ids, target_ids in loader:
        if input_ids.shape != expected_shape or target_ids.shape != expected_shape:
            raise AssertionError(
                f"shape contract violated: input={tuple(input_ids.shape)} "
                f"target={tuple(target_ids.shape)} expected={expected_shape}"
            )
        if input_ids.dtype != torch.long or target_ids.dtype != torch.long:
            raise AssertionError(f"dtype contract violated: {input_ids.dtype}, {target_ids.dtype}")
        if not torch.equal(input_ids[:, 1:], target_ids[:, :-1]):
            raise AssertionError("shift-by-one contract violated")

        starts.extend(int(token) for token in input_ids[:, 0].tolist())
        if first and report_first:
            print(
                f"epoch {epoch_index}: batch=1 input_shape={tuple(input_ids.shape)} "
                f"target_shape={tuple(target_ids.shape)} dtype={input_ids.dtype}"
            )
            print(f"  input_ids={input_ids[0].tolist()}")
            print(f"  target_ids={target_ids[0].tolist()}")
            first = False

    return starts, time.perf_counter() - started


def _run_demo(args: argparse.Namespace) -> int:
    corpus_path: Path = args.corpus
    try:
        text = corpus_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        print(f"error: corpus is not valid UTF-8: {corpus_path}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot read corpus {corpus_path}: {exc}", file=sys.stderr)
        return 2

    try:
        context_length = _require_int("context_length", args.context_length, 1)
        stride = None if args.stride is None else _require_int("stride", args.stride, 1)
        batch_size = _require_int("batch_size", args.batch_size, 1)
        epochs = _require_int("epochs", args.epochs, 1)
        base_seed = _require_int("base_seed", args.seed, 0)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    tokenizer = ByteTokenizer()
    encode_started = time.perf_counter()
    token_ids = _encode_corpus_to_ids(tokenizer, text)
    encode_elapsed = time.perf_counter() - encode_started
    dataset = SlidingWindowDataset(token_ids, context_length, stride)
    examples = len(dataset)
    if examples == 0:
        print(
            f"error: corpus yields no complete {context_length + 1}-token windows",
            file=sys.stderr,
        )
        return 2
    if examples < batch_size:
        print(
            f"error: {examples} examples is fewer than batch_size={batch_size} "
            "with drop_last=True",
            file=sys.stderr,
        )
        return 2

    print(
        f"corpus: {corpus_path} ({len(text)} chars, {token_ids.numel()} tokens, "
        f"vocab={tokenizer.vocab_size})"
    )
    print(
        f"config: context_length={dataset.context_length} stride={dataset.stride} "
        f"batch_size={batch_size} epochs={epochs} base_seed={base_seed} drop_last=True"
    )
    print(f"examples_per_epoch={examples}")
    print(f"batches_per_epoch={examples // batch_size}")
    print(f"encode_elapsed={encode_elapsed:.6f}s")

    non_overlap = SlidingWindowDataset.num_examples(
        token_ids.numel(), context_length, context_length
    )
    seen: set[int] = set()
    print(f"stride_sweep (context_length={context_length}):")
    for candidate in (
        context_length,
        max(1, context_length // 2),
        max(1, context_length // 4),
        1,
    ):
        if candidate in seen:
            continue
        seen.add(candidate)
        count = SlidingWindowDataset.num_examples(token_ids.numel(), context_length, candidate)
        ratio = count / non_overlap if non_overlap else 0.0
        print(f"  stride={candidate} examples={count} overlap={ratio:.2f}x")

    all_starts: dict[int, list[int]] = {}
    total_elapsed = 0.0
    for epoch in range(epochs):
        starts, elapsed = _run_epoch(
            dataset, batch_size, base_seed, epoch, report_first=True
        )
        all_starts[epoch] = starts
        total_elapsed += elapsed
        processed = len(starts)
        throughput = processed / elapsed if elapsed > 0 else 0.0
        print(
            f"epoch {epoch}: examples={processed} elapsed={elapsed:.4f}s "
            f"throughput={throughput:.1f} examples/s"
        )

    replay, _ = _run_epoch(dataset, batch_size, base_seed, 0, report_first=False)
    if replay != all_starts[0]:
        raise AssertionError("same seed and epoch did not reproduce the batch order")
    print(f"order_check=ok (epoch 0 replay with base_seed={base_seed})")

    if epochs > 1 and all_starts[0] == all_starts[1]:
        raise AssertionError("epoch 0 and epoch 1 produced the same batch order")
    if epochs > 1:
        print("epoch_order_change=ok")
    print("shift_contract=ok")
    print(f"total_elapsed={total_elapsed:.4f}s")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "demo":
        parser.error(f"unknown command: {args.command}")
    return _run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
