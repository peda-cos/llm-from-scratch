"""Sliding-window token dataset and deterministic batches for causal LM pretraining."""

from __future__ import annotations

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


class SlidingWindowDataset(Dataset):
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
