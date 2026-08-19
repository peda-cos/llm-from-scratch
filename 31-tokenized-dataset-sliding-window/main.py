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
