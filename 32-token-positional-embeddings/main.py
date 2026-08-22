"""Token and positional embedding tables and their composition for transformer inputs."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _require_positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _check_seq_len(seq_len: int, max_context_length: int) -> None:
    if isinstance(seq_len, bool) or not isinstance(seq_len, int):
        raise TypeError("seq_len must be an int")
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if seq_len > max_context_length:
        raise ValueError(
            f"seq_len {seq_len} exceeds max_context_length {max_context_length}"
        )


def _init_normal(weight: Tensor, std: float, seed: int | None) -> None:
    """Fill a weight tensor from N(0, std) using a local generator when seeded."""
    if seed is None:
        nn.init.normal_(weight, mean=0.0, std=std)
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int or None")
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    nn.init.normal_(weight, mean=0.0, std=std, generator=generator)


class TokenEmbedding(nn.Module):
    """Lookup table mapping vocabulary ids to dense vectors of shape (B, T, D)."""

    vocab_size: int
    dim: int
    embedding: nn.Embedding

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        init_std: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = _require_positive_int("vocab_size", vocab_size)
        self.dim = _require_positive_int("dim", dim)
        init_std = _require_positive_float("init_std", init_std)
        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        _init_normal(self.embedding.weight, init_std, seed)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must have shape (B, T), got {tuple(token_ids.shape)}"
            )
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must be an int32 or int64 tensor")
        return self.embedding(token_ids)


class LearnedPositionalEmbedding(nn.Module):
    """Lookup table with one learned D-vector per position in 0..L-1."""

    max_context_length: int
    dim: int
    embedding: nn.Embedding

    def __init__(
        self,
        max_context_length: int,
        dim: int,
        *,
        init_std: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.max_context_length = _require_positive_int(
            "max_context_length", max_context_length
        )
        self.dim = _require_positive_int("dim", dim)
        init_std = _require_positive_float("init_std", init_std)
        self.embedding = nn.Embedding(self.max_context_length, self.dim)
        _init_normal(self.embedding.weight, init_std, seed)

    def forward(self, seq_len: int) -> Tensor:
        _check_seq_len(seq_len, self.max_context_length)
        positions = torch.arange(
            seq_len, dtype=torch.long, device=self.embedding.weight.device
        )
        return self.embedding(positions).unsqueeze(0)


def parameter_count(module: nn.Module) -> int:
    """Number of scalar parameters in the module tree."""
    return sum(parameter.numel() for parameter in module.parameters())


def neighbour_cosine(rows: Tensor, gap: int = 1) -> Tensor:
    """Cosine similarity between rows[p] and rows[p + gap] for all valid p."""
    if rows.ndim != 2:
        raise ValueError("rows must be two-dimensional")
    if isinstance(gap, bool) or not isinstance(gap, int):
        raise TypeError("gap must be an int")
    if gap < 1:
        raise ValueError("gap must be >= 1")
    if gap >= rows.shape[0]:
        raise ValueError(f"need more than {gap} rows, got {rows.shape[0]}")
    return F.cosine_similarity(rows[:-gap], rows[gap:], dim=-1)
