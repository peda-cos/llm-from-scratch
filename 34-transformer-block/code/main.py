"""Transformer block primitives built from PyTorch tensor operations.

The module implements a hand-rolled LayerNorm, a fused-QKV causal multi-head
attention layer, a position-wise GELU feed-forward network, and a block that
can place the two normalizations before the sublayers (pre-LN) or after the
residual adds (post-LN). A small diagnostic stack and gradient probe make the
normalization-placement difference measurable without a training run.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _require_positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return converted


def _require_probability(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return converted


def _require_seed(seed: object) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int or None")
    return seed


def _init_linear(
    linear: nn.Linear, std: float, generator: torch.Generator | None
) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=std, generator=generator)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _validate_sequence_input(
    x: Tensor, d_model: int, max_context_length: int | None = None
) -> tuple[int, int, int]:
    if x.ndim != 3:
        raise ValueError(f"x must have shape (B, T, D), got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"x must be a floating-point tensor, got {x.dtype}")
    batch, length, dim = x.shape
    if dim != d_model:
        raise ValueError(
            f"x last dimension {dim} does not match d_model {d_model}"
        )
    if length < 1:
        raise ValueError("sequence length must be >= 1")
    if max_context_length is not None and length > max_context_length:
        raise ValueError(
            f"sequence length {length} exceeds "
            f"max_context_length {max_context_length}"
        )
    return batch, length, dim


def parameter_count(module: nn.Module) -> int:
    """Number of scalar parameters in the module tree."""
    return sum(parameter.numel() for parameter in module.parameters())


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension with learnable affine terms."""

    d_model: int
    eps: float
    weight: nn.Parameter
    bias: nn.Parameter

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = _require_positive_int("d_model", d_model)
        self.eps = _require_positive_float("eps", eps)
        self.weight = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f"x must have shape (..., d_model), got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError(
                f"x must be a floating-point tensor, got {x.dtype}"
            )
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"x last dimension {x.shape[-1]} does not match "
                f"d_model {self.d_model}"
            )
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normalized: Tensor = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class FeedForward(nn.Module):
    """Position-wise two-layer GELU network with dropout after the second linear."""

    d_model: int
    hidden_dim: int
    linear1: nn.Linear
    linear2: nn.Linear
    dropout: nn.Dropout

    def __init__(
        self,
        d_model: int,
        *,
        dropout: float = 0.0,
        init_std: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_positive_int("d_model", d_model)
        dropout_probability = _require_probability("dropout", dropout)
        init_std = _require_positive_float("init_std", init_std)
        seed_value = _require_seed(seed)
        self.hidden_dim = 4 * self.d_model

        self.linear1 = nn.Linear(self.d_model, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.d_model)
        self.dropout = nn.Dropout(dropout_probability)

        generator: torch.Generator | None = None
        if seed_value is not None:
            generator = torch.Generator(
                device=self.linear1.weight.device
            ).manual_seed(seed_value)
        _init_linear(self.linear1, init_std, generator)
        _init_linear(self.linear2, init_std, generator)

    def forward(self, x: Tensor) -> Tensor:
        _validate_sequence_input(x, self.d_model)
        output: Tensor = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return output
