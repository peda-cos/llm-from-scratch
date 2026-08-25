"""Causal multi-head self-attention implemented with explicit tensor operations.

The block follows the shape contract used by decoder-only transformers:
``(B, T, D)`` in, ``(B, T, D)`` out, with ``H`` parallel heads of width
``D // H`` inside. One fused projection produces Q, K, and V; the causal mask,
softmax, and weighted sum are written out rather than delegated to a fused
kernel so the mechanism is inspectable.
"""

from __future__ import annotations

import math
from typing import Literal, overload

import torch
from torch import Tensor, nn


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _require_positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return converted


def _init_linear(
    linear: nn.Linear, std: float, generator: torch.Generator | None
) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=std, generator=generator)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def parameter_count(module: nn.Module) -> int:
    """Number of scalar parameters in the module tree."""
    return sum(parameter.numel() for parameter in module.parameters())


def attention_entropy(weights: Tensor) -> Tensor:
    """Mean entropy of each head's attention rows, shape ``(H,)``.

    ``weights`` has shape ``(B, H, Q, S)``. Exact zeros contribute zero
    entropy; the clamp only keeps ``log(0)`` out of the computation.
    """
    if weights.ndim != 4:
        raise ValueError("weights must have shape (B, H, Q, S)")
    if not weights.is_floating_point():
        raise TypeError("weights must be a floating-point tensor")
    if weights.shape[-1] < 1:
        raise ValueError("weights last dimension must be >= 1")
    safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    return -(weights * safe.log()).sum(dim=-1).mean(dim=(0, 2))


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with a fused QKV projection."""

    d_model: int
    n_heads: int
    d_head: int
    max_context_length: int
    qkv_projection: nn.Linear
    output_projection: nn.Linear
    causal_mask: Tensor

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_context_length: int,
        *,
        init_std: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_positive_int("d_model", d_model)
        self.n_heads = _require_positive_int("n_heads", n_heads)
        self.max_context_length = _require_positive_int(
            "max_context_length", max_context_length
        )
        init_std = _require_positive_float("init_std", init_std)
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an int or None")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads: "
                f"{self.d_model} % {self.n_heads} != 0"
            )
        self.d_head = self.d_model // self.n_heads

        self.qkv_projection = nn.Linear(self.d_model, 3 * self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.d_model)

        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(
                device=self.qkv_projection.weight.device
            ).manual_seed(seed)
        _init_linear(self.qkv_projection, init_std, generator)
        _init_linear(self.output_projection, init_std, generator)

        # True means "blocked"; the derived table is not checkpoint state.
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    self.max_context_length,
                    self.max_context_length,
                    dtype=torch.bool,
                ),
                diagonal=1,
            ),
            persistent=False,
        )

    def _split_heads(self, projected: Tensor) -> Tensor:
        batch = projected.shape[0]
        length = projected.shape[1]
        return projected.reshape(
            batch, length, self.n_heads, self.d_head
        ).transpose(1, 2)

    @overload
    def forward(
        self, x: Tensor, return_weights: Literal[False] = False
    ) -> Tensor: ...

    @overload
    def forward(
        self, x: Tensor, return_weights: Literal[True]
    ) -> tuple[Tensor, Tensor]: ...

    def forward(
        self, x: Tensor, return_weights: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (B, T, D), got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError(
                f"x must be a floating-point tensor, got {x.dtype}"
            )
        batch = x.shape[0]
        length = x.shape[1]
        dim = x.shape[2]
        if dim != self.d_model:
            raise ValueError(
                f"x last dimension {dim} does not match d_model {self.d_model}"
            )
        if length < 1:
            raise ValueError("sequence length must be >= 1")
        if length > self.max_context_length:
            raise ValueError(
                f"sequence length {length} exceeds "
                f"max_context_length {self.max_context_length}"
            )

        qkv = self.qkv_projection(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        queries = self._split_heads(q)
        keys = self._split_heads(k)
        values = self._split_heads(v)

        scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(
            self.causal_mask[:length, :length], float("-inf")
        )
        weights = torch.softmax(scores, dim=-1)
        context = weights @ values
        context = context.transpose(1, 2).reshape(
            batch, length, self.d_model
        )
        output: Tensor = self.output_projection(context)

        if return_weights:
            return output, weights
        return output
