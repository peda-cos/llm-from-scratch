"""GPT-2-style model assembly built from local PyTorch tensor operations.

The transformer block is reimplemented locally rather than imported from a
sibling lesson. The completed module assembles token and learned position
embeddings, twelve pre-LN blocks, a final LayerNorm, and a language-model head
that shares storage with the token embedding. A small demo counts unique
parameters and generates synthetic token ids offline.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def _require_positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return converted


def _require_nonnegative_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0:
        raise ValueError(f"{name} must be finite and >= 0")
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


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _seed_offset(seed: int | None, offset: int) -> int | None:
    if seed is None:
        return None
    return seed + offset


def _make_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    return torch.Generator().manual_seed(seed)


def _init_linear(
    linear: nn.Linear, std: float, generator: torch.Generator | None
) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=std, generator=generator)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _init_embedding(
    embedding: nn.Embedding, std: float, generator: torch.Generator | None
) -> None:
    nn.init.normal_(embedding.weight, mean=0.0, std=std, generator=generator)


def _validate_sequence_input(
    x: Tensor, d_model: int, context_length: int
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
    if length > context_length:
        raise ValueError(
            f"sequence length {length} exceeds context_length {context_length}"
        )
    return batch, length, dim


def _validate_token_input(
    idx: Tensor, context_length: int | None = None
) -> tuple[int, int]:
    if idx.ndim != 2:
        raise ValueError(f"idx must have shape (B, T), got {tuple(idx.shape)}")
    if idx.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"idx must be an integer tensor, got {idx.dtype}")
    batch, length = idx.shape
    if batch < 1:
        raise ValueError("batch size must be >= 1")
    if length < 1:
        raise ValueError("sequence length must be >= 1")
    if context_length is not None and length > context_length:
        raise ValueError(
            f"sequence length {length} exceeds context_length {context_length}"
        )
    return batch, length


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension with affine parameters."""

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


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with one fused QKV projection."""

    d_model: int
    num_heads: int
    head_dim: int
    context_length: int
    qkv_projection: nn.Linear
    output_projection: nn.Linear
    attention_dropout: nn.Dropout
    causal_mask: Tensor

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_length: int,
        *,
        use_bias: bool = True,
        dropout: float = 0.0,
        init_std: float = 0.02,
        residual_std: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_positive_int("d_model", d_model)
        self.num_heads = _require_positive_int("num_heads", num_heads)
        self.context_length = _require_positive_int(
            "context_length", context_length
        )
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads: "
                f"{self.d_model} % {self.num_heads} != 0"
            )
        self.head_dim = self.d_model // self.num_heads
        use_bias_value = _require_bool("use_bias", use_bias)
        dropout_value = _require_probability("dropout", dropout)
        init_std_value = _require_positive_float("init_std", init_std)
        residual_std_value = (
            None
            if residual_std is None
            else _require_positive_float("residual_std", residual_std)
        )
        seed_value = _require_seed(seed)

        self.qkv_projection = nn.Linear(
            self.d_model, 3 * self.d_model, bias=use_bias_value
        )
        self.output_projection = nn.Linear(
            self.d_model, self.d_model, bias=use_bias_value
        )
        self.attention_dropout = nn.Dropout(dropout_value)
        generator = _make_generator(seed_value)
        _init_linear(self.qkv_projection, init_std_value, generator)
        _init_linear(
            self.output_projection,
            residual_std_value
            if residual_std_value is not None
            else init_std_value,
            generator,
        )
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(
                    self.context_length,
                    self.context_length,
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
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, _ = _validate_sequence_input(
            x, self.d_model, self.context_length
        )
        qkv = self.qkv_projection(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        queries = self._split_heads(q)
        keys = self._split_heads(k)
        values = self._split_heads(v)

        scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            self.causal_mask[:length, :length], float("-inf")
        )
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        context = weights @ values
        context = context.transpose(1, 2).reshape(
            batch, length, self.d_model
        )
        output: Tensor = self.output_projection(context)
        return output


class FeedForward(nn.Module):
    """Position-wise GELU feed-forward network with dropout."""

    d_model: int
    expansion: int
    input_projection: nn.Linear
    output_projection: nn.Linear
    activation: nn.GELU
    dropout: nn.Dropout

    def __init__(
        self,
        d_model: int,
        expansion: int = 4,
        *,
        use_bias: bool = True,
        dropout: float = 0.0,
        init_std: float = 0.02,
        residual_std: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_positive_int("d_model", d_model)
        self.expansion = _require_positive_int("expansion", expansion)
        use_bias_value = _require_bool("use_bias", use_bias)
        dropout_value = _require_probability("dropout", dropout)
        init_std_value = _require_positive_float("init_std", init_std)
        residual_std_value = (
            None
            if residual_std is None
            else _require_positive_float("residual_std", residual_std)
        )
        seed_value = _require_seed(seed)

        hidden = self.expansion * self.d_model
        self.input_projection = nn.Linear(
            self.d_model, hidden, bias=use_bias_value
        )
        self.output_projection = nn.Linear(
            hidden, self.d_model, bias=use_bias_value
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_value)
        generator = _make_generator(seed_value)
        _init_linear(self.input_projection, init_std_value, generator)
        _init_linear(
            self.output_projection,
            residual_std_value
            if residual_std_value is not None
            else init_std_value,
            generator,
        )

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.input_projection(x)
        activated = self.activation(hidden)
        projected: Tensor = self.output_projection(activated)
        output: Tensor = self.dropout(projected)
        return output


class TransformerBlock(nn.Module):
    """One pre-LN decoder block with attention and feed-forward residuals."""

    d_model: int
    context_length: int
    norm1: LayerNorm
    attention: MultiHeadAttention
    norm2: LayerNorm
    feed_forward: FeedForward

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_length: int,
        *,
        mlp_expansion: int = 4,
        use_bias: bool = True,
        dropout: float = 0.0,
        init_std: float = 0.02,
        residual_std: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_positive_int("d_model", d_model)
        self.context_length = _require_positive_int(
            "context_length", context_length
        )
        mlp_expansion_value = _require_positive_int(
            "mlp_expansion", mlp_expansion
        )
        use_bias_value = _require_bool("use_bias", use_bias)
        dropout_value = _require_probability("dropout", dropout)
        init_std_value = _require_positive_float("init_std", init_std)
        residual_std_value = (
            None
            if residual_std is None
            else _require_positive_float("residual_std", residual_std)
        )
        seed_value = _require_seed(seed)

        self.norm1 = LayerNorm(self.d_model)
        self.attention = MultiHeadAttention(
            self.d_model,
            num_heads,
            self.context_length,
            use_bias=use_bias_value,
            dropout=dropout_value,
            init_std=init_std_value,
            residual_std=residual_std_value,
            seed=seed_value,
        )
        self.norm2 = LayerNorm(self.d_model)
        self.feed_forward = FeedForward(
            self.d_model,
            mlp_expansion_value,
            use_bias=use_bias_value,
            dropout=dropout_value,
            init_std=init_std_value,
            residual_std=residual_std_value,
            seed=_seed_offset(seed_value, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        _validate_sequence_input(x, self.d_model, self.context_length)
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x
