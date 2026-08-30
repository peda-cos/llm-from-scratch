"""A measured GPT training loop built from local PyTorch operations.

The decoder, generation helper, next-token batches, weighted AdamW groups,
warmup-cosine schedule, held-out evaluation, qualitative probes, and JSONL
logging all live here so this lesson is independently verifiable. The demo
trains a tiny model on a deterministic toy phrase, not on natural language.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _require_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_float(name: str, value: object, *, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return converted


def _require_probability(name: str, value: object) -> float:
    converted = _require_float(name, value, minimum=0.0)
    if converted > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return converted


def _require_seed(seed: object) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int or None")
    return seed


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


def _validate_sequence(x: Tensor, d_model: int, context_length: int) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must have shape (B, T, D), got {tuple(x.shape)}")
    if not x.is_floating_point():
        raise TypeError(f"x must be a floating-point tensor, got {x.dtype}")
    if x.shape[2] != d_model:
        raise ValueError(
            f"x last dimension {x.shape[2]} does not match d_model {d_model}"
        )
    if x.shape[1] < 1:
        raise ValueError("sequence length must be >= 1")
    if x.shape[1] > context_length:
        raise ValueError(
            f"sequence length {x.shape[1]} exceeds context_length "
            f"{context_length}"
        )


def _validate_token_ids(
    name: str, tokens: Tensor, context_length: int | None = None
) -> None:
    if tokens.ndim != 2:
        raise ValueError(
            f"{name} must have shape (B, T), got {tuple(tokens.shape)}"
        )
    if tokens.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must be an integer tensor, got {tokens.dtype}")
    if tokens.shape[0] < 1:
        raise ValueError(f"{name} batch size must be >= 1")
    if tokens.shape[1] < 1:
        raise ValueError(f"{name} sequence length must be >= 1")
    if context_length is not None and tokens.shape[1] > context_length:
        raise ValueError(
            f"{name} sequence length {tokens.shape[1]} exceeds "
            f"context_length {context_length}"
        )


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension with affine parameters."""

    d_model: int
    eps: float
    weight: nn.Parameter
    bias: nn.Parameter

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = _require_int("d_model", d_model, minimum=1)
        self.eps = _require_float("eps", eps, minimum=0.0)
        if self.eps == 0.0:
            raise ValueError("eps must be > 0")
        self.weight = nn.Parameter(torch.ones(self.d_model))
        self.bias = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(
                f"x must have shape (..., d_model), got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError(f"x must be a floating-point tensor, got {x.dtype}")
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
        dropout: float = 0.0,
        residual_std: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_int("d_model", d_model, minimum=1)
        self.num_heads = _require_int("num_heads", num_heads, minimum=1)
        self.context_length = _require_int(
            "context_length", context_length, minimum=1
        )
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads: "
                f"{self.d_model} % {self.num_heads} != 0"
            )
        self.head_dim = self.d_model // self.num_heads
        dropout_value = _require_probability("dropout", dropout)
        output_std = (
            0.02
            if residual_std is None
            else _require_float("residual_std", residual_std, minimum=0.0)
        )
        generator = _make_generator(_require_seed(seed))

        self.qkv_projection = nn.Linear(
            self.d_model, 3 * self.d_model, bias=True
        )
        self.output_projection = nn.Linear(
            self.d_model, self.d_model, bias=True
        )
        self.attention_dropout = nn.Dropout(dropout_value)
        _init_linear(self.qkv_projection, 0.02, generator)
        _init_linear(self.output_projection, output_std, generator)
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
        batch, length = projected.shape[0], projected.shape[1]
        return projected.reshape(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        _validate_sequence(x, self.d_model, self.context_length)
        qkv = self.qkv_projection(x)
        queries, keys, values = qkv.split(self.d_model, dim=-1)
        query_heads = self._split_heads(queries)
        key_heads = self._split_heads(keys)
        value_heads = self._split_heads(values)

        scores = (
            query_heads @ key_heads.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            self.causal_mask[: x.shape[1], : x.shape[1]], float("-inf")
        )
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        context = weights @ value_heads
        context = context.transpose(1, 2).reshape(
            x.shape[0], x.shape[1], self.d_model
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
        dropout: float = 0.0,
        residual_std: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_int("d_model", d_model, minimum=1)
        self.expansion = _require_int("expansion", expansion, minimum=1)
        dropout_value = _require_probability("dropout", dropout)
        output_std = (
            0.02
            if residual_std is None
            else _require_float("residual_std", residual_std, minimum=0.0)
        )
        generator = _make_generator(_require_seed(seed))
        hidden = self.expansion * self.d_model

        self.input_projection = nn.Linear(self.d_model, hidden, bias=True)
        self.output_projection = nn.Linear(hidden, self.d_model, bias=True)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_value)
        _init_linear(self.input_projection, 0.02, generator)
        _init_linear(self.output_projection, output_std, generator)

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.activation(self.input_projection(x))
        projected: Tensor = self.output_projection(hidden)
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
        dropout: float = 0.0,
        residual_std: float | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_int("d_model", d_model, minimum=1)
        self.context_length = _require_int(
            "context_length", context_length, minimum=1
        )
        seed_value = _require_seed(seed)
        residual_std_value = (
            0.02
            if residual_std is None
            else _require_float("residual_std", residual_std, minimum=0.0)
        )
        self.norm1 = LayerNorm(self.d_model)
        self.attention = MultiHeadAttention(
            self.d_model,
            num_heads,
            self.context_length,
            dropout=dropout,
            residual_std=residual_std_value,
            seed=seed_value,
        )
        self.norm2 = LayerNorm(self.d_model)
        self.feed_forward = FeedForward(
            self.d_model,
            mlp_expansion,
            dropout=dropout,
            residual_std=residual_std_value,
            seed=None if seed_value is None else seed_value + 1,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


@dataclass(frozen=True)
class GPTConfig:
    """Compact GPT-2-style decoder shape used by the training demo."""

    vocab_size: int = 8
    context_length: int = 16
    d_model: int = 32
    num_heads: int = 4
    num_layers: int = 2
    mlp_expansion: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        _require_int("vocab_size", self.vocab_size, minimum=1)
        _require_int("context_length", self.context_length, minimum=1)
        _require_int("d_model", self.d_model, minimum=1)
        _require_int("num_heads", self.num_heads, minimum=1)
        _require_int("num_layers", self.num_layers, minimum=1)
        _require_int("mlp_expansion", self.mlp_expansion, minimum=1)
        _require_probability("dropout", self.dropout)
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads: "
                f"{self.d_model} % {self.num_heads} != 0"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


def count_parameters(model: nn.Module) -> int:
    """Count unique scalar parameters, honoring tied embedding storage."""
    seen: set[int] = set()
    total = 0
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        total += parameter.numel()
    return total


class GPTModel(nn.Module):
    """Token/position embeddings, pre-LN blocks, final norm, tied head."""

    config: GPTConfig
    token_embedding: nn.Embedding
    position_embedding: nn.Embedding
    embedding_dropout: nn.Dropout
    blocks: nn.ModuleList
    final_norm: LayerNorm
    lm_head: nn.Linear
    position_ids: Tensor

    def __init__(self, config: GPTConfig, *, seed: int | None = 0) -> None:
        super().__init__()
        if not isinstance(config, GPTConfig):
            raise TypeError("config must be a GPTConfig")
        seed_value = _require_seed(seed)
        self.config = config
        generator = _make_generator(seed_value)

        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model
        )
        self.position_embedding = nn.Embedding(
            config.context_length, config.d_model
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        _init_embedding(self.token_embedding, 0.02, generator)
        _init_embedding(self.position_embedding, 0.02, generator)

        residual_std = 0.02 / math.sqrt(2 * config.num_layers)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.context_length,
                    mlp_expansion=config.mlp_expansion,
                    dropout=config.dropout,
                    residual_std=residual_std,
                    seed=(
                        None
                        if seed_value is None
                        else seed_value + 2 + index
                    ),
                )
                for index in range(config.num_layers)
            ]
        )
        self.final_norm = LayerNorm(config.d_model)
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )
        self.lm_head.weight = self.token_embedding.weight
        self.register_buffer(
            "position_ids",
            torch.arange(config.context_length, dtype=torch.long),
            persistent=False,
        )

    @property
    def context_length(self) -> int:
        return self.config.context_length

    def forward(self, idx: Tensor) -> Tensor:
        _validate_token_ids("idx", idx, self.config.context_length)
        token_vectors = self.token_embedding(idx)
        position_vectors = self.position_embedding(
            self.position_ids[: idx.shape[1]]
        )
        x: Tensor = self.embedding_dropout(token_vectors + position_vectors)
        for block in self.blocks:
            x = cast(TransformerBlock, block)(x)
        x = self.final_norm(x)
        logits: Tensor = self.lm_head(x)
        return logits


def _model_context_length(model: nn.Module) -> int:
    context = getattr(model, "context_length", None)
    if isinstance(context, bool) or not isinstance(context, int):
        raise TypeError("model must expose an integer context_length")
    if context < 1:
        raise ValueError("model.context_length must be >= 1")
    return context


def _sample_next_token(
    logits: Tensor,
    *,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator | None,
) -> Tensor:
    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape (B, V), got {tuple(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise TypeError(f"logits must be floating point, got {logits.dtype}")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    scaled = logits / temperature
    if not bool(torch.isfinite(scaled).all()):
        raise ValueError("temperature is too small for the logit range")
    if top_k is not None and top_k < scaled.shape[-1]:
        values, indices = torch.topk(scaled, top_k, dim=-1)
        filtered = torch.full_like(scaled, float("-inf"))
        filtered.scatter_(1, indices, values)
        scaled = filtered
    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(
        probabilities, num_samples=1, generator=generator
    )


def generate(
    model: nn.Module,
    idx: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Autoregressively extend an integer prompt with a sliding context."""
    _validate_token_ids("idx", idx)
    _require_int("max_new_tokens", max_new_tokens, minimum=0)
    temperature_value = _require_float(
        "temperature", temperature, minimum=0.0
    )
    if top_k is not None:
        _require_int("top_k", top_k, minimum=1)
    context_length = _model_context_length(model)
    if max_new_tokens == 0:
        return idx.clone()

    was_training = model.training
    model.eval()
    result = idx
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                active = result[:, -context_length:]
                logits = model(active)
                if logits.ndim != 3:
                    raise ValueError(
                        "model must return logits with shape (B, T, V), "
                        f"got {tuple(logits.shape)}"
                    )
                if logits.shape[:2] != active.shape:
                    raise ValueError(
                        "model logits batch/sequence shape does not match "
                        f"input: {tuple(logits.shape[:2])} vs "
                        f"{tuple(active.shape)}"
                    )
                next_token = _sample_next_token(
                    logits[:, -1, :],
                    temperature=temperature_value,
                    top_k=top_k,
                    generator=generator,
                )
                result = torch.cat((result, next_token), dim=1)
    finally:
        model.train(was_training)
    return result


def generate_and_print_sample(
    model: nn.Module,
    prompt: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 0.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
    prefix: str = "sample",
) -> Tensor:
    """Generate from a prompt, print the ids, and return the full sequence."""
    if not isinstance(prefix, str):
        raise TypeError("prefix must be a string")
    generated = generate(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        generator=generator,
    )
    rows = generated.tolist()
    shown = rows[0] if len(rows) == 1 else rows
    print(f"{prefix} ids={shown}", flush=True)
    return generated


def make_batches(
    token_ids: Tensor, batch_size: int, context_length: int
) -> list[tuple[Tensor, Tensor]]:
    """Slice a token tensor into full batches with targets shifted by one."""
    if token_ids.ndim != 1:
        raise ValueError(
            f"token_ids must have shape (N,), got {tuple(token_ids.shape)}"
        )
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"token_ids must be an integer tensor, got {token_ids.dtype}"
        )
    batch_value = _require_int("batch_size", batch_size, minimum=1)
    context_value = _require_int(
        "context_length", context_length, minimum=1
    )
    required = batch_value * context_value + 1
    if token_ids.numel() < required:
        raise ValueError(
            f"token_ids has {token_ids.numel()} tokens; at least {required} "
            f"are required for batch_size={batch_value} and "
            f"context_length={context_value}"
        )

    windows = (token_ids.numel() - context_value) // context_value
    batches: list[tuple[Tensor, Tensor]] = []
    for start_window in range(0, windows - batch_value + 1, batch_value):
        inputs = torch.stack(
            [
                token_ids[
                    (start_window + offset) * context_value :
                    (start_window + offset + 1) * context_value
                ]
                for offset in range(batch_value)
            ]
        )
        targets = torch.stack(
            [
                token_ids[
                    (start_window + offset) * context_value + 1 :
                    (start_window + offset + 1) * context_value + 1
                ]
                for offset in range(batch_value)
            ]
        )
        batches.append((inputs, targets))
    return batches


def calc_loss_batch(
    model: nn.Module, inputs: Tensor, targets: Tensor
) -> Tensor:
    """Flatten logits and pre-shifted targets into one cross-entropy loss."""
    _validate_token_ids("inputs", inputs)
    _validate_token_ids("targets", targets)
    if inputs.shape != targets.shape:
        raise ValueError(
            f"inputs shape {tuple(inputs.shape)} does not match targets "
            f"shape {tuple(targets.shape)}"
        )
    logits = model(inputs)
    if logits.ndim != 3:
        raise ValueError(
            f"model logits must have shape (B, T, V), got "
            f"{tuple(logits.shape)}"
        )
    if logits.shape[:2] != inputs.shape:
        raise ValueError(
            f"model logits batch/sequence shape {tuple(logits.shape[:2])} "
            f"does not match inputs {tuple(inputs.shape)}"
        )
    loss: Tensor = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    return loss


def evaluate_model(
    model: nn.Module,
    val_loader: Sequence[tuple[Tensor, Tensor]],
    max_batches: int,
) -> float:
    """Average loss over a fixed number of held-out batches without grads."""
    batch_count = _require_int("max_batches", max_batches, minimum=1)
    if len(val_loader) < batch_count:
        raise ValueError(
            f"val_loader has {len(val_loader)} batches; "
            f"max_batches={batch_count} requested"
        )
    was_training = model.training
    model.eval()
    total = 0.0
    try:
        with torch.no_grad():
            for index in range(batch_count):
                inputs, targets = val_loader[index]
                total += float(calc_loss_batch(model, inputs, targets).item())
    finally:
        model.train(was_training)
    return total / batch_count


def build_param_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, Any]]:
    """Split matrix-shaped parameters from 1-D scales and biases for AdamW."""
    decay_value = _require_float(
        "weight_decay", weight_decay, minimum=0.0
    )
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return [
        {"params": decay, "weight_decay": decay_value},
        {"params": no_decay, "weight_decay": 0.0},
    ]
