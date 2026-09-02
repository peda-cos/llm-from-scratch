"""Load published GPT-2 safetensors weights into a self-contained GPT model.

The module has four parts:

* a GPT-2-style decoder whose local parameter paths match the lesson's
  ``tok_embed/pos_embed/blocks.N.attn.qkv/...`` naming convention;
* a fixed name map from the published GPT-2 convention to those local names;
* a two-phase safetensors loader that validates every name and shape before it
  copies a single tensor, and returns a ``LoadReport``;
* a deterministic demo that writes a GPT-2-format stub checkpoint, samples a
  continuation before and after loading it, and exercises the mismatch path.

No network access and no third-party model loader are used.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Small validation helpers shared by the model, loader, and demo.
# ---------------------------------------------------------------------------


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _require_nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _require_nonnegative_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if converted < 0.0:
        raise ValueError(f"{name} must be >= 0, got {converted}")
    return converted


def _require_positive_float(name: str, value: object) -> float:
    converted = _require_nonnegative_float(name, value)
    if converted == 0.0:
        raise ValueError(f"{name} must be > 0")
    return converted


def _require_probability(name: str, value: object) -> float:
    converted = _require_nonnegative_float(name, value)
    if converted > 1.0:
        raise ValueError(f"{name} must be <= 1, got {converted}")
    return converted


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


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


def _seed_offset(seed: int | None, offset: int) -> int | None:
    return None if seed is None else seed + offset


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
        raise TypeError(f"x must be floating-point, got {x.dtype}")
    if x.shape[-1] != d_model:
        raise ValueError(
            f"x last dimension {x.shape[-1]} does not match d_model {d_model}"
        )
    if x.shape[1] > context_length:
        raise ValueError(
            f"sequence length {x.shape[1]} exceeds context {context_length}"
        )
    return int(x.shape[0]), int(x.shape[1]), int(x.shape[2])


def _validate_token_input(
    idx: Tensor, context_length: int | None
) -> tuple[int, int]:
    if idx.ndim != 2:
        raise ValueError(f"idx must have shape (B, T), got {tuple(idx.shape)}")
    if idx.dtype not in (torch.long, torch.int):
        raise TypeError(f"idx must be an integer tensor, got {idx.dtype}")
    if context_length is not None and idx.shape[1] > context_length:
        raise ValueError(
            f"sequence length {idx.shape[1]} exceeds context {context_length}"
        )
    return int(idx.shape[0]), int(idx.shape[1])


# ---------------------------------------------------------------------------
# Local GPT model. Attribute names are the local names from the lesson table.
# ---------------------------------------------------------------------------


class ScaleShiftNorm(nn.Module):
    """LayerNorm with parameters named ``scale`` and ``shift``."""

    d_model: int
    eps: float
    scale: nn.Parameter
    shift: nn.Parameter

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.d_model = _require_positive_int("d_model", d_model)
        self.eps = _require_positive_float("eps", eps)
        self.scale = nn.Parameter(torch.ones(self.d_model))
        self.shift = nn.Parameter(torch.zeros(self.d_model))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim < 2:
            raise ValueError(f"x must have shape (..., D), got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError(f"x must be floating-point, got {x.dtype}")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"x last dimension {x.shape[-1]} does not match d_model {self.d_model}"
            )
        mean = x.mean(dim=-1, keepdim=True)
        variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        normalized: Tensor = (x - mean) / torch.sqrt(variance + self.eps)
        return normalized * self.scale + self.shift


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with one fused QKV projection."""

    d_model: int
    num_heads: int
    head_dim: int
    context_length: int
    qkv: nn.Linear
    out_proj: nn.Linear
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

        self.qkv = nn.Linear(
            self.d_model, 3 * self.d_model, bias=use_bias_value
        )
        self.out_proj = nn.Linear(
            self.d_model, self.d_model, bias=use_bias_value
        )
        self.attention_dropout = nn.Dropout(dropout_value)
        generator = _make_generator(seed_value)
        _init_linear(self.qkv, init_std_value, generator)
        _init_linear(
            self.out_proj,
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
        qkv = self.qkv(x)
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
        output: Tensor = self.out_proj(context)
        return output


class FeedForward(nn.Module):
    """Position-wise GELU feed-forward network."""

    d_model: int
    expansion: int
    fc1: nn.Linear
    fc2: nn.Linear
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
        self.fc1 = nn.Linear(self.d_model, hidden, bias=use_bias_value)
        self.fc2 = nn.Linear(hidden, self.d_model, bias=use_bias_value)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_value)
        generator = _make_generator(seed_value)
        _init_linear(self.fc1, init_std_value, generator)
        _init_linear(
            self.fc2,
            residual_std_value
            if residual_std_value is not None
            else init_std_value,
            generator,
        )

    def forward(self, x: Tensor) -> Tensor:
        hidden = self.activation(self.fc1(x))
        projected: Tensor = self.fc2(hidden)
        output: Tensor = self.dropout(projected)
        return output


class TransformerBlock(nn.Module):
    """One pre-LN decoder block with attention and feed-forward residuals."""

    d_model: int
    context_length: int
    ln1: ScaleShiftNorm
    attn: CausalSelfAttention
    ln2: ScaleShiftNorm
    mlp: FeedForward

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

        self.ln1 = ScaleShiftNorm(self.d_model)
        self.attn = CausalSelfAttention(
            self.d_model,
            num_heads,
            self.context_length,
            use_bias=use_bias_value,
            dropout=dropout_value,
            init_std=init_std_value,
            residual_std=residual_std_value,
            seed=seed_value,
        )
        self.ln2 = ScaleShiftNorm(self.d_model)
        self.mlp = FeedForward(
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
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


@dataclass(frozen=True)
class GPTConfig:
    """GPT-2-small shape with a tied token embedding and LM head by default."""

    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_expansion: int = 4
    dropout: float = 0.1
    use_bias: bool = True
    weight_tying: bool = True

    def __post_init__(self) -> None:
        _require_positive_int("vocab_size", self.vocab_size)
        _require_positive_int("context_length", self.context_length)
        _require_positive_int("d_model", self.d_model)
        _require_positive_int("num_heads", self.num_heads)
        _require_positive_int("num_layers", self.num_layers)
        _require_positive_int("mlp_expansion", self.mlp_expansion)
        _require_probability("dropout", self.dropout)
        _require_bool("use_bias", self.use_bias)
        _require_bool("weight_tying", self.weight_tying)
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads: "
                f"{self.d_model} % {self.num_heads} != 0"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads


def count_parameters(model: nn.Module) -> int:
    """Count unique scalar parameters, honoring parameter-level weight tying."""
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
    """Token/position embeddings, pre-LN blocks, final norm, and a tied head."""

    config: GPTConfig
    tok_embed: nn.Embedding
    pos_embed: nn.Embedding
    embedding_dropout: nn.Dropout
    blocks: nn.ModuleList
    final_ln: ScaleShiftNorm
    lm_head: nn.Linear
    position_ids: Tensor

    def __init__(
        self, config: GPTConfig, *, seed: int | None = None
    ) -> None:
        super().__init__()
        if not isinstance(config, GPTConfig):
            raise TypeError("config must be a GPTConfig")
        seed_value = _require_seed(seed)
        self.config = config

        self.tok_embed = nn.Embedding(
            config.vocab_size, config.d_model
        )
        self.pos_embed = nn.Embedding(
            config.context_length, config.d_model
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        generator = _make_generator(seed_value)
        _init_embedding(self.tok_embed, 0.02, generator)
        _init_embedding(self.pos_embed, 0.02, generator)

        residual_std = 0.02 / math.sqrt(2 * config.num_layers)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.context_length,
                    mlp_expansion=config.mlp_expansion,
                    use_bias=config.use_bias,
                    dropout=config.dropout,
                    residual_std=residual_std,
                    seed=_seed_offset(seed_value, 2 + index),
                )
                for index in range(config.num_layers)
            ]
        )
        self.final_ln = ScaleShiftNorm(config.d_model)
        self.lm_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )
        if config.weight_tying:
            self.lm_head.weight = self.tok_embed.weight
        else:
            _init_linear(self.lm_head, 0.02, generator)
        self.register_buffer(
            "position_ids",
            torch.arange(config.context_length, dtype=torch.long),
            persistent=False,
        )

    @property
    def context_length(self) -> int:
        return self.config.context_length

    def forward(self, idx: Tensor) -> Tensor:
        _, length = _validate_token_input(idx, self.config.context_length)
        token_vectors = self.tok_embed(idx)
        position_vectors = self.pos_embed(self.position_ids[:length])
        x: Tensor = self.embedding_dropout(token_vectors + position_vectors)
        for block in self.blocks:
            x = cast(TransformerBlock, block)(x)
        x = self.final_ln(x)
        logits: Tensor = self.lm_head(x)
        return logits


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def sample_next_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one token per batch row from last-position logits."""
    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape (B, V), got {tuple(logits.shape)}"
        )
    if not logits.is_floating_point():
        raise TypeError(f"logits must be floating-point, got {logits.dtype}")
    if logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ValueError("logits batch and vocabulary dimensions must be >= 1")
    greedy_value = _require_bool("greedy", greedy)
    temperature_value = _require_nonnegative_float(
        "temperature", temperature
    )
    top_k_value = (
        None if top_k is None else _require_positive_int("top_k", top_k)
    )
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite")

    if greedy_value or temperature_value == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    scaled = logits / temperature_value
    if not bool(torch.isfinite(scaled).all()):
        raise ValueError("temperature is too small for the logit range")
    if top_k_value is not None and top_k_value < scaled.shape[-1]:
        values, indices = torch.topk(scaled, top_k_value, dim=-1)
        filtered = torch.full_like(scaled, float("-inf"))
        filtered.scatter_(1, indices, values)
        scaled = filtered

    probabilities = torch.softmax(scaled, dim=-1)
    return torch.multinomial(
        probabilities, num_samples=1, generator=generator
    )


def generate(
    model: GPTModel,
    idx: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Autoregressively sample tokens, sliding the active context window."""
    if not isinstance(model, GPTModel):
        raise TypeError("model must be a GPTModel")
    _validate_token_input(idx, None)
    max_new_tokens_value = _require_nonnegative_int(
        "max_new_tokens", max_new_tokens
    )
    temperature_value = _require_nonnegative_float(
        "temperature", temperature
    )
    top_k_value = (
        None if top_k is None else _require_positive_int("top_k", top_k)
    )
    greedy_value = _require_bool("greedy", greedy)
    context_length = model.context_length
    if max_new_tokens_value == 0:
        return idx.clone()

    was_training = model.training
    model.eval()
    result = idx
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens_value):
                idx_cond = result[:, -context_length:]
                logits = model(idx_cond)
                if logits.ndim != 3:
                    raise ValueError(
                        "model must return logits with shape (B, T, V), "
                        f"got {tuple(logits.shape)}"
                    )
                if logits.shape[:2] != idx_cond.shape:
                    raise ValueError(
                        "model logits batch/sequence shape does not match "
                        f"input: {tuple(logits.shape[:2])} vs "
                        f"{tuple(idx_cond.shape)}"
                    )
                next_token = sample_next_token(
                    logits[:, -1, :],
                    temperature=temperature_value,
                    top_k=top_k_value,
                    greedy=greedy_value,
                    generator=generator,
                )
                result = torch.cat((result, next_token), dim=1)
    finally:
        model.train(was_training)
    return result


# ---------------------------------------------------------------------------
# Pretrained name map
# ---------------------------------------------------------------------------


_PER_LAYER_MAP: tuple[tuple[str, str], ...] = (
    ("ln_1.weight", "ln1.scale"),
    ("ln_1.bias", "ln1.shift"),
    ("attn.c_attn.weight", "attn.qkv.weight"),
    ("attn.c_attn.bias", "attn.qkv.bias"),
    ("attn.c_proj.weight", "attn.out_proj.weight"),
    ("attn.c_proj.bias", "attn.out_proj.bias"),
    ("ln_2.weight", "ln2.scale"),
    ("ln_2.bias", "ln2.shift"),
    ("mlp.c_fc.weight", "mlp.fc1.weight"),
    ("mlp.c_fc.bias", "mlp.fc1.bias"),
    ("mlp.c_proj.weight", "mlp.fc2.weight"),
    ("mlp.c_proj.bias", "mlp.fc2.bias"),
)

_TRANSPOSE_SUFFIXES: tuple[str, ...] = (
    ".attn.qkv.weight",
    ".attn.out_proj.weight",
    ".mlp.fc1.weight",
    ".mlp.fc2.weight",
)


def make_pretrained_to_local(num_layers: int) -> dict[str, str]:
    """Expand the fixed per-layer GPT-2 name map over ``num_layers`` blocks."""
    layer_count = _require_positive_int("num_layers", num_layers)
    mapping: dict[str, str] = {
        "wte.weight": "tok_embed.weight",
        "wpe.weight": "pos_embed.weight",
        "ln_f.weight": "final_ln.scale",
        "ln_f.bias": "final_ln.shift",
    }
    for index in range(layer_count):
        for source_suffix, target_suffix in _PER_LAYER_MAP:
            mapping[f"h.{index}.{source_suffix}"] = (
                f"blocks.{index}.{target_suffix}"
            )
    return mapping


def _transpose_required(local_name: str) -> bool:
    return local_name.endswith(_TRANSPOSE_SUFFIXES)


def _expected_pretrained_shape(
    local_name: str, local_shape: tuple[int, ...]
) -> tuple[int, ...]:
    if _transpose_required(local_name):
        return tuple(reversed(local_shape))
    return local_shape


# ---------------------------------------------------------------------------
# Stub fixture writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StubReport:
    path: Path
    tensor_count: int
    parameter_count: int
    size_bytes: int
    seed: int
    corrupt_name: str | None


def _normal(
    generator: torch.Generator,
    shape: tuple[int, ...],
    std: float,
) -> Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float32) * std


def _wrong_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 2:
        return (shape[0], max(1, shape[1] // 2))
    first = shape[0] if shape else 1
    return (max(1, first // 2),)


def make_stub_safetensors(
    path: str | Path,
    config: GPTConfig,
    *,
    seed: int = 2026,
    corrupt_name: str | None = None,
) -> StubReport:
    """Write a GPT-2-name-compatible stub checkpoint with deterministic values."""
    if not isinstance(config, GPTConfig):
        raise TypeError("config must be a GPTConfig")
    seed_value = _require_seed(seed)
    if seed_value is None:
        raise TypeError("seed must be an int")
    mapping = make_pretrained_to_local(config.num_layers)
    if corrupt_name is not None and corrupt_name not in mapping:
        raise ValueError(
            f"corrupt_name {corrupt_name!r} is not a tensor in the stub"
        )

    generator = torch.Generator().manual_seed(seed_value)
    d_model = config.d_model
    hidden = config.mlp_expansion * d_model
    tensors: dict[str, Tensor] = {
        "wte.weight": _normal(
            generator, (config.vocab_size, d_model), 0.05
        ),
        "wpe.weight": _normal(
            generator, (config.context_length, d_model), 0.02
        ),
    }
    for index in range(config.num_layers):
        prefix = f"h.{index}"
        tensors[f"{prefix}.ln_1.weight"] = 1.0 + _normal(
            generator, (d_model,), 0.01
        )
        tensors[f"{prefix}.ln_1.bias"] = _normal(
            generator, (d_model,), 0.01
        )
        tensors[f"{prefix}.attn.c_attn.weight"] = _normal(
            generator, (d_model, 3 * d_model), 0.03
        )
        tensors[f"{prefix}.attn.c_attn.bias"] = _normal(
            generator, (3 * d_model,), 0.01
        )
        tensors[f"{prefix}.attn.c_proj.weight"] = _normal(
            generator, (d_model, d_model), 0.03
        )
        tensors[f"{prefix}.attn.c_proj.bias"] = _normal(
            generator, (d_model,), 0.01
        )
        tensors[f"{prefix}.ln_2.weight"] = 1.0 + _normal(
            generator, (d_model,), 0.01
        )
        tensors[f"{prefix}.ln_2.bias"] = _normal(
            generator, (d_model,), 0.01
        )
        tensors[f"{prefix}.mlp.c_fc.weight"] = _normal(
            generator, (d_model, hidden), 0.03
        )
        tensors[f"{prefix}.mlp.c_fc.bias"] = _normal(
            generator, (hidden,), 0.01
        )
        tensors[f"{prefix}.mlp.c_proj.weight"] = _normal(
            generator, (hidden, d_model), 0.03
        )
        tensors[f"{prefix}.mlp.c_proj.bias"] = _normal(
            generator, (d_model,), 0.01
        )
    tensors["ln_f.weight"] = 1.0 + _normal(generator, (d_model,), 0.01)
    tensors["ln_f.bias"] = _normal(generator, (d_model,), 0.01)

    if corrupt_name is not None:
        tensors[corrupt_name] = _normal(
            generator, _wrong_shape(tuple(tensors[corrupt_name].shape)), 0.03
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(destination),
        metadata={
            "format": "pt",
            "source": "module-37-stub",
            "seed": str(seed_value),
        },
    )
    return StubReport(
        path=destination,
        tensor_count=len(tensors),
        parameter_count=sum(tensor.numel() for tensor in tensors.values()),
        size_bytes=destination.stat().st_size,
        seed=seed_value,
        corrupt_name=corrupt_name,
    )


def _inspect_stub(path: Path, seed: int) -> StubReport:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        parameter_count = 0
        for name in keys:
            shape = tuple(int(dim) for dim in handle.get_slice(name).get_shape())
            parameter_count += math.prod(shape)
    return StubReport(
        path=path,
        tensor_count=len(keys),
        parameter_count=parameter_count,
        size_bytes=path.stat().st_size,
        seed=seed,
        corrupt_name=None,
    )


# ---------------------------------------------------------------------------
# Strict loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedTensor:
    source: str
    target: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    transposed: bool


@dataclass(frozen=True)
class ShapeMismatch:
    source: str
    target: str
    source_shape: tuple[int, ...]
    expected_source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


@dataclass
class LoadReport:
    loaded: list[LoadedTensor] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    shape_mismatch: list[ShapeMismatch] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.missing or self.unexpected or self.shape_mismatch
        )

    def summary(self) -> str:
        return (
            f"loaded={len(self.loaded)} "
            f"missing={len(self.missing)} "
            f"unexpected={len(self.unexpected)} "
            f"ignored={len(self.ignored)} "
            f"shape_mismatch={len(self.shape_mismatch)}"
        )


class CheckpointLoadError(RuntimeError):
    """Raised when the checkpoint does not exactly cover the model parameters."""

    report: LoadReport

    def __init__(self, report: LoadReport) -> None:
        self.report = report
        lines = [
            "checkpoint validation failed: "
            f"{len(report.missing)} missing, "
            f"{len(report.unexpected)} unexpected, "
            f"{len(report.shape_mismatch)} shape mismatch"
        ]
        for name in report.missing:
            lines.append(f"missing: {name}")
        for name in report.unexpected:
            lines.append(f"unexpected: {name}")
        for mismatch in report.shape_mismatch:
            lines.append(
                f"shape mismatch: {mismatch.source} "
                f"source_shape={mismatch.source_shape} -> "
                f"{mismatch.target} expects pretrained "
                f"{mismatch.expected_source_shape} "
                f"(local {mismatch.target_shape})"
            )
        super().__init__("\n".join(lines))


_ATTENTION_BIAS = re.compile(r"h\.\d+\.attn\.bias")


def _canonical_source(name: str) -> str:
    prefix = "transformer."
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def load_safetensors(
    model: GPTModel, path: str | Path
) -> LoadReport:
    """Validate a GPT-2-format checkpoint, then assign every mapped tensor.

    Validation covers names, orientations, and shapes before any copy. Missing,
    unexpected, or shape-mismatched tensors raise ``CheckpointLoadError`` and
    leave the model untouched.
    """
    if not isinstance(model, GPTModel):
        raise TypeError("model must be a GPTModel")
    if not model.config.use_bias:
        raise ValueError(
            "the GPT-2 name map includes biases; use_bias=False is unsupported"
        )
    if not model.config.weight_tying:
        raise ValueError(
            "GPT-2 checkpoints omit the LM head; weight_tying=False is unsupported"
        )
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    name_map = make_pretrained_to_local(model.config.num_layers)
    params = dict(model.named_parameters())
    mapped_targets = set(name_map.values())
    model_targets = set(params)
    if mapped_targets != model_targets:
        missing_targets = sorted(model_targets - mapped_targets)
        extra_targets = sorted(mapped_targets - model_targets)
        raise ValueError(
            "name map does not match model parameters: "
            f"unmapped={missing_targets} absent={extra_targets}"
        )

    report = LoadReport()
    plan: list[tuple[str, str, bool, tuple[int, ...], tuple[int, ...]]] = []
    present: set[str] = set()
    seen_sources: dict[str, str] = {}

    with safe_open(
        str(checkpoint), framework="pt", device="cpu"
    ) as handle:
        for source in sorted(handle.keys()):
            canonical = _canonical_source(source)
            previous = seen_sources.get(canonical)
            if previous is not None:
                raise ValueError(
                    "checkpoint contains duplicate names after prefix "
                    f"normalization: {previous!r} and {source!r}"
                )
            seen_sources[canonical] = source
            present.add(canonical)

            if _ATTENTION_BIAS.fullmatch(canonical) is not None:
                report.ignored.append(source)
                continue
            if canonical == "lm_head.weight":
                report.ignored.append(source)
                continue

            target = name_map.get(canonical)
            if target is None:
                report.unexpected.append(source)
                continue
            parameter = params[target]
            local_shape = tuple(int(dim) for dim in parameter.shape)
            expected_source = _expected_pretrained_shape(
                target, local_shape
            )
            source_shape = tuple(
                int(dim) for dim in handle.get_slice(source).get_shape()
            )
            if source_shape != expected_source:
                report.shape_mismatch.append(
                    ShapeMismatch(
                        source=source,
                        target=target,
                        source_shape=source_shape,
                        expected_source_shape=expected_source,
                        target_shape=local_shape,
                    )
                )
                continue
            plan.append(
                (
                    source,
                    target,
                    _transpose_required(target),
                    source_shape,
                    local_shape,
                )
            )

        report.missing = sorted(
            target
            for source, target in name_map.items()
            if source not in present
        )
        if not report.ok:
            raise CheckpointLoadError(report)

        with torch.no_grad():
            for (
                source,
                target,
                transpose,
                source_shape,
                local_shape,
            ) in plan:
                tensor = handle.get_tensor(source)
                if transpose:
                    tensor = tensor.t()
                parameter = params[target]
                parameter.copy_(tensor.to(dtype=parameter.dtype))
                report.loaded.append(
                    LoadedTensor(
                        source=source,
                        target=target,
                        source_shape=source_shape,
                        target_shape=local_shape,
                        transposed=transpose,
                    )
                )
    return report


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoResult:
    config: GPTConfig
    parameters: int
    fixture_path: Path
    stub_report: StubReport
    report: LoadReport
    before_ids: tuple[int, ...]
    after_ids: tuple[int, ...]
    logit_delta: float
    bad_fixture_path: Path
    failure_report: LoadReport
    model_unchanged: bool


_PROMPT_IDS = (1, 2, 3, 4)


def _next_token_logits(model: GPTModel, prompt: Tensor) -> Tensor:
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits: Tensor = model(prompt)
        return logits[:, -1, :].clone()
    finally:
        model.train(was_training)


def _generate_ids(
    model: GPTModel,
    prompt: Tensor,
    *,
    sample_seed: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    greedy: bool,
) -> tuple[int, ...]:
    generator = torch.Generator().manual_seed(sample_seed)
    generated = generate(
        model,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        greedy=greedy,
        generator=generator,
    )
    return tuple(int(value) for value in generated[0].tolist())


def _ensure_stub(
    path: Path,
    config: GPTConfig,
    *,
    seed: int,
    corrupt_name: str | None,
    regenerate: bool,
) -> StubReport:
    if regenerate or not path.is_file():
        return make_stub_safetensors(
            path, config, seed=seed, corrupt_name=corrupt_name
        )
    return _inspect_stub(path, seed)


def run_demo(
    config: GPTConfig,
    fixture_dir: str | Path,
    *,
    seed: int = 1234,
    stub_seed: int = 2026,
    sample_seed: int = 7,
    max_new_tokens: int = 12,
    temperature: float = 1.0,
    top_k: int | None = 50,
    greedy: bool = False,
    regenerate: bool = False,
) -> DemoResult:
    """Run the before/after load demo and the shape-mismatch failure path."""
    if not isinstance(config, GPTConfig):
        raise TypeError("config must be a GPTConfig")
    seed_value = _require_seed(seed)
    stub_seed_value = _require_seed(stub_seed)
    sample_seed_value = _require_seed(sample_seed)
    if seed_value is None or stub_seed_value is None or sample_seed_value is None:
        raise TypeError("demo seeds must be ints")
    max_new_tokens_value = _require_nonnegative_int(
        "max_new_tokens", max_new_tokens
    )
    temperature_value = _require_nonnegative_float(
        "temperature", temperature
    )
    top_k_value = (
        None if top_k is None else _require_positive_int("top_k", top_k)
    )
    greedy_value = _require_bool("greedy", greedy)
    regenerate_value = _require_bool("regenerate", regenerate)
    if config.vocab_size < max(_PROMPT_IDS) + 1:
        raise ValueError(
            f"vocab_size must be >= {max(_PROMPT_IDS) + 1} for the demo prompt"
        )
    if config.context_length < len(_PROMPT_IDS):
        raise ValueError(
            "context_length must be >= "
            f"{len(_PROMPT_IDS)} for the demo prompt"
        )

    directory = Path(fixture_dir)
    fixture_path = directory / "gpt2-stub.safetensors"
    bad_fixture_path = directory / "gpt2-stub-bad.safetensors"
    stub_report = _ensure_stub(
        fixture_path,
        config,
        seed=stub_seed_value,
        corrupt_name=None,
        regenerate=regenerate_value,
    )
    _ensure_stub(
        bad_fixture_path,
        config,
        seed=stub_seed_value,
        corrupt_name="h.0.attn.c_attn.weight",
        regenerate=regenerate_value,
    )

    model = GPTModel(config, seed=seed_value)
    prompt = torch.tensor([list(_PROMPT_IDS)], dtype=torch.long)
    before_ids = _generate_ids(
        model,
        prompt,
        sample_seed=sample_seed_value,
        max_new_tokens=max_new_tokens_value,
        temperature=temperature_value,
        top_k=top_k_value,
        greedy=greedy_value,
    )
    before_logits = _next_token_logits(model, prompt)

    report = load_safetensors(model, fixture_path)

    after_ids = _generate_ids(
        model,
        prompt,
        sample_seed=sample_seed_value,
        max_new_tokens=max_new_tokens_value,
        temperature=temperature_value,
        top_k=top_k_value,
        greedy=greedy_value,
    )
    after_logits = _next_token_logits(model, prompt)
    logit_delta = float(
        (after_logits - before_logits).abs().max().item()
    )

    failure_model = GPTModel(config, seed=seed_value)
    snapshot = {
        name: parameter.detach().clone()
        for name, parameter in failure_model.named_parameters()
    }
    try:
        load_safetensors(failure_model, bad_fixture_path)
    except CheckpointLoadError as error:
        failure_report = error.report
    else:
        raise RuntimeError(
            "the deliberately corrupted fixture loaded without an error"
        )
    model_unchanged = True
    for name, parameter in failure_model.named_parameters():
        original = snapshot.get(name)
        if original is None or not torch.equal(original, parameter):
            model_unchanged = False
            break

    if not report.ok:
        raise RuntimeError("valid stub produced a non-ok load report")
    if not failure_report.shape_mismatch:
        raise RuntimeError("corrupt stub produced no shape mismatch")

    return DemoResult(
        config=config,
        parameters=count_parameters(model),
        fixture_path=fixture_path,
        stub_report=stub_report,
        report=report,
        before_ids=before_ids,
        after_ids=after_ids,
        logit_delta=logit_delta,
        bad_fixture_path=bad_fixture_path,
        failure_report=failure_report,
        model_unchanged=model_unchanged,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a GPT-2-format safetensors stub into a local GPT model, "
            "sample before and after, and exercise the mismatch path."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="demo",
        choices=("demo",),
        help="command to run (default: demo)",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=Path("outputs"),
        help="directory for generated safetensors fixtures",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="rewrite fixtures even when they already exist",
    )
    parser.add_argument("--seed", type=int, default=1234, help="model seed")
    parser.add_argument(
        "--stub-seed", type=int, default=2026, help="stub weight seed"
    )
    parser.add_argument(
        "--sample-seed", type=int, default=7, help="generation seed"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=12, help="tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="sampling temperature"
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="top-k sampling cutoff"
    )
    parser.add_argument(
        "--greedy", action="store_true", help="use argmax instead of sampling"
    )
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--mlp-expansion", type=int, default=4)
    return parser


def _print_result(result: DemoResult) -> bool:
    stub = result.stub_report
    report = result.report
    failure = result.failure_report
    config = result.config

    print(
        f"fixture: path={result.fixture_path} tensors={stub.tensor_count} "
        f"parameters={stub.parameter_count} bytes={stub.size_bytes}"
    )
    print(
        f"model: parameters={result.parameters} "
        f"d_model={config.d_model} num_heads={config.num_heads} "
        f"num_layers={config.num_layers} "
        f"context_length={config.context_length} "
        f"vocab_size={config.vocab_size}"
    )
    print(f"prompt_ids={list(_PROMPT_IDS)}")
    for entry in report.loaded:
        print(
            f"load: {entry.source} -> {entry.target} "
            f"source_shape={entry.source_shape} "
            f"target_shape={entry.target_shape} "
            f"transposed={entry.transposed}"
        )
    print(f"report: {report.summary()}")
    print(f"continuation_before_ids={list(result.before_ids)}")
    print(f"continuation_after_ids={list(result.after_ids)}")
    print(f"logits: max_abs_delta={result.logit_delta:.6f}")
    tokens_changed = (
        result.before_ids != result.after_ids and result.logit_delta > 0.0
    )
    print(f"check: tokens_changed={str(tokens_changed).lower()}")
    print(f"failure_fixture: path={result.bad_fixture_path}")
    print(
        "load_error: checkpoint validation failed: "
        f"{len(failure.missing)} missing, "
        f"{len(failure.unexpected)} unexpected, "
        f"{len(failure.shape_mismatch)} shape mismatch"
    )
    print(f"report: {failure.summary()}")
    for mismatch in failure.shape_mismatch:
        print(
            f"mismatch: source={mismatch.source} "
            f"source_shape={mismatch.source_shape} "
            f"expected_source_shape={mismatch.expected_source_shape} "
            f"target={mismatch.target} target_shape={mismatch.target_shape}"
        )
    print(f"check: model_unchanged={str(result.model_unchanged).lower()}")
    print(
        f"summary: loaded={len(report.loaded)} "
        f"ignored={len(report.ignored)} "
        f"tokens_changed={str(tokens_changed).lower()} "
        f"model_unchanged={str(result.model_unchanged).lower()}"
    )
    return report.ok and tokens_changed and result.model_unchanged


def _run_demo(args: argparse.Namespace) -> int:
    try:
        config = GPTConfig(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            mlp_expansion=args.mlp_expansion,
            dropout=0.1,
            use_bias=True,
            weight_tying=True,
        )
        result = run_demo(
            config,
            args.fixture_dir,
            seed=args.seed,
            stub_seed=args.stub_seed,
            sample_seed=args.sample_seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            greedy=args.greedy,
            regenerate=args.regenerate,
        )
    except (TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except CheckpointLoadError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0 if _print_result(result) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return _run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
