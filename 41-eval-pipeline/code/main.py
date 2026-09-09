"""A unified, offline evaluation pipeline for tiny byte-level language models.

The module implements four heterogeneous evaluations (held-out perplexity,
short-form exact match, open-form token F1, and a deterministic mock
LLM-as-judge), a configurable weighted aggregator, and a stable-key JSON
report. It trains a small decoder-only transformer from scratch on a synthetic
instruction corpus so the whole loop runs on CPU without a network, a dataset
download, or a provider API.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Vocabulary and validation helpers
# ---------------------------------------------------------------------------

BYTE_VOCAB_SIZE = 256
INST_ID = 256
RESP_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259
IGNORE_INDEX = -100
NEWLINE_ID = 10
MAX_SEQUENCE_LENGTH = 128

EVAL_NAMES: tuple[str, ...] = ("perplexity", "exact_match", "token_f1", "judge")


def _require_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_float(
    name: str,
    value: object,
    *,
    minimum: float,
    maximum: float | None = None,
    maximum_inclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted < minimum:
        raise ValueError(f"{name} must be finite and >= {minimum}")
    if maximum is not None:
        if maximum_inclusive:
            if converted > maximum:
                raise ValueError(f"{name} must be <= {maximum}")
        elif converted >= maximum:
            raise ValueError(f"{name} must be < {maximum}")
    return converted


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty or blank")
    return value


def _require_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    return seed


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class InstructionTokenizer:
    """UTF-8 byte tokenizer with instruction, response, and padding specials.

    Answers are terminated with a newline byte. The newline is both the
    generation stop marker and a supervised training target.
    """

    def __init__(self, max_len: int = MAX_SEQUENCE_LENGTH) -> None:
        self._max_len = _require_int("max_len", max_len, minimum=4)

    @property
    def max_len(self) -> int:
        return self._max_len

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def inst_id(self) -> int:
        return INST_ID

    @property
    def resp_id(self) -> int:
        return RESP_ID

    @property
    def pad_id(self) -> int:
        return PAD_ID

    def encode_prefix(self, instruction: str) -> list[int]:
        """Encode ``[INST] instruction [RESP]`` and leave room to answer."""
        text = _require_text("instruction", instruction)
        ids = [INST_ID] + list(text.encode("utf-8")) + [RESP_ID]
        if len(ids) >= self._max_len:
            raise ValueError(
                f"instruction prefix length {len(ids)} must be < max_len "
                f"{self._max_len}"
            )
        return ids

    def encode_example(
        self, instruction: str, response: str
    ) -> tuple[list[int], int]:
        """Encode a prompt, answer bytes, and the newline terminator."""
        answer = _require_text("response", response)
        if "\n" in answer or "\r" in answer:
            raise ValueError("response must be a single line")
        prefix = self.encode_prefix(instruction)
        ids = prefix + list(answer.encode("utf-8")) + [NEWLINE_ID]
        if len(ids) > self._max_len:
            raise ValueError(
                f"prompt plus response length {len(ids)} exceeds max_len "
                f"{self._max_len}"
            )
        return ids, len(prefix)

    def decode(self, ids: Sequence[int]) -> str:
        """Decode byte ids; control ids are not valid response bytes."""
        if isinstance(ids, (str, bytes)):
            raise TypeError("ids must be a sequence of ints")
        values: list[int] = []
        for value in ids:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("ids must contain ints")
            if value < 0 or value >= BYTE_VOCAB_SIZE:
                raise ValueError(f"id {value} is not a byte id")
            values.append(value)
        return bytes(values).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Tiny decoder-only transformer
# ---------------------------------------------------------------------------


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


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 96
    num_heads: int = 4
    num_layers: int = 2
    max_len: int = MAX_SEQUENCE_LENGTH
    dropout: float = 0.0

    def __post_init__(self) -> None:
        hidden = _require_int("hidden", self.hidden, minimum=1)
        heads = _require_int("num_heads", self.num_heads, minimum=1)
        _require_int("num_layers", self.num_layers, minimum=1)
        _require_int("max_len", self.max_len, minimum=2)
        dropout = _require_float("dropout", self.dropout, minimum=0.0)
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by num_heads")
        if dropout >= 1.0:
            raise ValueError("dropout must be < 1")


class TokenBlock(nn.Module):
    """Pre-norm causal transformer block with fused QKV attention."""

    def __init__(
        self,
        hidden: int,
        num_heads: int,
        max_len: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden = _require_int("hidden", hidden, minimum=1)
        self.num_heads = _require_int("num_heads", num_heads, minimum=1)
        if self.hidden % self.num_heads != 0:
            raise ValueError("hidden must be divisible by num_heads")
        self.max_len = _require_int("max_len", max_len, minimum=1)
        self.dropout = _require_float("dropout", dropout, minimum=0.0)
        if self.dropout >= 1.0:
            raise ValueError("dropout must be < 1")
        self.head_dim = self.hidden // self.num_heads

        self.ln1 = nn.LayerNorm(self.hidden)
        self.qkv = nn.Linear(self.hidden, 3 * self.hidden, bias=False)
        self.out_proj = nn.Linear(self.hidden, self.hidden, bias=False)
        self.ln2 = nn.LayerNorm(self.hidden)
        self.fc1 = nn.Linear(self.hidden, 4 * self.hidden, bias=False)
        self.fc2 = nn.Linear(4 * self.hidden, self.hidden, bias=False)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.causal_mask: Tensor
        self.register_buffer(
            "causal_mask",
            torch.triu(
                torch.ones(self.max_len, self.max_len, dtype=torch.bool),
                diagonal=1,
            ),
            persistent=False,
        )

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 3:
            raise ValueError(
                f"hidden must be (B, T, D), got {tuple(hidden.shape)}"
            )
        if hidden.shape[-1] != self.hidden:
            raise ValueError(
                f"hidden last dimension {hidden.shape[-1]} does not match "
                f"{self.hidden}"
            )
        batch, length, _ = hidden.shape
        if length < 1 or length > self.max_len:
            raise ValueError(
                f"sequence length {length} must be between 1 and {self.max_len}"
            )

        query, key, value = self.qkv(self.ln1(hidden)).chunk(3, dim=-1)
        query = query.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            self.causal_mask[:length, :length], float("-inf")
        )
        attention = self.attn_dropout(torch.softmax(scores, dim=-1)) @ value
        attention = attention.transpose(1, 2).contiguous().view(
            batch, length, self.hidden
        )
        hidden = hidden + self.resid_dropout(self.out_proj(attention))
        feed_forward = self.fc2(F.gelu(self.fc1(self.ln2(hidden))))
        output: Tensor = hidden + self.resid_dropout(feed_forward)
        return output


class TinyGPT(nn.Module):
    """Decoder-only transformer with a tied language-model head."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        self._config = config
        seed_value = _require_seed(seed)

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE, config.hidden, padding_idx=PAD_ID
        )
        self.position_embedding = nn.Embedding(config.max_len, config.hidden)
        self.blocks = nn.ModuleList(
            [
                TokenBlock(
                    config.hidden,
                    config.num_heads,
                    config.max_len,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.hidden)
        self.head = nn.Linear(config.hidden, VOCAB_SIZE, bias=False)
        self._init_weights(_make_generator(seed_value))
        self.head.weight = self.token_embedding.weight
        with torch.no_grad():
            self.token_embedding.weight[PAD_ID].zero_()

    def _init_weights(self, generator: torch.Generator | None) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _init_linear(module, 0.02, generator)
            elif isinstance(module, nn.Embedding):
                _init_embedding(module, 0.02, generator)

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def context_length(self) -> int:
        return self._config.max_len

    def forward(self, token_ids: Tensor) -> Tensor:
        if not isinstance(token_ids, Tensor):
            raise TypeError("token_ids must be a tensor")
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must be (B, T), got {tuple(token_ids.shape)}"
            )
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"token_ids must be integer, got {token_ids.dtype}")
        batch, length = token_ids.shape
        if batch < 1:
            raise ValueError("batch size must be >= 1")
        if length < 1 or length > self.context_length:
            raise ValueError(
                f"sequence length {length} must be between 1 and "
                f"{self.context_length}"
            )
        if bool(torch.any(token_ids < 0)) or bool(
            torch.any(token_ids >= VOCAB_SIZE)
        ):
            raise ValueError("token ids must be in [0, vocab_size)")

        positions = torch.arange(length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(
            positions
        )
        for block in self.blocks:
            hidden = block(hidden)
        logits: Tensor = self.head(self.final_norm(hidden))
        return logits


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad or not trainable_only
    )
