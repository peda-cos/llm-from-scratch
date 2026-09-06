"""Instruction tuning by supervised fine-tuning.

A byte-level Alpaca-style template is trained with a causal language-model
objective whose cross-entropy targets are ``-100`` on the instruction, the
``<RESP>`` boundary, and padding, and real token ids on the response. The same
tiny transformer is then used for greedy and temperature-sampled generation and
held-out exact-match evaluation.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants and small validation helpers
# ---------------------------------------------------------------------------

BYTE_VOCAB_SIZE = 256
INST_ID = 256
RESP_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259
IGNORE_INDEX = -100
RESPONSE_TERMINATOR_ID = 10  # newline byte
MAX_SEQUENCE_LENGTH = 64
CATEGORY_ORDER = (
    "capitals",
    "arithmetic",
    "lists",
    "summaries",
    "code",
    "definitions",
)
SENTENCE_END_BYTES = frozenset({ord("."), ord("!"), ord("?")})


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
    """UTF-8 byte tokenizer with instruction/response/padding specials."""

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
        """Encode ``[INST] instruction [RESP]`` for generation."""
        text = _require_text("instruction", instruction)
        ids = [INST_ID] + list(text.encode("utf-8")) + [RESP_ID]
        if len(ids) >= self._max_len:
            raise ValueError(
                f"instruction prefix length {len(ids)} must be < max_len "
                f"{self._max_len}"
            )
        return ids

    def encode_pair(self, instruction: str, response: str) -> tuple[list[int], int]:
        """Encode a full training pair and return ``(ids, response_start)``."""
        instruction_text = _require_text("instruction", instruction)
        response_text = _require_text("response", response)
        inst_bytes = list(instruction_text.encode("utf-8"))
        ids = (
            [INST_ID]
            + inst_bytes
            + [RESP_ID]
            + list(response_text.encode("utf-8"))
            + [RESPONSE_TERMINATOR_ID]
        )
        if len(ids) > self._max_len:
            raise ValueError(
                f"pair length {len(ids)} exceeds max_len {self._max_len}"
            )
        response_start = 2 + len(inst_bytes)
        return ids, response_start

    def decode_response(self, ids: Sequence[int]) -> str:
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
# Deterministic six-task fixture
# ---------------------------------------------------------------------------


_CAPITAL_BASES: tuple[tuple[str, str], ...] = (
    ("France", "Paris"),
    ("Spain", "Madrid"),
    ("Italy", "Rome"),
)

_CAPITAL_TEMPLATES: tuple[str, ...] = (
    "What is the capital of {c}?",
    "Name the capital city of {c}.",
    "Capital of {c}?",
    "Tell me the capital of {c}.",
    "{c}'s capital is?",
    "Which city is the capital of {c}?",
    "Give me the capital of {c}.",
    "The capital of {c} is?",
    "What city is {c}'s capital?",
    "Can you name {c}'s capital?",
    "Capital city of {c}?",
    "State the capital of {c}.",
    "Where is the capital of {c}?",
    "What's the capital of {c}?",
)

_ARITHMETIC_BASES: tuple[tuple[int, int, str], ...] = (
    (2, 3, "+"),
    (5, 4, "*"),
    (9, 7, "-"),
)

_ARITHMETIC_TEMPLATES: tuple[str, ...] = (
    "Compute {a} {op} {b}.",
    "What is {a} {op} {b}?",
    "{a} {op} {b}?",
    "Calculate {a} {op} {b}.",
    "Evaluate {a} {op} {b}.",
    "Solve {a} {op} {b}.",
    "Result of {a} {op} {b}?",
    "How much is {a} {op} {b}?",
    "{a} {op} {b} equals?",
    "Find {a} {op} {b}.",
)

_LIST_BASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("colors", ("red", "green", "blue")),
    ("planets", ("mercury", "venus", "earth")),
    ("vowels", ("a", "e", "i")),
)

_LIST_TEMPLATES: tuple[str, ...] = (
    "List three {n}.",
    "Give me three {n}.",
    "Name three {n}.",
    "Three examples of {n}?",
    "What are three {n}?",
    "Provide three {n}.",
    "Say three {n}.",
    "Which three {n}?",
    "Examples of {n}?",
    "Three {n}?",
)

_SUMMARY_BASES: tuple[tuple[str, str], ...] = (
    ("cats sleep a lot.", "cats sleep a lot."),
    ("water boils at 100c.", "water boils at 100c."),
    ("the moon orbits earth.", "the moon orbits earth."),
)

_SUMMARY_TEMPLATES: tuple[str, ...] = (
    "Summarise: {t}",
    "In brief: {t}",
    "TLDR: {t}",
    "Shorten: {t}",
    "Summary: {t}",
    "Condense: {t}",
    "Summarize: {t}",
    "In short: {t}",
    "Briefly: {t}",
    "One line: {t}",
)

_CODE_BASES: tuple[tuple[str, str], ...] = (
    ("print hello", "print('hello')"),
    ("sort items ascending", "items.sort()"),
    ("reverse items", "items.reverse()"),
)

_CODE_TEMPLATES: tuple[str, ...] = (
    "Write python code to {k}.",
    "Python: {k}.",
    "Code that will {k}.",
    "Show python code to {k}.",
    "How do I {k} in python?",
    "Python code: {k}.",
    "Code to {k}.",
    "Write code to {k}.",
    "Python snippet to {k}.",
    "Give python code to {k}.",
)

_DEFINITION_BASES: tuple[tuple[str, str], ...] = (
    ("variable", "a name bound to a value."),
    ("function", "a reusable block of code."),
    ("loop", "code that repeats a block."),
)

_DEFINITION_TEMPLATES: tuple[str, ...] = (
    "Define {x}.",
    "What is a {x}?",
    "Explain the term {x}.",
    "{x}: what is it?",
    "Give a definition of {x}.",
    "What does {x} mean?",
    "Describe a {x}.",
    "Define the word {x}.",
    "Tell me what a {x} is.",
    "{x} definition?",
    "Explain {x}.",
    "What is {x}?",
    "Meaning of {x}?",
    "Define {x} in one sentence.",
)


@dataclass(frozen=True)
class InstructionPair:
    instruction: str
    response: str
    category: str
    base: str
    template: int

    def __post_init__(self) -> None:
        _require_text("instruction", self.instruction)
        _require_text("response", self.response)
        _require_text("base", self.base)
        _require_int("template", self.template, minimum=0)
        if self.category not in CATEGORY_ORDER:
            raise ValueError(f"unknown category {self.category!r}")


@dataclass(frozen=True)
class DataSplit:
    train: tuple[InstructionPair, ...]
    test: tuple[InstructionPair, ...]


def _make_pairs() -> list[InstructionPair]:
    pairs: list[InstructionPair] = []

    for index, (country, city) in enumerate(_CAPITAL_BASES):
        count = 14 if index == 0 else 13
        for template_index in range(count):
            pairs.append(
                InstructionPair(
                    instruction=_CAPITAL_TEMPLATES[template_index].format(c=country),
                    response=f"{city}.",
                    category="capitals",
                    base=country,
                    template=template_index,
                )
            )

    for first, second, operator in _ARITHMETIC_BASES:
        if operator == "+":
            result = first + second
        elif operator == "*":
            result = first * second
        elif operator == "-":
            result = first - second
        else:  # pragma: no cover - the fixture has only three operators
            raise ValueError(f"unsupported operator {operator!r}")
        response = f"{first} {operator} {second} = {result}."
        for template_index, template in enumerate(_ARITHMETIC_TEMPLATES):
            pairs.append(
                InstructionPair(
                    instruction=template.format(a=first, b=second, op=operator),
                    response=response,
                    category="arithmetic",
                    base=f"{first} {operator} {second}",
                    template=template_index,
                )
            )

    for name, items in _LIST_BASES:
        response = ", ".join(items) + "."
        for template_index, template in enumerate(_LIST_TEMPLATES):
            pairs.append(
                InstructionPair(
                    instruction=template.format(n=name),
                    response=response,
                    category="lists",
                    base=name,
                    template=template_index,
                )
            )

    for text, summary in _SUMMARY_BASES:
        for template_index, template in enumerate(_SUMMARY_TEMPLATES):
            pairs.append(
                InstructionPair(
                    instruction=template.format(t=text),
                    response=summary,
                    category="summaries",
                    base=text,
                    template=template_index,
                )
            )

    for task, code in _CODE_BASES:
        for template_index, template in enumerate(_CODE_TEMPLATES):
            pairs.append(
                InstructionPair(
                    instruction=template.format(k=task),
                    response=code,
                    category="code",
                    base=task,
                    template=template_index,
                )
            )

    for index, (term, definition) in enumerate(_DEFINITION_BASES):
        count = 14 if index == 0 else 13
        for template_index in range(count):
            pairs.append(
                InstructionPair(
                    instruction=_DEFINITION_TEMPLATES[template_index].format(x=term),
                    response=definition,
                    category="definitions",
                    base=term,
                    template=template_index,
                )
            )

    return pairs


def make_dataset(seed: int = 0) -> tuple[InstructionPair, ...]:
    """Build the 200-pair fixture, deterministically shuffled."""
    seed_value = _require_seed(seed)
    pairs = _make_pairs()
    if len(pairs) != 200:
        raise ValueError(f"fixture must contain 200 pairs, found {len(pairs)}")
    instructions = {pair.instruction for pair in pairs}
    if len(instructions) != len(pairs):
        raise ValueError("fixture instructions must be unique")

    tokenizer = InstructionTokenizer()
    for pair in pairs:
        tokenizer.encode_pair(pair.instruction, pair.response)

    random.Random(seed_value).shuffle(pairs)
    return tuple(pairs)


def split_dataset(
    pairs: Sequence[InstructionPair],
    test_fraction: float = 0.2,
    seed: int = 0,
) -> DataSplit:
    """Stratified held-out split with category and base-coverage checks."""
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise TypeError("pairs must be a sequence of InstructionPair")
    values = tuple(pairs)
    if not values:
        raise ValueError("pairs must not be empty")
    if not all(isinstance(pair, InstructionPair) for pair in values):
        raise TypeError("pairs must contain InstructionPair values")
    fraction = _require_float("test_fraction", test_fraction, minimum=0.0)
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError("test_fraction must be strictly between 0 and 1")
    seed_value = _require_seed(seed)

    by_category: dict[str, list[InstructionPair]] = {}
    for pair in values:
        by_category.setdefault(pair.category, []).append(pair)
    missing = [category for category in CATEGORY_ORDER if category not in by_category]
    if missing:
        raise ValueError(f"missing categories: {missing}")

    rng = random.Random(seed_value)
    train: list[InstructionPair] = []
    test: list[InstructionPair] = []
    for category in CATEGORY_ORDER:
        group = by_category[category]
        if len(group) < 2:
            raise ValueError(f"category {category!r} needs at least two pairs")
        order = list(range(len(group)))
        rng.shuffle(order)
        test_count = max(1, round(len(group) * fraction))
        if test_count >= len(group):
            raise ValueError(f"test_fraction leaves no training pairs in {category}")
        chosen = set(order[:test_count])
        for index, pair in enumerate(group):
            if index in chosen:
                test.append(pair)
            else:
                train.append(pair)

    for category in CATEGORY_ORDER:
        if not any(pair.category == category for pair in train):
            raise ValueError(f"training split has no {category} pairs")
        if not any(pair.category == category for pair in test):
            raise ValueError(f"test split has no {category} pairs")

    train_bases = {(pair.category, pair.base) for pair in train}
    for pair in test:
        if (pair.category, pair.base) not in train_bases:
            raise ValueError(
                f"test base {pair.base!r} in {pair.category!r} has no training pair"
            )

    if {pair.instruction for pair in train} & {pair.instruction for pair in test}:
        raise ValueError("train and test instructions overlap")
    return DataSplit(train=tuple(train), test=tuple(test))


# ---------------------------------------------------------------------------
# Dataset, collate, and response-only loss
# ---------------------------------------------------------------------------


class SFTDataset(Dataset[tuple[Tensor, Tensor]]):
    """Full token ids and response-only labels, before dynamic padding."""

    def __init__(
        self,
        pairs: Sequence[InstructionPair],
        tokenizer: InstructionTokenizer,
    ) -> None:
        if not isinstance(tokenizer, InstructionTokenizer):
            raise TypeError("tokenizer must be an InstructionTokenizer")
        if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
            raise TypeError("pairs must be a sequence of InstructionPair")
        values = tuple(pairs)
        if not values:
            raise ValueError("pairs must not be empty")
        if not all(isinstance(pair, InstructionPair) for pair in values):
            raise TypeError("pairs must contain InstructionPair values")
        self._pairs = values
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        pair = self._pairs[index]
        encoded, response_start = self._tokenizer.encode_pair(
            pair.instruction, pair.response
        )
        ids = torch.tensor(encoded, dtype=torch.long)
        labels = ids.clone()
        labels[:response_start] = IGNORE_INDEX
        return ids, labels


def sft_collate(
    batch: Sequence[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """Right-pad a batch and mask instruction, boundary, and padding labels."""
    if isinstance(batch, (str, bytes)) or not isinstance(batch, Sequence):
        raise TypeError("batch must be a sequence of (ids, labels) tensors")
    if not batch:
        raise ValueError("batch must not be empty")

    rows: list[tuple[Tensor, Tensor]] = []
    width = 0
    for item in batch:
        if len(item) != 2:
            raise ValueError("each batch item must be (ids, labels)")
        ids, labels = item
        if not isinstance(ids, Tensor) or not isinstance(labels, Tensor):
            raise TypeError("ids and labels must be tensors")
        if ids.ndim != 1 or labels.ndim != 1:
            raise ValueError("ids and labels must be 1-D")
        if ids.shape[0] != labels.shape[0]:
            raise ValueError("ids and labels must have the same length")
        if ids.shape[0] < 1:
            raise ValueError("sequences must not be empty")
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"ids must be integer, got {ids.dtype}")
        if labels.dtype not in (torch.int32, torch.int64):
            raise TypeError(f"labels must be integer, got {labels.dtype}")
        if bool(torch.any(ids < 0)) or bool(torch.any(ids >= VOCAB_SIZE)):
            raise ValueError("ids must be in [0, vocab_size)")
        valid_labels = labels != IGNORE_INDEX
        if bool(torch.any(valid_labels & ((labels < 0) | (labels >= VOCAB_SIZE)))):
            raise ValueError("labels must be -100 or a valid token id")
        rows.append((ids.to(torch.long), labels.to(torch.long)))
        width = max(width, ids.shape[0])

    padded_ids: list[Tensor] = []
    padded_labels: list[Tensor] = []
    for ids, labels in rows:
        padding = width - ids.shape[0]
        padded_ids.append(
            torch.cat(
                (ids, torch.full((padding,), PAD_ID, dtype=torch.long))
            )
        )
        padded_labels.append(
            torch.cat(
                (labels, torch.full((padding,), IGNORE_INDEX, dtype=torch.long))
            )
        )
    return torch.stack(padded_ids), torch.stack(padded_labels)


def sft_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Causal cross-entropy with the explicit one-token label shift."""
    if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
        raise TypeError("logits and labels must be tensors")
    if logits.ndim != 3:
        raise ValueError(f"logits must be (B, T, V), got {tuple(logits.shape)}")
    if logits.shape[-1] != VOCAB_SIZE:
        raise ValueError(f"logits last dimension must be {VOCAB_SIZE}")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if labels.ndim != 2 or tuple(labels.shape) != tuple(logits.shape[:2]):
        raise ValueError(
            f"labels must be {tuple(logits.shape[:2])}, got {tuple(labels.shape)}"
        )
    if labels.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"labels must be integer, got {labels.dtype}")
    valid = labels != IGNORE_INDEX
    if bool(torch.any(valid & ((labels < 0) | (labels >= VOCAB_SIZE)))):
        raise ValueError("labels must be -100 or a valid token id")

    targets = labels[:, 1:]
    if not bool(torch.any(targets != IGNORE_INDEX)):
        raise ValueError("batch has no response targets")
    return F.cross_entropy(
        logits[:, :-1, :].reshape(-1, VOCAB_SIZE),
        targets.reshape(-1).to(torch.long),
        ignore_index=IGNORE_INDEX,
    )


# ---------------------------------------------------------------------------
# Tiny transformer
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


class Block(nn.Module):
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
        attention = (self.attn_dropout(torch.softmax(scores, dim=-1)) @ value)
        attention = attention.transpose(1, 2).contiguous().view(
            batch, length, self.hidden
        )
        hidden = hidden + self.resid_dropout(self.out_proj(attention))
        feed_forward = self.fc2(F.gelu(self.fc1(self.ln2(hidden))))
        output: Tensor = hidden + self.resid_dropout(feed_forward)
        return output


class TinyGPT(nn.Module):
    """Decoder-only transformer body with a tied language-model head."""

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
                Block(
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
        if not trainable_only or parameter.requires_grad
    )


# ---------------------------------------------------------------------------
# Generation, exact match, and evaluation
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    return " ".join(text.lower().split())


def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 when normalized strings match, otherwise 0.0."""
    if normalize_text(prediction) == normalize_text(reference):
        return 1.0
    return 0.0


@torch.no_grad()
def generate(
    model: nn.Module,
    tokenizer: InstructionTokenizer,
    instruction: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    seed: int = 0,
) -> str:
    """Causal decode a response prefix with greedy or sampled tokens."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(tokenizer, InstructionTokenizer):
        raise TypeError("tokenizer must be an InstructionTokenizer")
    max_tokens = _require_int("max_new_tokens", max_new_tokens, minimum=1)
    temp = _require_float("temperature", temperature, minimum=0.0)
    seed_value = _require_seed(seed)
    context_length = getattr(model, "context_length", None)
    if isinstance(context_length, bool) or not isinstance(context_length, int):
        raise TypeError("model must expose an int context_length")
    if context_length < 2:
        raise ValueError("model context_length must be >= 2")

    ids = tokenizer.encode_prefix(instruction)
    if len(ids) >= context_length:
        return ""

    generator = (
        torch.Generator().manual_seed(seed_value) if temp > 0.0 else None
    )
    emitted: list[int] = []
    was_training = model.training
    try:
        model.eval()
        for _ in range(max_tokens):
            if len(ids) >= context_length:
                break
            token_ids = torch.tensor([ids], dtype=torch.long)
            logits = cast(Tensor, model(token_ids))
            if logits.ndim != 3 or logits.shape[0] != 1:
                raise ValueError("model must return logits of shape (1, T, V)")
            if logits.shape[-1] != VOCAB_SIZE:
                raise ValueError(f"model logits must have vocabulary {VOCAB_SIZE}")
            next_logits = logits[0, -1, :]
            if temp == 0.0:
                next_id = int(torch.argmax(next_logits).item())
            else:
                probabilities = torch.softmax(next_logits / temp, dim=-1)
                next_id = int(
                    torch.multinomial(
                        probabilities, num_samples=1, generator=generator
                    ).item()
                )
            if next_id in (INST_ID, RESP_ID, PAD_ID, RESPONSE_TERMINATOR_ID):
                break
            ids.append(next_id)
            emitted.append(next_id)
            if (
                len(emitted) >= 2
                and emitted[-1] in SENTENCE_END_BYTES
                and emitted[-2] in SENTENCE_END_BYTES
            ):
                break
    finally:
        model.train(was_training)
    return tokenizer.decode_response(emitted)


@dataclass(frozen=True)
class CategoryScore:
    category: str
    correct: int
    total: int
    exact_match: float


@dataclass(frozen=True)
class EvalResult:
    correct: int
    total: int
    exact_match: float
    categories: tuple[CategoryScore, ...]
    predictions: tuple[str, ...]


@dataclass(frozen=True)
class SFTReport:
    losses: tuple[float, ...]
    evals: tuple[tuple[int, EvalResult], ...]
    final_eval: EvalResult | None


@dataclass(frozen=True)
class DemoConfig:
    hidden: int = 96
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.0
    max_len: int = MAX_SEQUENCE_LENGTH
    batch_size: int = 8
    epochs: int = 60
    learning_rate: float = 1e-3
    eval_every: int = 5
    max_new_tokens: int = 64
    test_fraction: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        hidden = _require_int("hidden", self.hidden, minimum=1)
        heads = _require_int("num_heads", self.num_heads, minimum=1)
        _require_int("num_layers", self.num_layers, minimum=1)
        dropout = _require_float("dropout", self.dropout, minimum=0.0)
        _require_int("max_len", self.max_len, minimum=4)
        _require_int("batch_size", self.batch_size, minimum=1)
        _require_int("epochs", self.epochs, minimum=1)
        rate = _require_float("learning_rate", self.learning_rate, minimum=0.0)
        _require_int("eval_every", self.eval_every, minimum=1)
        _require_int("max_new_tokens", self.max_new_tokens, minimum=1)
        fraction = _require_float("test_fraction", self.test_fraction, minimum=0.0)
        _require_seed(self.seed)
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by num_heads")
        if dropout >= 1.0:
            raise ValueError("dropout must be < 1")
        if rate == 0.0:
            raise ValueError("learning_rate must be > 0")
        if fraction <= 0.0 or fraction >= 1.0:
            raise ValueError("test_fraction must be strictly between 0 and 1")


@dataclass(frozen=True)
class DemoReport:
    config: DemoConfig
    losses: tuple[float, ...]
    evals: tuple[tuple[int, EvalResult], ...]
    baseline: EvalResult
    final: EvalResult
    response_targets: int
    ignored_targets: int
    mask_ok: bool


def evaluate(
    model: nn.Module,
    tokenizer: InstructionTokenizer,
    pairs: Sequence[InstructionPair],
    *,
    max_new_tokens: int = 64,
) -> EvalResult:
    """Greedy held-out exact-match, overall and per category."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(tokenizer, InstructionTokenizer):
        raise TypeError("tokenizer must be an InstructionTokenizer")
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise TypeError("pairs must be a sequence of InstructionPair")
    values = tuple(pairs)
    if not values:
        raise ValueError("pairs must not be empty")
    if not all(isinstance(pair, InstructionPair) for pair in values):
        raise TypeError("pairs must contain InstructionPair values")
    _require_int("max_new_tokens", max_new_tokens, minimum=1)

    predictions: list[str] = []
    category_counts: dict[str, list[int]] = {}
    correct = 0
    was_training = model.training
    try:
        for pair in values:
            prediction = generate(
                model,
                tokenizer,
                pair.instruction,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )
            hit = int(exact_match(prediction, pair.response))
            correct += hit
            predictions.append(prediction)
            counts = category_counts.setdefault(pair.category, [0, 0])
            counts[0] += hit
            counts[1] += 1
    finally:
        model.train(was_training)

    categories = tuple(
        CategoryScore(
            category=category,
            correct=category_counts[category][0],
            total=category_counts[category][1],
            exact_match=category_counts[category][0]
            / category_counts[category][1],
        )
        for category in sorted(category_counts)
    )
    total = len(values)
    return EvalResult(
        correct=correct,
        total=total,
        exact_match=correct / total,
        categories=categories,
        predictions=tuple(predictions),
    )


def train_sft(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    eval_every: int = 5,
    eval_fn: Callable[[], EvalResult] | None = None,
    log: Callable[[str], None] = print,
) -> SFTReport:
    """Response-only SFT loop with optional periodic held-out evaluation."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    epoch_count = _require_int("epochs", epochs, minimum=1)
    interval = _require_int("eval_every", eval_every, minimum=1)

    losses: list[float] = []
    evals: list[tuple[int, EvalResult]] = []
    final_eval: EvalResult | None = None

    for epoch in range(1, epoch_count + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for input_ids, labels in loader:
            logits = cast(Tensor, model(input_ids))
            loss = sft_loss(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        if batches == 0:
            raise ValueError("training loader produced no batches")
        average = total_loss / batches
        losses.append(average)
        if eval_fn is not None and epoch % interval == 0:
            result = eval_fn()
            evals.append((epoch, result))
            final_eval = result
            log(
                f"epoch {epoch:>3d}: loss={average:.4f} "
                f"exact_match={result.exact_match:.3f}"
            )

    if eval_fn is not None and (not evals or evals[-1][0] != epoch_count):
        final_eval = eval_fn()
    return SFTReport(tuple(losses), tuple(evals), final_eval)


# ---------------------------------------------------------------------------
# Demo and CLI
# ---------------------------------------------------------------------------


def run_demo(config: DemoConfig | None = None) -> DemoReport:
    """Build the fixture, train the default model, and print measurements."""
    if config is None:
        config = DemoConfig()
    if not isinstance(config, DemoConfig):
        raise TypeError("config must be a DemoConfig")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    tokenizer = InstructionTokenizer(config.max_len)
    pairs = make_dataset(config.seed)
    split = split_dataset(
        pairs, test_fraction=config.test_fraction, seed=config.seed
    )
    train_dataset = SFTDataset(split.train, tokenizer)
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=sft_collate,
    )
    model = TinyGPT(
        ModelConfig(
            hidden=config.hidden,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_len=config.max_len,
            dropout=config.dropout,
        ),
        seed=config.seed,
    )

    baseline = evaluate(
        model, tokenizer, split.test, max_new_tokens=config.max_new_tokens
    )

    category_counts = {
        category: sum(1 for pair in pairs if pair.category == category)
        for category in CATEGORY_ORDER
    }
    print(
        f"instruction-tuning-sft: seed={config.seed} "
        f"train={len(split.train)} test={len(split.test)} "
        f"max_len={config.max_len}"
    )
    print(
        "dataset: pairs="
        + str(len(pairs))
        + " "
        + " ".join(
            f"{category}={category_counts[category]}"
            for category in CATEGORY_ORDER
        )
    )
    print(
        f"tokenizer: vocab={VOCAB_SIZE} inst={INST_ID} resp={RESP_ID} "
        f"pad={PAD_ID} terminator={RESPONSE_TERMINATOR_ID}"
    )
    print(
        f"model: parameters={count_parameters(model)} "
        f"context_length={model.context_length} hidden={config.hidden} "
        f"heads={config.num_heads} layers={config.num_layers}"
    )

    response_targets = 0
    total_targets = 0
    for index in range(len(train_dataset)):
        _, labels = train_dataset[index]
        total_targets += labels.numel()
        response_targets += int((labels != IGNORE_INDEX).sum().item())
    ignored_targets = total_targets - response_targets
    print(
        f"mask: targets={total_targets} response={response_targets} "
        f"ignored={ignored_targets} "
        f"response_fraction={response_targets / total_targets:.3f}"
    )
    print(
        f"baseline: exact_match={baseline.exact_match:.3f} "
        f"correct={baseline.correct}/{baseline.total}"
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate
    )
    eval_fn = lambda: evaluate(  # noqa: E731
        model, tokenizer, split.test, max_new_tokens=config.max_new_tokens
    )
    report = train_sft(
        model,
        loader,
        optimizer,
        epochs=config.epochs,
        eval_every=config.eval_every,
        eval_fn=eval_fn,
    )
    final = report.final_eval
    if final is None:
        raise RuntimeError("training did not produce a final evaluation")

    print(
        f"final: exact_match={final.exact_match:.3f} "
        f"correct={final.correct}/{final.total}"
    )
    for category in final.categories:
        print(
            f"per-category: {category.category}={category.exact_match:.3f} "
            f"({category.correct}/{category.total})"
        )
    for pair in split.test[:3]:
        prediction = generate(
            model,
            tokenizer,
            pair.instruction,
            max_new_tokens=config.max_new_tokens,
        )
        status = "MATCH" if exact_match(prediction, pair.response) == 1.0 else "MISS"
        print(f"sample: [{status}] instruction={pair.instruction}")
        print(f"sample:         gold={pair.response}")
        print(f"sample:         prediction={prediction}")

    mask_ok = True
    for index, pair in enumerate(split.train):
        encoded, response_start = tokenizer.encode_pair(
            pair.instruction, pair.response
        )
        ids, labels = train_dataset[index]
        if ids.tolist() != encoded:
            mask_ok = False
        if bool((labels[:response_start] != IGNORE_INDEX).any()):
            mask_ok = False
        if bool((labels[response_start:] == IGNORE_INDEX).any()):
            mask_ok = False
    all_ids, all_labels = sft_collate(
        [train_dataset[index] for index in range(len(train_dataset))]
    )
    if bool(((all_ids == PAD_ID) & (all_labels != IGNORE_INDEX)).any()):
        mask_ok = False
    loss_finite = all(math.isfinite(value) for value in report.losses)
    print(
        f"check: loss_finite={str(loss_finite).lower()} "
        f"mask_only_response={str(mask_ok).lower()}"
    )

    return DemoReport(
        config=config,
        losses=report.losses,
        evals=report.evals,
        baseline=baseline,
        final=final,
        response_targets=response_targets,
        ignored_targets=ignored_targets,
        mask_ok=mask_ok,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Instruction tuning by supervised fine-tuning."
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--max-len", type=int, default=MAX_SEQUENCE_LENGTH)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        config = DemoConfig(
            hidden=args.hidden,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            max_len=args.max_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            eval_every=args.eval_every,
            max_new_tokens=args.max_new_tokens,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        report = run_demo(config)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not all(math.isfinite(value) for value in report.losses):
        print("error: training produced a non-finite loss", file=sys.stderr)
        return 1
    if not report.mask_ok:
        print("error: response-only mask audit failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
