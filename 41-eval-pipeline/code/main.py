"""A unified, offline evaluation pipeline for tiny byte-level language models.

The module implements four heterogeneous evaluations (held-out perplexity,
short-form exact match, open-form token F1, and a deterministic mock
LLM-as-judge), a configurable weighted aggregator, and a stable-key JSON
report. It trains a small decoder-only transformer from scratch on a synthetic
instruction corpus so the whole loop runs on CPU without a network, a dataset
download, or a provider API.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeAlias

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


# ---------------------------------------------------------------------------
# Deterministic synthetic fixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QaExample:
    """One instruction with a single reference answer."""

    instruction: str
    reference: str
    category: str

    def __post_init__(self) -> None:
        instruction = _require_text("instruction", self.instruction)
        reference = _require_text("reference", self.reference)
        if "\n" in instruction or "\r" in instruction:
            raise ValueError("instruction must be a single line")
        if "\n" in reference or "\r" in reference:
            raise ValueError("reference must be a single line")
        _require_text("category", self.category)


_CAPITALS: tuple[tuple[str, str], ...] = (
    ("France", "Paris"),
    ("Spain", "Madrid"),
    ("Italy", "Rome"),
    ("Japan", "Tokyo"),
    ("Kenya", "Nairobi"),
    ("Canada", "Ottawa"),
    ("Brazil", "Brasilia"),
    ("Egypt", "Cairo"),
)

_CAPITAL_TRAIN_TEMPLATES: tuple[str, ...] = (
    "What is the capital of {country}?",
    "Name the capital city of {country}.",
    "Tell me {country}'s capital.",
)
_CAPITAL_EM_TEMPLATE = "The capital of {country} is?"
_CAPITAL_COVERAGE: tuple[tuple[str, str], ...] = (
    ("Germany", "Berlin"),
    ("Portugal", "Lisbon"),
    ("Greece", "Athens"),
    ("Norway", "Oslo"),
)

_ARITHMETIC: tuple[tuple[int, int, str], ...] = (
    (2, 3, "+"),
    (7, 6, "*"),
    (9, 4, "-"),
    (12, 3, "/"),
    (5, 5, "+"),
    (8, 2, "*"),
)

_ARITHMETIC_TRAIN_TEMPLATES: tuple[str, ...] = (
    "What is {a} {op} {b}?",
    "Compute {a} {op} {b}.",
    "How much is {a} {op} {b}?",
)
_ARITHMETIC_EM_TEMPLATE = "{a} {op} {b} equals?"
_ARITHMETIC_COVERAGE: tuple[tuple[int, int, str], ...] = (
    (3, 4, "+"),
    (6, 2, "*"),
    (10, 3, "-"),
)

_LISTS: tuple[tuple[str, str], ...] = (
    ("colors", "red, green, blue."),
    ("planets", "mercury, venus, earth."),
    ("vowels", "a, e, i, o, u."),
    ("days", "monday, tuesday, wednesday."),
    ("animals", "cat, dog, bird."),
    ("fruits", "apple, banana, cherry."),
    ("oceans", "pacific, atlantic, indian."),
    ("shapes", "circle, square, triangle."),
    ("instruments", "piano, guitar, violin."),
    ("metals", "iron, copper, gold."),
    ("seasons", "spring, summer, autumn."),
    ("languages", "python, rust, go."),
    ("vehicles", "car, train, plane."),
    ("trees", "oak, pine, maple."),
)
_EM_LIST_COUNT = 6

_LIST_TRAIN_TEMPLATES: tuple[str, ...] = (
    "List three {name}.",
    "Name three {name}.",
    "Give three examples of {name}.",
)
_LIST_EVAL_TEMPLATE = "Which three {name} come to mind?"
_LIST_COVERAGE: tuple[tuple[str, str], ...] = (
    ("berries", "strawberry, blueberry, raspberry."),
    ("tools", "hammer, saw, drill."),
    ("sports", "soccer, tennis, swimming."),
    ("drinks", "water, tea, coffee."),
)

_DEFINITIONS: tuple[tuple[str, str], ...] = (
    ("variable", "a name bound to a value."),
    ("function", "a reusable block of code."),
    ("loop", "code that repeats a block."),
    ("token", "a unit of text."),
    ("gradient", "a slope of a loss."),
    ("tensor", "a multidimensional array."),
    ("optimizer", "a rule that updates weights."),
    ("checkpoint", "a saved model state."),
    ("dataset", "a collection of examples."),
    ("inference", "running a trained model."),
    ("prompt", "an instruction given to a model."),
    ("embedding", "a learned vector representation."),
)
_JUDGE_DEFINITION_COUNT = 10

_DEFINITION_TRAIN_TEMPLATES: tuple[str, ...] = (
    "Define {term}.",
    "What is a {term}?",
    "Explain the term {term}.",
)
_F1_DEFINITION_TEMPLATE = "Give a definition of {term}."
_JUDGE_DEFINITION_TEMPLATE = "In one sentence, define {term}."
_DEFINITION_COVERAGE: tuple[tuple[str, str], ...] = (
    ("attention", "a weighting over tokens."),
    ("epoch", "one pass over the data."),
    ("batch", "a group of examples."),
    ("loss", "a training objective."),
    ("layer", "a stack of operations."),
    ("weight", "a learned parameter."),
)

_SUMMARIES: tuple[str, ...] = (
    "cats sleep a lot.",
    "water boils at 100c.",
    "the moon orbits earth.",
    "plants need sunlight.",
    "ice is frozen water.",
    "birds have feathers.",
    "the sun is a star.",
    "fish live in water.",
    "rocks are hard.",
    "fire is hot.",
)

_SUMMARY_TRAIN_TEMPLATES: tuple[str, ...] = (
    "Summarise: {text}",
    "In brief: {text}",
    "Shorten: {text}",
)
_SUMMARY_JUDGE_TEMPLATE = "Summary: {text}"
_SUMMARY_COVERAGE: tuple[str, ...] = (
    "snow is cold.",
    "metal is heavy.",
    "glass is fragile.",
)


def _arithmetic_response(first: int, second: int, operator: str) -> str:
    if operator == "+":
        result = first + second
    elif operator == "*":
        result = first * second
    elif operator == "-":
        result = first - second
    elif operator == "/":
        if second == 0 or first % second != 0:
            raise ValueError("division fixture must be exact")
        result = first // second
    else:
        raise ValueError(f"unsupported operator {operator!r}")
    return str(result)


def _make_pairs() -> list[QaExample]:
    pairs: list[QaExample] = []
    for country, city in (*_CAPITALS, *_CAPITAL_COVERAGE):
        for template in _CAPITAL_TRAIN_TEMPLATES:
            pairs.append(
                QaExample(
                    template.format(country=country),
                    f"{city}.",
                    "capitals",
                )
            )
    for country, city in _CAPITAL_COVERAGE:
        pairs.append(
            QaExample(
                _CAPITAL_EM_TEMPLATE.format(country=country),
                f"{city}.",
                "capitals",
            )
        )
    for first, second, operator in (*_ARITHMETIC, *_ARITHMETIC_COVERAGE):
        response = _arithmetic_response(first, second, operator)
        for template in _ARITHMETIC_TRAIN_TEMPLATES:
            pairs.append(
                QaExample(
                    template.format(a=first, op=operator, b=second),
                    response,
                    "arithmetic",
                )
            )
    for first, second, operator in _ARITHMETIC_COVERAGE:
        pairs.append(
            QaExample(
                _ARITHMETIC_EM_TEMPLATE.format(a=first, op=operator, b=second),
                _arithmetic_response(first, second, operator),
                "arithmetic",
            )
        )
    for name, items in (*_LISTS, *_LIST_COVERAGE):
        for template in _LIST_TRAIN_TEMPLATES:
            pairs.append(
                QaExample(
                    template.format(name=name),
                    items,
                    "lists",
                )
            )
    for name, items in _LIST_COVERAGE:
        pairs.append(
            QaExample(
                _LIST_EVAL_TEMPLATE.format(name=name),
                items,
                "lists",
            )
        )
    for term, definition in (*_DEFINITIONS, *_DEFINITION_COVERAGE):
        for template in _DEFINITION_TRAIN_TEMPLATES:
            pairs.append(
                QaExample(
                    template.format(term=term),
                    definition,
                    "definitions",
                )
            )
    for term, definition in _DEFINITION_COVERAGE:
        pairs.append(
            QaExample(
                _F1_DEFINITION_TEMPLATE.format(term=term),
                definition,
                "definitions",
            )
        )
    for term, definition in _DEFINITION_COVERAGE[:3]:
        pairs.append(
            QaExample(
                _JUDGE_DEFINITION_TEMPLATE.format(term=term),
                definition,
                "definitions",
            )
        )
    for text in (*_SUMMARIES, *_SUMMARY_COVERAGE):
        for template in _SUMMARY_TRAIN_TEMPLATES:
            pairs.append(
                QaExample(
                    template.format(text=text),
                    text,
                    "summaries",
                )
            )
    for text in _SUMMARY_COVERAGE:
        pairs.append(
            QaExample(
                _SUMMARY_JUDGE_TEMPLATE.format(text=text),
                text,
                "summaries",
            )
        )
    return pairs


def make_training_pairs() -> tuple[QaExample, ...]:
    """Return the fixed 233-pair training split."""
    pairs = _make_pairs()
    if len(pairs) != 233:
        raise ValueError(
            f"training fixture must contain 233 pairs, found {len(pairs)}"
        )
    instructions = {pair.instruction for pair in pairs}
    if len(instructions) != len(pairs):
        raise ValueError("training instructions must be unique")
    return tuple(pairs)


def make_lm_training_corpus() -> tuple[str, ...]:
    """Return 30 declarative training sentences."""
    sentences: list[str] = [
        f"The capital of {country} is {city}."
        for country, city in _CAPITALS
    ]
    sentences.extend(
        f"{first} {operator} {second} equals "
        f"{_arithmetic_response(first, second, operator)}."
        for first, second, operator in _ARITHMETIC
    )
    sentences.extend(
        f"{items.rstrip('.')} are examples of {name}."
        for name, items in _LISTS[:6]
    )
    sentences.extend(
        f"{term} is {definition}"
        for term, definition in _DEFINITIONS[:5]
    )
    sentences.extend(_SUMMARIES[:5])
    if len(sentences) != 30:
        raise ValueError(f"LM training corpus must have 30 sentences, found {len(sentences)}")
    return tuple(sentences)


def make_lm_fixture() -> tuple[str, ...]:
    """Return 20 held-out declarative sentences."""
    sentences: list[str] = [
        f"{city} is the capital city of {country}."
        for country, city in _CAPITALS
    ]
    sentences.extend(
        f"When you compute {first} {operator} {second}, you get "
        f"{_arithmetic_response(first, second, operator)}."
        for first, second, operator in _ARITHMETIC
    )
    sentences.extend(
        f"Examples of {name} include {items.rstrip('.')}."
        for name, items in _LISTS[:6]
    )
    if len(sentences) != 20:
        raise ValueError(f"LM fixture must have 20 sentences, found {len(sentences)}")
    return tuple(sentences)


def make_em_fixture() -> tuple[QaExample, ...]:
    """Return 20 held-out short-form factual prompts."""
    examples: list[QaExample] = [
        QaExample(
            _CAPITAL_EM_TEMPLATE.format(country=country),
            f"{city}.",
            "capitals",
        )
        for country, city in _CAPITALS
    ]
    examples.extend(
        QaExample(
            _ARITHMETIC_EM_TEMPLATE.format(a=first, op=operator, b=second),
            _arithmetic_response(first, second, operator),
            "arithmetic",
        )
        for first, second, operator in _ARITHMETIC
    )
    examples.extend(
        QaExample(
            _LIST_EVAL_TEMPLATE.format(name=name),
            items,
            "lists",
        )
        for name, items in _LISTS[:_EM_LIST_COUNT]
    )
    return _validate_fixture(examples, expected=20, name="EM")


def make_f1_fixture() -> tuple[QaExample, ...]:
    """Return 20 held-out open-form prompts with multi-token references."""
    examples: list[QaExample] = [
        QaExample(
            _LIST_EVAL_TEMPLATE.format(name=name),
            items,
            "lists",
        )
        for name, items in _LISTS[_EM_LIST_COUNT:]
    ]
    examples.extend(
        QaExample(
            _F1_DEFINITION_TEMPLATE.format(term=term),
            definition,
            "definitions",
        )
        for term, definition in _DEFINITIONS
    )
    return _validate_fixture(examples, expected=20, name="F1")


def make_judge_fixture() -> tuple[QaExample, ...]:
    """Return 20 held-out prompts for qualitative judging."""
    examples: list[QaExample] = [
        QaExample(
            _JUDGE_DEFINITION_TEMPLATE.format(term=term),
            definition,
            "definitions",
        )
        for term, definition in _DEFINITIONS[:_JUDGE_DEFINITION_COUNT]
    ]
    examples.extend(
        QaExample(
            _SUMMARY_JUDGE_TEMPLATE.format(text=text),
            text,
            "summaries",
        )
        for text in _SUMMARIES
    )
    return _validate_fixture(examples, expected=20, name="judge")


def _validate_fixture(
    examples: Sequence[QaExample], *, expected: int, name: str
) -> tuple[QaExample, ...]:
    values = tuple(examples)
    if len(values) != expected:
        raise ValueError(f"{name} fixture must have {expected} examples")
    instructions = {example.instruction for example in values}
    if len(instructions) != len(values):
        raise ValueError(f"{name} prompts must be unique")
    tokenizer = InstructionTokenizer()
    for example in values:
        tokenizer.encode_example(example.instruction, example.reference)
    return values


# ---------------------------------------------------------------------------
# Exact match, token F1, and the deterministic mock judge
# ---------------------------------------------------------------------------


def normalise_text(text: str) -> str:
    """Lowercase, collapse whitespace, and drop trailing terminal punctuation."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    collapsed = " ".join(text.lower().split())
    while collapsed and collapsed[-1] in ".!?":
        collapsed = collapsed[:-1].rstrip()
    return collapsed


def exact_match(prediction: str, reference: str) -> bool:
    """Compare two answers after normalisation."""
    return normalise_text(prediction) == normalise_text(reference)


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 over a whitespace bag-of-tokens."""
    prediction_tokens = normalise_text(prediction).split()
    reference_tokens = normalise_text(reference).split()
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    matches = sum(common.values())
    if matches == 0:
        return 0.0
    precision = matches / len(prediction_tokens)
    recall = matches / len(reference_tokens)
    return 2.0 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class JudgeVerdict:
    score: int
    rationale: str

    def __post_init__(self) -> None:
        score = _require_int("score", self.score, minimum=1)
        if score > 5:
            raise ValueError("score must be <= 5")
        _require_text("rationale", self.rationale)


def mock_judge(
    instruction: str, prediction: str, reference: str
) -> JudgeVerdict:
    """Score a prediction deterministically against a reference on 1-5."""
    _require_text("instruction", instruction)
    if not isinstance(prediction, str):
        raise TypeError("prediction must be a string")
    if not isinstance(reference, str):
        raise TypeError("reference must be a string")
    if normalise_text(prediction) == normalise_text(reference):
        return JudgeVerdict(5, "exact normalised match")
    score = token_f1(prediction, reference)
    if score >= 0.8:
        return JudgeVerdict(4, f"token F1 {score:.3f} >= 0.80")
    if score >= 0.5:
        return JudgeVerdict(3, f"token F1 {score:.3f} in [0.50, 0.80)")
    if score >= 0.2:
        return JudgeVerdict(2, f"token F1 {score:.3f} in [0.20, 0.50)")
    return JudgeVerdict(1, f"token F1 {score:.3f} < 0.20")


# ---------------------------------------------------------------------------
# Evaluation results, predictors, and the four evals
# ---------------------------------------------------------------------------

JsonValue: TypeAlias = (
    "str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]"
)


@dataclass(frozen=True)
class TextExampleResult:
    instruction: str
    reference: str
    prediction: str
    score: float
    normalised_prediction: str
    normalised_reference: str
    rationale: str
    kind: str = "text"


@dataclass(frozen=True)
class PerplexityExampleResult:
    document: str
    tokens: int
    mean_nll: float
    perplexity: float
    kind: str = "perplexity"


EvalExample: TypeAlias = "TextExampleResult | PerplexityExampleResult"


@dataclass(frozen=True)
class HistogramBin:
    label: str
    count: int

    def __post_init__(self) -> None:
        _require_text("label", self.label)
        _require_int("count", self.count, minimum=0)


@dataclass(frozen=True)
class EvalResult:
    name: str
    metric: str
    value: float
    examples: tuple[EvalExample, ...]
    histogram: tuple[HistogramBin, ...] = ()

    def __post_init__(self) -> None:
        if self.name not in EVAL_NAMES:
            raise ValueError(f"unknown eval name {self.name!r}")
        _require_text("metric", self.metric)
        value = _require_float("value", self.value, minimum=0.0)
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        if not self.examples:
            raise ValueError("examples must not be empty")
        if self.name == "perplexity" and self.value <= 0.0:
            raise ValueError("perplexity must be > 0")


class Predictor(Protocol):
    def predict(self, instruction: str) -> str: ...


JudgeFn: TypeAlias = "Callable[[str, str, str], JudgeVerdict]"
LossBins: tuple[tuple[str, float], ...] = (
    ("[0,0.5)", 0.5),
    ("[0.5,1)", 1.0),
    ("[1,2)", 2.0),
    ("[2,4)", 4.0),
    ("[4,8)", 8.0),
    ("[8,inf)", math.inf),
)


def _validate_examples(
    examples: Sequence[QaExample], name: str
) -> tuple[QaExample, ...]:
    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
        raise TypeError(f"{name} examples must be a sequence")
    values = tuple(examples)
    if not values:
        raise ValueError(f"{name} examples must not be empty")
    if not all(isinstance(example, QaExample) for example in values):
        raise TypeError(f"{name} examples must contain QaExample values")
    return values


def _loss_bin(value: float) -> str:
    for label, upper in LossBins:
        if value < upper:
            return label
    return LossBins[-1][0]


def _predict_text(predictor: Predictor, instruction: str) -> str:
    prediction = predictor.predict(instruction)
    if not isinstance(prediction, str):
        raise TypeError("predictor must return a string")
    return prediction


def perplexity_eval(
    model: nn.Module,
    tokenizer: InstructionTokenizer,
    corpus: Sequence[str],
) -> EvalResult:
    """Compute document-level held-out perplexity with pad masking."""
    if not isinstance(tokenizer, InstructionTokenizer):
        raise TypeError("tokenizer must be an InstructionTokenizer")
    if isinstance(corpus, (str, bytes)) or not isinstance(corpus, Sequence):
        raise TypeError("corpus must be a sequence of strings")
    documents = tuple(corpus)
    if not documents:
        raise ValueError("corpus must not be empty")

    encoded: list[list[int]] = []
    for document in documents:
        text = _require_text("document", document)
        ids = list(text.encode("utf-8"))
        if len(ids) < 2:
            raise ValueError("each document must contain at least two bytes")
        if len(ids) > tokenizer.max_len:
            raise ValueError(
                f"document length {len(ids)} exceeds max_len {tokenizer.max_len}"
            )
        encoded.append(ids)

    max_length = max(len(ids) for ids in encoded)
    input_ids = torch.full(
        (len(encoded), max_length),
        PAD_ID,
        dtype=torch.long,
    )
    for row, ids in enumerate(encoded):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            logits = model(input_ids)
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            targets = input_ids[:, 1:]
            mask = targets != PAD_ID
            token_nll = -log_probs.gather(
                2, targets.unsqueeze(-1)
            ).squeeze(-1)
            nll_values = token_nll[mask]
    finally:
        model.train(was_training)

    token_count = int(mask.sum().item())
    if token_count == 0:
        raise ValueError("no scoreable tokens in the corpus")
    total_nll = float(nll_values.sum().item())
    if not math.isfinite(total_nll):
        raise ValueError("model produced non-finite negative log-likelihood")
    perplexity = math.exp(total_nll / token_count)
    if not math.isfinite(perplexity) or perplexity <= 0.0:
        raise ValueError("computed perplexity is not finite and positive")

    examples: list[PerplexityExampleResult] = []
    for row, (document, ids) in enumerate(zip(documents, encoded)):
        row_mask = mask[row]
        row_count = int(row_mask.sum().item())
        if row_count != len(ids) - 1:
            raise ValueError("internal mask accounting mismatch")
        row_nll = float(token_nll[row][row_mask].sum().item())
        mean_nll = row_nll / row_count
        examples.append(
            PerplexityExampleResult(
                document=document,
                tokens=row_count,
                mean_nll=mean_nll,
                perplexity=math.exp(mean_nll),
            )
        )

    counts = Counter(_loss_bin(float(value)) for value in nll_values.tolist())
    histogram = tuple(
        HistogramBin(label=label, count=counts[label])
        for label, _ in LossBins
    )
    return EvalResult(
        name="perplexity",
        metric="perplexity",
        value=perplexity,
        examples=tuple(examples),
        histogram=histogram,
    )


def exact_match_eval(
    predictor: Predictor, examples: Sequence[QaExample]
) -> EvalResult:
    """Mean exact match after answer normalisation."""
    values = _validate_examples(examples, "exact match")
    records: list[TextExampleResult] = []
    total = 0.0
    for example in values:
        prediction = _predict_text(predictor, example.instruction)
        matched = exact_match(prediction, example.reference)
        score = 1.0 if matched else 0.0
        total += score
        records.append(
            TextExampleResult(
                instruction=example.instruction,
                reference=example.reference,
                prediction=prediction,
                score=score,
                normalised_prediction=normalise_text(prediction),
                normalised_reference=normalise_text(example.reference),
                rationale="match" if matched else "mismatch",
            )
        )
    return EvalResult(
        name="exact_match",
        metric="accuracy",
        value=total / len(values),
        examples=tuple(records),
    )


def token_f1_eval(
    predictor: Predictor, examples: Sequence[QaExample]
) -> EvalResult:
    """Mean token-level F1 across open-form answers."""
    values = _validate_examples(examples, "token F1")
    records: list[TextExampleResult] = []
    total = 0.0
    for example in values:
        prediction = _predict_text(predictor, example.instruction)
        score = token_f1(prediction, example.reference)
        total += score
        records.append(
            TextExampleResult(
                instruction=example.instruction,
                reference=example.reference,
                prediction=prediction,
                score=score,
                normalised_prediction=normalise_text(prediction),
                normalised_reference=normalise_text(example.reference),
                rationale=f"token F1 {score:.3f}",
            )
        )
    return EvalResult(
        name="token_f1",
        metric="mean_token_f1",
        value=total / len(values),
        examples=tuple(records),
    )


def judge_eval(
    predictor: Predictor,
    examples: Sequence[QaExample],
    *,
    judge: JudgeFn = mock_judge,
) -> EvalResult:
    """Mean 1-5 judge score across model predictions."""
    values = _validate_examples(examples, "judge")
    if not callable(judge):
        raise TypeError("judge must be callable")
    records: list[TextExampleResult] = []
    total = 0.0
    for example in values:
        prediction = _predict_text(predictor, example.instruction)
        verdict = judge(
            example.instruction, prediction, example.reference
        )
        if not isinstance(verdict, JudgeVerdict):
            raise TypeError("judge must return a JudgeVerdict")
        total += verdict.score
        records.append(
            TextExampleResult(
                instruction=example.instruction,
                reference=example.reference,
                prediction=prediction,
                score=float(verdict.score),
                normalised_prediction=normalise_text(prediction),
                normalised_reference=normalise_text(example.reference),
                rationale=verdict.rationale,
            )
        )
    return EvalResult(
        name="judge",
        metric="mean_judge_score",
        value=total / len(values),
        examples=tuple(records),
    )


def generate_greedy(
    model: nn.Module,
    tokenizer: InstructionTokenizer,
    instruction: str,
    max_new_tokens: int,
) -> str:
    """Generate a greedy answer until a newline or a non-byte special."""
    if not isinstance(tokenizer, InstructionTokenizer):
        raise TypeError("tokenizer must be an InstructionTokenizer")
    limit = _require_int("max_new_tokens", max_new_tokens, minimum=1)
    prefix = tokenizer.encode_prefix(instruction)
    if len(prefix) + limit > tokenizer.max_len:
        raise ValueError(
            f"prefix length {len(prefix)} plus max_new_tokens {limit} exceeds "
            f"max_len {tokenizer.max_len}"
        )

    ids = list(prefix)
    was_training = model.training
    model.eval()
    generated: list[int] = []
    try:
        with torch.no_grad():
            for _ in range(limit):
                token_ids = torch.tensor([ids], dtype=torch.long)
                logits = model(token_ids)
                next_id = int(torch.argmax(logits[0, -1]).item())
                if next_id == NEWLINE_ID or next_id == PAD_ID or next_id >= 256:
                    break
                generated.append(next_id)
                ids.append(next_id)
    finally:
        model.train(was_training)
    return tokenizer.decode(generated).strip()


@dataclass(frozen=True)
class TinyGptPredictor:
    model: nn.Module
    tokenizer: InstructionTokenizer
    max_new_tokens: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer, InstructionTokenizer):
            raise TypeError("tokenizer must be an InstructionTokenizer")
        _require_int(
            "max_new_tokens", self.max_new_tokens, minimum=1
        )

    def predict(self, instruction: str) -> str:
        return generate_greedy(
            self.model,
            self.tokenizer,
            instruction,
            self.max_new_tokens,
        )


@dataclass(frozen=True)
class EvalDatasets:
    lm_corpus: tuple[str, ...]
    em: tuple[QaExample, ...]
    f1: tuple[QaExample, ...]
    judge: tuple[QaExample, ...]

    def __post_init__(self) -> None:
        if not self.lm_corpus:
            raise ValueError("lm_corpus must not be empty")
        _validate_examples(self.em, "EM")
        _validate_examples(self.f1, "F1")
        _validate_examples(self.judge, "judge")


class EvalSuite:
    """Thin orchestrator that runs the selected evals in a stable order."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: InstructionTokenizer,
        datasets: EvalDatasets,
        *,
        max_new_tokens: int = 32,
        enabled: Sequence[str] = EVAL_NAMES,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        if not isinstance(tokenizer, InstructionTokenizer):
            raise TypeError("tokenizer must be an InstructionTokenizer")
        if not isinstance(datasets, EvalDatasets):
            raise TypeError("datasets must be an EvalDatasets")
        _require_int("max_new_tokens", max_new_tokens, minimum=1)
        names = tuple(enabled)
        if not names:
            raise ValueError("enabled must not be empty")
        if len(set(names)) != len(names):
            raise ValueError("enabled must not contain duplicates")
        unknown = [name for name in names if name not in EVAL_NAMES]
        if unknown:
            raise ValueError(f"unknown eval names: {unknown}")
        self.model = model
        self.tokenizer = tokenizer
        self.datasets = datasets
        self.max_new_tokens = max_new_tokens
        self.enabled = names

    def run(self) -> tuple[EvalResult, ...]:
        predictor = TinyGptPredictor(
            self.model, self.tokenizer, self.max_new_tokens
        )
        results: list[EvalResult] = []
        for name in EVAL_NAMES:
            if name not in self.enabled:
                continue
            if name == "perplexity":
                results.append(
                    perplexity_eval(
                        self.model, self.tokenizer, self.datasets.lm_corpus
                    )
                )
            elif name == "exact_match":
                results.append(
                    exact_match_eval(predictor, self.datasets.em)
                )
            elif name == "token_f1":
                results.append(token_f1_eval(predictor, self.datasets.f1))
            else:
                results.append(judge_eval(predictor, self.datasets.judge))
        return tuple(results)


# ---------------------------------------------------------------------------
# Weighted aggregation and the JSON report
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "perplexity": 0.2,
    "exact_match": 0.3,
    "token_f1": 0.3,
    "judge": 0.2,
}


@dataclass(frozen=True)
class ScoredEval:
    result: EvalResult
    normalised: float
    weight: float
    contribution: float


@dataclass(frozen=True)
class FinalReport:
    aggregate: float
    evals: tuple[ScoredEval, ...]
    weights: Mapping[str, float]
    metadata: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        aggregate = _require_float("aggregate", self.aggregate, minimum=0.0)
        if aggregate > 1.0:
            raise ValueError("aggregate must be <= 1")
        if not self.evals:
            raise ValueError("evals must not be empty")
        for scored in self.evals:
            _require_float("normalised", scored.normalised, minimum=0.0)
            _require_float("weight", scored.weight, minimum=0.0)
            _require_float("contribution", scored.contribution, minimum=0.0)

    def to_json_dict(self) -> dict[str, JsonValue]:
        eval_payload: dict[str, JsonValue] = {}
        for scored in self.evals:
            result = scored.result
            examples: list[JsonValue] = [
                _example_to_json(example) for example in result.examples
            ]
            payload: dict[str, JsonValue] = {
                "metric": result.metric,
                "value": result.value,
                "normalised": scored.normalised,
                "weight": scored.weight,
                "contribution": scored.contribution,
                "num_examples": len(result.examples),
                "examples": examples,
            }
            if result.name == "perplexity":
                payload["scored_tokens"] = sum(
                    example.tokens
                    for example in result.examples
                    if isinstance(example, PerplexityExampleResult)
                )
                payload["histogram"] = [
                    {"bin": bin.label, "count": bin.count}
                    for bin in result.histogram
                ]
            eval_payload[result.name] = payload
        return {
            "schema_version": 1,
            "aggregate": self.aggregate,
            "weights": {
                name: weight
                for name, weight in self.weights.items()
                if name in EVAL_NAMES
            },
            "evals": eval_payload,
            "metadata": dict(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_json_dict(), indent=indent)


def _example_to_json(example: EvalExample) -> dict[str, JsonValue]:
    if isinstance(example, TextExampleResult):
        return {
            "kind": example.kind,
            "instruction": example.instruction,
            "reference": example.reference,
            "prediction": example.prediction,
            "score": example.score,
            "normalised_prediction": example.normalised_prediction,
            "normalised_reference": example.normalised_reference,
            "rationale": example.rationale,
        }
    return {
        "kind": example.kind,
        "document": example.document,
        "tokens": example.tokens,
        "mean_nll": example.mean_nll,
        "perplexity": example.perplexity,
    }


class Aggregator:
    """Normalise each eval to [0,1] and combine it with configurable weights."""

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        source = DEFAULT_WEIGHTS if weights is None else weights
        if not isinstance(source, Mapping):
            raise TypeError("weights must be a mapping")
        copied: dict[str, float] = {}
        for name, value in source.items():
            if name not in EVAL_NAMES:
                raise ValueError(f"unknown eval name {name!r}")
            weight = _require_float(f"weight[{name}]", value, minimum=0.0)
            copied[name] = weight
        if not copied:
            raise ValueError("weights must not be empty")
        if sum(copied.values()) <= 0.0:
            raise ValueError("weights must have a positive total")
        self._weights = copied

    @property
    def weights(self) -> Mapping[str, float]:
        return dict(self._weights)

    @staticmethod
    def normalise(name: str, value: float) -> float:
        if name not in EVAL_NAMES:
            raise ValueError(f"unknown eval name {name!r}")
        if (
            name == "perplexity"
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isinf(float(value))
            and value > 0
        ):
            return 0.0
        number = _require_float("value", value, minimum=0.0)
        if name == "perplexity":
            if number < 1.0:
                raise ValueError("perplexity must be >= 1")
            return 1.0 / (1.0 + math.log(number))
        if name == "judge":
            if number < 1.0 or number > 5.0:
                raise ValueError("judge score must be in [1, 5]")
            return number / 5.0
        if number > 1.0:
            raise ValueError(f"{name} value must be in [0, 1]")
        return number

    def aggregate(
        self,
        results: Sequence[EvalResult],
        *,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> FinalReport:
        if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
            raise TypeError("results must be a sequence")
        values = tuple(results)
        if not values:
            raise ValueError("results must not be empty")
        for result in values:
            if not isinstance(result, EvalResult):
                raise TypeError("results must contain EvalResult values")
        names = [result.name for result in values]
        if len(set(names)) != len(names):
            raise ValueError("result names must be unique")
        if set(names) != set(self._weights):
            raise ValueError(
                "result names must match the configured weight names"
            )

        total_weight = sum(self._weights.values())
        scored: list[ScoredEval] = []
        aggregate = 0.0
        for result in values:
            normalised = self.normalise(result.name, result.value)
            weight = self._weights[result.name]
            contribution = weight * normalised / total_weight
            aggregate += contribution
            scored.append(
                ScoredEval(
                    result=result,
                    normalised=normalised,
                    weight=weight,
                    contribution=contribution,
                )
            )
        if aggregate < 0.0 or aggregate > 1.0 + 1e-9:
            raise ValueError("aggregate must be in [0, 1]")
        metadata_value = {} if metadata is None else dict(metadata)
        return FinalReport(
            aggregate=min(aggregate, 1.0),
            evals=tuple(scored),
            weights=dict(self._weights),
            metadata=metadata_value,
        )


# ---------------------------------------------------------------------------
# Training and the offline demo
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    lm_steps: int = 80
    lm_batch_size: int = 8
    lm_learning_rate: float = 1e-3
    sft_epochs: int = 60
    sft_batch_size: int = 8
    sft_learning_rate: float = 1e-3
    hidden: int = 96
    num_heads: int = 4
    num_layers: int = 2
    max_new_tokens: int = 32

    def __post_init__(self) -> None:
        _require_seed(self.seed)
        _require_int("lm_steps", self.lm_steps, minimum=1)
        _require_int("lm_batch_size", self.lm_batch_size, minimum=1)
        _require_float(
            "lm_learning_rate", self.lm_learning_rate, minimum=0.0
        )
        if self.lm_learning_rate == 0.0:
            raise ValueError("lm_learning_rate must be > 0")
        _require_int("sft_epochs", self.sft_epochs, minimum=1)
        _require_int("sft_batch_size", self.sft_batch_size, minimum=1)
        _require_float(
            "sft_learning_rate", self.sft_learning_rate, minimum=0.0
        )
        if self.sft_learning_rate == 0.0:
            raise ValueError("sft_learning_rate must be > 0")
        hidden = _require_int("hidden", self.hidden, minimum=1)
        heads = _require_int("num_heads", self.num_heads, minimum=1)
        _require_int("num_layers", self.num_layers, minimum=1)
        _require_int("max_new_tokens", self.max_new_tokens, minimum=1)
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by num_heads")
        if self.max_new_tokens >= MAX_SEQUENCE_LENGTH:
            raise ValueError("max_new_tokens must fit in the context")


@dataclass(frozen=True)
class TrainedModel:
    model: TinyGPT
    tokenizer: InstructionTokenizer
    parameters: int
    lm_losses: tuple[float, ...]
    sft_losses: tuple[float, ...]
    config: TrainConfig


def _lm_batches(
    corpus: Sequence[str],
    tokenizer: InstructionTokenizer,
    config: TrainConfig,
) -> list[tuple[Tensor, Tensor]]:
    context = min(64, tokenizer.max_len - 1)
    stream: list[int] = []
    for document in corpus:
        text = _require_text("document", document)
        if "\n" in text or "\r" in text:
            raise ValueError("LM documents must be single-line")
        stream.extend(text.encode("utf-8"))
        stream.append(NEWLINE_ID)
    if len(stream) <= context:
        raise ValueError("LM corpus is too short for one training window")

    windows: list[tuple[Tensor, Tensor]] = []
    for start in range(0, len(stream) - context, context):
        inputs = torch.tensor(stream[start : start + context], dtype=torch.long)
        targets = torch.tensor(
            stream[start + 1 : start + context + 1], dtype=torch.long
        )
        windows.append((inputs, targets))
    batches: list[tuple[Tensor, Tensor]] = []
    for start in range(0, len(windows), config.lm_batch_size):
        group = windows[start : start + config.lm_batch_size]
        batches.append(
            (
                torch.stack([item[0] for item in group]),
                torch.stack([item[1] for item in group]),
            )
        )
    return batches


def pretrain_lm(
    model: TinyGPT,
    tokenizer: InstructionTokenizer,
    corpus: Sequence[str],
    config: TrainConfig,
    log: Callable[[str], None] | None = None,
) -> tuple[float, ...]:
    """Train the model as a causal LM on the held-in synthetic corpus."""
    batches = _lm_batches(corpus, tokenizer, config)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lm_learning_rate
    )
    losses: list[float] = []
    model.train()
    for step in range(1, config.lm_steps + 1):
        inputs, targets = batches[(step - 1) % len(batches)]
        logits = model(inputs)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        optimizer.step()
        value = float(loss.item())
        losses.append(value)
        if log is not None and (step == 1 or step % 20 == 0):
            log(f"lm step {step:3d}: loss={value:.4f}")
    return tuple(losses)


def sft_train(
    model: TinyGPT,
    tokenizer: InstructionTokenizer,
    pairs: Sequence[QaExample],
    config: TrainConfig,
    log: Callable[[str], None] | None = None,
) -> tuple[float, ...]:
    """Fit instruction/answer pairs with completion-only loss."""
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, Sequence):
        raise TypeError("pairs must be a sequence")
    values = tuple(pairs)
    if not values:
        raise ValueError("pairs must not be empty")
    if not all(isinstance(pair, QaExample) for pair in values):
        raise TypeError("pairs must contain QaExample values")

    sequences: list[list[int]] = []
    starts: list[int] = []
    for pair in values:
        ids, completion_start = tokenizer.encode_example(
            pair.instruction, pair.reference
        )
        sequences.append(ids)
        starts.append(completion_start)
    max_length = max(len(ids) for ids in sequences)
    input_ids = torch.full(
        (len(sequences), max_length), PAD_ID, dtype=torch.long
    )
    labels = torch.full(
        (len(sequences), max_length), IGNORE_INDEX, dtype=torch.long
    )
    for row, (ids, completion_start) in enumerate(zip(sequences, starts)):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[row, completion_start : len(ids)] = torch.tensor(
            ids[completion_start:], dtype=torch.long
        )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.sft_learning_rate
    )
    losses: list[float] = []
    model.train()
    for epoch in range(1, config.sft_epochs + 1):
        epoch_loss = 0.0
        batch_count = 0
        for start in range(0, len(sequences), config.sft_batch_size):
            batch_inputs = input_ids[start : start + config.sft_batch_size]
            batch_labels = labels[start : start + config.sft_batch_size]
            logits = model(batch_inputs)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, VOCAB_SIZE),
                batch_labels[:, 1:].reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
            optimizer.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            optimizer.step()
            epoch_loss += float(loss.item())
            batch_count += 1
        if batch_count == 0:
            raise ValueError("SFT training produced no batches")
        average = epoch_loss / batch_count
        losses.append(average)
        if log is not None and (epoch == 1 or epoch % 10 == 0):
            log(f"sft epoch {epoch:3d}: loss={average:.4f}")
    return tuple(losses)


def train_model(config: TrainConfig) -> TrainedModel:
    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    tokenizer = InstructionTokenizer()
    model = TinyGPT(
        ModelConfig(
            hidden=config.hidden,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
        ),
        seed=config.seed,
    )
    lm_losses = pretrain_lm(
        model,
        tokenizer,
        make_lm_training_corpus(),
        config,
        log=print,
    )
    sft_losses = sft_train(
        model,
        tokenizer,
        make_training_pairs(),
        config,
        log=print,
    )
    model.eval()
    return TrainedModel(
        model=model,
        tokenizer=tokenizer,
        parameters=count_parameters(model),
        lm_losses=lm_losses,
        sft_losses=sft_losses,
        config=config,
    )


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 0
    lm_steps: int = 80
    lm_batch_size: int = 8
    lm_learning_rate: float = 1e-3
    sft_epochs: int = 60
    sft_batch_size: int = 8
    sft_learning_rate: float = 1e-3
    hidden: int = 96
    num_heads: int = 4
    num_layers: int = 2
    max_new_tokens: int = 32
    output_path: Path = Path("outputs/report.json")

    def __post_init__(self) -> None:
        self.to_train_config()
        if not isinstance(self.output_path, Path):
            raise TypeError("output_path must be a Path")

    def to_train_config(self) -> TrainConfig:
        return TrainConfig(
            seed=self.seed,
            lm_steps=self.lm_steps,
            lm_batch_size=self.lm_batch_size,
            lm_learning_rate=self.lm_learning_rate,
            sft_epochs=self.sft_epochs,
            sft_batch_size=self.sft_batch_size,
            sft_learning_rate=self.sft_learning_rate,
            hidden=self.hidden,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_new_tokens=self.max_new_tokens,
        )


@dataclass(frozen=True)
class DemoReport:
    final: FinalReport
    results: tuple[EvalResult, ...]
    training: TrainedModel
    output_path: Path
    success: bool


def _print_report(report: FinalReport) -> None:
    print(
        f"{'eval':<14}{'metric':<18}{'value':>10}{'normalised':>12}"
        f"{'weight':>9}{'contribution':>14}",
        flush=True,
    )
    for scored in report.evals:
        result = scored.result
        print(
            f"{result.name:<14}{result.metric:<18}{result.value:>10.4f}"
            f"{scored.normalised:>12.4f}{scored.weight:>9.3f}"
            f"{scored.contribution:>14.4f}",
            flush=True,
        )
    print(f"aggregate: {report.aggregate:.4f}", flush=True)


def _print_diagnostics(results: Sequence[EvalResult]) -> None:
    for result in results:
        if result.name == "perplexity":
            perplexity_examples = [
                example
                for example in result.examples
                if isinstance(example, PerplexityExampleResult)
            ]
            if perplexity_examples:
                worst_document = max(
                    perplexity_examples,
                    key=lambda example: example.perplexity,
                )
                print(
                    f"sample: perplexity "
                    f"worst={worst_document.perplexity:.2f} "
                    f"tokens={worst_document.tokens} "
                    f"text={worst_document.document!r}",
                    flush=True,
                )
            continue
        text_examples = [
            example
            for example in result.examples
            if isinstance(example, TextExampleResult)
        ]
        if not text_examples:
            continue
        worst_text = min(text_examples, key=lambda example: example.score)
        print(
            f"sample: {result.name} score={worst_text.score:.3f} "
            f"instruction={worst_text.instruction!r} "
            f"gold={worst_text.reference!r} "
            f"prediction={worst_text.prediction!r}",
            flush=True,
        )


def _read_report(path: Path) -> Mapping[str, object]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("report JSON must be an object")
    return parsed


def run_demo(config: DemoConfig | None = None) -> DemoReport:
    """Train, evaluate, print the report table, and write the JSON report."""
    if config is None:
        config = DemoConfig()
    if not isinstance(config, DemoConfig):
        raise TypeError("config must be a DemoConfig")

    datasets = EvalDatasets(
        lm_corpus=make_lm_fixture(),
        em=make_em_fixture(),
        f1=make_f1_fixture(),
        judge=make_judge_fixture(),
    )
    training = train_model(config.to_train_config())
    print(
        f"eval-pipeline: seed={config.seed} "
        f"train_pairs={len(make_training_pairs())} "
        f"lm_train={len(make_lm_training_corpus())} "
        f"eval_lm={len(datasets.lm_corpus)} "
        f"eval_em={len(datasets.em)} eval_f1={len(datasets.f1)} "
        f"eval_judge={len(datasets.judge)}",
        flush=True,
    )
    print(
        f"model: parameters={training.parameters} hidden={config.hidden} "
        f"heads={config.num_heads} layers={config.num_layers} "
        f"context={training.tokenizer.max_len}",
        flush=True,
    )
    print(
        f"training: lm_steps={config.lm_steps} "
        f"lm_loss={training.lm_losses[0]:.4f}->{training.lm_losses[-1]:.4f} "
        f"sft_epochs={config.sft_epochs} "
        f"sft_loss={training.sft_losses[0]:.4f}->{training.sft_losses[-1]:.4f}",
        flush=True,
    )

    suite = EvalSuite(
        training.model,
        training.tokenizer,
        datasets,
        max_new_tokens=config.max_new_tokens,
    )
    results = suite.run()
    metadata: dict[str, JsonValue] = {
        "model": {
            "parameters": training.parameters,
            "hidden": config.hidden,
            "heads": config.num_heads,
            "layers": config.num_layers,
        },
        "tokenizer": {
            "kind": "byte",
            "vocab_size": VOCAB_SIZE,
            "context_length": training.tokenizer.max_len,
        },
        "training": {
            "seed": config.seed,
            "lm_steps": config.lm_steps,
            "sft_epochs": config.sft_epochs,
            "sft_batch_size": config.sft_batch_size,
            "pairs": len(make_training_pairs()),
            "lm_documents": len(make_lm_training_corpus()),
            "lm_first_loss": training.lm_losses[0],
            "lm_final_loss": training.lm_losses[-1],
            "sft_first_loss": training.sft_losses[0],
            "sft_final_loss": training.sft_losses[-1],
        },
        "fixtures": {
            "lm": len(datasets.lm_corpus),
            "exact_match": len(datasets.em),
            "token_f1": len(datasets.f1),
            "judge": len(datasets.judge),
        },
    }
    final = Aggregator().aggregate(results, metadata=metadata)
    _print_report(final)
    _print_diagnostics(results)

    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final.to_json() + "\n", encoding="utf-8")
    print(f"report: {output_path}", flush=True)

    success = (
        len(results) == len(EVAL_NAMES)
        and {result.name for result in results} == set(EVAL_NAMES)
        and all(math.isfinite(result.value) for result in results)
        and 0.0 <= final.aggregate <= 1.0
    )
    try:
        parsed = _read_report(output_path)
        evals = parsed.get("evals")
        success = (
            success
            and parsed.get("schema_version") == 1
            and isinstance(evals, dict)
            and set(evals) == set(EVAL_NAMES)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        success = False
    return DemoReport(
        final=final,
        results=results,
        training=training,
        output_path=output_path,
        success=success,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny GPT and run perplexity, exact-match, token-F1, "
            "and mock-judge evaluations."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lm-steps", type=int, default=80)
    parser.add_argument("--sft-epochs", type=int, default=60)
    parser.add_argument("--sft-batch-size", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/report.json")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        try:
            config = DemoConfig(
                seed=args.seed,
                lm_steps=args.lm_steps,
                sft_epochs=args.sft_epochs,
                sft_batch_size=args.sft_batch_size,
                hidden=args.hidden,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
                max_new_tokens=args.max_new_tokens,
                output_path=args.output,
            )
            report = run_demo(config)
        except (TypeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr, flush=True)
            return 2
        if not report.success:
            print("error: demo self-check failed", file=sys.stderr, flush=True)
            return 1
        return 0
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    raise SystemExit(main())
