"""Classifier fine-tuning by swapping the language-model head.

The module keeps a tiny pre-norm transformer body, pretrains it with a causal
language-model objective, then replaces the token-prediction head with a
two-class linear head on top of masked mean pooling. The same classifier is
trained once with a frozen body and once with the whole body trainable.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants and small validation helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 260
PAD_ID = 256
BYTE_VOCAB_SIZE = 256


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


def _require_seed(seed: object) -> int | None:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int or None")
    if seed < 0:
        raise ValueError("seed must be >= 0")
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


# ---------------------------------------------------------------------------
# Byte tokenizer
# ---------------------------------------------------------------------------


class ByteTokenizer:
    """UTF-8 byte tokenizer with a reserved padding id."""

    @property
    def vocab_size(self) -> int:
        return VOCAB_SIZE

    @property
    def pad_id(self) -> int:
        return PAD_ID

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        values = list(ids)
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("ids must contain ints")
            if value < 0 or value >= BYTE_VOCAB_SIZE:
                raise ValueError(f"id {value} is not a byte id")
        return bytes(values).decode("utf-8")

    def pad_batch(
        self, texts: Sequence[str], max_length: int
    ) -> tuple[Tensor, Tensor]:
        length = _require_int("max_length", max_length, minimum=1)
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not a string")
        if not texts:
            raise ValueError("texts must not be empty")

        id_rows: list[list[int]] = []
        mask_rows: list[list[bool]] = []
        for text in texts:
            encoded = self.encode(text)[:length]
            real = len(encoded)
            id_rows.append(encoded + [PAD_ID] * (length - real))
            mask_rows.append([True] * real + [False] * (length - real))
        return (
            torch.tensor(id_rows, dtype=torch.long),
            torch.tensor(mask_rows, dtype=torch.bool),
        )


# ---------------------------------------------------------------------------
# Deterministic SMS fixture and stratified split
# ---------------------------------------------------------------------------

_SPAM_TEMPLATES = (
    "win {a} {b}",
    "free {b} {n}",
    "claim {a} {b}",
    "urgent {b} {n}",
    "you won {a} {n}",
    "txt {w} {n}",
    "last {b} {a}",
    "{b} {n} now",
    "cash {a} {n}",
    "acct {n} {b}",
)

_HAM_TEMPLATES = (
    "lunch {t} {p}",
    "call {p} {t}",
    "see {p} {t}",
    "can you {v} {p}",
    "meet {p} {t}",
    "dinner {d} {p}",
    "on my {t} {p}",
    "thanks {p} {o}",
    "are you {adj} {p}",
    "got {o} {p}",
)

_SLOT_FILLERS: dict[str, tuple[str, ...]] = {
    "a": ("500", "100", "50", "250", "1000", "20", "75", "900"),
    "b": (
        "gift",
        "trip",
        "ring",
        "cash",
        "prize",
        "deal",
        "voucher",
        "phone",
    ),
    "n": ("555", "123", "900", "777", "000", "321", "888", "456"),
    "t": ("noon", "6pm", "8", "1", "5pm", "7", "later", "9"),
    "v": ("talk", "come", "call", "meet", "chat", "help", "stop", "wait"),
    "p": ("sam", "ana", "jo", "max", "kim", "lu", "ben", "eve"),
    "d": ("fri", "sat", "sun", "mon", "tue", "wed", "thu"),
    "adj": ("free", "busy", "ok", "home", "up", "done", "late"),
    "o": ("keys", "bag", "book", "mail", "coat", "phone", "note", "file"),
    "w": ("go", "yes", "now", "free", "win", "cash", "hi", "deal"),
}

_SLOT_PATTERN = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class SMSFixture:
    messages: tuple[str, ...]
    labels: tuple[int, ...]


@dataclass(frozen=True)
class DataSplit:
    train_texts: tuple[str, ...]
    train_labels: tuple[int, ...]
    test_texts: tuple[str, ...]
    test_labels: tuple[int, ...]


def _candidate_messages(templates: Sequence[str]) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()
    for template in templates:
        slots = _SLOT_PATTERN.findall(template)
        if len(slots) != 2:
            raise ValueError(f"template must have two slots: {template!r}")
        first, second = slots
        for left in _SLOT_FILLERS[first]:
            for right in _SLOT_FILLERS[second]:
                message = template.replace(
                    "{" + first + "}", left
                ).replace("{" + second + "}", right)
                size = len(message.encode("utf-8"))
                if 5 <= size <= 25 and message not in seen:
                    seen.add(message)
                    candidates.append(message)
    return candidates


def make_sms_fixture(count: int = 800, seed: int = 0) -> SMSFixture:
    total = _require_int("count", count, minimum=2)
    if total % 2 != 0:
        raise ValueError("count must be even")
    seed_value = _require_seed(seed)
    assert seed_value is not None
    per_class = total // 2
    rng = random.Random(seed_value)

    spam = _candidate_messages(_SPAM_TEMPLATES)
    ham = _candidate_messages(_HAM_TEMPLATES)
    if len(spam) < per_class or len(ham) < per_class:
        raise ValueError(
            f"count={total} needs {per_class} messages per class, but only "
            f"{len(spam)} spam and {len(ham)} ham candidates exist"
        )
    rng.shuffle(spam)
    rng.shuffle(ham)

    pairs = list(zip(spam[:per_class], [1] * per_class, strict=True))
    pairs.extend(zip(ham[:per_class], [0] * per_class, strict=True))
    rng.shuffle(pairs)
    messages = tuple(message for message, _ in pairs)
    labels = tuple(label for _, label in pairs)
    if len(set(messages)) != total:
        raise ValueError("fixture messages are not unique")
    return SMSFixture(messages=messages, labels=labels)


def stratified_split(
    messages: Sequence[str],
    labels: Sequence[int],
    train_fraction: float = 0.8,
    seed: int = 0,
) -> DataSplit:
    if len(messages) != len(labels):
        raise ValueError("messages and labels must have the same length")
    if not messages:
        raise ValueError("messages must not be empty")
    fraction = _require_float(
        "train_fraction", train_fraction, minimum=0.0
    )
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    seed_value = _require_seed(seed)
    assert seed_value is not None
    rng = random.Random(seed_value)

    by_class: dict[int, list[int]] = {0: [], 1: []}
    for index, label in enumerate(labels):
        if isinstance(label, bool) or label not in (0, 1):
            raise ValueError("labels must be 0 or 1")
        by_class[label].append(index)
    if not by_class[0] or not by_class[1]:
        raise ValueError("both classes must be present")

    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in (0, 1):
        indices = by_class[label]
        if len(indices) < 2:
            raise ValueError("each class needs at least two examples")
        rng.shuffle(indices)
        test_count = min(
            len(indices) - 1,
            max(1, round(len(indices) * (1.0 - fraction))),
        )
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    train_texts = tuple(messages[index] for index in train_indices)
    train_labels = tuple(labels[index] for index in train_indices)
    test_texts = tuple(messages[index] for index in test_indices)
    test_labels = tuple(labels[index] for index in test_indices)
    if set(train_texts) & set(test_texts):
        raise ValueError("train and test messages overlap")
    return DataSplit(
        train_texts=train_texts,
        train_labels=train_labels,
        test_texts=test_texts,
        test_labels=test_labels,
    )


class ClassificationDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Right-padded byte ids, an attention mask, and a binary label."""

    _ids: Tensor
    _mask: Tensor
    _labels: Tensor

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: ByteTokenizer,
        max_length: int,
    ) -> None:
        if len(texts) != len(labels):
            raise ValueError("texts and labels must have the same length")
        if not texts:
            raise ValueError("texts must not be empty")
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("texts must contain strings")
            if not text.strip():
                raise ValueError("texts must not be empty or blank")
        for label in labels:
            if isinstance(label, bool) or label not in (0, 1):
                raise ValueError("labels must be 0 or 1")
        self._ids, self._mask = tokenizer.pad_batch(texts, max_length)
        self._labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self._ids[index], self._mask[index], self._labels[index]


# ---------------------------------------------------------------------------
# Transformer body, pooling, and classifier head
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    context_length: int = 32
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        context = _require_int(
            "context_length", self.context_length, minimum=2
        )
        model_width = _require_int("d_model", self.d_model, minimum=1)
        heads = _require_int("num_heads", self.num_heads, minimum=1)
        layers = _require_int("num_layers", self.num_layers, minimum=1)
        dropout = _require_float("dropout", self.dropout, minimum=0.0)
        if model_width % heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if dropout >= 1.0:
            raise ValueError("dropout must be < 1")
        if context < 2 or layers < 1:
            raise ValueError("invalid model dimensions")


class Block(nn.Module):
    """Pre-norm causal transformer block with fused QKV attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = _require_int("d_model", d_model, minimum=1)
        self.num_heads = _require_int("num_heads", num_heads, minimum=1)
        self.context_length = _require_int(
            "context_length", context_length, minimum=1
        )
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.dropout = _require_float("dropout", dropout, minimum=0.0)
        if self.dropout >= 1.0:
            raise ValueError("dropout must be < 1")
        self.head_dim = self.d_model // self.num_heads

        self.ln1 = nn.LayerNorm(self.d_model)
        self.qkv = nn.Linear(self.d_model, 3 * self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)
        self.fc1 = nn.Linear(self.d_model, 4 * self.d_model)
        self.fc2 = nn.Linear(4 * self.d_model, self.d_model)
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.causal_mask: Tensor
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

    def forward(self, x: Tensor) -> Tensor:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a tensor")
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, T, D), got {tuple(x.shape)}")
        if not x.is_floating_point():
            raise TypeError("x must be floating point")
        if x.shape[-1] != self.d_model:
            raise ValueError(
                f"x last dimension {x.shape[-1]} does not match "
                f"d_model {self.d_model}"
            )
        batch, length, _ = x.shape
        if length < 1 or length > self.context_length:
            raise ValueError(
                f"sequence length {length} must be between 1 and "
                f"{self.context_length}"
            )

        query, key, value = self.qkv(self.ln1(x)).chunk(3, dim=-1)
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
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        attention = (weights @ value).transpose(1, 2).contiguous()
        attention = attention.view(batch, length, self.d_model)
        x = x + self.resid_dropout(self.out_proj(attention))

        feed_forward = self.fc2(F.gelu(self.fc1(self.ln2(x))))
        output: Tensor = x + self.resid_dropout(feed_forward)
        return output


class LMBody(nn.Module):
    """Token/position embeddings, transformer blocks, and a final norm."""

    _config: ModelConfig

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig")
        self._config = config
        seed_value = _require_seed(seed)
        generator = _make_generator(seed_value)

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE, config.d_model, padding_idx=PAD_ID
        )
        self.position_embedding = nn.Embedding(
            config.context_length, config.d_model
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    config.d_model,
                    config.num_heads,
                    config.context_length,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self._init_weights(generator)
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
        return self._config.context_length

    def forward(self, token_ids: Tensor) -> Tensor:
        if not isinstance(token_ids, Tensor):
            raise TypeError("token_ids must be a tensor")
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must have shape (B, T), got "
                f"{tuple(token_ids.shape)}"
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
        output: Tensor = self.final_norm(hidden)
        return output


class MeanPool(nn.Module):
    """Mask-weighted average of hidden states over the sequence axis."""

    def forward(self, hidden: Tensor, mask: Tensor) -> Tensor:
        if not isinstance(hidden, Tensor) or not isinstance(mask, Tensor):
            raise TypeError("hidden and mask must be tensors")
        if hidden.ndim != 3:
            raise ValueError(
                f"hidden must have shape (B, T, D), got {tuple(hidden.shape)}"
            )
        if not hidden.is_floating_point():
            raise TypeError("hidden must be floating point")
        if mask.ndim != 2 or tuple(mask.shape) != tuple(hidden.shape[:2]):
            raise ValueError(
                f"mask must have shape {tuple(hidden.shape[:2])}, got "
                f"{tuple(mask.shape)}"
            )
        if mask.dtype == torch.bool:
            weights = mask.to(hidden.dtype)
        elif mask.dtype in (
            torch.float16,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            if bool(torch.any((mask != 0) & (mask != 1))):
                raise ValueError("mask values must be 0 or 1")
            weights = mask.to(hidden.dtype)
        else:
            raise TypeError(f"unsupported mask dtype {mask.dtype}")

        counts = weights.sum(dim=1)
        if bool(torch.any(counts == 0)):
            raise ValueError("every sequence needs at least one real token")
        return (hidden * weights.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(
            -1
        )


class Classifier(nn.Module):
    """A shared LMBody, masked mean pool, and a two-class linear head."""

    def __init__(
        self,
        body: LMBody,
        *,
        num_classes: int = 2,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(body, LMBody):
            raise TypeError("body must be an LMBody")
        classes = _require_int("num_classes", num_classes, minimum=2)
        if classes != 2:
            raise ValueError("num_classes must be 2 for binary classification")
        self.body = body
        self.pool = MeanPool()
        self.head = nn.Linear(body.config.d_model, classes)
        _init_linear(self.head, 0.02, _make_generator(_require_seed(seed)))

    def forward(
        self, token_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        hidden = self.body(token_ids)
        mask = token_ids != PAD_ID if attention_mask is None else attention_mask
        logits: Tensor = self.head(self.pool(hidden, mask))
        return logits


def freeze_body(classifier: Classifier) -> int:
    if not isinstance(classifier, Classifier):
        raise TypeError("classifier must be a Classifier")
    total = 0
    for parameter in classifier.body.parameters():
        parameter.requires_grad_(False)
        total += parameter.numel()
    return total


def unfreeze_body(classifier: Classifier) -> int:
    if not isinstance(classifier, Classifier):
        raise TypeError("classifier must be a Classifier")
    total = 0
    for parameter in classifier.body.parameters():
        parameter.requires_grad_(True)
        total += parameter.numel()
    return total


def count_parameters(module: nn.Module, *, trainable_only: bool = False) -> int:
    if not isinstance(module, nn.Module):
        raise TypeError("module must be an nn.Module")
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


# ---------------------------------------------------------------------------
# Pretraining, shared training loop, and evaluation
# ---------------------------------------------------------------------------


def pretrain_quick(
    body: LMBody,
    tokenizer: ByteTokenizer,
    texts: Sequence[str],
    *,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 3e-3,
    seed: int = 0,
) -> tuple[float, ...]:
    """Train the body briefly with a discarded causal LM head."""
    if not isinstance(body, LMBody):
        raise TypeError("body must be an LMBody")
    if not isinstance(tokenizer, ByteTokenizer):
        raise TypeError("tokenizer must be a ByteTokenizer")
    epoch_count = _require_int("epochs", epochs, minimum=0)
    batch_size_value = _require_int("batch_size", batch_size, minimum=1)
    rate = _require_float("learning_rate", learning_rate, minimum=0.0)
    if rate == 0.0:
        raise ValueError("learning_rate must be > 0")
    seed_value = _require_seed(seed)
    assert seed_value is not None
    if not texts:
        raise ValueError("texts must not be empty")
    if any(len(tokenizer.encode(text)) < 2 for text in texts):
        raise ValueError("every text must encode to at least two bytes")
    if epoch_count == 0:
        return ()

    ids, _ = tokenizer.pad_batch(texts, body.config.context_length)
    lm_head = nn.Linear(body.config.d_model, VOCAB_SIZE)
    _init_linear(
        lm_head, 0.02, torch.Generator().manual_seed(seed_value)
    )
    optimizer = torch.optim.Adam(
        list(body.parameters()) + list(lm_head.parameters()), lr=rate
    )
    was_training = body.training
    losses: list[float] = []
    try:
        body.train()
        lm_head.train()
        for epoch in range(epoch_count):
            order = torch.randperm(
                ids.shape[0],
                generator=torch.Generator().manual_seed(seed_value + epoch),
            )
            epoch_loss = 0.0
            batches = 0
            for start in range(0, order.numel(), batch_size_value):
                batch = ids[order[start : start + batch_size_value]]
                inputs = batch[:, :-1]
                targets = batch[:, 1:]
                logits = lm_head(body(inputs))
                loss = F.cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    targets.reshape(-1),
                    ignore_index=PAD_ID,
                )
                optimizer.zero_grad(set_to_none=True)
                torch.autograd.backward(loss)
                optimizer.step()
                epoch_loss += float(loss.item())
                batches += 1
            losses.append(epoch_loss / max(1, batches))
    finally:
        body.train(was_training)
    return tuple(losses)


@dataclass(frozen=True)
class TrainingHistory:
    losses: tuple[float, ...]
    accuracies: tuple[float, ...]


def train_classifier(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> TrainingHistory:
    """One training loop shared by frozen-body and full fine-tuning regimes."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    epoch_count = _require_int("epochs", epochs, minimum=0)
    if epoch_count == 0:
        return TrainingHistory(losses=(), accuracies=())

    losses: list[float] = []
    accuracies: list[float] = []
    model.train()
    for _ in range(epoch_count):
        total_loss = 0.0
        correct = 0
        examples = 0
        for ids, mask, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits: Tensor = model(ids, mask)
            loss = F.cross_entropy(logits, labels)
            torch.autograd.backward(loss)
            optimizer.step()
            size = int(labels.shape[0])
            total_loss += float(loss.item()) * size
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            examples += size
        if examples == 0:
            raise ValueError("loader produced no batches")
        losses.append(total_loss / examples)
        accuracies.append(correct / examples)
    return TrainingHistory(losses=tuple(losses), accuracies=tuple(accuracies))


@dataclass(frozen=True)
class ConfusionMatrix:
    tn: int
    fp: int
    fn: int
    tp: int

    def as_rows(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return ((self.tn, self.fp), (self.fn, self.tp))


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    confusion: ConfusionMatrix


def evaluate(
    model: nn.Module,
    loader: Iterable[tuple[Tensor, Tensor, Tensor]],
) -> Metrics:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an nn.Module")
    was_training = model.training
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    try:
        with torch.no_grad():
            for ids, mask, labels in loader:
                logits: Tensor = model(ids, mask)
                if logits.ndim != 2 or logits.shape[1] != 2:
                    raise ValueError(
                        "logits must have shape (B, 2), got "
                        f"{tuple(logits.shape)}"
                    )
                if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
                    raise ValueError(
                        "labels must have shape (B,) matching logits"
                    )
                for value in labels.tolist():
                    if value not in (0, 1):
                        raise ValueError("labels must be 0 or 1")
                predictions.extend(
                    int(value) for value in logits.argmax(dim=-1).tolist()
                )
                targets.extend(int(value) for value in labels.tolist())
    finally:
        model.train(was_training)

    if not targets:
        raise ValueError("loader produced no samples")
    tn = fp = fn = tp = 0
    for target, prediction in zip(targets, predictions, strict=True):
        if prediction == 1 and target == 1:
            tp += 1
        elif prediction == 1 and target == 0:
            fp += 1
        elif prediction == 0 and target == 1:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=ConfusionMatrix(tn=tn, fp=fp, fn=fn, tp=tp),
    )


def format_metrics(name: str, metrics: Metrics) -> str:
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not isinstance(metrics, Metrics):
        raise TypeError("metrics must be Metrics")
    confusion = metrics.confusion
    return (
        f"{name}: precision={metrics.precision:.4f} "
        f"recall={metrics.recall:.4f} f1={metrics.f1:.4f}\n"
        f"{name} confusion (rows=true, cols=pred):\n"
        f"  ham  tn={confusion.tn} fp={confusion.fp}\n"
        f"  spam fn={confusion.fn} tp={confusion.tp}"
    )


# ---------------------------------------------------------------------------
# Demo and CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 0
    messages: int = 800
    train_fraction: float = 0.8
    context_length: int = 32
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    pretrain_epochs: int = 5
    head_epochs: int = 20
    full_epochs: int = 20
    batch_size: int = 32
    head_learning_rate: float = 1e-3
    full_learning_rate: float = 2e-4

    def __post_init__(self) -> None:
        _require_int("seed", self.seed, minimum=0)
        messages = _require_int("messages", self.messages, minimum=4)
        if messages % 2 != 0:
            raise ValueError("messages must be even")
        fraction = _require_float(
            "train_fraction", self.train_fraction, minimum=0.0
        )
        if fraction <= 0.0 or fraction >= 1.0:
            raise ValueError("train_fraction must be strictly between 0 and 1")
        _require_int("context_length", self.context_length, minimum=2)
        model_width = _require_int("d_model", self.d_model, minimum=1)
        heads = _require_int("num_heads", self.num_heads, minimum=1)
        _require_int("num_layers", self.num_layers, minimum=1)
        if model_width % heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        dropout = _require_float("dropout", self.dropout, minimum=0.0)
        if dropout >= 1.0:
            raise ValueError("dropout must be < 1")
        _require_int("pretrain_epochs", self.pretrain_epochs, minimum=0)
        _require_int("head_epochs", self.head_epochs, minimum=1)
        _require_int("full_epochs", self.full_epochs, minimum=1)
        _require_int("batch_size", self.batch_size, minimum=1)
        head_rate = _require_float(
            "head_learning_rate", self.head_learning_rate, minimum=0.0
        )
        full_rate = _require_float(
            "full_learning_rate", self.full_learning_rate, minimum=0.0
        )
        if head_rate == 0.0 or full_rate == 0.0:
            raise ValueError("learning rates must be > 0")


@dataclass(frozen=True)
class DemoResult:
    config: DemoConfig
    body_parameters: int
    head_parameters: int
    head_only_trainable: int
    full_trainable: int
    pretrain_losses: tuple[float, ...]
    head_history: TrainingHistory
    full_history: TrainingHistory
    head_metrics: Metrics
    full_metrics: Metrics
    body_unchanged_in_head_only: bool
    body_changed_in_full: bool
    pretrain_seconds: float
    head_seconds: float
    full_seconds: float


def run_demo(config: DemoConfig | None = None) -> DemoResult:
    if config is None:
        config = DemoConfig()
    if not isinstance(config, DemoConfig):
        raise TypeError("config must be a DemoConfig")
    torch.manual_seed(config.seed)

    fixture = make_sms_fixture(config.messages, config.seed)
    split = stratified_split(
        fixture.messages,
        fixture.labels,
        config.train_fraction,
        config.seed,
    )
    tokenizer = ByteTokenizer()
    train_dataset = ClassificationDataset(
        split.train_texts,
        split.train_labels,
        tokenizer,
        config.context_length,
    )
    test_dataset = ClassificationDataset(
        split.test_texts,
        split.test_labels,
        tokenizer,
        config.context_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed + 2),
    )
    full_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed + 3),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )

    model_config = ModelConfig(
        context_length=config.context_length,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    body = LMBody(model_config, seed=config.seed)

    lengths = [len(message.encode("utf-8")) for message in fixture.messages]
    spam_count = sum(fixture.labels)
    print(
        f"demo: seed={config.seed} messages={len(fixture.messages)} "
        f"train={len(split.train_texts)} test={len(split.test_texts)}"
    )
    print(
        f"fixture: spam={spam_count} ham={len(fixture.labels) - spam_count} "
        f"byte_lengths={min(lengths)}..{max(lengths)}"
    )
    print(
        f"split: train={len(split.train_texts)} "
        f"test={len(split.test_texts)} "
        f"train_spam={sum(split.train_labels)} "
        f"train_ham={len(split.train_labels) - sum(split.train_labels)} "
        f"test_spam={sum(split.test_labels)} "
        f"test_ham={len(split.test_labels) - sum(split.test_labels)}"
    )
    print(f"tokenizer: vocab={tokenizer.vocab_size} pad_id={tokenizer.pad_id}")
    print(
        f"model: body_parameters={count_parameters(body)} "
        f"head_parameters={config.d_model * 2 + 2} "
        f"context_length={config.context_length} d_model={config.d_model} "
        f"num_heads={config.num_heads} num_layers={config.num_layers}"
    )

    pretrain_seconds = 0.0
    if config.pretrain_epochs > 0:
        start = time.perf_counter()
        pretrain_losses = pretrain_quick(
            body,
            tokenizer,
            split.train_texts,
            epochs=config.pretrain_epochs,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        pretrain_seconds = time.perf_counter() - start
    else:
        pretrain_losses = ()
    shown_losses = ", ".join(f"{loss:.4f}" for loss in pretrain_losses)
    print(
        f"pretrain: epochs={config.pretrain_epochs} "
        f"losses=[{shown_losses}] seconds={pretrain_seconds:.2f}"
    )

    classifier = Classifier(body, seed=config.seed + 1)
    body_parameters = count_parameters(body)
    head_parameters = count_parameters(classifier.head)

    freeze_body(classifier)
    head_only_trainable = count_parameters(classifier, trainable_only=True)
    head_optimizer = torch.optim.Adam(
        [p for p in classifier.parameters() if p.requires_grad],
        lr=config.head_learning_rate,
    )
    body_before_head = tuple(
        parameter.detach().clone() for parameter in body.parameters()
    )
    print(
        f"regime: head-only trainable_parameters={head_only_trainable} "
        f"body_frozen=true epochs={config.head_epochs}"
    )
    head_start = time.perf_counter()
    head_history = train_classifier(
        classifier, train_loader, head_optimizer, config.head_epochs
    )
    head_seconds = time.perf_counter() - head_start
    head_metrics = evaluate(classifier, test_loader)
    body_unchanged = all(
        torch.equal(before, after)
        for before, after in zip(
            body_before_head, body.parameters(), strict=True
        )
    )
    print(
        f"train: head-only epochs={config.head_epochs} "
        f"final_loss={head_history.losses[-1]:.4f} "
        f"final_accuracy={head_history.accuracies[-1]:.4f}"
    )
    print(format_metrics("head-only", head_metrics))
    print(f"head-only seconds={head_seconds:.2f}")

    unfreeze_body(classifier)
    full_trainable = count_parameters(classifier, trainable_only=True)
    full_optimizer = torch.optim.Adam(
        [p for p in classifier.parameters() if p.requires_grad],
        lr=config.full_learning_rate,
    )
    body_before_full = tuple(
        parameter.detach().clone() for parameter in body.parameters()
    )
    print(
        f"regime: full trainable_parameters={full_trainable} "
        f"body_frozen=false epochs={config.full_epochs}"
    )
    full_start = time.perf_counter()
    full_history = train_classifier(
        classifier, full_loader, full_optimizer, config.full_epochs
    )
    full_seconds = time.perf_counter() - full_start
    full_metrics = evaluate(classifier, test_loader)
    body_changed = any(
        not torch.equal(before, after)
        for before, after in zip(
            body_before_full, body.parameters(), strict=True
        )
    )
    print(
        f"train: full epochs={config.full_epochs} "
        f"final_loss={full_history.losses[-1]:.4f} "
        f"final_accuracy={full_history.accuracies[-1]:.4f}"
    )
    print(format_metrics("full", full_metrics))
    print(f"full seconds={full_seconds:.2f}")

    ratio = full_seconds / head_seconds if head_seconds > 0.0 else 0.0
    print(
        f"check: body_unchanged_in_head_only={str(body_unchanged).lower()} "
        f"body_changed_in_full={str(body_changed).lower()}"
    )
    print(
        f"summary: head_only_trainable={head_only_trainable} "
        f"full_trainable={full_trainable} "
        f"head_seconds={head_seconds:.2f} "
        f"full_seconds={full_seconds:.2f} ratio={ratio:.2f}"
    )
    return DemoResult(
        config=config,
        body_parameters=body_parameters,
        head_parameters=head_parameters,
        head_only_trainable=head_only_trainable,
        full_trainable=full_trainable,
        pretrain_losses=pretrain_losses,
        head_history=head_history,
        full_history=full_history,
        head_metrics=head_metrics,
        full_metrics=full_metrics,
        body_unchanged_in_head_only=body_unchanged,
        body_changed_in_full=body_changed,
        pretrain_seconds=pretrain_seconds,
        head_seconds=head_seconds,
        full_seconds=full_seconds,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a two-class head on a frozen or full transformer body."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--messages", type=int, default=800)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--full-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--full-lr", type=float, default=2e-4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = DemoConfig(
            seed=args.seed,
            messages=args.messages,
            train_fraction=args.train_fraction,
            context_length=args.context_length,
            d_model=args.d_model,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
            pretrain_epochs=args.pretrain_epochs,
            head_epochs=args.head_epochs,
            full_epochs=args.full_epochs,
            batch_size=args.batch_size,
            head_learning_rate=args.head_lr,
            full_learning_rate=args.full_lr,
        )
        result = run_demo(config)
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0 if result.body_unchanged_in_head_only and result.body_changed_in_full else 1


if __name__ == "__main__":
    raise SystemExit(main())
