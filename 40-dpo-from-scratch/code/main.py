"""Direct Preference Optimization from scratch on a tiny byte-level transformer.

The module trains a policy against twelve synthetic preference triples with a
frozen reference copy of an SFT-like warm-up model. It implements completion-only
log-probabilities, the DPO sigmoid loss, reference invariance, and a
deterministic CPU demo. No external services, datasets, or pretrained weights
are used.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Constants and small validation helpers
# ---------------------------------------------------------------------------

BYTE_VOCAB_SIZE = 256
INST_ID = 256
RESP_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259
IGNORE_INDEX = -100
MAX_SEQUENCE_LENGTH = 128
CATEGORY_ORDER = ("capital", "arithmetic", "list")


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
        """Encode ``[INST] instruction [RESP]``."""
        text = _require_text("instruction", instruction)
        ids = [INST_ID] + list(text.encode("utf-8")) + [RESP_ID]
        if len(ids) >= self._max_len:
            raise ValueError(
                f"instruction prefix length {len(ids)} must be < max_len "
                f"{self._max_len}"
            )
        return ids

    def encode_completion(
        self, instruction: str, completion: str
    ) -> tuple[list[int], int]:
        """Encode a prompt plus completion and return ``(ids, completion_start)``.

        No terminator is appended: the module scores exactly the supplied
        completion bytes and does not generate text.
        """
        completion_text = _require_text("completion", completion)
        prefix = self.encode_prefix(instruction)
        ids = prefix + list(completion_text.encode("utf-8"))
        if len(ids) > self._max_len:
            raise ValueError(
                f"prompt plus completion length {len(ids)} exceeds max_len "
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
# Deterministic twelve-triple preference fixture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preference:
    """One prompt with a chosen and a rejected completion."""

    prompt: str
    chosen: str
    rejected: str
    category: str

    def __post_init__(self) -> None:
        _require_text("prompt", self.prompt)
        _require_text("chosen", self.chosen)
        _require_text("rejected", self.rejected)
        if not isinstance(self.category, str):
            raise TypeError("category must be a string")
        if self.category not in CATEGORY_ORDER:
            raise ValueError(f"unknown category {self.category!r}")
        if self.chosen == self.rejected:
            raise ValueError("chosen and rejected must differ")


_PREFERENCE_ROWS: tuple[tuple[str, str, str, str], ...] = (
    (
        "capital",
        "What is the capital of France?",
        "Paris.",
        "France is a large country with many cities and museums, and its "
        "capital is Paris.",
    ),
    (
        "capital",
        "What is the capital of Spain?",
        "Madrid.",
        "Spain has many cities and a famous football league, so the answer is "
        "not short.",
    ),
    (
        "capital",
        "What is the capital of Italy?",
        "Rome.",
        "Italy has a long history, fashion, and food, and I would rather "
        "discuss those.",
    ),
    (
        "capital",
        "What is the capital of Japan?",
        "Tokyo.",
        "Japan is an island nation in East Asia with many temples and a rich "
        "culture.",
    ),
    (
        "arithmetic",
        "What is 2 + 3?",
        "5.",
        "Let me think: two plus three is a sum, and sums are useful in many "
        "situations.",
    ),
    (
        "arithmetic",
        "What is 7 * 6?",
        "42.",
        "Forty-two is the answer to everything, so the product might be 42 or "
        "maybe 46.",
    ),
    (
        "arithmetic",
        "What is 9 - 4?",
        "5.",
        "Subtraction takes one number away from another, and there are many "
        "ways to explain it.",
    ),
    (
        "arithmetic",
        "What is 12 / 3?",
        "4.",
        "Division is repeated subtraction; I could show a long derivation, but "
        "the result is 6.",
    ),
    (
        "list",
        "List the primary colors.",
        "red, blue, yellow.",
        "Colors are everywhere: red, green, blue, orange, purple, and many "
        "more, so the list is long.",
    ),
    (
        "list",
        "Name the planets in order.",
        "Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune.",
        "Space has stars, moons, asteroids, and planets, and the order is "
        "complicated.",
    ),
    (
        "list",
        "List the vowels.",
        "a, e, i, o, u.",
        "The alphabet contains consonants and vowels, and English spelling is "
        "complicated.",
    ),
    (
        "list",
        "List the days of the week.",
        "Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday.",
        "Time is divided into days, weeks, months, and years, and calendars "
        "vary.",
    ),
)


def make_preferences() -> tuple[Preference, ...]:
    """Return the fixed twelve preference triples."""
    return tuple(
        Preference(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
            category=category,
        )
        for category, prompt, chosen, rejected in _PREFERENCE_ROWS
    )


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
        if parameter.requires_grad or not trainable_only
    )


# ---------------------------------------------------------------------------
# Response-only warm-up
# ---------------------------------------------------------------------------


def warmup_pretrain(
    model: TinyGPT,
    tokenizer: InstructionTokenizer,
    preferences: Sequence[Preference],
    *,
    epochs: int = 60,
    lr: float = 1e-3,
    log: Callable[[str], None] | None = None,
) -> tuple[float, ...]:
    """Warm up a model on the chosen completions with response-only loss."""
    if not preferences:
        raise ValueError("preferences must not be empty")
    epochs_value = _require_int("epochs", epochs, minimum=1)
    rate = _require_float("lr", lr, minimum=0.0)
    if rate == 0.0:
        raise ValueError("lr must be > 0")

    device = next(model.parameters()).device
    sequences: list[list[int]] = []
    label_rows: list[list[int]] = []
    for preference in preferences:
        ids, completion_start = tokenizer.encode_completion(
            preference.prompt, preference.chosen
        )
        sequences.append(ids)
        label_rows.append(
            [IGNORE_INDEX] * completion_start + ids[completion_start:]
        )

    max_length = max(len(ids) for ids in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        PAD_ID,
        dtype=torch.long,
        device=device,
    )
    labels = torch.full(
        (len(label_rows), max_length),
        IGNORE_INDEX,
        dtype=torch.long,
        device=device,
    )
    for row, (ids, label_row) in enumerate(zip(sequences, label_rows)):
        input_ids[row, : len(ids)] = torch.tensor(
            ids, dtype=torch.long, device=device
        )
        labels[row, : len(label_row)] = torch.tensor(
            label_row, dtype=torch.long, device=device
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=rate)
    losses: list[float] = []
    model.train()
    for epoch in range(1, epochs_value + 1):
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB_SIZE),
            labels[:, 1:].reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        optimizer.step()
        value = float(loss.item())
        losses.append(value)
        if log is not None and (epoch == 1 or epoch % 10 == 0):
            log(f"warmup epoch {epoch:3d}: loss={value:.4f}")
    return tuple(losses)


# ---------------------------------------------------------------------------
# Completion log-probabilities and the DPO loss
# ---------------------------------------------------------------------------


def sequence_log_prob(
    model: TinyGPT,
    tokenizer: InstructionTokenizer,
    prompt: str,
    completion: str,
) -> Tensor:
    """Return the sum of completion-token log-probabilities under ``model``.

    The prompt and the ``RESP`` boundary contribute no direct terms. The
    ``RESP`` position predicts the first completion byte.
    """
    ids, completion_start = tokenizer.encode_completion(prompt, completion)
    device = next(model.parameters()).device
    token_ids = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(token_ids)
    log_probs = F.log_softmax(logits[0, :-1], dim=-1)
    targets = token_ids[0, completion_start:]
    rows = torch.arange(
        completion_start - 1,
        token_ids.shape[1] - 1,
        device=device,
    )
    return log_probs[rows].gather(1, targets.unsqueeze(1)).sum()


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    beta: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Return per-example DPO losses and implicit reward deltas.

    The implicit reward delta is
    ``beta * ((log pi_w - log pi_ref_w) - (log pi_l - log pi_ref_l))``,
    which is the argument to the Bradley-Terry sigmoid. The loss is
    ``-logsigmoid(delta)``.
    """
    tensors = (
        ("policy_chosen_logps", policy_chosen_logps),
        ("policy_rejected_logps", policy_rejected_logps),
        ("reference_chosen_logps", reference_chosen_logps),
        ("reference_rejected_logps", reference_rejected_logps),
    )
    for name, tensor in tensors:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if tensor.numel() == 0:
            raise ValueError(f"{name} must not be empty")
    shapes = {tuple(tensor.shape) for _, tensor in tensors}
    if len(shapes) != 1:
        raise ValueError("all four log-probability tensors must have equal shape")

    rate = _require_float("beta", beta, minimum=0.0)
    if rate == 0.0:
        raise ValueError("beta must be > 0")

    log_ratio_chosen = policy_chosen_logps - reference_chosen_logps
    log_ratio_rejected = policy_rejected_logps - reference_rejected_logps
    reward_delta = rate * (log_ratio_chosen - log_ratio_rejected)
    losses = -F.logsigmoid(reward_delta)
    return losses, reward_delta


# ---------------------------------------------------------------------------
# Reference/policy construction and DPO training
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferencePair:
    reference: TinyGPT
    policy: TinyGPT
    warmup_losses: tuple[float, ...]


def build_reference_and_policy(
    config: ModelConfig,
    tokenizer: InstructionTokenizer,
    preferences: Sequence[Preference],
    *,
    seed: int = 0,
    warmup_epochs: int = 60,
    warmup_lr: float = 1e-3,
) -> ReferencePair:
    """Warm a reference model, freeze it, and copy it into a trainable policy."""
    seed_value = _require_seed(seed)
    reference = TinyGPT(config, seed=seed_value)
    warmup_losses = warmup_pretrain(
        reference,
        tokenizer,
        preferences,
        epochs=warmup_epochs,
        lr=warmup_lr,
    )
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    policy = TinyGPT(config, seed=seed_value + 1)
    policy.load_state_dict(reference.state_dict())
    policy.train()
    return ReferencePair(
        reference=reference,
        policy=policy,
        warmup_losses=warmup_losses,
    )


@dataclass(frozen=True)
class DPOReport:
    losses: tuple[float, ...]
    policy_margins: tuple[float, ...]
    implicit_reward_margins: tuple[float, ...]


def train_dpo(
    policy: TinyGPT,
    reference: TinyGPT,
    tokenizer: InstructionTokenizer,
    preferences: Sequence[Preference],
    *,
    beta: float = 0.1,
    lr: float = 1e-5,
    steps: int = 30,
    log: Callable[[str], None] | None = None,
) -> DPOReport:
    """Train the policy on preference triples with the DPO loss."""
    if not preferences:
        raise ValueError("preferences must not be empty")
    steps_value = _require_int("steps", steps, minimum=1)
    rate = _require_float("lr", lr, minimum=0.0)
    if rate == 0.0:
        raise ValueError("lr must be > 0")
    beta_value = _require_float("beta", beta, minimum=0.0)
    if beta_value == 0.0:
        raise ValueError("beta must be > 0")
    if policy.config != reference.config:
        raise ValueError("policy and reference must share a ModelConfig")
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise ValueError("reference parameters must be frozen")

    reference.eval()
    with torch.no_grad():
        reference_chosen = torch.stack(
            [
                sequence_log_prob(
                    reference, tokenizer, preference.prompt, preference.chosen
                )
                for preference in preferences
            ]
        ).detach()
        reference_rejected = torch.stack(
            [
                sequence_log_prob(
                    reference, tokenizer, preference.prompt, preference.rejected
                )
                for preference in preferences
            ]
        ).detach()

    optimizer = torch.optim.Adam(policy.parameters(), lr=rate)
    policy.train()
    losses: list[float] = []
    policy_margins: list[float] = []
    implicit_reward_margins: list[float] = []
    for step in range(1, steps_value + 1):
        policy_chosen = torch.stack(
            [
                sequence_log_prob(
                    policy, tokenizer, preference.prompt, preference.chosen
                )
                for preference in preferences
            ]
        )
        policy_rejected = torch.stack(
            [
                sequence_log_prob(
                    policy, tokenizer, preference.prompt, preference.rejected
                )
                for preference in preferences
            ]
        )
        step_losses, reward_deltas = dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta_value,
        )
        loss = step_losses.mean()
        loss_value = float(loss.detach().item())
        margin_value = float(
            (policy_chosen.detach() - policy_rejected.detach()).mean().item()
        )
        reward_value = float(reward_deltas.detach().mean().item())

        optimizer.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        optimizer.step()

        losses.append(loss_value)
        policy_margins.append(margin_value)
        implicit_reward_margins.append(reward_value)
        if log is not None:
            log(
                f"step {step:3d}: loss={loss_value:.5f} "
                f"margin={margin_value:.4f} reward_delta={reward_value:.4f}"
            )

    policy.eval()
    return DPOReport(
        losses=tuple(losses),
        policy_margins=tuple(policy_margins),
        implicit_reward_margins=tuple(implicit_reward_margins),
    )


@torch.no_grad()
def evaluate_margins(
    model: TinyGPT,
    tokenizer: InstructionTokenizer,
    preferences: Sequence[Preference],
) -> float:
    """Return the mean chosen-minus-rejected log-probability margin."""
    if not preferences:
        raise ValueError("preferences must not be empty")
    was_training = model.training
    model.eval()
    try:
        chosen = torch.stack(
            [
                sequence_log_prob(
                    model, tokenizer, preference.prompt, preference.chosen
                )
                for preference in preferences
            ]
        )
        rejected = torch.stack(
            [
                sequence_log_prob(
                    model, tokenizer, preference.prompt, preference.rejected
                )
                for preference in preferences
            ]
        )
        return float((chosen - rejected).mean().item())
    finally:
        model.train(was_training)


@torch.no_grad()
def _mean_completion_logps(
    model: TinyGPT,
    tokenizer: InstructionTokenizer,
    preferences: Sequence[Preference],
) -> tuple[float, float]:
    if not preferences:
        raise ValueError("preferences must not be empty")
    chosen = torch.stack(
        [
            sequence_log_prob(
                model, tokenizer, preference.prompt, preference.chosen
            )
            for preference in preferences
        ]
    )
    rejected = torch.stack(
        [
            sequence_log_prob(
                model, tokenizer, preference.prompt, preference.rejected
            )
            for preference in preferences
        ]
    )
    return float(chosen.mean().item()), float(rejected.mean().item())


# ---------------------------------------------------------------------------
# Demo and CLI
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 0
    warmup_epochs: int = 60
    warmup_lr: float = 1e-3
    steps: int = 30
    beta: float = 0.1
    lr: float = 1e-5
    hidden: int = 96
    num_heads: int = 4
    num_layers: int = 2

    def __post_init__(self) -> None:
        _require_seed(self.seed)
        _require_int("warmup_epochs", self.warmup_epochs, minimum=1)
        warmup_rate = _require_float(
            "warmup_lr", self.warmup_lr, minimum=0.0
        )
        _require_int("steps", self.steps, minimum=1)
        beta = _require_float("beta", self.beta, minimum=0.0)
        rate = _require_float("lr", self.lr, minimum=0.0)
        hidden = _require_int("hidden", self.hidden, minimum=1)
        heads = _require_int("num_heads", self.num_heads, minimum=1)
        _require_int("num_layers", self.num_layers, minimum=1)
        if hidden % heads != 0:
            raise ValueError("hidden must be divisible by num_heads")
        if warmup_rate == 0.0:
            raise ValueError("warmup_lr must be > 0")
        if beta == 0.0:
            raise ValueError("beta must be > 0")
        if rate == 0.0:
            raise ValueError("lr must be > 0")


@dataclass(frozen=True)
class DemoReport:
    config: DemoConfig
    pair: ReferencePair
    dpo: DPOReport
    warmup_chosen_mean: float
    warmup_rejected_mean: float
    initial_margin: float
    final_margin: float
    policy_chosen_mean: float
    policy_rejected_mean: float
    reference_unchanged: bool
    loss_decreased: bool
    margin_increased: bool
    success: bool
    parameters: int


def run_demo(config: DemoConfig | None = None) -> DemoReport:
    """Build, warm up, train, and print the deterministic DPO demo."""
    if config is None:
        config = DemoConfig()
    if not isinstance(config, DemoConfig):
        raise TypeError("config must be a DemoConfig")

    tokenizer = InstructionTokenizer(MAX_SEQUENCE_LENGTH)
    preferences = make_preferences()
    model_config = ModelConfig(
        hidden=config.hidden,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )
    pair = build_reference_and_policy(
        model_config,
        tokenizer,
        preferences,
        seed=config.seed,
        warmup_epochs=config.warmup_epochs,
        warmup_lr=config.warmup_lr,
    )
    parameters = count_parameters(pair.policy)

    initial_margin = evaluate_margins(pair.policy, tokenizer, preferences)
    warmup_chosen_mean, warmup_rejected_mean = _mean_completion_logps(
        pair.reference, tokenizer, preferences
    )
    category_counts = {
        category: sum(
            1 for preference in preferences if preference.category == category
        )
        for category in CATEGORY_ORDER
    }
    print(
        f"dpo-from-scratch: seed={config.seed} pairs={len(preferences)} "
        f"max_len={MAX_SEQUENCE_LENGTH}"
    )
    print(
        "data: "
        + " ".join(
            f"{category}={category_counts[category]}"
            for category in CATEGORY_ORDER
        )
    )
    print(
        f"model: parameters={parameters} hidden={config.hidden} "
        f"heads={config.num_heads} layers={config.num_layers}"
    )
    print(
        f"warmup: epochs={config.warmup_epochs} "
        f"loss={pair.warmup_losses[0]:.4f} -> {pair.warmup_losses[-1]:.4f}"
    )
    print(
        f"reference: chosen_mean={warmup_chosen_mean:.4f} "
        f"rejected_mean={warmup_rejected_mean:.4f} "
        f"margin={warmup_chosen_mean - warmup_rejected_mean:.4f}"
    )
    print(
        f"dpo: beta={config.beta:.4f} lr={config.lr:g} "
        f"steps={config.steps}"
    )

    dpo_report = train_dpo(
        pair.policy,
        pair.reference,
        tokenizer,
        preferences,
        beta=config.beta,
        lr=config.lr,
        steps=config.steps,
        log=print,
    )
    final_margin = evaluate_margins(pair.policy, tokenizer, preferences)
    reference_chosen_after, reference_rejected_after = _mean_completion_logps(
        pair.reference, tokenizer, preferences
    )
    policy_chosen_mean, policy_rejected_mean = _mean_completion_logps(
        pair.policy, tokenizer, preferences
    )
    reference_unchanged = math.isclose(
        warmup_chosen_mean, reference_chosen_after, rel_tol=0.0, abs_tol=1e-6
    ) and math.isclose(
        warmup_rejected_mean,
        reference_rejected_after,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    loss_decreased = dpo_report.losses[-1] < dpo_report.losses[0]
    margin_increased = final_margin > initial_margin
    finite = all(
        math.isfinite(value)
        for value in (
            *pair.warmup_losses,
            *dpo_report.losses,
            *dpo_report.policy_margins,
            *dpo_report.implicit_reward_margins,
            warmup_chosen_mean,
            warmup_rejected_mean,
            initial_margin,
            final_margin,
            policy_chosen_mean,
            policy_rejected_mean,
        )
    )
    success = reference_unchanged and loss_decreased and margin_increased and finite
    print(
        f"final: policy_chosen={policy_chosen_mean:.4f} "
        f"policy_rejected={policy_rejected_mean:.4f} "
        f"margin={final_margin:.4f}"
    )
    print(
        f"check: ref_unchanged={str(reference_unchanged).lower()} "
        f"loss_decreased={str(loss_decreased).lower()} "
        f"margin_increased={str(margin_increased).lower()}"
    )
    return DemoReport(
        config=config,
        pair=pair,
        dpo=dpo_report,
        warmup_chosen_mean=warmup_chosen_mean,
        warmup_rejected_mean=warmup_rejected_mean,
        initial_margin=initial_margin,
        final_margin=final_margin,
        policy_chosen_mean=policy_chosen_mean,
        policy_rejected_mean=policy_rejected_mean,
        reference_unchanged=reference_unchanged,
        loss_decreased=loss_decreased,
        margin_increased=margin_increased,
        success=success,
        parameters=parameters,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Direct Preference Optimization from scratch."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-epochs", type=int, default=60)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    args = parser.parse_args(argv)

    # Keep the printed CLI measurements reproducible across CPU thread counts.
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        try:
            config = DemoConfig(
                seed=args.seed,
                warmup_epochs=args.warmup_epochs,
                steps=args.steps,
                beta=args.beta,
                lr=args.lr,
                hidden=args.hidden,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
            )
            report = run_demo(config)
        except (TypeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        if not report.success:
            print("error: demo self-check failed", file=sys.stderr)
            return 1
        return 0
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    raise SystemExit(main())
