"""Token and positional embedding tables and their composition for transformer inputs."""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _require_positive_float(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


def _check_seq_len(seq_len: int, max_context_length: int) -> None:
    if isinstance(seq_len, bool) or not isinstance(seq_len, int):
        raise TypeError("seq_len must be an int")
    if seq_len < 1:
        raise ValueError("seq_len must be >= 1")
    if seq_len > max_context_length:
        raise ValueError(
            f"seq_len {seq_len} exceeds max_context_length {max_context_length}"
        )


def _init_normal(weight: Tensor, std: float, seed: int | None) -> None:
    """Fill a weight tensor from N(0, std) using a local generator when seeded."""
    if seed is None:
        nn.init.normal_(weight, mean=0.0, std=std)
        return
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an int or None")
    generator = torch.Generator(device=weight.device).manual_seed(seed)
    nn.init.normal_(weight, mean=0.0, std=std, generator=generator)


class TokenEmbedding(nn.Module):
    """Lookup table mapping vocabulary ids to dense vectors of shape (B, T, D)."""

    vocab_size: int
    dim: int
    embedding: nn.Embedding

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        *,
        init_std: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = _require_positive_int("vocab_size", vocab_size)
        self.dim = _require_positive_int("dim", dim)
        init_std = _require_positive_float("init_std", init_std)
        self.embedding = nn.Embedding(self.vocab_size, self.dim)
        _init_normal(self.embedding.weight, init_std, seed)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError(
                f"token_ids must have shape (B, T), got {tuple(token_ids.shape)}"
            )
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must be an int32 or int64 tensor")
        embedded: Tensor = self.embedding(token_ids)
        return embedded


class LearnedPositionalEmbedding(nn.Module):
    """Lookup table with one learned D-vector per position in 0..L-1."""

    max_context_length: int
    dim: int
    embedding: nn.Embedding

    def __init__(
        self,
        max_context_length: int,
        dim: int,
        *,
        init_std: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.max_context_length = _require_positive_int(
            "max_context_length", max_context_length
        )
        self.dim = _require_positive_int("dim", dim)
        init_std = _require_positive_float("init_std", init_std)
        self.embedding = nn.Embedding(self.max_context_length, self.dim)
        _init_normal(self.embedding.weight, init_std, seed)

    def forward(self, seq_len: int) -> Tensor:
        _check_seq_len(seq_len, self.max_context_length)
        positions = torch.arange(
            seq_len, dtype=torch.long, device=self.embedding.weight.device
        )
        embedded: Tensor = self.embedding(positions)
        return embedded.unsqueeze(0)


def _build_sinusoidal_table(max_context_length: int, dim: int) -> Tensor:
    """The fixed sin/cos table indexed by position and feature."""
    positions = torch.arange(max_context_length, dtype=torch.float64)
    pair_index = torch.arange((dim + 1) // 2, dtype=torch.float64)
    angles = positions[:, None] / (10000.0 ** (2.0 * pair_index / dim))
    table = torch.zeros((max_context_length, dim), dtype=torch.float64)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles[:, : dim // 2])
    return table.to(torch.float32)


class SinusoidalPositionalEmbedding(nn.Module):
    """Fixed sinusoidal table with one D-vector per position in 0..L-1."""

    max_context_length: int
    dim: int
    table: Tensor

    def __init__(self, max_context_length: int, dim: int) -> None:
        super().__init__()
        self.max_context_length = _require_positive_int(
            "max_context_length", max_context_length
        )
        self.dim = _require_positive_int("dim", dim)
        self.register_buffer(
            "table",
            _build_sinusoidal_table(self.max_context_length, self.dim),
            persistent=False,
        )

    def forward(self, seq_len: int) -> Tensor:
        _check_seq_len(seq_len, self.max_context_length)
        return self.table[:seq_len].unsqueeze(0)

    def wavelengths(self) -> Tensor:
        """Wavelength in positions for each sin/cos pair, shortest first."""
        pair_count = (self.dim + 1) // 2
        pair_index = torch.arange(
            pair_count, dtype=self.table.dtype, device=self.table.device
        )
        return 2.0 * math.pi * torch.exp(
            pair_index * (2.0 / self.dim) * math.log(10000.0)
        )


class EmbeddingComposer(nn.Module):
    """Sum token and positional vectors into a (B, T, D) transformer input."""

    token_embedding: TokenEmbedding
    positional_embedding: LearnedPositionalEmbedding | SinusoidalPositionalEmbedding

    def __init__(
        self,
        token_embedding: TokenEmbedding,
        positional_embedding: LearnedPositionalEmbedding
        | SinusoidalPositionalEmbedding,
    ) -> None:
        super().__init__()
        if token_embedding.dim != positional_embedding.dim:
            raise ValueError(
                f"dim mismatch: token {token_embedding.dim}, "
                f"positional {positional_embedding.dim}"
            )
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding

    def forward(self, token_ids: Tensor) -> Tensor:
        tokens: Tensor = self.token_embedding(token_ids)
        positions: Tensor = self.positional_embedding(tokens.shape[1])
        return tokens + positions


def parameter_count(module: nn.Module) -> int:
    """Number of scalar parameters in the module tree."""
    return sum(parameter.numel() for parameter in module.parameters())


def neighbour_cosine(rows: Tensor, gap: int = 1) -> Tensor:
    """Cosine similarity between rows[p] and rows[p + gap] for all valid p."""
    if rows.ndim != 2:
        raise ValueError("rows must be two-dimensional")
    if isinstance(gap, bool) or not isinstance(gap, int):
        raise TypeError("gap must be an int")
    if gap < 1:
        raise ValueError("gap must be >= 1")
    if gap >= rows.shape[0]:
        raise ValueError(f"need more than {gap} rows, got {rows.shape[0]}")
    return F.cosine_similarity(rows[:-gap], rows[gap:], dim=-1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Token and positional embedding demo for transformer inputs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser(
        "demo", help="print shapes, parameter counts, and similarity diagnostics"
    )
    demo.add_argument("--vocab-size", type=int, default=64, help="vocabulary size (V)")
    demo.add_argument("--dim", type=int, default=64, help="model dimension (D)")
    demo.add_argument(
        "--context-length", type=int, default=16, help="tokens per sequence (T)"
    )
    demo.add_argument("--batch-size", type=int, default=4, help="batch size (B)")
    demo.add_argument("--seed", type=int, default=1234, help="base seed")
    return parser


def _run_demo(args: argparse.Namespace) -> int:
    try:
        vocab_size = _require_positive_int("vocab_size", args.vocab_size)
        dim = _require_positive_int("dim", args.dim)
        context_length = _require_positive_int("context_length", args.context_length)
        batch_size = _require_positive_int("batch_size", args.batch_size)
        seed = args.seed
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an int")
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    token = TokenEmbedding(vocab_size, dim, seed=seed)
    learned = LearnedPositionalEmbedding(context_length, dim, seed=seed + 1)
    sinusoidal_context = max(4 * context_length, context_length + 1)
    sinusoidal = SinusoidalPositionalEmbedding(sinusoidal_context, dim)
    learned_composer = EmbeddingComposer(token, learned)
    sinusoidal_composer = EmbeddingComposer(token, sinusoidal)

    generator = torch.Generator().manual_seed(seed + 2)
    token_ids = torch.randint(
        0, vocab_size, (batch_size, context_length), generator=generator
    )
    tokens = token(token_ids)
    learned_positions = learned(context_length)
    sinusoidal_positions = sinusoidal(context_length)
    learned_input = learned_composer(token_ids)
    sinusoidal_input = sinusoidal_composer(token_ids)

    print(f"token_ids: shape={tuple(token_ids.shape)} dtype={token_ids.dtype}")
    print(f"token_embeddings: shape={tuple(tokens.shape)}")
    print(f"learned_positions: shape={tuple(learned_positions.shape)}")
    print(f"sinusoidal_positions: shape={tuple(sinusoidal_positions.shape)}")
    print(
        f"composed_inputs: learned={tuple(learned_input.shape)} "
        f"sinusoidal={tuple(sinusoidal_input.shape)}"
    )

    token_params = parameter_count(token)
    learned_position_params = parameter_count(learned)
    sinusoidal_position_params = parameter_count(sinusoidal)
    learned_total = parameter_count(learned_composer)
    sinusoidal_total = parameter_count(sinusoidal_composer)
    overhead = (learned_total - sinusoidal_total) / sinusoidal_total * 100.0
    print("parameter_counts:")
    print(f"  token={token_params}")
    print(f"  learned_positional={learned_position_params}")
    print(f"  sinusoidal_positional={sinusoidal_position_params}")
    print(f"  learned_total={learned_total}")
    print(f"  sinusoidal_total={sinusoidal_total}")
    print(f"  overhead={overhead:.1f}%")

    if not torch.allclose(learned_input, tokens + learned_positions):
        raise AssertionError("learned composition is not token + positional")
    if not torch.allclose(sinusoidal_input, tokens + sinusoidal_positions):
        raise AssertionError("sinusoidal composition is not token + positional")
    print("composition_sum=ok")

    learned_delta = learned_input - tokens
    sinusoidal_delta = sinusoidal_input - tokens
    if not torch.allclose(learned_delta, learned_positions.expand_as(learned_delta)):
        raise AssertionError("learned positions did not broadcast across the batch")
    if not torch.allclose(sinusoidal_delta, sinusoidal_positions.expand_as(sinusoidal_delta)):
        raise AssertionError("sinusoidal positions did not broadcast across the batch")
    print("batch_broadcast=ok")

    learned_rows = learned.embedding.weight.detach()
    sinusoidal_rows = sinusoidal.table[:context_length]
    print("neighbour_cosine:")
    for gap in (1, 2, 4, 8):
        if gap >= context_length:
            continue
        sinusoidal_cosine = neighbour_cosine(sinusoidal_rows, gap)
        learned_cosine = neighbour_cosine(learned_rows, gap)
        print(
            f"  gap={gap} sinusoidal={float(sinusoidal_cosine.mean()):.4f} "
            f"learned={float(learned_cosine.mean()):.4f}"
        )
    sinusoidal_gap1 = neighbour_cosine(sinusoidal_rows, 1)
    learned_gap1 = neighbour_cosine(learned_rows, 1)
    print(
        f"  gap1_std: sinusoidal={float(sinusoidal_gap1.std()):.6f} "
        f"learned={float(learned_gap1.std()):.6f}"
    )

    try:
        learned(context_length + 1)
    except ValueError as exc:
        print(f"learned_positional: rejected at T+1 ({exc})")
    else:
        raise AssertionError("learned positional embedding accepted T+1")

    accepted = sinusoidal(context_length + 1)
    print(
        "sinusoidal_positional: accepted at T+1 "
        f"(shape={tuple(accepted.shape)})"
    )
    try:
        sinusoidal(4 * context_length + 1)
    except ValueError as exc:
        print(f"sinusoidal_positional: rejected at max+1 ({exc})")
    else:
        raise AssertionError("sinusoidal positional embedding accepted max+1")
    print(
        "length_note: table-level bound only; downstream model length "
        "generalization is not tested here"
    )

    wavelengths = sinusoidal.wavelengths()
    pair_count = (dim + 1) // 2
    bands = sorted({0, 1, 2, 4, 8, dim // 2 - 1})
    print(f"wavelength_bands (dim={dim}):")
    for band in bands:
        if 0 <= band < pair_count:
            print(f"  band={band} wavelength={float(wavelengths[band]):.6g}")

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "demo":
        parser.error(f"unknown command: {args.command}")
    return _run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
