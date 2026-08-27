"""Transformer block primitives built from PyTorch tensor operations.

The module implements a hand-rolled LayerNorm, a fused-QKV causal multi-head
attention layer, a position-wise GELU feed-forward network, and a block that
can place the two normalizations before the sublayers (pre-LN) or after the
residual adds (post-LN). A small diagnostic stack and gradient probe make the
normalization-placement difference measurable without a training run.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from typing import Literal, NamedTuple, cast, overload

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _require_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


def _require_positive_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and > 0")
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


def _init_linear(
    linear: nn.Linear, std: float, generator: torch.Generator | None
) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=std, generator=generator)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def _validate_sequence_input(
    x: Tensor, d_model: int, max_context_length: int | None = None
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
    if max_context_length is not None and length > max_context_length:
        raise ValueError(
            f"sequence length {length} exceeds "
            f"max_context_length {max_context_length}"
        )
    return batch, length, dim


def parameter_count(module: nn.Module) -> int:
    """Number of scalar parameters in the module tree."""
    return sum(parameter.numel() for parameter in module.parameters())


class LayerNorm(nn.Module):
    """Layer normalization over the last dimension with learnable affine terms."""

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


class FeedForward(nn.Module):
    """Position-wise two-layer GELU network with dropout after the second linear."""

    d_model: int
    hidden_dim: int
    linear1: nn.Linear
    linear2: nn.Linear
    dropout: nn.Dropout

    def __init__(
        self,
        d_model: int,
        *,
        dropout: float = 0.0,
        init_std: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_positive_int("d_model", d_model)
        dropout_probability = _require_probability("dropout", dropout)
        init_std = _require_positive_float("init_std", init_std)
        seed_value = _require_seed(seed)
        self.hidden_dim = 4 * self.d_model

        self.linear1 = nn.Linear(self.d_model, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.d_model)
        self.dropout = nn.Dropout(dropout_probability)

        generator: torch.Generator | None = None
        if seed_value is not None:
            generator = torch.Generator(
                device=self.linear1.weight.device
            ).manual_seed(seed_value)
        _init_linear(self.linear1, init_std, generator)
        _init_linear(self.linear2, init_std, generator)

    def forward(self, x: Tensor) -> Tensor:
        _validate_sequence_input(x, self.d_model)
        output: Tensor = self.linear2(self.dropout(F.gelu(self.linear1(x))))
        return output


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention with one fused QKV projection."""

    d_model: int
    num_heads: int
    head_dim: int
    max_context_length: int
    qkv_projection: nn.Linear
    output_projection: nn.Linear
    attention_dropout: nn.Dropout
    causal_mask: Tensor

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_context_length: int,
        *,
        dropout: float = 0.0,
        init_std: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_positive_int("d_model", d_model)
        self.num_heads = _require_positive_int("num_heads", num_heads)
        self.max_context_length = _require_positive_int(
            "max_context_length", max_context_length
        )
        dropout_probability = _require_probability("dropout", dropout)
        init_std = _require_positive_float("init_std", init_std)
        seed_value = _require_seed(seed)
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads: "
                f"{self.d_model} % {self.num_heads} != 0"
            )
        self.head_dim = self.d_model // self.num_heads

        self.qkv_projection = nn.Linear(self.d_model, 3 * self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.d_model)
        self.attention_dropout = nn.Dropout(dropout_probability)

        generator: torch.Generator | None = None
        if seed_value is not None:
            generator = torch.Generator(
                device=self.qkv_projection.weight.device
            ).manual_seed(seed_value)
        _init_linear(self.qkv_projection, init_std, generator)
        _init_linear(self.output_projection, init_std, generator)

        # True means "blocked"; the table is derived state, not checkpoint state.
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
            batch, length, self.num_heads, self.head_dim
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
        batch, length, _ = _validate_sequence_input(
            x, self.d_model, self.max_context_length
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

        if return_weights:
            return output, weights
        return output


def _seed_offset(seed: int | None, offset: int) -> int | None:
    if seed is None:
        return None
    return seed + offset


class TransformerBlock(nn.Module):
    """One decoder block with the LayerNorms placed pre- or post-residual."""

    d_model: int
    max_context_length: int
    pre_ln: bool
    norm1: LayerNorm
    attention: MultiHeadAttention
    norm2: LayerNorm
    feed_forward: FeedForward

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_context_length: int,
        *,
        pre_ln: bool = True,
        dropout: float = 0.0,
        init_std: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_positive_int("d_model", d_model)
        self.max_context_length = _require_positive_int(
            "max_context_length", max_context_length
        )
        if not isinstance(pre_ln, bool):
            raise TypeError("pre_ln must be a bool")
        dropout_probability = _require_probability("dropout", dropout)
        init_std = _require_positive_float("init_std", init_std)
        seed_value = _require_seed(seed)
        self.pre_ln = pre_ln

        self.norm1 = LayerNorm(self.d_model)
        self.attention = MultiHeadAttention(
            self.d_model,
            num_heads,
            self.max_context_length,
            dropout=dropout_probability,
            init_std=init_std,
            seed=seed_value,
        )
        self.norm2 = LayerNorm(self.d_model)
        self.feed_forward = FeedForward(
            self.d_model,
            dropout=dropout_probability,
            init_std=init_std,
            seed=_seed_offset(seed_value, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        _validate_sequence_input(x, self.d_model, self.max_context_length)
        if self.pre_ln:
            x = x + self.attention(self.norm1(x))
            x = x + self.feed_forward(self.norm2(x))
        else:
            x = self.norm1(x + self.attention(x))
            x = self.norm2(x + self.feed_forward(x))
        return x


class _DiagnosticStack(nn.Module):
    """Embedding plus blocks, with no terminal norm, used by the gradient probe."""

    d_model: int
    max_context_length: int
    vocabulary_size: int
    pre_ln: bool
    embedding: nn.Embedding
    blocks: nn.ModuleList

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_context_length: int,
        vocabulary_size: int,
        *,
        pre_ln: bool,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_positive_int("d_model", d_model)
        num_layers = _require_positive_int("num_layers", num_layers)
        self.max_context_length = _require_positive_int(
            "max_context_length", max_context_length
        )
        self.vocabulary_size = _require_positive_int(
            "vocabulary_size", vocabulary_size
        )
        if not isinstance(pre_ln, bool):
            raise TypeError("pre_ln must be a bool")
        seed_value = _require_seed(seed)
        self.pre_ln = pre_ln

        self.embedding = nn.Embedding(self.vocabulary_size, self.d_model)
        generator: torch.Generator | None = None
        if seed_value is not None:
            generator = torch.Generator().manual_seed(seed_value)
        nn.init.normal_(
            self.embedding.weight,
            mean=0.0,
            std=0.02,
            generator=generator,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.d_model,
                    num_heads,
                    self.max_context_length,
                    pre_ln=pre_ln,
                    dropout=0.0,
                    seed=_seed_offset(seed_value, 1 + layer_index),
                )
                for layer_index in range(num_layers)
            ]
        )

    def forward(self, ids: Tensor) -> Tensor:
        if ids.ndim != 2:
            raise ValueError(f"ids must have shape (B, T), got {tuple(ids.shape)}")
        if ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("ids must be an int32 or int64 tensor")
        length = ids.shape[1]
        if length < 1:
            raise ValueError("sequence length must be >= 1")
        if length > self.max_context_length:
            raise ValueError(
                f"sequence length {length} exceeds "
                f"max_context_length {self.max_context_length}"
            )
        hidden: Tensor = self.embedding(ids)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class GradientProbe(NamedTuple):
    pre_output_shape: tuple[int, ...]
    post_output_shape: tuple[int, ...]
    pre_probe_loss: float
    post_probe_loss: float
    pre_embedding_grad_norm: float
    post_embedding_grad_norm: float
    embedding_grad_ratio: float
    pre_block_grad_norms: tuple[float, ...]
    post_block_grad_norms: tuple[float, ...]
    max_output_difference: float
    causal_future_weight_max: float
    causal_row_sum_max_error: float
    pre_parameter_count: int
    post_parameter_count: int


def _require_finite(tensor: Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise AssertionError(f"{name} contains non-finite values")


def _embedding_grad_norm(stack: _DiagnosticStack) -> float:
    gradient = stack.embedding.weight.grad
    if gradient is None:
        raise AssertionError("embedding gradient is missing")
    _require_finite(gradient, "embedding gradient")
    norm = float(gradient.norm())
    if norm <= 0:
        raise AssertionError("embedding gradient norm must be positive")
    return norm


def _block_grad_norm(block: nn.Module) -> float:
    total = 0.0
    for parameter in block.parameters():
        gradient = parameter.grad
        if gradient is None:
            raise AssertionError("a block parameter did not receive a gradient")
        _require_finite(gradient, "block parameter gradient")
        total += float(gradient.norm()) ** 2
    norm = math.sqrt(total)
    if not math.isfinite(norm):
        raise AssertionError("block gradient norm is not finite")
    return norm


def _probe_stack(
    stack: _DiagnosticStack, ids: Tensor
) -> tuple[Tensor, float, float, tuple[float, ...]]:
    stack.eval()
    stack.zero_grad(set_to_none=True)
    output = stack(ids)
    _require_finite(output, "stack output")
    loss = output.square().mean()
    loss.backward()
    embedding_norm = _embedding_grad_norm(stack)
    block_norms = tuple(_block_grad_norm(block) for block in stack.blocks)
    return output, float(loss.detach()), embedding_norm, block_norms


def gradient_probe(
    *,
    d_model: int,
    num_heads: int,
    num_layers: int,
    max_context_length: int,
    batch_size: int,
    vocabulary_size: int,
    seed: int = 1234,
) -> GradientProbe:
    """Compare pre-LN and post-LN at initialization with one backward pass."""
    d_model = _require_positive_int("d_model", d_model)
    num_heads = _require_positive_int("num_heads", num_heads)
    num_layers = _require_positive_int("num_layers", num_layers)
    max_context_length = _require_positive_int(
        "max_context_length", max_context_length
    )
    batch_size = _require_positive_int("batch_size", batch_size)
    vocabulary_size = _require_positive_int("vocabulary_size", vocabulary_size)
    seed_value = _require_seed(seed)
    if seed_value is None:
        raise TypeError("seed must be an int")

    if num_layers < 2:
        raise ValueError(
            "num_layers must be >= 2 for the pre/post gradient comparison"
        )
    if max_context_length < 2:
        raise ValueError("max_context_length must be >= 2")
    if vocabulary_size < 2:
        raise ValueError("vocabulary_size must be >= 2")
    if d_model % num_heads != 0:
        raise ValueError(
            "d_model must be divisible by num_heads: "
            f"{d_model} % {num_heads} != 0"
        )
    if num_heads > d_model:
        raise ValueError("num_heads must be <= d_model")

    pre = _DiagnosticStack(
        d_model,
        num_heads,
        num_layers,
        max_context_length,
        vocabulary_size,
        pre_ln=True,
        seed=seed_value,
    )
    post = _DiagnosticStack(
        d_model,
        num_heads,
        num_layers,
        max_context_length,
        vocabulary_size,
        pre_ln=False,
        seed=seed_value,
    )
    post.load_state_dict(pre.state_dict(), strict=True)

    ids = torch.randint(
        0,
        vocabulary_size,
        (batch_size, max_context_length),
        generator=torch.Generator().manual_seed(seed_value + 7),
    )
    pre_output, pre_loss, pre_norm, pre_block_norms = _probe_stack(pre, ids)
    post_output, post_loss, post_norm, post_block_norms = _probe_stack(post, ids)

    if pre_output.shape != post_output.shape:
        raise AssertionError("pre-LN and post-LN output shapes differ")
    max_output_difference = float(
        (pre_output.detach() - post_output.detach()).abs().max()
    )
    if max_output_difference == 0.0:
        raise AssertionError("pre-LN and post-LN produced identical outputs")

    first_block = cast(TransformerBlock, pre.blocks[0])
    with torch.no_grad():
        attention_input = first_block.norm1(pre.embedding(ids))
        _, weights = first_block.attention(attention_input, return_weights=True)
    causal_future_weight_max = float(
        torch.triu(weights, diagonal=1).abs().max()
    )
    causal_row_sum_max_error = float(
        (weights.sum(dim=-1) - 1.0).abs().max()
    )
    if causal_future_weight_max != 0.0:
        raise AssertionError("causal mask leaked future attention weight")
    if causal_row_sum_max_error > 1e-5:
        raise AssertionError("attention rows do not sum to one")

    return GradientProbe(
        pre_output_shape=tuple(pre_output.shape),
        post_output_shape=tuple(post_output.shape),
        pre_probe_loss=pre_loss,
        post_probe_loss=post_loss,
        pre_embedding_grad_norm=pre_norm,
        post_embedding_grad_norm=post_norm,
        embedding_grad_ratio=pre_norm / post_norm,
        pre_block_grad_norms=pre_block_norms,
        post_block_grad_norms=post_block_norms,
        max_output_difference=max_output_difference,
        causal_future_weight_max=causal_future_weight_max,
        causal_row_sum_max_error=causal_row_sum_max_error,
        pre_parameter_count=parameter_count(pre),
        post_parameter_count=parameter_count(post),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-LN vs post-LN transformer block gradient probe."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="demo",
        choices=("demo",),
        help="command to run (default: demo)",
    )
    parser.add_argument("--d-model", type=int, default=128, help="model dimension")
    parser.add_argument("--heads", type=int, default=4, help="attention heads")
    parser.add_argument("--layers", type=int, default=6, help="blocks per stack")
    parser.add_argument(
        "--context-length", type=int, default=16, help="tokens per sequence"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="sequences per probe"
    )
    parser.add_argument(
        "--vocab-size", type=int, default=256, help="synthetic vocabulary size"
    )
    parser.add_argument("--seed", type=int, default=1234, help="base seed")
    return parser


def _run_demo(args: argparse.Namespace) -> int:
    try:
        d_model = _require_positive_int("d_model", args.d_model)
        num_heads = _require_positive_int("num_heads", args.heads)
        num_layers = _require_positive_int("num_layers", args.layers)
        context_length = _require_positive_int(
            "max_context_length", args.context_length
        )
        batch_size = _require_positive_int("batch_size", args.batch_size)
        vocabulary_size = _require_positive_int(
            "vocabulary_size", args.vocab_size
        )
        seed_value = _require_seed(args.seed)
        if seed_value is None:
            raise TypeError("seed must be an int")
        if num_layers < 2:
            raise ValueError(
                "num_layers must be >= 2 for the pre/post gradient comparison"
            )
        if context_length < 2:
            raise ValueError("max_context_length must be >= 2")
        if vocabulary_size < 2:
            raise ValueError("vocabulary_size must be >= 2")
        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads: "
                f"{d_model} % {num_heads} != 0"
            )
        if num_heads > d_model:
            raise ValueError("num_heads must be <= d_model")
        probe = gradient_probe(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            max_context_length=context_length,
            batch_size=batch_size,
            vocabulary_size=vocabulary_size,
            seed=seed_value,
        )
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("transformer block gradient probe")
    print(
        f"config: layers={num_layers} d_model={d_model} "
        f"num_heads={num_heads} head_dim={d_model // num_heads} "
        f"max_context_length={context_length} batch_size={batch_size} "
        f"vocabulary_size={vocabulary_size} seed={seed_value}"
    )
    print(
        "probe: loss=output.square().mean(), dropout=0.0, no final norm, "
        "identical token ids and copied weights"
    )
    print(
        f"pre_ln: output_shape={probe.pre_output_shape} "
        f"embedding_grad_norm={probe.pre_embedding_grad_norm:.6e} "
        f"probe_loss={probe.pre_probe_loss:.6e}"
    )
    print(
        f"post_ln: output_shape={probe.post_output_shape} "
        f"embedding_grad_norm={probe.post_embedding_grad_norm:.6e} "
        f"probe_loss={probe.post_probe_loss:.6e}"
    )
    print(f"embedding_grad_ratio={probe.embedding_grad_ratio:.6e}")
    print(f"max_output_difference={probe.max_output_difference:.6e}")
    pre_block_text = " ".join(
        f"block_{index}={norm:.6e}"
        for index, norm in enumerate(probe.pre_block_grad_norms)
    )
    post_block_text = " ".join(
        f"block_{index}={norm:.6e}"
        for index, norm in enumerate(probe.post_block_grad_norms)
    )
    print(f"pre_ln block_grad_norms: {pre_block_text}")
    print(f"post_ln block_grad_norms: {post_block_text}")
    print(
        f"causal_check: future_weight_max={probe.causal_future_weight_max:.1f} "
        f"row_sum_max_error={probe.causal_row_sum_max_error:.2e}"
    )
    print(
        f"parameter_counts: pre_ln={probe.pre_parameter_count} "
        f"post_ln={probe.post_parameter_count}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "demo":
        parser.error(f"unknown command: {args.command}")
    return _run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())



