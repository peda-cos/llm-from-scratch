"""Causal multi-head self-attention implemented with explicit tensor operations.

The block follows the shape contract used by decoder-only transformers:
``(B, T, D)`` in, ``(B, T, D)`` out, with ``H`` parallel heads of width
``D // H`` inside. One fused projection produces Q, K, and V; the causal mask,
softmax, and weighted sum are written out rather than delegated to a fused
kernel so the mechanism is inspectable.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Sequence
from typing import Literal, NamedTuple, overload

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
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return converted


def _init_linear(
    linear: nn.Linear, std: float, generator: torch.Generator | None
) -> None:
    nn.init.normal_(linear.weight, mean=0.0, std=std, generator=generator)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def parameter_count(module: nn.Module) -> int:
    """Number of scalar parameters in the module tree."""
    return sum(parameter.numel() for parameter in module.parameters())


def attention_entropy(weights: Tensor) -> Tensor:
    """Mean entropy of each head's attention rows, shape ``(H,)``.

    ``weights`` has shape ``(B, H, Q, S)``. Exact zeros contribute zero
    entropy; the clamp only keeps ``log(0)`` out of the computation.
    """
    if weights.ndim != 4:
        raise ValueError("weights must have shape (B, H, Q, S)")
    if not weights.is_floating_point():
        raise TypeError("weights must be a floating-point tensor")
    if weights.shape[-1] < 1:
        raise ValueError("weights last dimension must be >= 1")
    safe = weights.clamp_min(torch.finfo(weights.dtype).tiny)
    return -(weights * safe.log()).sum(dim=-1).mean(dim=(0, 2))


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with a fused QKV projection."""

    d_model: int
    n_heads: int
    d_head: int
    max_context_length: int
    qkv_projection: nn.Linear
    output_projection: nn.Linear
    causal_mask: Tensor

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_context_length: int,
        *,
        init_std: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _require_positive_int("d_model", d_model)
        self.n_heads = _require_positive_int("n_heads", n_heads)
        self.max_context_length = _require_positive_int(
            "max_context_length", max_context_length
        )
        init_std = _require_positive_float("init_std", init_std)
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an int or None")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads: "
                f"{self.d_model} % {self.n_heads} != 0"
            )
        self.d_head = self.d_model // self.n_heads

        self.qkv_projection = nn.Linear(self.d_model, 3 * self.d_model)
        self.output_projection = nn.Linear(self.d_model, self.d_model)

        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator(
                device=self.qkv_projection.weight.device
            ).manual_seed(seed)
        _init_linear(self.qkv_projection, init_std, generator)
        _init_linear(self.output_projection, init_std, generator)

        # True means "blocked"; the derived table is not checkpoint state.
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
            batch, length, self.n_heads, self.d_head
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
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (B, T, D), got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError(
                f"x must be a floating-point tensor, got {x.dtype}"
            )
        batch = x.shape[0]
        length = x.shape[1]
        dim = x.shape[2]
        if dim != self.d_model:
            raise ValueError(
                f"x last dimension {dim} does not match d_model {self.d_model}"
            )
        if length < 1:
            raise ValueError("sequence length must be >= 1")
        if length > self.max_context_length:
            raise ValueError(
                f"sequence length {length} exceeds "
                f"max_context_length {self.max_context_length}"
            )

        qkv = self.qkv_projection(x)
        q, k, v = qkv.split(self.d_model, dim=-1)
        queries = self._split_heads(q)
        keys = self._split_heads(k)
        values = self._split_heads(v)

        scores = (queries @ keys.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores.masked_fill(
            self.causal_mask[:length, :length], float("-inf")
        )
        weights = torch.softmax(scores, dim=-1)
        context = weights @ values
        context = context.transpose(1, 2).reshape(
            batch, length, self.d_model
        )
        output: Tensor = self.output_projection(context)

        if return_weights:
            return output, weights
        return output


class _ToyLanguageModel(nn.Module):
    """Token/position embeddings, one attention block, and an LM head."""

    token_embedding: nn.Embedding
    position_embedding: nn.Embedding
    attention: MultiHeadSelfAttention
    lm_head: nn.Linear
    max_context_length: int

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        max_context_length: int,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        vocab_size = _require_positive_int("vocab_size", vocab_size)
        d_model = _require_positive_int("d_model", d_model)
        n_heads = _require_positive_int("n_heads", n_heads)
        max_context_length = _require_positive_int(
            "max_context_length", max_context_length
        )
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an int or None")
        self.max_context_length = max_context_length

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_context_length, d_model)
        self.attention = MultiHeadSelfAttention(
            d_model,
            n_heads,
            max_context_length,
            seed=None if seed is None else seed + 1,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        generator: torch.Generator | None = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        nn.init.normal_(
            self.token_embedding.weight,
            mean=0.0,
            std=0.02,
            generator=generator,
        )
        nn.init.normal_(
            self.position_embedding.weight,
            mean=0.0,
            std=0.02,
            generator=generator,
        )
        nn.init.normal_(
            self.lm_head.weight,
            mean=0.0,
            std=0.02,
            generator=generator,
        )

    def embed(self, ids: Tensor) -> Tensor:
        if ids.ndim != 2:
            raise ValueError(
                f"ids must have shape (B, T), got {tuple(ids.shape)}"
            )
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
        positions = torch.arange(length, device=ids.device)
        embedded: Tensor = self.token_embedding(
            ids
        ) + self.position_embedding(positions).unsqueeze(0)
        return embedded

    def forward(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.embed(ids)
        output, weights = self.attention(hidden, return_weights=True)
        logits: Tensor = self.lm_head(output)
        return logits, weights


class _TrainingHistory(NamedTuple):
    initial_loss: float
    epoch_losses: tuple[float, ...]
    epoch_validation_losses: tuple[float, ...]
    final_loss: float
    final_accuracy: float
    gradients_ok: bool


def _gradients_are_finite(module: nn.Module) -> bool:
    for parameter in module.parameters():
        grad = parameter.grad
        if grad is None or not bool(torch.isfinite(grad).all()):
            return False
    return True


def _evaluate(model: _ToyLanguageModel, ids: Tensor) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits, _ = model(ids)
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab_size),
            ids[:, 1:].reshape(-1),
        )
        accuracy = (
            logits[:, :-1].argmax(-1) == ids[:, 1:]
        ).float().mean()
    return float(loss), float(accuracy)


def _train_repeat_task(
    model: _ToyLanguageModel,
    train_ids: Tensor,
    validation_ids: Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> _TrainingHistory:
    train_size = train_ids.shape[0]
    initial_loss, _ = _evaluate(model, validation_ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    epoch_losses: list[float] = []
    epoch_validation_losses: list[float] = []
    gradients_ok = False
    first_batch = True

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(
            train_size,
            generator=torch.Generator().manual_seed(seed + 100 + epoch),
        )
        total = 0.0
        for start in range(0, train_size, batch_size):
            batch = train_ids[permutation[start : start + batch_size]]
            logits, _ = model(batch)
            vocab_size = logits.shape[-1]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                batch[:, 1:].reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]  # torch stub is unannotated
            if first_batch:
                gradients_ok = _gradients_are_finite(model)
                first_batch = False
            optimizer.step()
            total += float(loss.detach()) * batch.shape[0]
        epoch_losses.append(total / train_size)
        validation_loss, _ = _evaluate(model, validation_ids)
        epoch_validation_losses.append(validation_loss)

    final_loss, final_accuracy = _evaluate(model, validation_ids)
    return _TrainingHistory(
        initial_loss=initial_loss,
        epoch_losses=tuple(epoch_losses),
        epoch_validation_losses=tuple(epoch_validation_losses),
        final_loss=final_loss,
        final_accuracy=final_accuracy,
        gradients_ok=gradients_ok,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Causal multi-head self-attention block and repeat-task "
            "training demo."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser(
        "demo",
        help="inspect a block and train it on the repeat task",
    )
    demo.add_argument("--dim", type=int, default=32, help="model dimension (D)")
    demo.add_argument("--heads", type=int, default=4, help="attention heads (H)")
    demo.add_argument(
        "--context-length", type=int, default=12, help="tokens per sequence (T)"
    )
    demo.add_argument(
        "--vocab-size", type=int, default=64, help="vocabulary size (V)"
    )
    demo.add_argument("--epochs", type=int, default=3, help="training epochs")
    demo.add_argument(
        "--batch-size", type=int, default=64, help="sequences per batch"
    )
    demo.add_argument(
        "--train-size",
        type=int,
        default=2048,
        help="number of synthetic training sequences",
    )
    demo.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="AdamW learning rate",
    )
    demo.add_argument(
        "--heatmap-head",
        type=int,
        default=0,
        help="head index to render as a character heatmap",
    )
    demo.add_argument("--seed", type=int, default=1234, help="base seed")
    return parser


def _run_demo(args: argparse.Namespace) -> int:
    try:
        dim = _require_positive_int("dim", args.dim)
        heads = _require_positive_int("heads", args.heads)
        context_length = _require_positive_int(
            "context_length", args.context_length
        )
        vocab_size = _require_positive_int("vocab_size", args.vocab_size)
        epochs = _require_positive_int("epochs", args.epochs)
        batch_size = _require_positive_int("batch_size", args.batch_size)
        train_size = _require_positive_int("train_size", args.train_size)
        learning_rate = _require_positive_float(
            "learning_rate", args.learning_rate
        )
        seed = args.seed
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an int")
        heatmap_head = args.heatmap_head
        if isinstance(heatmap_head, bool) or not isinstance(heatmap_head, int):
            raise TypeError("heatmap_head must be an int")
        if dim % heads != 0:
            raise ValueError(
                "d_model must be divisible by n_heads: "
                f"{dim} % {heads} != 0"
            )
        if context_length < 2:
            raise ValueError(
                "context_length must be >= 2 for the next-token demo"
            )
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        if not 0 <= heatmap_head < heads:
            raise ValueError("heatmap_head must be in [0, n_heads)")
    except (TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    model = _ToyLanguageModel(
        vocab_size,
        dim,
        heads,
        context_length,
        seed=seed,
    )
    train_ids = torch.randint(
        0,
        vocab_size,
        (train_size, 1),
        generator=torch.Generator().manual_seed(seed + 2),
    ).repeat(1, context_length)
    validation_ids = torch.randint(
        0,
        vocab_size,
        (256, 1),
        generator=torch.Generator().manual_seed(seed + 3),
    ).repeat(1, context_length)
    inspection_ids = torch.randint(
        0,
        vocab_size,
        (1, context_length),
        generator=torch.Generator().manual_seed(seed + 7),
    )

    print("training (repeat task):")
    history = _train_repeat_task(
        model,
        train_ids,
        validation_ids,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    losses = (
        history.initial_loss,
        history.final_loss,
        *history.epoch_losses,
        *history.epoch_validation_losses,
    )
    if not all(math.isfinite(loss) for loss in losses):
        raise AssertionError("training produced a non-finite loss")
    print(f"initial_val_loss={history.initial_loss:.4f}")
    for epoch, (train_loss, validation_loss) in enumerate(
        zip(history.epoch_losses, history.epoch_validation_losses),
        start=1,
    ):
        print(
            f"  epoch={epoch} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f}"
        )
    print(f"final_val_loss={history.final_loss:.4f}")
    print(f"final_val_accuracy={history.final_accuracy:.4f}")
    if not history.gradients_ok:
        raise AssertionError("gradients did not reach every parameter")
    print("grad_flow=ok")

    inspection_input = model.embed(inspection_ids)
    with torch.no_grad():
        attention_output, attention_weights = model.attention(
            inspection_input, return_weights=True
        )
    if not bool(torch.isfinite(attention_output).all()) or not bool(
        torch.isfinite(attention_weights).all()
    ):
        raise AssertionError("attention produced non-finite values")
    causal_violation = float(
        torch.triu(attention_weights, diagonal=1).abs().max()
    )
    if causal_violation != 0.0:
        raise AssertionError(
            f"causal mask leaked future weight {causal_violation}"
        )
    row_error = float((attention_weights.sum(dim=-1) - 1.0).abs().max())
    if row_error > 1e-5:
        raise AssertionError(
            f"attention rows do not sum to 1: max error {row_error}"
        )

    attention_params = parameter_count(model.attention)
    model_params = parameter_count(model)
    print(
        f"attention_config: d_model={dim} n_heads={heads} "
        f"d_head={dim // heads} max_context_length={context_length}"
    )
    print(
        f"inspection_input: shape={tuple(inspection_input.shape)} "
        f"dtype={inspection_input.dtype}"
    )
    print(f"attention_output: shape={tuple(attention_output.shape)}")
    print(f"attention_weights: shape={tuple(attention_weights.shape)}")
    print(f"causal_violation_max={causal_violation:.1f}")
    print(f"row_sum_max_error={row_error:.2e}")
    print(
        f"parameter_counts: attention={attention_params} "
        f"model={model_params} "
        f"attention_share={attention_params / model_params * 100.0:.1f}%"
    )
    print(f"inspection_ids={inspection_ids[0].tolist()}")

    query_weights = attention_weights[:, :, 1:, :]
    entropy = attention_entropy(query_weights)
    self_weight = attention_weights.diagonal(
        dim1=-2, dim2=-1
    )[:, :, 1:].mean(dim=(0, 2))
    previous = attention_weights.diagonal(
        dim1=-2, dim2=-1, offset=-1
    ).mean(dim=(0, 2))
    first = query_weights[..., 0].mean(dim=(0, 2))
    print("head_stats (queries 1..T-1):")
    for head in range(heads):
        print(
            f"  head={head} entropy={float(entropy[head]):.3f} "
            f"self={float(self_weight[head]):.3f} "
            f"previous={float(previous[head]):.3f} "
            f"first={float(first[head]):.3f}"
        )

    heatmap_chars = " .:-=+*#%@"
    head_weights = attention_weights[0, heatmap_head].detach()
    print(
        f"heatmap_head={heatmap_head} "
        "(rows=query, cols=key; chars scale 0..9):"
    )
    for query_index, row in enumerate(head_weights):
        cells = " ".join(
            heatmap_chars[
                min(int(float(value) * 10), len(heatmap_chars) - 1)
            ]
            for value in row
        )
        print(f"  q{query_index:02d} | {cells}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command != "demo":
        parser.error(f"unknown command: {args.command}")
    return _run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())
