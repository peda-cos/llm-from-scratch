from __future__ import annotations

import math
import re
from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from main import (
    MultiHeadSelfAttention,
    attention_entropy,
    main,
    parameter_count,
)


def _randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype)


def reference_attention(
    block: MultiHeadSelfAttention, x: Tensor
) -> tuple[Tensor, Tensor]:
    """Independent per-head reference built from the block's parameters."""
    batch = x.shape[0]
    length = x.shape[1]
    d_model = block.d_model
    n_heads = block.n_heads
    d_head = block.d_head

    weight = block.qkv_projection.weight
    bias = block.qkv_projection.bias
    assert bias is not None
    q_weight = weight[:d_model]
    k_weight = weight[d_model : 2 * d_model]
    v_weight = weight[2 * d_model :]
    q_bias = bias[:d_model]
    k_bias = bias[d_model : 2 * d_model]
    v_bias = bias[2 * d_model :]

    q = x @ q_weight.T + q_bias
    k = x @ k_weight.T + k_bias
    v = x @ v_weight.T + v_bias
    q = q.reshape(batch, length, n_heads, d_head).transpose(1, 2)
    k = k.reshape(batch, length, n_heads, d_head).transpose(1, 2)
    v = v.reshape(batch, length, n_heads, d_head).transpose(1, 2)

    mask = torch.triu(
        torch.ones(length, length, dtype=torch.bool), diagonal=1
    )
    contexts: list[Tensor] = []
    heads: list[Tensor] = []
    for head in range(n_heads):
        scores = (q[:, head] @ k[:, head].transpose(-2, -1)) / math.sqrt(
            d_head
        )
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        contexts.append(weights @ v[:, head])
        heads.append(weights)

    context = torch.cat(contexts, dim=-1)
    output = block.output_projection(context)
    return output, torch.stack(heads, dim=1)


def test_attention_shape_contract() -> None:
    block = MultiHeadSelfAttention(32, 4, 12, seed=0)
    x = _randn((2, 5, 32), seed=1)
    output, weights = block(x, return_weights=True)
    assert output.shape == (2, 5, 32)
    assert output.dtype == torch.float32
    assert output.requires_grad
    assert weights.shape == (2, 4, 5, 5)


def test_attention_parameter_count() -> None:
    block = MultiHeadSelfAttention(8, 2, 4, seed=0)
    assert parameter_count(block) == 4 * 8 * 8 + 4 * 8
    assert len(list(block.named_parameters())) == 4
    assert "causal_mask" in dict(block.named_buffers())
    assert "causal_mask" not in block.state_dict()


def test_mask_buffer_is_causal_and_non_persistent() -> None:
    block = MultiHeadSelfAttention(8, 2, 4, seed=0)
    expected = torch.triu(
        torch.ones(4, 4, dtype=torch.bool), diagonal=1
    )
    assert torch.equal(block.causal_mask, expected)
    assert block.causal_mask.dtype == torch.bool
    assert not block.causal_mask.requires_grad


def test_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadSelfAttention(10, 4, 4)
    with pytest.raises(ValueError, match="n_heads"):
        MultiHeadSelfAttention(8, 0, 4)
    with pytest.raises(TypeError, match="d_model"):
        MultiHeadSelfAttention(cast(Any, 3.0), 1, 4)
    with pytest.raises(ValueError, match="max_context_length"):
        MultiHeadSelfAttention(8, 2, 0)


def test_rejects_invalid_input() -> None:
    block = MultiHeadSelfAttention(8, 2, 4, seed=0)
    with pytest.raises(ValueError, match="shape"):
        block(_randn((2, 8), seed=2))
    with pytest.raises(ValueError, match="d_model"):
        block(_randn((1, 3, 7), seed=3))
    with pytest.raises(ValueError, match="sequence length"):
        block(torch.empty((1, 0, 8)))
    with pytest.raises(ValueError, match="max_context_length"):
        block(_randn((1, 5, 8), seed=4))
    with pytest.raises(TypeError, match="floating-point"):
        block(torch.ones((1, 2, 8), dtype=torch.int64))


def test_fused_qkv_equals_three_linear_projections() -> None:
    block = MultiHeadSelfAttention(8, 2, 4, seed=0).to(torch.float64)
    x = _randn((2, 4, 8), seed=1, dtype=torch.float64)
    fused = block.qkv_projection(x)
    d_model = block.d_model
    qkv_bias = block.qkv_projection.bias
    assert qkv_bias is not None
    projections: list[Tensor] = []
    for start in (0, d_model, 2 * d_model):
        projection = nn.Linear(d_model, d_model, dtype=torch.float64)
        projection.weight.data.copy_(
            block.qkv_projection.weight[start : start + d_model]
        )
        projection_bias = projection.bias
        assert projection_bias is not None
        projection.bias.data.copy_(qkv_bias[start : start + d_model])
        projections.append(projection(x))

    q, k, v = fused.split(d_model, dim=-1)
    assert torch.allclose(q, projections[0], atol=1e-10)
    assert torch.allclose(k, projections[1], atol=1e-10)
    assert torch.allclose(v, projections[2], atol=1e-10)


def test_head_split_matches_explicit_per_head_reference() -> None:
    block = MultiHeadSelfAttention(8, 2, 4, seed=2).to(torch.float64)
    x = _randn((2, 4, 8), seed=3, dtype=torch.float64)
    output, weights = block(x, return_weights=True)
    reference_output, reference_weights = reference_attention(block, x)
    assert torch.allclose(output, reference_output, atol=1e-10)
    assert torch.allclose(weights, reference_weights, atol=1e-10)


def test_scaling_mask_and_softmax_weights() -> None:
    block = MultiHeadSelfAttention(8, 2, 4, seed=4).to(torch.float64)
    x = _randn((2, 4, 8), seed=5, dtype=torch.float64)
    _, weights = block(x, return_weights=True)
    qkv = block.qkv_projection(x)
    q_raw, k_raw, _ = qkv.split(block.d_model, dim=-1)
    q = q_raw.reshape(
        x.shape[0], x.shape[1], block.n_heads, block.d_head
    ).transpose(1, 2)
    k = k_raw.reshape(
        x.shape[0], x.shape[1], block.n_heads, block.d_head
    ).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(block.d_head)
    mask = torch.triu(
        torch.ones(x.shape[1], x.shape[1], dtype=torch.bool), diagonal=1
    )
    expected = torch.softmax(
        scores.masked_fill(mask, float("-inf")), dim=-1
    )
    assert torch.allclose(weights, expected, atol=1e-10)


def test_matches_pytorch_scaled_dot_product_attention() -> None:
    block = MultiHeadSelfAttention(8, 2, 4, seed=6)
    x = _randn((2, 4, 8), seed=7)
    qkv = block.qkv_projection(x)
    q, k, v = qkv.split(block.d_model, dim=-1)

    def to_heads(projected: Tensor) -> Tensor:
        return projected.reshape(
            x.shape[0], x.shape[1], block.n_heads, block.d_head
        ).transpose(1, 2)

    context = F.scaled_dot_product_attention(
        to_heads(q), to_heads(k), to_heads(v), is_causal=True
    )
    reference = block.output_projection(
        context.transpose(1, 2).reshape(
            x.shape[0], x.shape[1], block.d_model
        )
    )
    assert torch.allclose(block(x), reference, atol=1e-6, rtol=1e-5)


def test_causality_blocks_future_tokens() -> None:
    block = MultiHeadSelfAttention(8, 2, 6, seed=8)
    x1 = _randn((1, 6, 8), seed=9)
    x2 = x1.clone()
    x2[:, 3] += 1.0
    y1 = block(x1)
    y2 = block(x2)
    assert torch.allclose(y1[:, :3], y2[:, :3], atol=1e-6)
    assert not torch.allclose(y1[:, 3:], y2[:, 3:], atol=1e-6)


def test_attention_weights_are_causal_and_normalized() -> None:
    block = MultiHeadSelfAttention(8, 2, 4, seed=10)
    x = _randn((2, 4, 8), seed=11)
    _, weights = block(x, return_weights=True)
    assert bool((weights >= 0).all())
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    assert torch.allclose(
        weights[..., 0, 0], torch.ones_like(weights[..., 0, 0])
    )
    assert torch.allclose(
        weights[..., 1, 2:], torch.zeros_like(weights[..., 1, 2:])
    )


def test_gradients_flow_to_every_parameter() -> None:
    block = MultiHeadSelfAttention(8, 2, 4, seed=12)
    x = _randn((2, 4, 8), seed=13).requires_grad_(True)
    output, weights = block(x, return_weights=True)
    (output.square().sum() + weights.square().sum()).backward()

    input_grad = x.grad
    assert input_grad is not None
    assert bool(torch.isfinite(input_grad).all())
    assert float(input_grad.abs().sum()) > 0.0
    for _, parameter in block.named_parameters():
        grad = parameter.grad
        assert grad is not None
        assert bool(torch.isfinite(grad).all())
        assert float(grad.abs().sum()) > 0.0


def test_float64_forward_preserves_dtype() -> None:
    block = MultiHeadSelfAttention(8, 2, 4, seed=14).to(torch.float64)
    x = _randn((2, 3, 8), seed=15, dtype=torch.float64)
    output, weights = block(x, return_weights=True)
    assert output.dtype == torch.float64
    assert weights.dtype == torch.float64
    assert block.causal_mask.dtype == torch.bool
    assert block.causal_mask.device == output.device
    assert bool(torch.isfinite(output).all())
    assert bool(torch.isfinite(weights).all())
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0


def test_seeded_initialization_is_reproducible() -> None:
    first = MultiHeadSelfAttention(64, 4, 8, seed=11)
    replay = MultiHeadSelfAttention(64, 4, 8, seed=11)
    other = MultiHeadSelfAttention(64, 4, 8, seed=12)
    assert torch.equal(
        first.qkv_projection.weight, replay.qkv_projection.weight
    )
    assert torch.equal(
        first.output_projection.weight, replay.output_projection.weight
    )
    assert not torch.equal(
        first.qkv_projection.weight, other.qkv_projection.weight
    )
    qkv_bias = first.qkv_projection.bias
    assert qkv_bias is not None
    assert torch.count_nonzero(qkv_bias) == 0
    weight = first.qkv_projection.weight.detach()
    assert 0.018 <= float(weight.std()) <= 0.022
    assert abs(float(weight.mean())) <= 0.002


def test_attention_entropy_matches_uniform_reference() -> None:
    uniform = torch.full((1, 1, 2, 4), 0.25)
    assert float(attention_entropy(uniform)[0]) == pytest.approx(math.log(4))

    block = MultiHeadSelfAttention(8, 2, 4, seed=16)
    x = _randn((1, 3, 8), seed=17)
    _, weights = block(x, return_weights=True)
    entropy = attention_entropy(weights)
    assert entropy.shape == (2,)
    assert bool(torch.isfinite(entropy).all())
    assert bool((entropy <= math.log(4) + 1e-6).all())
    with pytest.raises(ValueError, match="shape"):
        attention_entropy(torch.ones((2, 3, 4)))


def test_demo_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "demo",
            "--dim",
            "8",
            "--heads",
            "2",
            "--context-length",
            "4",
            "--vocab-size",
            "16",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--train-size",
            "32",
            "--learning-rate",
            "0.02",
            "--seed",
            "3",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    for expected in (
        "attention_weights: shape=(1, 2, 4, 4)",
        "causal_violation_max=0.0",
        "grad_flow=ok",
        "final_val_loss=",
        "heatmap_head=0",
        "head_stats (queries 1..T-1):",
        "epoch=1 train_loss=",
    ):
        assert expected in output


def test_demo_heads_eight(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "demo",
            "--heads",
            "8",
            "--train-size",
            "64",
            "--batch-size",
            "32",
            "--epochs",
            "1",
            "--seed",
            "5",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "d_head=4" in output
    assert "attention_weights: shape=(1, 8, 12, 12)" in output
    assert "heatmap_head=0" in output


def test_demo_is_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "demo",
        "--dim",
        "8",
        "--heads",
        "2",
        "--context-length",
        "4",
        "--vocab-size",
        "16",
        "--epochs",
        "1",
        "--batch-size",
        "8",
        "--train-size",
        "32",
        "--learning-rate",
        "0.02",
        "--seed",
        "3",
    ]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert main(args) == 0
    second = capsys.readouterr().out
    assert first == second


def test_demo_training_reduces_loss(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["demo"]) == 0
    output = capsys.readouterr().out
    initial_match = re.search(r"initial_val_loss=([0-9.]+)", output)
    final_match = re.search(r"final_val_loss=([0-9.]+)", output)
    assert initial_match is not None
    assert final_match is not None
    initial = float(initial_match.group(1))
    final = float(final_match.group(1))
    assert math.isfinite(initial)
    assert math.isfinite(final)
    assert final < initial
    assert final < 1.0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--dim", "30", "--heads", "4"], "divisible"),
        (["--context-length", "1"], "context_length must be >= 2"),
        (["--epochs", "0"], "epochs"),
        (["--heatmap-head", "4"], "heatmap_head"),
    ],
)
def test_demo_rejects_invalid_configuration(
    args: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["demo", *args]) == 2
    error = capsys.readouterr().err
    assert error.startswith("error:")
    assert message in error
