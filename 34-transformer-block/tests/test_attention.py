from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from main import MultiHeadAttention, parameter_count


def _randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype)


def reference_attention(
    block: MultiHeadAttention, x: Tensor
) -> tuple[Tensor, Tensor]:
    """Independent per-head reference built from the block's parameters."""
    batch = x.shape[0]
    length = x.shape[1]
    d_model = block.d_model
    num_heads = block.num_heads
    head_dim = block.head_dim

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
    q = q.reshape(batch, length, num_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch, length, num_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch, length, num_heads, head_dim).transpose(1, 2)

    mask = block.causal_mask[:length, :length]
    contexts: list[Tensor] = []
    heads: list[Tensor] = []
    for head in range(num_heads):
        scores = (q[:, head] @ k[:, head].transpose(-2, -1)) / math.sqrt(
            head_dim
        )
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        contexts.append(weights @ v[:, head])
        heads.append(weights)

    context = torch.cat(contexts, dim=-1)
    output = block.output_projection(context)
    return output, torch.stack(heads, dim=1)


def test_attention_shape_and_parameter_count() -> None:
    block = MultiHeadAttention(32, 4, 12, seed=0)
    x = _randn((2, 5, 32), seed=1)
    output, weights = block(x, return_weights=True)

    assert output.shape == (2, 5, 32)
    assert output.dtype == torch.float32
    assert output.requires_grad
    assert weights.shape == (2, 4, 5, 5)
    assert parameter_count(block) == 4 * 32**2 + 4 * 32
    assert len(list(block.named_parameters())) == 4
    assert "causal_mask" in dict(block.named_buffers())
    assert "causal_mask" not in block.state_dict()


def test_fused_qkv_equals_three_linears() -> None:
    block = MultiHeadAttention(8, 2, 4, seed=0).to(torch.float64)
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
    torch.testing.assert_close(q, projections[0], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(k, projections[1], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(v, projections[2], atol=1e-10, rtol=1e-10)


def test_attention_weights_match_explicit_per_head_reference() -> None:
    block = MultiHeadAttention(8, 2, 4, seed=2).to(torch.float64)
    x = _randn((2, 4, 8), seed=3, dtype=torch.float64)
    output, weights = block(x, return_weights=True)
    reference_output, reference_weights = reference_attention(block, x)

    torch.testing.assert_close(output, reference_output, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        weights, reference_weights, atol=1e-10, rtol=1e-10
    )


def test_causal_mask_is_nonpersistent_and_blocks_future() -> None:
    block = MultiHeadAttention(8, 2, 4, seed=0)
    expected = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1)
    assert torch.equal(block.causal_mask, expected)
    assert block.causal_mask.dtype == torch.bool
    assert not block.causal_mask.requires_grad
    assert "causal_mask" not in block.state_dict()

    block.eval()
    x = _randn((2, 4, 8), seed=1)
    with torch.no_grad():
        _, weights = block(x, return_weights=True)
    upper = torch.triu(weights, diagonal=1)
    assert float(upper.abs().max()) == 0.0
    row_error = float((weights.sum(dim=-1) - 1.0).abs().max())
    assert row_error <= 1e-5
    assert block.to(torch.float64).causal_mask.dtype == torch.bool
