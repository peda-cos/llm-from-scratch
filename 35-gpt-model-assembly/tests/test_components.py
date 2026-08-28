from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from main import LayerNorm, MultiHeadAttention


def _randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype)


def test_layernorm_matches_torch() -> None:
    x = _randn((2, 5, 16), seed=1, dtype=torch.float64)
    ours = LayerNorm(16).to(torch.float64)
    reference = nn.LayerNorm(16, eps=1e-5).to(torch.float64)
    generator = torch.Generator().manual_seed(2)
    with torch.no_grad():
        ours.weight.copy_(
            torch.randn(16, generator=generator, dtype=torch.float64)
        )
        ours.bias.copy_(
            torch.randn(16, generator=generator, dtype=torch.float64)
        )
        reference.weight.copy_(ours.weight)
        reference.bias.copy_(ours.bias)

    torch.testing.assert_close(
        ours(x), reference(x), atol=1e-12, rtol=1e-12
    )


def test_attention_is_causal_and_shaped() -> None:
    attention = MultiHeadAttention(
        16, 2, 6, dropout=0.0, seed=0
    ).to(torch.float64)
    attention.eval()
    x = _randn((2, 6, 16), seed=1, dtype=torch.float64)
    changed = x.clone()
    changed[:, 3:] = changed[:, 3:] + 5.0

    with torch.no_grad():
        original = attention(x)
        perturbed = attention(changed)

    assert original.shape == (2, 6, 16)
    torch.testing.assert_close(
        original[:, :3],
        perturbed[:, :3],
        atol=1e-12,
        rtol=1e-12,
    )
    assert not torch.allclose(
        original[:, 3:], perturbed[:, 3:], atol=1e-8
    )


def test_attention_rejects_bad_shapes() -> None:
    attention = MultiHeadAttention(16, 2, 6, seed=0)

    with pytest.raises(ValueError, match="shape"):
        attention(_randn((2, 6), seed=1))
    with pytest.raises(ValueError, match="does not match"):
        attention(_randn((2, 6, 12), seed=1))
    with pytest.raises(ValueError, match="context_length"):
        attention(_randn((2, 7, 16), seed=1))
    with pytest.raises(TypeError, match="floating-point"):
        attention(torch.ones((2, 6, 16), dtype=torch.int64))
