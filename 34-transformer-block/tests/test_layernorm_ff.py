from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from main import FeedForward, LayerNorm, parameter_count


def _randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype)


def test_layer_norm_matches_pytorch_reference() -> None:
    layer_norm = LayerNorm(8, eps=1e-5).to(torch.float64)
    with torch.no_grad():
        layer_norm.weight.copy_(_randn((8,), seed=1, dtype=torch.float64))
        layer_norm.bias.copy_(_randn((8,), seed=2, dtype=torch.float64))

    reference = nn.LayerNorm(8, eps=1e-5, dtype=torch.float64)
    with torch.no_grad():
        reference.weight.copy_(layer_norm.weight)
        reference.bias.copy_(layer_norm.bias)

    x = _randn((2, 5, 8), seed=3, dtype=torch.float64) * 3.0 + 1.0
    torch.testing.assert_close(layer_norm(x), reference(x), atol=1e-10, rtol=1e-10)


def test_layer_norm_rejects_invalid_arguments_and_input() -> None:
    with pytest.raises(ValueError, match="d_model"):
        LayerNorm(0)
    with pytest.raises(TypeError, match="d_model"):
        LayerNorm(cast(Any, 3.0))
    with pytest.raises(ValueError, match="eps"):
        LayerNorm(8, eps=0.0)
    with pytest.raises(ValueError, match="shape"):
        LayerNorm(8)(_randn((8,), seed=4))
    with pytest.raises(ValueError, match="d_model"):
        LayerNorm(8)(_randn((1, 3, 7), seed=5))
    with pytest.raises(TypeError, match="floating-point"):
        LayerNorm(8)(torch.ones((1, 3, 8), dtype=torch.int64))


def test_feed_forward_shape_and_parameter_count() -> None:
    d_model = 16
    feed_forward = FeedForward(d_model, seed=0)
    x = _randn((2, 4, d_model), seed=1)
    output = feed_forward(x)

    assert output.shape == (2, 4, d_model)
    assert output.dtype == torch.float32
    assert parameter_count(feed_forward) == 8 * d_model**2 + 5 * d_model
    assert len(list(feed_forward.named_parameters())) == 4

    output.square().mean().backward()
    for parameter in feed_forward.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())


def test_feed_forward_dropout_only_in_training() -> None:
    feed_forward = FeedForward(32, dropout=0.5, seed=0)
    x = _randn((8, 16, 32), seed=1)

    feed_forward.eval()
    with torch.no_grad():
        first = feed_forward(x)
        second = feed_forward(x)
    assert torch.equal(first, second)

    feed_forward.train()
    torch.manual_seed(0)
    first = feed_forward(x)
    second = feed_forward(x)
    assert not torch.allclose(first, second)
