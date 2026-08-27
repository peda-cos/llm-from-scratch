from __future__ import annotations

from typing import Any, cast

import pytest
import torch
from torch import Tensor, nn

from main import TransformerBlock, parameter_count


def _randn(
    shape: tuple[int, ...],
    *,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype)


def test_pre_and_post_ln_shapes_and_equal_parameter_counts() -> None:
    d_model = 16
    pre = TransformerBlock(d_model, 2, 8, pre_ln=True, seed=0)
    post = TransformerBlock(d_model, 2, 8, pre_ln=False, seed=0)
    x = _randn((2, 5, d_model), seed=1)

    assert pre(x).shape == (2, 5, d_model)
    assert post(x).shape == (2, 5, d_model)
    assert set(pre.state_dict()) == set(post.state_dict())
    assert parameter_count(pre) == 12 * d_model**2 + 13 * d_model
    assert parameter_count(pre) == parameter_count(post)
    assert len(list(pre.named_parameters())) == 12


def test_pre_and_post_ln_differ_with_identical_weights() -> None:
    pre = TransformerBlock(32, 4, 8, pre_ln=True, seed=0)
    post = TransformerBlock(32, 4, 8, pre_ln=False, seed=0)
    post.load_state_dict(pre.state_dict())
    pre.eval()
    post.eval()
    x = _randn((2, 6, 32), seed=1)

    with torch.no_grad():
        pre_output = pre(x)
        post_output = post(x)
    assert not torch.allclose(pre_output, post_output)
    difference = float((pre_output - post_output).abs().max())
    assert difference > 1e-3


def test_gradients_reach_every_parameter() -> None:
    x = _randn((2, 4, 16), seed=1)
    for pre_ln in (True, False):
        block = TransformerBlock(16, 2, 8, pre_ln=pre_ln, seed=0)
        output = block(x)
        output.square().mean().backward()
        nonzero = False
        for parameter in block.parameters():
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())
            nonzero = nonzero or bool(parameter.grad.abs().sum() > 0)
        assert nonzero


def test_future_tokens_do_not_change_past_outputs() -> None:
    x = _randn((2, 6, 16), seed=1, dtype=torch.float64)
    perturbation = _randn((2, 3, 16), seed=2, dtype=torch.float64) * 5.0
    changed = x.clone()
    changed[:, 3:] = changed[:, 3:] + perturbation

    for pre_ln in (True, False):
        block = TransformerBlock(
            16, 2, 8, pre_ln=pre_ln, seed=0
        ).to(torch.float64)
        block.eval()
        with torch.no_grad():
            original_output = block(x)
            changed_output = block(changed)

        torch.testing.assert_close(
            original_output[:, :3],
            changed_output[:, :3],
            atol=1e-10,
            rtol=1e-10,
        )
        assert not torch.allclose(
            original_output[:, 3:], changed_output[:, 3:], atol=1e-8
        )


def test_dropout_appears_in_two_places() -> None:
    block = TransformerBlock(16, 2, 8, dropout=0.5, seed=0)
    dropouts = [
        module for module in block.modules() if isinstance(module, nn.Dropout)
    ]
    assert len(dropouts) == 2

    x = _randn((8, 8, 16), seed=1)
    block.eval()
    with torch.no_grad():
        first = block(x)
        second = block(x)
    assert torch.equal(first, second)

    block.train()
    torch.manual_seed(0)
    first = block(x)
    second = block(x)
    assert not torch.allclose(first, second)


def test_block_rejects_invalid_configuration_and_input() -> None:
    with pytest.raises(TypeError, match="pre_ln"):
        TransformerBlock(16, 2, 8, pre_ln=cast(Any, 1))
    with pytest.raises(ValueError, match="dropout"):
        TransformerBlock(16, 2, 8, dropout=1.5)

    block = TransformerBlock(16, 2, 4, seed=0)
    with pytest.raises(ValueError, match="max_context_length"):
        block(_randn((1, 5, 16), seed=1))
    with pytest.raises(ValueError, match="shape"):
        block(_randn((2, 16), seed=2))
    with pytest.raises(TypeError, match="floating-point"):
        block(torch.ones((1, 2, 16), dtype=torch.int64))
