from __future__ import annotations

import torch
from torch import Tensor

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
