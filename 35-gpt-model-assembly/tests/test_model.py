from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from torch import nn

from main import GPTConfig, GPTModel, TransformerBlock, count_parameters


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        context_length=8,
        d_model=16,
        num_heads=4,
        num_layers=2,
        mlp_expansion=4,
        dropout=0.0,
        use_bias=True,
        weight_tying=True,
    )


def _expected_parameters(config: GPTConfig, *, tied: bool) -> int:
    per_block = (
        (4 + 2 * config.mlp_expansion) * config.d_model**2
        + 4 * config.d_model
    )
    if config.use_bias:
        per_block += (config.mlp_expansion + 5) * config.d_model
    total = (
        config.vocab_size * config.d_model
        + config.context_length * config.d_model
        + 2 * config.d_model
        + config.num_layers * per_block
    )
    if not tied:
        total += config.vocab_size * config.d_model
    return total


def test_parameter_count_matches_reference() -> None:
    with torch.device("meta"):
        model = GPTModel(GPTConfig())
    assert count_parameters(model) == 124_439_808


def test_parameter_formula_and_tying_delta() -> None:
    config = _tiny_config()
    tied = GPTModel(config, seed=0)
    untied = GPTModel(replace(config, weight_tying=False), seed=0)

    assert count_parameters(tied) == _expected_parameters(config, tied=True)
    assert count_parameters(untied) == _expected_parameters(
        config, tied=False
    )
    assert (
        count_parameters(untied) - count_parameters(tied)
        == config.vocab_size * config.d_model
        == 32 * 16
    )

    no_bias = replace(config, use_bias=False)
    tied_no_bias = GPTModel(no_bias, seed=0)
    untied_no_bias = GPTModel(
        replace(no_bias, weight_tying=False), seed=0
    )
    assert count_parameters(tied_no_bias) == _expected_parameters(
        no_bias, tied=True
    )
    assert count_parameters(untied_no_bias) == _expected_parameters(
        no_bias, tied=False
    )


def test_weight_tying_shares_parameter_and_gradient() -> None:
    model = GPTModel(_tiny_config(), seed=0)
    assert model.lm_head.weight is model.token_embedding.weight
    assert model.lm_head.bias is None

    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    logits = model(ids)
    logits.square().mean().backward()
    gradient = model.token_embedding.weight.grad
    assert gradient is not None
    assert model.lm_head.weight.grad is gradient

    untied = GPTModel(
        replace(_tiny_config(), weight_tying=False), seed=0
    )
    assert untied.lm_head.weight is not untied.token_embedding.weight


def test_forward_shape_and_validation() -> None:
    model = GPTModel(_tiny_config(), seed=0)
    model.eval()
    ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]],
        dtype=torch.long,
    )
    with torch.no_grad():
        logits = model(ids)

    assert logits.shape == (2, 8, 32)
    assert bool(torch.isfinite(logits).all())

    with pytest.raises(ValueError, match="shape"):
        model(torch.tensor([1, 2, 3], dtype=torch.long))
    with pytest.raises(TypeError, match="integer"):
        model(torch.ones((1, 3), dtype=torch.float32))
    with pytest.raises(ValueError, match="batch"):
        model(torch.empty((0, 3), dtype=torch.long))
    with pytest.raises(ValueError, match="sequence length"):
        model(torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(ValueError, match="context_length"):
        model(torch.zeros((1, 9), dtype=torch.long))


def test_full_model_is_causal() -> None:
    model = GPTModel(_tiny_config(), seed=0).double()
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    changed = ids.clone()
    changed[:, -1] = 7

    with torch.no_grad():
        original = model(ids)
        perturbed = model(changed)

    torch.testing.assert_close(
        original[:, :-1],
        perturbed[:, :-1],
        atol=1e-10,
        rtol=1e-10,
    )
    assert not torch.allclose(
        original[:, -1], perturbed[:, -1], atol=1e-8
    )


def test_position_ids_are_cached_and_nonpersistent() -> None:
    model = GPTModel(_tiny_config(), seed=0)
    assert model.position_ids.tolist() == list(range(8))
    assert model.position_ids.dtype == torch.long
    assert "position_ids" not in model.state_dict()


def test_residual_projections_use_scaled_init() -> None:
    config = replace(
        _tiny_config(), d_model=64, num_heads=4, num_layers=4
    )
    model = GPTModel(config, seed=0)
    expected = 0.02 / math.sqrt(2 * config.num_layers)

    for module in model.blocks:
        block = cast(TransformerBlock, module)
        attention_std = float(
            block.attention.output_projection.weight.detach().std()
        )
        feed_forward_std = float(
            block.feed_forward.output_projection.weight.detach().std()
        )
        qkv_std = float(
            block.attention.qkv_projection.weight.detach().std()
        )
        assert abs(attention_std - expected) < expected * 0.25
        assert abs(feed_forward_std - expected) < expected * 0.25
        assert 0.015 < qkv_std < 0.025


def test_use_bias_false_and_mlp_expansion() -> None:
    config = replace(
        _tiny_config(), use_bias=False, mlp_expansion=2
    )
    model = GPTModel(config, seed=0)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            assert module.bias is None
    assert model.final_norm.bias is not None

    first_block = cast(TransformerBlock, model.blocks[0])
    assert (
        first_block.feed_forward.input_projection.out_features
        == 2 * config.d_model
    )
    assert count_parameters(model) == _expected_parameters(
        config, tied=True
    )


def test_config_rejects_invalid_fields() -> None:
    with pytest.raises(ValueError, match="vocab_size"):
        GPTConfig(vocab_size=0)
    with pytest.raises(ValueError, match="context_length"):
        GPTConfig(context_length=0)
    with pytest.raises(ValueError, match="d_model"):
        GPTConfig(d_model=0)
    with pytest.raises(ValueError, match="num_heads"):
        GPTConfig(num_heads=0)
    with pytest.raises(ValueError, match="num_layers"):
        GPTConfig(num_layers=0)
    with pytest.raises(ValueError, match="mlp_expansion"):
        GPTConfig(mlp_expansion=0)
    with pytest.raises(ValueError, match="dropout"):
        GPTConfig(dropout=1.5)
    with pytest.raises(TypeError, match="use_bias"):
        GPTConfig(use_bias=cast(Any, 1))
    with pytest.raises(TypeError, match="weight_tying"):
        GPTConfig(weight_tying=cast(Any, "yes"))
    with pytest.raises(ValueError, match="divisible"):
        GPTConfig(d_model=10, num_heads=3)
