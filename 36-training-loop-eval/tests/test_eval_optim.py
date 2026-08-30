from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from main import GPTConfig, GPTModel, build_param_groups, evaluate_model


class TrackingLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, inputs: Tensor) -> Tensor:
        if torch.is_grad_enabled():
            raise AssertionError("evaluation must run without gradients")
        self.calls += 1
        return torch.zeros(
            (inputs.shape[0], inputs.shape[1], 5), dtype=torch.float32
        )


def _loader(batches: int) -> list[tuple[Tensor, Tensor]]:
    inputs = torch.zeros((2, 3), dtype=torch.long)
    targets = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)
    return [(inputs, targets) for _ in range(batches)]


def test_build_param_groups_splits_matrix_and_1d_params() -> None:
    model = GPTModel(
        GPTConfig(
            vocab_size=16,
            context_length=8,
            d_model=16,
            num_heads=4,
            num_layers=2,
        ),
        seed=2,
    )
    groups = build_param_groups(model, 0.2)

    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.2
    assert groups[1]["weight_decay"] == 0.0
    decay_params = groups[0]["params"]
    no_decay_params = groups[1]["params"]
    assert all(parameter.ndim >= 2 for parameter in decay_params)
    assert all(parameter.ndim == 1 for parameter in no_decay_params)
    assert any(parameter is model.token_embedding.weight for parameter in decay_params)
    assert any(parameter is model.final_norm.weight for parameter in no_decay_params)

    unique = {id(parameter) for parameter in model.parameters()}
    all_params = decay_params + no_decay_params
    grouped = [id(parameter) for parameter in all_params]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == unique
    assert sum(
        parameter is model.token_embedding.weight for parameter in all_params
    ) == 1

    optimizer = torch.optim.AdamW(groups, lr=1e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_evaluate_model_uses_exactly_max_batches_and_no_grad() -> None:
    model = TrackingLogits()
    model.train()
    value = evaluate_model(model, _loader(3), max_batches=2)

    assert model.calls == 2
    assert value == pytest.approx(math.log(5.0))
    assert model.training

    model.eval()
    evaluate_model(model, _loader(1), max_batches=1)
    assert not model.training


def test_evaluate_model_rejects_invalid_requests() -> None:
    model = TrackingLogits()
    with pytest.raises(ValueError, match="max_batches"):
        evaluate_model(model, _loader(2), max_batches=0)
    with pytest.raises(ValueError, match="max_batches=3"):
        evaluate_model(model, _loader(2), max_batches=3)
    assert model.calls == 0


def test_evaluate_model_does_not_change_parameters() -> None:
    model = GPTModel(
        GPTConfig(
            vocab_size=16,
            context_length=8,
            d_model=16,
            num_heads=4,
            num_layers=1,
        ),
        seed=6,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    evaluate_model(model, _loader(2), max_batches=2)

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name])
