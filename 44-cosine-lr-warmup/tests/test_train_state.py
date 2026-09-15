from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from main import CosineWithWarmup, ScheduleConfig, TrainState, grad_l2_norm
from support import tiny_state


class _GradModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(torch.zeros(2))
        self.second = nn.Parameter(torch.zeros(2))


def test_grad_norm_is_concatenated_l2_norm() -> None:
    module = _GradModule()
    module.first.grad = torch.tensor([3.0, 4.0])
    module.second.grad = torch.tensor([12.0, 16.0])
    assert grad_l2_norm(module) == pytest.approx(math.sqrt(425.0))

    module.second.grad = None
    assert grad_l2_norm(module) == pytest.approx(5.0)

    empty = _GradModule()
    assert grad_l2_norm(empty) == 0.0


def test_train_state_applies_schedule_and_advances_global_step() -> None:
    state = tiny_state(
        ScheduleConfig(warmup_steps=2, total_steps=5, lr_max=0.1, lr_min=0.01)
    )
    assert state.global_step == 0

    first = state.step()
    assert first.step == 0
    assert first.lr == 0.0
    assert state.optimizer.param_groups[0]["lr"] == 0.0
    assert state.global_step == 1
    assert math.isfinite(first.loss)
    assert first.grad_l2_norm > 0.0

    second = state.step()
    assert second.step == 1
    assert second.lr == pytest.approx(0.05)
    assert state.optimizer.param_groups[0]["lr"] == pytest.approx(0.05)
    assert state.global_step == 2
    assert math.isfinite(second.loss)


def test_zero_lr_first_step_leaves_parameters_unchanged() -> None:
    state = tiny_state()
    before = [parameter.detach().clone() for parameter in state.model.parameters()]
    state.step()
    after_zero = [
        parameter.detach().clone() for parameter in state.model.parameters()
    ]
    assert all(
        torch.equal(left, right)
        for left, right in zip(before, after_zero, strict=True)
    )

    state.step()
    after_one = [
        parameter.detach().clone() for parameter in state.model.parameters()
    ]
    assert any(
        not torch.equal(left, right)
        for left, right in zip(before, after_one, strict=True)
    )


def test_initial_global_step_drives_schedule_and_past_total_warns() -> None:
    state = tiny_state(
        ScheduleConfig(warmup_steps=1, total_steps=3, lr_max=0.1, lr_min=0.01),
        initial_step=3,
    )
    first = state.step()
    assert first.step == 3
    assert first.lr == 0.01
    assert state.global_step == 4

    with pytest.warns(RuntimeWarning, match="pinned at lr_min"):
        past_total = state.step()
    assert past_total.step == 4
    assert past_total.lr == 0.01
    assert state.global_step == 5


def test_train_state_validates_batch_shapes() -> None:
    config = ScheduleConfig(warmup_steps=1, total_steps=5, lr_max=0.1, lr_min=0.01)
    schedule = CosineWithWarmup(config)
    torch.manual_seed(4)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_max)
    valid_inputs = torch.zeros(4, 3)
    valid_targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="same batch size"):
        TrainState(
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            inputs=valid_inputs,
            targets=valid_targets[:3],
        )
    with pytest.raises(ValueError, match="2-D"):
        TrainState(
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            inputs=torch.zeros(4),
            targets=valid_targets,
        )
    with pytest.raises(ValueError, match="1-D"):
        TrainState(
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            inputs=valid_inputs,
            targets=valid_targets.reshape(1, 4),
        )
    with pytest.raises(ValueError, match="torch.long"):
        TrainState(
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            inputs=valid_inputs,
            targets=valid_targets.float(),
        )
