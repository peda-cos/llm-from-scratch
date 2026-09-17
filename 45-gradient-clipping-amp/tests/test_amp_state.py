from __future__ import annotations

import math
from typing import cast

import pytest
import torch
from torch import Tensor, nn
from torch.amp import GradScaler

from main import AmpTrainState, SkipLog, StepLog
from support import tiny_inputs, tiny_state


class _ProbeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)
        self.autocast_enabled = False
        self.autocast_dtype: torch.dtype | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        self.autocast_enabled = torch.is_autocast_enabled("cpu")
        self.autocast_dtype = (
            torch.get_autocast_dtype("cpu")
            if self.autocast_enabled
            else None
        )
        return cast(Tensor, self.linear(inputs))


class _NanLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.full(
            (inputs.shape[0], 2),
            float("nan"),
            dtype=torch.float32,
            device=inputs.device,
        )


def test_clean_step_updates_parameters_and_logs_norms() -> None:
    state = tiny_state()
    inputs, targets = tiny_inputs()
    before = [
        parameter.detach().clone() for parameter in state.model.parameters()
    ]

    record = state.step(inputs, targets)

    assert state.scaler.is_enabled()
    assert isinstance(record, StepLog)
    assert record.step == 0
    assert state.global_step == 1
    assert math.isfinite(record.loss)
    assert math.isfinite(record.grad_l2_pre_clip)
    assert math.isfinite(record.grad_l2_post_clip)
    assert record.clipped is (record.grad_l2_pre_clip > state.max_grad_norm)
    assert any(
        not torch.equal(parameter.detach(), old)
        for parameter, old in zip(
            state.model.parameters(), before, strict=True
        )
    )


def test_step_uses_autocast_with_configured_dtype() -> None:
    probe = _ProbeModel()
    state = tiny_state(model=probe, autocast_dtype=torch.float16)
    inputs, targets = tiny_inputs()

    record = state.step(inputs, targets)

    assert isinstance(record, StepLog)
    assert probe.autocast_enabled is True
    assert probe.autocast_dtype is torch.float16


def test_logged_norm_is_unscaled_not_scaled() -> None:
    state = tiny_state(max_grad_norm=1e9)
    inputs, targets = tiny_inputs()

    record = state.step(inputs, targets)

    assert isinstance(record, StepLog)
    assert record.scaler_scale == pytest.approx(1024.0)
    assert record.grad_l2_pre_clip < 100.0
    assert record.clipped is False


def test_tight_clip_bounds_post_norm() -> None:
    state = tiny_state(max_grad_norm=1e-3)
    inputs, targets = tiny_inputs()

    record = state.step(inputs, targets)

    assert isinstance(record, StepLog)
    assert record.clipped is True
    assert record.grad_l2_post_clip <= 1e-3 + 1e-6


def test_inf_gradient_skips_step_and_next_step_recovers() -> None:
    state = tiny_state(inject_inf_step=0)
    inputs, targets = tiny_inputs()
    before = [
        parameter.detach().clone() for parameter in state.model.parameters()
    ]

    record = state.step(inputs, targets)

    assert isinstance(record, SkipLog)
    assert record.step == 0
    assert record.reason == "non_finite_grad"
    assert record.grad_l2_pre_clip is not None
    assert math.isinf(record.grad_l2_pre_clip)
    assert state.global_step == 1
    assert state.scaler.get_scale() == pytest.approx(512.0)
    for parameter, old in zip(
        state.model.parameters(), before, strict=True
    ):
        assert torch.equal(parameter.detach(), old)

    recovered = state.step(inputs, targets)

    assert isinstance(recovered, StepLog)
    assert recovered.step == 1
    assert state.global_step == 2
    assert state.scaler.get_scale() == pytest.approx(512.0)


def test_non_finite_loss_skips_without_backward_and_halves_scale() -> None:
    state = tiny_state(model=_NanLossModel())
    inputs, targets = tiny_inputs()
    before = [
        parameter.detach().clone() for parameter in state.model.parameters()
    ]

    record = state.step(inputs, targets)

    assert isinstance(record, SkipLog)
    assert record.reason == "non_finite_loss"
    assert math.isnan(record.loss)
    assert record.grad_l2_pre_clip is None
    assert state.global_step == 1
    assert state.scaler.get_scale() == pytest.approx(512.0)
    assert all(
        parameter.grad is None for parameter in state.model.parameters()
    )
    for parameter, old in zip(
        state.model.parameters(), before, strict=True
    ):
        assert torch.equal(parameter.detach(), old)


def test_state_rejects_wrong_optimizer_scaler_and_dtype() -> None:
    torch.manual_seed(3)
    model = nn.Linear(3, 2)
    inputs, targets = tiny_inputs()

    with pytest.raises(TypeError, match="AdamW"):
        AmpTrainState(
            model=model,
            optimizer=cast(
                torch.optim.AdamW,
                torch.optim.SGD(model.parameters(), lr=0.1),
            ),
            scaler=GradScaler("cpu"),
            device=torch.device("cpu"),
            autocast_dtype=torch.float16,
            max_grad_norm=0.5,
            lr=0.1,
            backoff_factor=0.5,
        )

    with pytest.raises(TypeError, match="GradScaler"):
        AmpTrainState(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.1),
            scaler=cast(GradScaler, object()),
            device=torch.device("cpu"),
            autocast_dtype=torch.float16,
            max_grad_norm=0.5,
            lr=0.1,
            backoff_factor=0.5,
        )

    with pytest.raises(ValueError, match="float16 or torch.bfloat16"):
        tiny_state(autocast_dtype=torch.float32)

    with pytest.raises(ValueError, match="max_grad_norm must be > 0"):
        tiny_state(max_grad_norm=0.0)

    state = tiny_state()
    with pytest.raises(ValueError, match="same batch size"):
        state.step(inputs, targets[:3])
    with pytest.raises(ValueError, match="2-D"):
        state.step(torch.zeros(4), targets)
    with pytest.raises(ValueError, match="1-D"):
        state.step(inputs, targets.reshape(1, 4))
    with pytest.raises(ValueError, match="torch.long"):
        state.step(inputs, targets.float())
