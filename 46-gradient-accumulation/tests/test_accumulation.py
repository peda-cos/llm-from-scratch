from __future__ import annotations

import copy
from collections.abc import Generator
from contextlib import AbstractContextManager, contextmanager
from typing import cast

import pytest
import torch
from torch import Tensor, nn

from main import (
    accumulate_gradients,
    build_model,
    chunk_micro_batches,
    grad_l2_norm,
    make_optimizer,
    max_abs_gradient_diff,
    max_abs_parameter_diff,
    mean_cross_entropy,
    train_one_optimizer_step,
)
from support import build_inputs, clone_parameters, tiny_config


class RecordingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.enters = 0
        self.exits = 0
        self.forwards_inside = 0
        self.forwards_outside = 0
        self.in_no_sync = False

    def forward(self, inputs: Tensor) -> Tensor:
        if self.in_no_sync:
            self.forwards_inside += 1
        else:
            self.forwards_outside += 1
        return cast(Tensor, self.linear(inputs))

    def no_sync(self) -> AbstractContextManager[None]:
        @contextmanager
        def context() -> Generator[None, None, None]:
            self.in_no_sync = True
            self.enters += 1
            try:
                yield
            finally:
                self.in_no_sync = False
                self.exits += 1

        return context()


def test_optimizer_state_advances_once_per_effective_batch() -> None:
    config = tiny_config(optimizer="adamw", accum_steps=(4,), num_steps=1, lr=0.01)
    model = build_model(config)
    optimizer = make_optimizer(model, config)
    initial = clone_parameters(model)
    inputs, targets = build_inputs(8)
    micro_batches = chunk_micro_batches(inputs, targets, 4)

    for _ in range(3):
        train_one_optimizer_step(model, optimizer, micro_batches)

    for parameter in model.parameters():
        assert int(optimizer.state[parameter]["step"].item()) == 3
    assert any(
        not torch.equal(before, after)
        for before, after in zip(initial, clone_parameters(model), strict=True)
    )


def test_stale_gradients_are_cleared_before_accumulation() -> None:
    config = tiny_config()
    dirty_model = build_model(config)
    clean_model = copy.deepcopy(dirty_model)
    for parameter in dirty_model.parameters():
        parameter.grad = torch.full_like(parameter, 5.0)
    dirty_optimizer = make_optimizer(dirty_model, config)
    clean_optimizer = make_optimizer(clean_model, config)
    inputs, targets = build_inputs(8)
    micro_batches = chunk_micro_batches(inputs, targets, 2)

    train_one_optimizer_step(dirty_model, dirty_optimizer, micro_batches)
    train_one_optimizer_step(clean_model, clean_optimizer, micro_batches)

    assert max_abs_gradient_diff(dirty_model, clean_model) < 1e-7
    assert max_abs_parameter_diff(dirty_model, clean_model) < 1e-7


def test_sync_on_last_step_wraps_only_non_final_micros() -> None:
    config = tiny_config()
    model = RecordingModel()
    optimizer = make_optimizer(model, config)
    inputs, targets = build_inputs(8)
    micro_batches = chunk_micro_batches(inputs, targets, 4)

    result = train_one_optimizer_step(model, optimizer, micro_batches)

    assert model.enters == 3
    assert model.exits == 3
    assert model.forwards_inside == 3
    assert model.forwards_outside == 1
    assert result.micro_batches == 4
    assert result.sync_calls == 1
    assert result.skipped_syncs == 3
    assert result.grad_norm > 0.0


def test_plain_module_no_sync_is_a_noop() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs, targets = build_inputs(8)
    micro_batches = chunk_micro_batches(inputs, targets, 4)
    assert not hasattr(model, "no_sync")

    result = train_one_optimizer_step(model, optimizer, micro_batches)

    assert result.sync_calls == 1
    assert result.skipped_syncs == 3
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_mean_loss_matches_full_batch_loss() -> None:
    config = tiny_config()
    model = build_model(config)
    reference = copy.deepcopy(model)
    inputs, targets = build_inputs(8)
    micro_batches = chunk_micro_batches(inputs, targets, 2)

    result = accumulate_gradients(
        model,
        make_optimizer(model, config),
        micro_batches,
    )
    full_loss = mean_cross_entropy(reference(inputs), targets)

    assert abs(result.mean_loss - float(full_loss.item())) < 1e-6
    assert result.grad_norm > 0.0


def test_grad_norm_skips_missing_gradients() -> None:
    model = nn.Linear(4, 3)
    assert grad_l2_norm(model.parameters()) == 0.0
    model(torch.randn(2, 4)).sum().backward()
    assert grad_l2_norm(model.parameters()) > 0.0


def test_invalid_micro_batches_are_rejected() -> None:
    model = nn.Linear(4, 3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    inputs, targets = build_inputs(8)

    with pytest.raises(ValueError, match="must not be empty"):
        accumulate_gradients(model, optimizer, [])

    with pytest.raises(ValueError, match="different lengths"):
        accumulate_gradients(model, optimizer, [(inputs[:3], targets[:2])])

    with pytest.raises(ValueError, match="2-D"):
        accumulate_gradients(
            model,
            optimizer,
            [(inputs[0], targets[:1])],
        )

    unequal = [
        (inputs[:2], targets[:2]),
        (inputs[2:], targets[2:]),
    ]
    with pytest.raises(ValueError, match="equal sample counts"):
        accumulate_gradients(model, optimizer, unequal)
