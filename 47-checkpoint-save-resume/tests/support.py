from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from main import TrainState


def assert_plain_python_value(value: object) -> None:
    """Assert that a value uses only types accepted by weights-only loading."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for item in value:
            assert_plain_python_value(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AssertionError(f"non-string mapping key: {key!r}")
            assert_plain_python_value(item)
        return
    raise AssertionError(f"non-plain value: {type(value).__name__}")


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def make_training_objects() -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    CosineAnnealingLR,
    TrainState,
]:
    torch.manual_seed(11)
    model: nn.Module = nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 3),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    state = TrainState(
        step=0,
        epoch=0,
        batch_in_epoch=0,
        batches_per_epoch=2,
        losses=(),
    )
    return model, optimizer, scheduler, state


def run_training_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    state: TrainState,
    steps: int,
) -> TrainState:
    generator = torch.Generator().manual_seed(3)
    inputs = torch.randn(8, 4, generator=generator)
    targets = torch.arange(8) % 3
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), targets)
        torch.autograd.backward(loss)
        optimizer.step()
        scheduler.step()
        state = state.advance(float(loss.item()))
    return state
