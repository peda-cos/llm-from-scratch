from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from main import CosineWithWarmup, ScheduleConfig, TrainState


def tiny_state(
    config: ScheduleConfig | None = None,
    *,
    initial_step: int = 0,
) -> TrainState:
    selected = config or ScheduleConfig(
        warmup_steps=2,
        total_steps=5,
        lr_max=0.1,
        lr_min=0.01,
    )
    torch.manual_seed(3)
    model = nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=selected.lr_max)
    inputs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    return TrainState(
        model=model,
        optimizer=optimizer,
        schedule=CosineWithWarmup(selected),
        inputs=inputs,
        targets=targets,
        global_step=initial_step,
    )


def write_config(path: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "warmup_steps": 4,
        "total_steps": 10,
        "lr_max": 1.0,
        "lr_min": 0.1,
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
