from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.amp import GradScaler

from main import AmpTrainState


def tiny_inputs() -> tuple[Tensor, Tensor]:
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
    return inputs, targets


def tiny_state(
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.AdamW | None = None,
    scaler: GradScaler | None = None,
    max_grad_norm: float = 0.5,
    lr: float = 0.1,
    backoff_factor: float = 0.5,
    inject_inf_step: int | None = None,
    autocast_dtype: torch.dtype = torch.float16,
    initial_step: int = 0,
) -> AmpTrainState:
    torch.manual_seed(3)
    selected_model = model if model is not None else nn.Linear(3, 2)
    selected_optimizer = (
        optimizer
        if optimizer is not None
        else torch.optim.AdamW(selected_model.parameters(), lr=lr)
    )
    selected_scaler = (
        scaler
        if scaler is not None
        else GradScaler(
            "cpu",
            init_scale=1024.0,
            growth_interval=2,
            backoff_factor=backoff_factor,
        )
    )
    return AmpTrainState(
        model=selected_model,
        optimizer=selected_optimizer,
        scaler=selected_scaler,
        device=torch.device("cpu"),
        autocast_dtype=autocast_dtype,
        max_grad_norm=max_grad_norm,
        lr=lr,
        backoff_factor=backoff_factor,
        inject_inf_step=inject_inf_step,
        global_step=initial_step,
    )


def write_config(path: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "lr": 0.1,
        "max_grad_norm": 0.5,
        "init_scale": 65536.0,
        "growth_interval": 4,
        "backoff_factor": 0.5,
        "steps": 20,
        "seed": 7,
        "inject_inf_step": 5,
        "autocast_dtype": "float16",
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
