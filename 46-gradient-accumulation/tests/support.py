from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor

from main import OptimizerName, SweepConfig, make_blobs


def tiny_config(
    *,
    micro_batch: int = 4,
    accum_steps: tuple[int, ...] = (1, 2),
    num_steps: int = 2,
    lr: float = 0.1,
    momentum: float = 0.9,
    optimizer: OptimizerName = "sgd",
    seed: int = 7,
    input_dim: int = 4,
    hidden_dim: int = 8,
    num_classes: int = 3,
    noise: float = 0.5,
) -> SweepConfig:
    return SweepConfig(
        micro_batch=micro_batch,
        accum_steps=accum_steps,
        num_steps=num_steps,
        lr=lr,
        momentum=momentum,
        optimizer=optimizer,
        seed=seed,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        noise=noise,
    )


def config_payload(**overrides: object) -> dict[str, object]:
    payload = tiny_config().as_dict()
    payload.update(overrides)
    return payload


def write_config(path: Path, **overrides: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_payload(**overrides)), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def build_inputs(size: int = 8) -> tuple[Tensor, Tensor]:
    return make_blobs(
        size,
        seed=11,
        input_dim=4,
        num_classes=3,
        noise=0.5,
    )


def clone_parameters(model: torch.nn.Module) -> list[Tensor]:
    return [parameter.detach().clone() for parameter in model.parameters()]
