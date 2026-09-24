from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import torch

from main import DemoConfig, ReduceMode

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, timeout: float = 180.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "code" / "main.py"), *args],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} did not contain a JSON object")
    return cast(dict[str, Any], payload)


def tiny_config(
    *,
    world_size: int = 2,
    steps: int = 2,
    batch_per_rank: int = 4,
    input_dim: int = 4,
    hidden_dim: int = 8,
    num_classes: int = 3,
    lr: float = 0.1,
    momentum: float = 0.9,
    seed: int = 7,
    reduce_mode: ReduceMode = "sum",
    timeout_seconds: int = 30,
) -> DemoConfig:
    return DemoConfig(
        world_size=world_size,
        steps=steps,
        batch_per_rank=batch_per_rank,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        lr=lr,
        momentum=momentum,
        seed=seed,
        reduce_mode=reduce_mode,
        timeout_seconds=timeout_seconds,
    )
