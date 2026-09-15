from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

import pytest

from main import (
    CosineWithWarmup,
    ScheduleConfig,
    load_schedule_config,
)
from support import write_config


def _schedule(
    warmup_steps: int = 4,
    total_steps: int = 10,
    lr_max: float = 1.0,
    lr_min: float = 0.1,
) -> CosineWithWarmup:
    return CosineWithWarmup(
        ScheduleConfig(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            lr_max=lr_max,
            lr_min=lr_min,
        )
    )


def test_warmup_cosine_floor_boundaries_are_exact() -> None:
    schedule = _schedule()
    assert schedule.lr(0) == 0.0
    assert schedule.lr(2) == 0.5
    assert schedule.lr(4) == 1.0
    assert schedule.lr(7) == pytest.approx(0.55)
    assert schedule.lr(10) == 0.1
    assert schedule.lr(11) == 0.1
    assert schedule.lr(1000) == 0.1


def test_zero_warmup_starts_at_peak_and_decays() -> None:
    schedule = _schedule(warmup_steps=0)
    assert schedule.lr(0) == 1.0
    assert schedule.lr(5) == pytest.approx(0.55)
    assert schedule.lr(10) == 0.1
    assert schedule.lr(11) == 0.1
    values = [schedule.lr(step) for step in range(11)]
    assert all(
        left + 1e-15 >= right
        for left, right in zip(values, values[1:])
    )


def test_schedule_continuity_honors_exercise_bound() -> None:
    for config in (
        ScheduleConfig(warmup_steps=7, total_steps=30, lr_max=0.01, lr_min=0.001),
        ScheduleConfig(warmup_steps=0, total_steps=30, lr_max=0.01, lr_min=0.001),
    ):
        schedule = CosineWithWarmup(config)
        cosine_bound = math.pi * (config.lr_max - config.lr_min) / (
            2 * (config.total_steps - config.warmup_steps)
        )
        warmup_bound = (
            config.lr_max / config.warmup_steps
            if config.warmup_steps > 0
            else 0.0
        )
        bound = max(cosine_bound, warmup_bound)
        for step in range(config.total_steps):
            delta = abs(schedule.lr(step + 1) - schedule.lr(step))
            assert delta <= bound + 1e-12


def test_schedule_is_stateless_across_query_orders() -> None:
    schedule = _schedule(total_steps=8)
    forward = [schedule.lr(step) for step in range(10)]
    backward = [schedule.lr(step) for step in reversed(range(10))]
    repeated = [schedule.lr(step) for step in reversed(range(10))]
    assert list(reversed(backward)) == forward
    assert repeated == backward


def test_schedule_rejects_invalid_steps() -> None:
    schedule = _schedule()
    for step in (-1, -100):
        with pytest.raises(ValueError, match="step"):
            schedule.lr(step)
    for bad_step in (True, 1.5):
        with pytest.raises((TypeError, ValueError), match="step"):
            schedule.lr(cast(int, bad_step))


@pytest.mark.parametrize(
    "warmup_steps,total_steps,lr_max,lr_min",
    [
        (-1, 10, 1.0, 0.1),
        (10, 10, 1.0, 0.1),
        (11, 10, 1.0, 0.1),
        (0, 0, 1.0, 0.1),
        (0, 10, 0.0, 0.0),
        (0, 10, -0.1, 0.0),
        (0, 10, float("nan"), 0.0),
        (0, 10, 1.0, -0.1),
        (0, 10, 1.0, 1.1),
        (True, 10, 1.0, 0.1),
    ],
)
def test_config_rejects_invalid_values(
    warmup_steps: object,
    total_steps: object,
    lr_max: object,
    lr_min: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        ScheduleConfig(
            warmup_steps=warmup_steps,  # type: ignore[arg-type]
            total_steps=total_steps,  # type: ignore[arg-type]
            lr_max=lr_max,  # type: ignore[arg-type]
            lr_min=lr_min,  # type: ignore[arg-type]
        )


def test_config_json_round_trip_and_invalid_documents(
    tmp_path: Path,
) -> None:
    config = ScheduleConfig(warmup_steps=4, total_steps=20, lr_max=0.05, lr_min=0.005)
    path = tmp_path / "nested" / "schedule.json"
    write_config(
        path,
        warmup_steps=config.warmup_steps,
        total_steps=config.total_steps,
        lr_max=config.lr_max,
        lr_min=config.lr_min,
    )
    loaded = load_schedule_config(path)
    assert loaded == config
    assert loaded.to_dict() == {
        "warmup_steps": 4,
        "total_steps": 20,
        "lr_max": 0.05,
        "lr_min": 0.005,
    }

    for payload in (
        {"total_steps": 20, "lr_max": 0.05, "lr_min": 0.005},
        {**config.to_dict(), "unexpected": 1},
        {**config.to_dict(), "warmup_steps": True},
    ):
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="invalid schedule config"):
            load_schedule_config(path)

    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid schedule config"):
        load_schedule_config(path)
