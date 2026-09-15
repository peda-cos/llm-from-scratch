from __future__ import annotations

import csv
from pathlib import Path

import pytest

from main import (
    CosineWithWarmup,
    ScheduleConfig,
    StepLog,
    plot_schedule_ascii,
    write_schedule_csv,
    write_training_log,
)


def test_schedule_csv_schema_and_inclusive_rows(tmp_path: Path) -> None:
    config = ScheduleConfig(warmup_steps=2, total_steps=5, lr_max=0.1, lr_min=0.01)
    schedule = CosineWithWarmup(config)
    path = tmp_path / "nested" / "schedule.csv"
    write_schedule_csv(schedule, path)

    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["step", "lr"]
    assert len(rows) == 7
    for index, row in enumerate(rows[1:]):
        assert len(row) == 2
        assert int(row[0]) == index
        assert float(row[1]) == pytest.approx(schedule.lr(index))


def test_training_log_csv_schema_and_order(tmp_path: Path) -> None:
    logs = (
        StepLog(step=0, lr=0.0, grad_l2_norm=1.25, loss=2.5),
        StepLog(step=1, lr=0.5, grad_l2_norm=0.75, loss=2.0),
        StepLog(step=2, lr=0.25, grad_l2_norm=0.5, loss=1.5),
    )
    path = tmp_path / "nested" / "training_log.csv"
    write_training_log(logs, path)

    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0] == ["step", "lr", "grad_l2_norm", "loss"]
    assert len(rows) == 4
    for row, log in zip(rows[1:], logs, strict=True):
        assert int(row[0]) == log.step
        assert float(row[1]) == pytest.approx(log.lr)
        assert float(row[2]) == pytest.approx(log.grad_l2_norm)
        assert float(row[3]) == pytest.approx(log.loss)

    empty_path = tmp_path / "empty.csv"
    write_training_log((), empty_path)
    assert empty_path.read_text(encoding="utf-8") == (
        "step,lr,grad_l2_norm,loss\n"
    )


def test_plot_is_readable_and_shows_floor_tail() -> None:
    config = ScheduleConfig(warmup_steps=3, total_steps=12, lr_max=1.0, lr_min=0.1)
    schedule = CosineWithWarmup(config)
    lines = plot_schedule_ascii(
        schedule, width=32, height=8, end_step=14
    ).splitlines()

    assert lines[0].startswith("lr schedule: steps 0..14")
    assert "warmup=3" in lines[0]
    assert "peak=1.000000" in lines[0]
    assert "floor=0.100000" in lines[0]

    grid_lines = lines[1:9]
    assert len(grid_lines) == 8
    for line in grid_lines:
        payload = line.split("|", 1)[1]
        assert len(payload) == 32
    assert "*" in grid_lines[0].split("|", 1)[1]
    assert "*" in grid_lines[-1].split("|", 1)[1]

    last_chars = [line.split("|", 1)[1][-1] for line in grid_lines]
    assert last_chars.count("*") == 1
    assert last_chars[6] == "*"
    assert lines[-1].startswith(" " * 8)
    assert lines[-1].rstrip().endswith("14")


def test_plot_rejects_invalid_dimensions() -> None:
    schedule = CosineWithWarmup(
        ScheduleConfig(warmup_steps=1, total_steps=4, lr_max=0.1, lr_min=0.01)
    )
    with pytest.raises(ValueError, match="width"):
        plot_schedule_ascii(schedule, width=7)
    with pytest.raises(ValueError, match="height"):
        plot_schedule_ascii(schedule, height=1)
    with pytest.raises(ValueError, match="end_step"):
        plot_schedule_ascii(schedule, end_step=3)
