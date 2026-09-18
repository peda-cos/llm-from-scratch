from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from main import (
    TRAINING_LOG_COLUMNS,
    SkipLog,
    StepLog,
    StepLogWriter,
)


def _read_rows(path: Path) -> list[list[str]]:
    return list(
        csv.reader(path.read_text(encoding="utf-8").splitlines())
    )


def test_csv_schema_and_clean_skip_rows(tmp_path: Path) -> None:
    clean = StepLog(
        step=0,
        lr=0.1,
        loss=1.5,
        grad_l2_pre_clip=2.0,
        grad_l2_post_clip=0.5,
        scaler_scale=65536.0,
        clipped=True,
    )
    gradient_skip = SkipLog(
        step=1,
        lr=0.1,
        loss=1.25,
        scaler_scale=32768.0,
        reason="non_finite_grad",
        grad_l2_pre_clip=float("inf"),
    )
    loss_skip = SkipLog(
        step=2,
        lr=0.1,
        loss=float("nan"),
        scaler_scale=16384.0,
        reason="non_finite_loss",
        grad_l2_pre_clip=None,
    )
    path = tmp_path / "nested" / "training_log.csv"

    with StepLogWriter(path) as writer:
        writer.write(clean)
        writer.write(gradient_skip)
        writer.write(loss_skip)

    rows = _read_rows(path)
    assert tuple(rows[0]) == TRAINING_LOG_COLUMNS
    assert len(rows) == 4

    assert rows[1][0] == "0"
    assert float(rows[1][1]) == pytest.approx(0.1)
    assert float(rows[1][2]) == pytest.approx(2.0)
    assert float(rows[1][3]) == pytest.approx(0.5)
    assert float(rows[1][4]) == pytest.approx(1.5)
    assert rows[1][5] == "0"
    assert rows[1][6] == ""
    assert float(rows[1][7]) == pytest.approx(65536.0)

    assert rows[2][0] == "1"
    assert math.isinf(float(rows[2][2]))
    assert rows[2][3] == ""
    assert rows[2][5] == "1"
    assert rows[2][6] == "non_finite_grad"

    assert rows[3][2] == ""
    assert math.isnan(float(rows[3][4]))
    assert rows[3][5] == "1"
    assert rows[3][6] == "non_finite_loss"


def test_writer_flushes_header_and_rows_before_exit(tmp_path: Path) -> None:
    path = tmp_path / "live.csv"
    row = StepLog(
        step=7,
        lr=0.05,
        loss=0.25,
        grad_l2_pre_clip=0.75,
        grad_l2_post_clip=0.25,
        scaler_scale=4096.0,
        clipped=True,
    )

    with StepLogWriter(path) as writer:
        writer.write(row)
        visible = _read_rows(path)
        assert visible[0] == list(TRAINING_LOG_COLUMNS)
        assert len(visible) == 2
        assert visible[1][0] == "7"


def test_writer_rejects_writes_outside_context(tmp_path: Path) -> None:
    path = tmp_path / "closed.csv"
    writer = StepLogWriter(path)
    with pytest.raises(RuntimeError, match="not open"):
        writer.write(
            StepLog(
                step=0,
                lr=0.0,
                loss=1.0,
                grad_l2_pre_clip=1.0,
                grad_l2_post_clip=1.0,
                scaler_scale=1.0,
                clipped=False,
            )
        )

    with writer:
        pass
    with pytest.raises(RuntimeError, match="not open"):
        writer.write(
            SkipLog(
                step=0,
                lr=0.0,
                loss=1.0,
                scaler_scale=1.0,
                reason="non_finite_loss",
                grad_l2_pre_clip=None,
            )
        )
