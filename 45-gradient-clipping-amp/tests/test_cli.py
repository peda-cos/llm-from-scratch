from __future__ import annotations

import csv
from pathlib import Path

import pytest

from main import TRAINING_LOG_COLUMNS, main
from support import write_config


def _read_rows(path: Path) -> list[list[str]]:
    return list(
        csv.reader(path.read_text(encoding="utf-8").splitlines())
    )


def _tagged_lines(stdout: str) -> list[str]:
    return [
        line
        for line in stdout.splitlines()
        if line.startswith(("STEP", "SKIP"))
    ]


def test_default_demo_prints_steps_and_one_skip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workdir = tmp_path / "default"
    assert main(["--workdir", str(workdir)]) == 0
    captured = capsys.readouterr()

    assert "error:" not in captured.err
    assert "gradient clipping + mixed precision" in captured.out
    assert "precision=float16" in captured.out
    assert "inject_inf_step=5" in captured.out
    assert "SKIP step=005" in captured.out
    assert "summary: steps=20 skipped=1 skip_rate=0.0500" in captured.out

    tagged = _tagged_lines(captured.out)
    assert len(tagged) == 20
    assert sum(line.startswith("SKIP") for line in tagged) == 1

    rows = _read_rows(workdir / "training_log.csv")
    assert tuple(rows[0]) == TRAINING_LOG_COLUMNS
    assert len(rows) == 21
    assert rows[6][0] == "5"
    assert rows[6][5] == "1"
    assert rows[6][6] == "non_finite_grad"
    assert rows[6][3] == ""
    assert float(rows[7][7]) == pytest.approx(
        float(rows[6][7]) * 0.5
    )
    losses = [float(row[4]) for row in rows[1:]]
    assert losses[-1] < losses[0]
    assert (workdir / "run_config.json").stat().st_size > 0


def test_demo_is_deterministic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workdir = tmp_path / "repeat"
    argv = ["--workdir", str(workdir), "--steps", "12"]

    assert main(argv) == 0
    first = capsys.readouterr()
    first_csv = (workdir / "training_log.csv").read_bytes()

    assert main(argv) == 0
    second = capsys.readouterr()
    second_csv = (workdir / "training_log.csv").read_bytes()

    assert first.out == second.out
    assert first.err == second.err
    assert first_csv == second_csv


def test_bf16_without_injection_has_no_skips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workdir = tmp_path / "bf16"
    assert (
        main(
            [
                "--workdir",
                str(workdir),
                "--no-inject",
                "--bf16",
            ]
        )
        == 0
    )
    captured = capsys.readouterr()

    assert "precision=bfloat16" in captured.out
    assert "inject_inf_step=none" in captured.out
    assert "skipped=0 skip_rate=0.0000" in captured.out
    rows = _read_rows(workdir / "training_log.csv")
    assert len(rows) == 21
    assert sum(row[5] == "1" for row in rows[1:]) == 0


def test_invalid_config_returns_two_without_run_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "bad.json"
    config_path.write_text('{"lr": 0.1}', encoding="utf-8")
    workdir = tmp_path / "never-created"

    assert (
        main(
            [
                "--config",
                str(config_path),
                "--workdir",
                str(workdir),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert not workdir.exists()


def test_skip_rate_failure_flag_returns_one_after_writing_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workdir = tmp_path / "alert"
    assert (
        main(
            [
                "--workdir",
                str(workdir),
                "--fail-on-skip-rate",
                "--skip-rate-window",
                "2",
                "--skip-rate-threshold",
                "0.0",
                "--skip-rate-fail-after",
                "1",
            ]
        )
        == 1
    )
    captured = capsys.readouterr()
    assert "alert=yes" in captured.out
    assert "rolling skip rate exceeded" in captured.err
    rows = _read_rows(workdir / "training_log.csv")
    assert len(rows) == 21
    assert any(row[5] == "1" for row in rows[1:])


def test_invalid_cli_overrides_return_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.json"
    write_config(config_path)
    for index, extra in enumerate(
        (
            ["--steps", "0"],
            ["--device", "not-a-device"],
            ["--skip-rate-window", "0"],
        )
    ):
        assert (
            main(
                [
                    "--config",
                    str(config_path),
                    "--workdir",
                    str(tmp_path / f"bad-{index}"),
                    *extra,
                ]
            )
            == 2
        )
        captured = capsys.readouterr()
        assert "error:" in captured.err
