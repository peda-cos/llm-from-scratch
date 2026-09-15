from __future__ import annotations

import csv
from pathlib import Path

import pytest

from main import DEFAULT_CONFIG_PATH, main
from support import write_config

ARTIFACT_NAMES = (
    "training_log.csv",
    "schedule.csv",
    "lr_schedule.txt",
    "run_config.json",
)


def _read_rows(path: Path) -> list[list[str]]:
    return list(csv.reader(path.read_text(encoding="utf-8").splitlines()))


def test_demo_is_deterministic_and_writes_contract_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workdir = tmp_path / "run"
    argv = [
        "--config",
        str(DEFAULT_CONFIG_PATH),
        "--steps",
        "8",
        "--workdir",
        str(workdir),
        "--seed",
        "11",
    ]

    assert main(argv) == 0
    first = capsys.readouterr()
    first_artifacts = {
        name: (workdir / name).read_bytes() for name in ARTIFACT_NAMES
    }

    assert main(argv) == 0
    second = capsys.readouterr()
    second_artifacts = {
        name: (workdir / name).read_bytes() for name in ARTIFACT_NAMES
    }

    assert first.out == second.out
    assert first.err == second.err == ""
    assert first_artifacts == second_artifacts
    assert len(_read_rows(workdir / "training_log.csv")) == 9
    assert len(_read_rows(workdir / "schedule.csv")) == 22
    plot_text = (workdir / "lr_schedule.txt").read_text(encoding="utf-8")
    assert first.out.endswith(plot_text)


def test_default_demo_reduces_loss_and_prints_boundaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workdir = tmp_path / "default"
    assert main(["--workdir", str(workdir)]) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    assert "cosine lr with linear warmup" in captured.out
    assert "config: warmup_steps=4 total_steps=20 " in captured.out
    assert "model: parameters=15 features=4 classes=3 batch=48" in captured.out
    assert "boundaries: lr(0)=0.000000 lr(warmup=4)=0.050000 " in captured.out
    assert "lr(total=20)=0.005000 lr(total+1)=0.005000" in captured.out
    assert "summary: steps=20 " in captured.out
    assert "lr schedule: steps 0..22" in captured.out
    assert captured.out.count("step=") == 20

    rows = _read_rows(workdir / "training_log.csv")
    assert rows[0] == ["step", "lr", "grad_l2_norm", "loss"]
    assert len(rows) == 21
    losses = [float(row[3]) for row in rows[1:]]
    assert losses[-1] < losses[0]

    schedule_rows = _read_rows(workdir / "schedule.csv")
    assert schedule_rows[0] == ["step", "lr"]
    assert len(schedule_rows) == 22
    for name in ARTIFACT_NAMES:
        assert (workdir / name).stat().st_size > 0


def test_main_rejects_invalid_config_and_step_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    invalid_documents = (
        '{"warmup_steps": 4, "total_steps": 20, "lr_max": 0.05}',
        "{not json",
        '{"warmup_steps": 4, "total_steps": 20, "lr_max": 0.05, '
        '"lr_min": 0.005, "extra": 1}',
    )
    for index, document in enumerate(invalid_documents):
        config_path = tmp_path / f"invalid-{index}.json"
        config_path.write_text(document, encoding="utf-8")
        workdir = tmp_path / f"invalid-run-{index}"
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

    workdir = tmp_path / "zero-steps"
    assert (
        main(
            [
                "--config",
                str(DEFAULT_CONFIG_PATH),
                "--steps",
                "0",
                "--workdir",
                str(workdir),
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error: steps must be >= 1" in captured.err
    assert not workdir.exists()


def test_main_rejects_invalid_workdir(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workdir = tmp_path / "not-a-directory"
    workdir.write_text("file", encoding="utf-8")
    assert main(["--workdir", str(workdir)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err


def test_config_written_by_helper_loads(tmp_path: Path) -> None:
    path = tmp_path / "config" / "schedule.json"
    write_config(path, warmup_steps=2, total_steps=5)
    workdir = tmp_path / "small-run"
    assert (
        main(
            [
                "--config",
                str(path),
                "--workdir",
                str(workdir),
                "--steps",
                "3",
            ]
        )
        == 0
    )
    rows = _read_rows(workdir / "training_log.csv")
    assert len(rows) == 4
    assert len(_read_rows(workdir / "schedule.csv")) == 7
