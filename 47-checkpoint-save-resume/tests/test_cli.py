from __future__ import annotations

from pathlib import Path

import pytest

from main import main
from support import read_json


def test_cli_writes_report_and_skill(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workdir = tmp_path / "artifacts"
    exit_code = main(
        [
            "--workdir",
            str(workdir),
            "--total-steps",
            "6",
            "--interrupt-at",
            "3",
            "--shards",
            "2",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "single:" in captured.out
    assert "sharded:" in captured.out
    assert "passed=yes" in captured.out
    assert f"wrote {(workdir / 'resume-demo.json').as_posix()}" in captured.out
    payload = read_json(workdir / "resume-demo.json")
    assert payload["single"]["passed"] is True
    assert payload["sharded"]["passed"] is True
    assert (workdir / "skill-checkpoint-save-resume.md").is_file()


def test_cli_invalid_interrupt_returns_two_without_workdir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workdir = tmp_path / "never-created"
    exit_code = main(
        [
            "--workdir",
            str(workdir),
            "--total-steps",
            "3",
            "--interrupt-at",
            "3",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "error:" in captured.err
    assert not workdir.exists()


def test_cli_rejects_shards_above_demo_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workdir = tmp_path / "never-created"
    exit_code = main(
        [
            "--workdir",
            str(workdir),
            "--shards",
            "5",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "error:" in captured.err
    assert not workdir.exists()


def test_cli_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    assert "Atomic checkpoint" in capsys.readouterr().out
