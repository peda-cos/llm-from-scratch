from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from main import _parse_accum_steps, main
from support import read_json, write_config


def test_cli_writes_curve_and_skill(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.json"
    workdir = tmp_path / "artifacts"
    write_config(config_path)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--workdir",
            str(workdir),
            "--num-steps",
            "2",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "equivalence:" in captured.out
    assert "passed=yes" in captured.out
    assert "samples_per_sec" in captured.out
    assert "sync_calls" in captured.out
    assert f"wrote {(workdir / 'accum-curve.json').as_posix()}" in captured.out
    assert (
        f"wrote {(workdir / 'skill-gradient-accumulation.md').as_posix()}"
        in captured.out
    )
    curve_path = workdir / "accum-curve.json"
    skill_path = workdir / "skill-gradient-accumulation.md"
    assert curve_path.is_file()
    assert skill_path.is_file()
    payload = read_json(curve_path)
    assert all(point["num_steps"] == 2 for point in payload["points"])


def test_cli_overrides_micro_batch_and_accum_steps(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.json"
    workdir = tmp_path / "artifacts"
    write_config(config_path)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--workdir",
            str(workdir),
            "--micro-batch",
            "4",
            "--accum-steps",
            "1,2",
        ]
    )
    payload = read_json(workdir / "accum-curve.json")

    assert exit_code == 0
    assert [point["effective_batch"] for point in payload["points"]] == [4, 8]


def test_cli_invalid_config_returns_two_without_workdir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.json"
    workdir = tmp_path / "never-created"
    write_config(config_path, typo_key=1)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--workdir",
            str(workdir),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "error:" in captured.err
    assert not workdir.exists()


def test_cli_invalid_override_returns_two(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    workdir = tmp_path / "never-created"
    write_config(config_path)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--workdir",
            str(workdir),
            "--num-steps",
            "0",
        ]
    )

    assert exit_code == 2
    assert not workdir.exists()


def test_invalid_accum_steps_exits_two() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--accum-steps", "1,,2"])

    assert exc_info.value.code == 2


def test_parse_accum_steps_accepts_whitespace() -> None:
    assert _parse_accum_steps("1, 2, 4") == (1, 2, 4)

    with pytest.raises(argparse.ArgumentTypeError, match="positive integers"):
        _parse_accum_steps("1,,2")
