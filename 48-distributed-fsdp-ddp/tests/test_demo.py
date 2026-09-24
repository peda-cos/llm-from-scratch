from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import pytest

from main import REPORT_FILENAME, SKILL_FILENAME, main
from support import read_json, run_cli


def _run_report(workdir: Path, *args: str) -> dict[str, Any]:
    result = run_cli("--workdir", str(workdir), *args)
    assert result.returncode == 0, result.stderr
    return read_json(workdir / REPORT_FILENAME)


def _ranks(report: dict[str, Any]) -> list[dict[str, Any]]:
    ranks = report["ranks"]
    assert isinstance(ranks, list)
    return ranks


def _assert_all_checks_pass(report: dict[str, Any]) -> None:
    checks = report["checks"]
    assert isinstance(checks, dict)
    assert checks["all_checks_passed"] is True
    assert all(bool(value) for value in checks.values())


def test_default_demo_proves_equivalence(tmp_path: Path) -> None:
    report = _run_report(tmp_path, "--steps", "3", "--world-size", "2")

    assert report["schema"] == "ddp-demo/v1"
    _assert_all_checks_pass(report)
    evidence = report["evidence"]
    assert isinstance(evidence, dict)
    assert float(evidence["gradient_max_abs_diff"]) < 1e-3
    assert float(evidence["loss_max_abs_diff"]) < 1e-3
    assert float(evidence["final_parameter_max_abs_diff"]) < 1e-3

    ranks = _ranks(report)
    assert len(ranks) == 2
    assert {int(rank["rank"]) for rank in ranks} == {0, 1}
    global_losses: list[list[float]] = []
    for rank in ranks:
        steps = rank["steps"]
        assert isinstance(steps, list)
        assert len(steps) == 3
        assert float(rank["state_max_abs_diff"]) < 1e-6
        assert float(rank["buffer_after_broadcast"]) == 1.0
        assert rank["steps_agreed"] is True
        fsdp = rank["fsdp"]
        assert isinstance(fsdp, dict)
        assert fsdp["bit_equal"] is True
        assert fsdp["reshard_bit_equal"] is True
        global_losses.append([float(step["global_loss"]) for step in steps])

    assert global_losses[0] == pytest.approx(global_losses[1], abs=1e-9)
    assert (tmp_path / SKILL_FILENAME).is_file()


def test_global_loss_is_the_mean_of_local_losses(tmp_path: Path) -> None:
    report = _run_report(tmp_path, "--steps", "1", "--world-size", "2")
    ranks = _ranks(report)
    step_zero = [rank["steps"][0] for rank in ranks]
    local_losses = [float(step["local_loss"]) for step in step_zero]
    global_loss = float(step_zero[0]["global_loss"])

    assert abs(statistics.fmean(local_losses) - global_loss) < 1e-6
    assert any(
        abs(float(step["local_loss"]) - float(step["global_loss"])) > 1e-6
        for step in step_zero
    )


def test_world_size_four_stays_in_sync(tmp_path: Path) -> None:
    report = _run_report(tmp_path, "--world-size", "4", "--steps", "2")
    ranks = _ranks(report)

    assert len(ranks) == 4
    _assert_all_checks_pass(report)
    for rank in ranks:
        assert float(rank["state_max_abs_diff"]) < 1e-6
        fsdp = rank["fsdp"]
        assert isinstance(fsdp, dict)
        assert fsdp["bit_equal"] is True
        assert fsdp["reshard_bit_equal"] is True
        assert float(fsdp["shard_fraction"]) < 0.5


def test_avg_reduce_matches_sum_reference(tmp_path: Path) -> None:
    sum_report = _run_report(
        tmp_path / "sum", "--steps", "2", "--world-size", "2", "--reduce", "sum"
    )
    avg_report = _run_report(
        tmp_path / "avg", "--steps", "2", "--world-size", "2", "--reduce", "avg"
    )

    for report in (sum_report, avg_report):
        _assert_all_checks_pass(report)
        evidence = report["evidence"]
        assert isinstance(evidence, dict)
        assert float(evidence["gradient_max_abs_diff"]) < 1e-3
        assert float(evidence["final_parameter_max_abs_diff"]) < 1e-3

    sum_ranks = _ranks(sum_report)
    avg_ranks = _ranks(avg_report)
    assert len(sum_ranks) == len(avg_ranks) == 2
    for sum_rank, avg_rank in zip(sum_ranks, avg_ranks):
        assert float(sum_rank["parameter_sum_after"]) == pytest.approx(
            float(avg_rank["parameter_sum_after"]), abs=1e-4
        )
        assert float(sum_rank["final_global_loss"]) == pytest.approx(
            float(avg_rank["final_global_loss"]), abs=1e-4
        )


def test_world_size_one_is_the_degenerate_reference(tmp_path: Path) -> None:
    report = _run_report(tmp_path, "--world-size", "1", "--steps", "2")
    ranks = _ranks(report)

    assert len(ranks) == 1
    _assert_all_checks_pass(report)
    fsdp = ranks[0]["fsdp"]
    assert isinstance(fsdp, dict)
    assert float(fsdp["shard_fraction"]) == 1.0
    assert int(fsdp["padded_params"]) == 0
    assert int(fsdp["padding_numel"]) == 0


def test_cli_invalid_config_exits_two_without_workdir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workdir = tmp_path / "never-created"
    exit_code = main(["--world-size", "0", "--workdir", str(workdir)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "error:" in captured.err
    assert not workdir.exists()

    result = run_cli(
        "--reduce", "mean", "--workdir", str(tmp_path / "never-created-either")
    )
    assert result.returncode == 2
