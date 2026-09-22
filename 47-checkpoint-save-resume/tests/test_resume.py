from __future__ import annotations

from pathlib import Path

import pytest

from main import (
    DemoConfig,
    _seed_training_rng,
    build_demo,
    load_checkpoint,
    render_skill_recipe,
    run_demo_steps,
    run_resume_demo,
    save_checkpoint,
)
from support import read_json


def test_resume_demo_matches_baseline_single_and_sharded(tmp_path: Path) -> None:
    config = DemoConfig()
    report = run_resume_demo(config, tmp_path / "artifacts")

    assert report.config == config
    assert report.pre_interrupt_max_abs_diff < 1e-12
    assert len(report.baseline_losses) == config.total_steps
    assert report.single.passed
    assert report.sharded.passed
    assert report.single.max_abs_diff < config.tolerance
    assert report.sharded.max_abs_diff < config.tolerance
    assert report.single.resumed_steps == config.total_steps - config.interrupt_at
    assert report.sharded.resumed_steps == config.total_steps - config.interrupt_at
    assert report.single.checkpoint_bytes > 0
    assert report.sharded.checkpoint_bytes > report.single.checkpoint_bytes
    assert report.sharded.shard_count == config.shards
    assert report.sharded.index_sha256 is not None
    assert len(report.sharded.index_sha256) == 64
    assert {"python", "numpy", "torch_cpu"} <= set(report.rng_sources)
    assert (tmp_path / "artifacts" / "resume-demo.json").is_file()
    assert (tmp_path / "artifacts" / "skill-checkpoint-save-resume.md").is_file()
    assert (tmp_path / "artifacts" / "demo-config.json").is_file()


def test_report_json_is_serializable_and_schema_versioned(tmp_path: Path) -> None:
    workdir = tmp_path / "artifacts"
    run_resume_demo(DemoConfig(), workdir)
    payload = read_json(workdir / "resume-demo.json")

    assert payload["schema_version"] == 1
    assert payload["config"]["total_steps"] == 12
    assert payload["single"]["format"] == "single"
    assert payload["sharded"]["format"] == "sharded"
    assert payload["single"]["passed"] is True
    assert payload["sharded"]["passed"] is True
    assert len(payload["sharded"]["index_sha256"]) == 64
    assert len(payload["baseline_losses"]) == 12


def test_skill_recipe_is_deterministic() -> None:
    config = DemoConfig()
    first = render_skill_recipe(config)
    second = render_skill_recipe(config)

    assert first == second
    for heading in (
        "## Payload shape",
        "## Atomic write",
        "## RNG capture and restore",
        "## Sharded layout",
    ):
        assert heading in first
    assert "max_abs_diff" not in first
    assert "wall_time" not in first


def test_resume_demo_rejects_interrupt_at_total_steps() -> None:
    with pytest.raises(ValueError, match="interrupt_at"):
        DemoConfig(total_steps=3, interrupt_at=3)


def test_mid_epoch_cursor_is_preserved(tmp_path: Path) -> None:
    config = DemoConfig()
    model, optimizer, scheduler, state = build_demo(config)
    _seed_training_rng(config.seed)
    state, _ = run_demo_steps(
        config,
        model,
        optimizer,
        scheduler,
        state,
        config.interrupt_at,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, state)

    loaded = load_checkpoint(path, model, optimizer, scheduler)

    assert loaded.step == 7
    assert loaded.epoch == 1
    assert loaded.batch_in_epoch == 3
    assert len(loaded.losses) == 7
