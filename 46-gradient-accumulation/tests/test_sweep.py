from __future__ import annotations

import math
from pathlib import Path

from main import (
    CurvePoint,
    render_skill_recipe,
    run_demo,
    sweep_effective_batches,
    write_curve,
)
from support import read_json, tiny_config


def test_curve_counters_and_batch_identity() -> None:
    config = tiny_config(accum_steps=(1, 2, 4), num_steps=2)
    report = sweep_effective_batches(config)

    assert [point.accum_steps for point in report.points] == [1, 2, 4]
    assert [point.effective_batch for point in report.points] == [4, 8, 16]
    assert all(point.world_size == 1 for point in report.points)
    for point in report.points:
        assert point.micro_batches == 2 * point.accum_steps
        assert point.optimizer_steps == 2
        assert point.sync_calls == 2
        assert point.skipped_syncs == 2 * (point.accum_steps - 1)
        assert point.samples == 4 * point.accum_steps * 2
        assert point.wall_time_sec > 0.0
        assert point.samples_per_sec > 0.0
        assert point.median_step_ms > 0.0
        assert all(
            math.isfinite(loss)
            for loss in (point.first_loss, point.avg_loss, point.final_loss)
        )

    effective_batches = [point.effective_batch for point in report.points]
    assert effective_batches == sorted(effective_batches)


def test_write_curve_creates_nested_parent_and_exact_schema(tmp_path: Path) -> None:
    config = tiny_config(accum_steps=(1, 2), num_steps=2)
    report = sweep_effective_batches(config)
    path = tmp_path / "nested" / "deeper" / "accum-curve.json"

    write_curve(path, report)
    payload = read_json(path)

    assert payload["schema_version"] == 1
    assert payload["config"]["accum_steps"] == [1, 2]
    assert payload["config"]["num_steps"] == 2
    assert len(payload["points"]) == 2
    expected_point_keys = {
        "accum_steps",
        "micro_batch",
        "effective_batch",
        "world_size",
        "num_steps",
        "micro_batches",
        "optimizer_steps",
        "sync_calls",
        "skipped_syncs",
        "samples",
        "wall_time_sec",
        "samples_per_sec",
        "median_step_ms",
        "first_loss",
        "avg_loss",
        "final_loss",
    }
    assert set(payload["points"][0]) == expected_point_keys


def test_demo_is_deterministic_except_for_timing(tmp_path: Path) -> None:
    config = tiny_config(accum_steps=(1, 2), num_steps=2)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_curve, first_skill, _ = run_demo(config, first_dir)
    second_curve, second_skill, _ = run_demo(config, second_dir)

    first_payload = read_json(first_curve)
    second_payload = read_json(second_curve)
    for payload in (first_payload, second_payload):
        for point in payload["points"]:
            del point["wall_time_sec"]
            del point["samples_per_sec"]
            del point["median_step_ms"]

    assert first_payload == second_payload
    assert first_skill.read_text(encoding="utf-8") == second_skill.read_text(
        encoding="utf-8"
    )


def test_skill_recipe_contains_contract(tmp_path: Path) -> None:
    config = tiny_config()
    text = render_skill_recipe(config)
    path = tmp_path / "skill.md"
    path.write_text(text, encoding="utf-8")

    assert "effective_batch = micro_batch * accum_steps * world_size" in text
    assert "loss / accum_steps" in text
    assert "no_sync" in text
    assert "accum-curve.json" in text
    assert f"micro_batch: {config.micro_batch}" in text


def test_curve_point_as_dict_has_all_fields() -> None:
    point = CurvePoint(
        accum_steps=2,
        micro_batch=4,
        effective_batch=8,
        world_size=1,
        num_steps=3,
        micro_batches=6,
        optimizer_steps=3,
        sync_calls=3,
        skipped_syncs=3,
        samples=24,
        wall_time_sec=1.0,
        samples_per_sec=24.0,
        median_step_ms=100.0,
        first_loss=1.0,
        avg_loss=0.5,
        final_loss=0.25,
    )

    payload = point.as_dict()

    assert payload["effective_batch"] == 8
    assert payload["optimizer_steps"] == 3
    assert len(payload) == 16
