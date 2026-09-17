from __future__ import annotations

import json
from pathlib import Path

import pytest

from main import (
    DEFAULT_CONFIG_PATH,
    SkipRateMonitor,
    TrainConfig,
    load_train_config,
)
from support import write_config


def test_committed_config_loads_expected_values() -> None:
    config = load_train_config(DEFAULT_CONFIG_PATH)

    assert config.lr == pytest.approx(0.1)
    assert config.max_grad_norm == pytest.approx(0.5)
    assert config.init_scale == pytest.approx(65536.0)
    assert config.growth_interval == 4
    assert config.backoff_factor == pytest.approx(0.5)
    assert config.steps == 20
    assert config.seed == 7
    assert config.inject_inf_step == 5
    assert config.autocast_dtype == "float16"
    assert config.autocast_dtype_torch() is not None
    assert config.to_dict()["steps"] == 20


def test_config_rejects_missing_unknown_and_invalid_values(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps({"lr": 0.1}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing train config keys"):
        load_train_config(missing)

    unknown = tmp_path / "unknown.json"
    write_config(unknown, surprise=1)
    with pytest.raises(ValueError, match="unexpected train config keys"):
        load_train_config(unknown)

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid train config"):
        load_train_config(invalid_json)

    not_object = tmp_path / "array.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_train_config(not_object)

    bool_steps = tmp_path / "bool.json"
    write_config(bool_steps, steps=True)
    with pytest.raises(ValueError, match="steps must be an int"):
        load_train_config(bool_steps)

    bad_dtype = tmp_path / "dtype.json"
    write_config(bad_dtype, autocast_dtype="float32")
    with pytest.raises(ValueError, match="float16"):
        load_train_config(bad_dtype)

    bad_backoff = tmp_path / "backoff.json"
    write_config(bad_backoff, backoff_factor=1.0)
    with pytest.raises(ValueError, match="backoff_factor"):
        load_train_config(bad_backoff)

    bad_growth = tmp_path / "growth.json"
    write_config(bad_growth, growth_interval=0)
    with pytest.raises(ValueError, match="growth_interval"):
        load_train_config(bad_growth)

    bad_scale = tmp_path / "scale.json"
    write_config(bad_scale, init_scale=0.0)
    with pytest.raises(ValueError, match="init_scale must be > 0"):
        load_train_config(bad_scale)

    bad_norm = tmp_path / "norm.json"
    write_config(bad_norm, max_grad_norm=0.0)
    with pytest.raises(ValueError, match="max_grad_norm must be > 0"):
        load_train_config(bad_norm)


def test_config_accepts_null_injection_and_bf16() -> None:
    config = TrainConfig(
        lr=0.0,
        max_grad_norm=1.0,
        init_scale=2.0,
        growth_interval=1,
        backoff_factor=0.5,
        steps=1,
        seed=0,
        inject_inf_step=None,
        autocast_dtype="bfloat16",
    )
    assert config.inject_inf_step is None
    assert config.autocast_dtype_torch() is not None


def test_skip_rate_monitor_requires_full_window_then_latches() -> None:
    monitor = SkipRateMonitor(window=3, threshold=0.5, fail_after=2)

    assert monitor.record(True) is False
    assert monitor.record(True) is False
    assert monitor.record(True) is False
    assert monitor.rate == pytest.approx(1.0)
    assert monitor.record(True) is True
    assert monitor.alerted is True
    assert monitor.record(False) is True


def test_skip_rate_monitor_resets_streak_below_threshold() -> None:
    monitor = SkipRateMonitor(window=3, threshold=0.5, fail_after=3)

    monitor.record(True)
    monitor.record(True)
    monitor.record(True)
    assert monitor.alerted is False
    monitor.record(False)
    assert monitor.rate == pytest.approx(2.0 / 3.0)
    assert monitor.alerted is False
    monitor.record(False)
    assert monitor.rate == pytest.approx(1.0 / 3.0)
    monitor.record(False)
    monitor.record(True)
    assert monitor.alerted is False


def test_skip_rate_monitor_validates_parameters() -> None:
    with pytest.raises(ValueError, match="window must be >= 1"):
        SkipRateMonitor(window=0)
    with pytest.raises(ValueError, match="threshold"):
        SkipRateMonitor(threshold=1.5)
    with pytest.raises(ValueError, match="fail_after must be >= 1"):
        SkipRateMonitor(fail_after=0)
