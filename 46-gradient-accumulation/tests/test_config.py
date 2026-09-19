from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from main import OptimizerName, load_config
from support import tiny_config, write_config


def test_valid_config_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    write_config(
        path,
        micro_batch=2,
        accum_steps=[2, 4],
        optimizer="adamw",
        seed=13,
    )

    config = load_config(path)

    assert config.micro_batch == 2
    assert config.accum_steps == (2, 4)
    assert config.optimizer == "adamw"
    assert config.seed == 13
    assert config.as_dict()["accum_steps"] == [2, 4]


@pytest.mark.parametrize("key", ["lr", "micro_batch", "noise"])
def test_missing_key_is_rejected(tmp_path: Path, key: str) -> None:
    payload = tiny_config().as_dict()
    del payload[key]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=key):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    write_config(path, typo_key=1)

    with pytest.raises(ValueError, match="typo_key"):
        load_config(path)


@pytest.mark.parametrize(
    "accum_steps",
    [[], [2, 2], [4, 2], [0, 2], ["1", 2], [True, 2]],
)
def test_invalid_accum_steps_are_rejected(
    tmp_path: Path,
    accum_steps: list[object],
) -> None:
    path = tmp_path / "config.json"
    write_config(path, accum_steps=accum_steps)

    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("lr", 0),
        ("momentum", 1.0),
        ("noise", float("nan")),
        ("micro_batch", True),
        ("input_dim", 2),
    ],
)
def test_invalid_scalar_config_is_rejected(
    tmp_path: Path,
    key: str,
    value: object,
) -> None:
    path = tmp_path / "config.json"
    write_config(path, **{key: value})

    with pytest.raises(ValueError):
        load_config(path)


def test_missing_file_is_a_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read config"):
        load_config(tmp_path / "missing.json")


def test_replace_revalidates() -> None:
    config = tiny_config()

    with pytest.raises(ValueError, match="strictly increasing"):
        replace(config, accum_steps=(2, 1))

    with pytest.raises(ValueError, match="optimizer"):
        replace(config, optimizer=cast(OptimizerName, "lamb"))


def test_input_dim_must_cover_classes() -> None:
    with pytest.raises(ValueError, match="input_dim"):
        tiny_config(input_dim=2, num_classes=3)


def test_boolean_integer_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="micro_batch"):
        tiny_config(micro_batch=True)
