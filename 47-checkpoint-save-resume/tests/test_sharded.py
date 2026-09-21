from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest
import torch

import main
from main import load_sharded_checkpoint, save_sharded_checkpoint
from support import make_training_objects, read_json, run_training_steps


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _save_valid(
    directory: Path,
    steps: int = 1,
    num_shards: int = 3,
) -> tuple[Any, Any, Any, Any, Path]:
    model, optimizer, scheduler, state = make_training_objects()
    state = run_training_steps(model, optimizer, scheduler, state, steps)
    index_path = save_sharded_checkpoint(
        directory,
        model,
        optimizer,
        scheduler,
        state,
        num_shards,
    )
    return model, optimizer, scheduler, state, index_path


def test_sharded_round_robin_keys_and_hashes(tmp_path: Path) -> None:
    model, _, _, _, index_path = _save_valid(tmp_path / "checkpoint")
    index = read_json(index_path)
    keys = list(model.state_dict())

    assert index["shard_count"] == 3
    assert len(index["shards"]) == 3
    seen: list[str] = []
    for shard_index, entry in enumerate(index["shards"]):
        shard_path = index_path.parent / entry["file"]
        assert _sha256(shard_path) == entry["sha256"]
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        assert set(shard) == set(keys[shard_index::3])
        seen.extend(shard)
    assert set(seen) == set(keys)

    meta_path = index_path.parent / index["meta"]["file"]
    assert _sha256(meta_path) == index["meta"]["sha256"]


def test_sharded_round_trip_restores_state(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    model, optimizer, scheduler, state, _ = _save_valid(directory, steps=2)

    parameter = next(model.parameters())
    saved_parameter = parameter.detach().clone()
    saved_exp_avg = optimizer.state[parameter]["exp_avg"].clone()
    saved_lr = scheduler.get_last_lr()[0]

    run_training_steps(model, optimizer, scheduler, state, 2)
    loaded = load_sharded_checkpoint(directory, model, optimizer, scheduler)

    assert loaded == state
    assert torch.equal(parameter.detach(), saved_parameter)
    assert torch.equal(optimizer.state[parameter]["exp_avg"], saved_exp_avg)
    assert scheduler.get_last_lr()[0] == pytest.approx(saved_lr)


def test_sharded_shard_hash_mismatch_raises(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    model, optimizer, scheduler, _, index_path = _save_valid(directory)
    index = read_json(index_path)
    shard_path = index_path.parent / index["shards"][0]["file"]
    with shard_path.open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_sharded_checkpoint(directory, model, optimizer, scheduler)


def test_sharded_meta_hash_mismatch_raises(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    model, optimizer, scheduler, _, index_path = _save_valid(directory)
    index = read_json(index_path)
    meta_path = index_path.parent / index["meta"]["file"]
    with meta_path.open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_sharded_checkpoint(directory, model, optimizer, scheduler)


@pytest.mark.parametrize("bad_name", ["../escape.pt", "/etc/passwd"])
def test_sharded_rejects_path_escape(
    tmp_path: Path,
    bad_name: str,
) -> None:
    directory = tmp_path / "checkpoint"
    model, optimizer, scheduler, _, index_path = _save_valid(directory)
    index = read_json(index_path)
    index["shards"][0]["file"] = bad_name
    main.atomic_write_json(index_path, index)

    with pytest.raises(ValueError, match="escapes directory"):
        load_sharded_checkpoint(directory, model, optimizer, scheduler)


def test_sharded_rejects_symlink_escape(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    model, optimizer, scheduler, _, index_path = _save_valid(directory)
    outside = tmp_path / "outside.pt"
    torch.save({"value": torch.tensor([1])}, outside)
    link = directory / "link.pt"
    link.symlink_to(outside)
    index = read_json(index_path)
    index["shards"][0]["file"] = "link.pt"
    main.atomic_write_json(index_path, index)

    with pytest.raises(ValueError, match="escapes directory"):
        load_sharded_checkpoint(directory, model, optimizer, scheduler)


def test_sharded_rejects_bad_shard_count(tmp_path: Path) -> None:
    model, optimizer, scheduler, state = make_training_objects()

    with pytest.raises(ValueError, match="num_shards"):
        save_sharded_checkpoint(
            tmp_path / "zero",
            model,
            optimizer,
            scheduler,
            state,
            0,
        )
    with pytest.raises(ValueError, match="num_shards"):
        save_sharded_checkpoint(
            tmp_path / "too-many",
            model,
            optimizer,
            scheduler,
            state,
            5,
        )


def test_old_generation_stays_loadable_after_new_save(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    model, optimizer, scheduler, state, index_path = _save_valid(directory, steps=1)
    old_index = index_path.read_bytes()
    old_step = state.step

    state = run_training_steps(model, optimizer, scheduler, state, 1)
    save_sharded_checkpoint(
        directory,
        model,
        optimizer,
        scheduler,
        state,
        3,
    )
    index_path.write_bytes(old_index)

    loaded = load_sharded_checkpoint(directory, model, optimizer, scheduler)

    assert loaded.step == old_step == 1


def test_duplicate_shard_keys_raise(tmp_path: Path) -> None:
    directory = tmp_path / "checkpoint"
    model, optimizer, scheduler, _, index_path = _save_valid(directory, num_shards=2)
    index = read_json(index_path)
    first_path = index_path.parent / index["shards"][0]["file"]
    second_path = index_path.parent / index["shards"][1]["file"]
    first_shard = cast(
        dict[str, torch.Tensor],
        torch.load(first_path, map_location="cpu", weights_only=True),
    )
    second_shard = cast(
        dict[str, torch.Tensor],
        torch.load(second_path, map_location="cpu", weights_only=True),
    )
    duplicate_key = next(iter(first_shard))
    second_shard[duplicate_key] = first_shard[duplicate_key]
    main.atomic_save(second_path, second_shard)
    index["shards"][1]["sha256"] = _sha256(second_path)
    main.atomic_write_json(index_path, index)

    with pytest.raises(ValueError, match="duplicate key"):
        load_sharded_checkpoint(directory, model, optimizer, scheduler)


def test_missing_index_raises(tmp_path: Path) -> None:
    model, optimizer, scheduler, _ = make_training_objects()

    with pytest.raises(FileNotFoundError, match="index"):
        load_sharded_checkpoint(tmp_path / "missing", model, optimizer, scheduler)
