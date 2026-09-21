from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import main

CODE_DIR = Path(__file__).resolve().parents[1] / "code"


def test_atomic_save_replaces_target_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.pt"
    main.atomic_save(target, {"value": torch.arange(3)})
    main.atomic_save(target, {"value": torch.arange(5)})

    loaded = torch.load(target, map_location="cpu", weights_only=True)
    assert loaded["value"].tolist() == [0, 1, 2, 3, 4]
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_write_json_round_trip_and_leaves_no_temp(tmp_path: Path) -> None:
    target = tmp_path / "index.json"
    main.atomic_write_json(target, {"shard_count": 2, "shards": ["a", "b"]})

    assert target.read_text(encoding="utf-8").endswith("\n")
    assert '"shard_count": 2' in target.read_text(encoding="utf-8")
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_save_creates_missing_parent(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "checkpoint.pt"
    main.atomic_save(target, {"value": torch.tensor([7])})

    assert target.is_file()
    assert torch.equal(
        torch.load(target, map_location="cpu", weights_only=True)["value"],
        torch.tensor([7]),
    )


def test_atomic_save_failure_keeps_previous_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "checkpoint.pt"
    main.atomic_save(target, {"value": torch.tensor([1])})
    before = target.read_bytes()

    def boom(payload: object, handle: object, *args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", boom)

    with pytest.raises(RuntimeError, match="disk full"):
        main.atomic_save(target, {"value": torch.tensor([2])})

    assert target.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_save_survives_process_kill_before_rename(tmp_path: Path) -> None:
    target = tmp_path / "checkpoint.pt"
    main.atomic_save(target, {"value": torch.arange(3)})
    before = target.read_bytes()

    script = """
import os
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
import main

def sabotage(payload, handle, *args, **kwargs):
    handle.write(b"partial garbage")
    handle.flush()
    os.fsync(handle.fileno())
    os._exit(17)

main.torch.save = sabotage
main.atomic_save(Path(sys.argv[1]), {"new": 1})
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(target), str(CODE_DIR)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(CODE_DIR)},
    )

    assert completed.returncode == 17
    assert target.read_bytes() == before
    loaded = torch.load(target, map_location="cpu", weights_only=True)
    assert loaded["value"].tolist() == [0, 1, 2]
    assert list(tmp_path.glob(".*.tmp"))
