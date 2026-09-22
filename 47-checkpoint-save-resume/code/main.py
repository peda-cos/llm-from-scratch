from __future__ import annotations

import argparse
import errno
import hashlib
import hmac
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler

CHECKPOINT_SCHEMA = "checkpoint/v1"
INDEX_SCHEMA = "checkpoint-index/v1"
SCHEMA_VERSION = 1
TOLERANCE = 1e-4
DEFAULT_WORKDIR = Path("outputs")
DEMO_CONFIG_FILENAME = "demo-config.json"
REPORT_FILENAME = "resume-demo.json"
SKILL_FILENAME = "skill-checkpoint-save-resume.md"
NUMPY_SEED_LIMIT = 2**32 - 1


def _require_int(
    name: str,
    value: object,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _require_float(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_min: bool = False,
    strict_max: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None:
        too_small = result <= minimum if strict_min else result < minimum
        if too_small:
            comparator = ">" if strict_min else ">="
            raise ValueError(f"{name} must be {comparator} {minimum}")
    if maximum is not None:
        too_large = result >= maximum if strict_max else result > maximum
        if too_large:
            comparator = "<" if strict_max else "<="
            raise ValueError(f"{name} must be {comparator} {maximum}")
    return result


def _require_str(name: str, value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _require_mapping(name: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    payload = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in payload):
        raise ValueError(f"{name} must use string keys")
    return cast(Mapping[str, object], payload)


def _require_int_list(
    name: str,
    value: object,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return [
        _require_int(f"{name}[{index}]", item, minimum=minimum, maximum=maximum)
        for index, item in enumerate(value)
    ]


def _parse_python_rng(value: object) -> tuple[int, tuple[int, ...], float | None]:
    payload = _require_mapping("rng.python", value)
    version = _require_int("rng.python.version", payload.get("version"), minimum=0)
    states = tuple(
        _require_int_list("rng.python.state", payload.get("state"), minimum=0)
    )
    gauss_raw = payload.get("gauss_next")
    gauss_next = (
        None
        if gauss_raw is None
        else _require_float("rng.python.gauss_next", gauss_raw)
    )
    return version, states, gauss_next


def _parse_numpy_rng(value: object) -> tuple[str, np.ndarray, int, int, float]:
    payload = _require_mapping("rng.numpy", value)
    bit_generator = _require_str(
        "rng.numpy.bit_generator", payload.get("bit_generator")
    )
    if bit_generator != "MT19937":
        raise ValueError(f"unsupported NumPy bit generator: {bit_generator!r}")
    key = _require_int_list(
        "rng.numpy.key",
        payload.get("key"),
        minimum=0,
        maximum=NUMPY_SEED_LIMIT,
    )
    if not key:
        raise ValueError("rng.numpy.key must not be empty")
    pos = _require_int("rng.numpy.pos", payload.get("pos"), minimum=0)
    has_gauss = _require_int("rng.numpy.has_gauss", payload.get("has_gauss"), minimum=0)
    cached_raw = payload.get("cached_gaussian")
    cached_gaussian = (
        0.0
        if cached_raw is None
        else _require_float("rng.numpy.cached_gaussian", cached_raw)
    )
    return (
        bit_generator,
        np.asarray(key, dtype=np.uint32),
        pos,
        has_gauss,
        cached_gaussian,
    )


def _parse_torch_cpu_rng(value: object) -> list[int]:
    state = _require_int_list("rng.torch_cpu", value, minimum=0, maximum=255)
    if not state:
        raise ValueError("rng.torch_cpu must not be empty")
    return state


def _parse_torch_cuda_rng(value: object) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError("rng.torch_cuda must be a list")
    states: list[list[int]] = []
    for index, raw_state in enumerate(value):
        states.append(
            _require_int_list(
                f"rng.torch_cuda[{index}]",
                raw_state,
                minimum=0,
                maximum=255,
            )
        )
    return states


def capture_rng_state() -> dict[str, object]:
    """Capture Python, NumPy, and Torch RNG state as plain Python values."""
    python_version, python_state, python_gauss = random.getstate()
    numpy_name, numpy_key, numpy_pos, numpy_has_gauss, numpy_cached = cast(
        tuple[str, np.ndarray, int, int, float], np.random.get_state()
    )
    if torch.cuda.is_available():
        torch_cuda = [
            [int(value) for value in device_state.tolist()]
            for device_state in torch.cuda.get_rng_state_all()
        ]
    else:
        torch_cuda = []
    return {
        "python": {
            "version": int(python_version),
            "state": [int(value) for value in python_state],
            "gauss_next": None if python_gauss is None else float(python_gauss),
        },
        "numpy": {
            "bit_generator": str(numpy_name),
            "key": [int(value) for value in numpy_key.tolist()],
            "pos": int(numpy_pos),
            "has_gauss": int(numpy_has_gauss),
            "cached_gaussian": float(numpy_cached),
        },
        "torch_cpu": [int(value) for value in torch.get_rng_state().tolist()],
        "torch_cuda": torch_cuda,
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""
    payload = _require_mapping("rng", state)
    python_version, python_state, python_gauss = _parse_python_rng(
        payload.get("python")
    )
    numpy_name, numpy_key, numpy_pos, numpy_has_gauss, numpy_cached = (
        _parse_numpy_rng(payload.get("numpy"))
    )
    torch_cpu = _parse_torch_cpu_rng(payload.get("torch_cpu"))
    torch_cuda = _parse_torch_cuda_rng(payload.get("torch_cuda"))
    if torch_cuda:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(torch_cuda) != torch.cuda.device_count():
            raise ValueError(
                "checkpoint CUDA RNG state does not match the visible device count"
            )

    try:
        random.setstate((python_version, python_state, python_gauss))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Python RNG state: {exc}") from exc
    try:
        np.random.set_state(
            (numpy_name, numpy_key, numpy_pos, numpy_has_gauss, numpy_cached)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid NumPy RNG state: {exc}") from exc
    try:
        torch.set_rng_state(torch.tensor(torch_cpu, dtype=torch.uint8))
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid Torch CPU RNG state: {exc}") from exc
    if torch_cuda:
        try:
            torch.cuda.set_rng_state_all(
                [torch.tensor(values, dtype=torch.uint8) for values in torch_cuda]
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid Torch CUDA RNG state: {exc}") from exc


def _seed_training_rng(seed: int) -> None:
    seed = _require_int("seed", seed, minimum=0, maximum=NUMPY_SEED_LIMIT)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        try:
            os.fsync(fd)
        except OSError as exc:
            # shortcut: directory fsync is unsupported on a few filesystems.
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(fd)


def _atomic_write(path: Path, writer: Callable[[BinaryIO], None]) -> None:
    """Write through a same-directory temp file, fsync, then atomically replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_save(path: Path, payload: object) -> None:
    _atomic_write(path, lambda handle: torch.save(payload, handle))


def atomic_write_json(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )

    def write(handle: BinaryIO) -> None:
        handle.write(encoded)

    _atomic_write(path, write)


def _atomic_write_text(path: Path, text: str) -> None:
    encoded = text.encode("utf-8")

    def write(handle: BinaryIO) -> None:
        handle.write(encoded)

    _atomic_write(path, write)


TRAIN_STATE_KEYS = frozenset(
    {"step", "epoch", "batch_in_epoch", "batches_per_epoch", "losses"}
)
CHECKPOINT_KEYS = frozenset(
    {"schema", "model", "optimizer", "scheduler", "train_state", "rng", "wall_saved_at"}
)


@dataclass(frozen=True)
class TrainState:
    step: int
    epoch: int
    batch_in_epoch: int
    batches_per_epoch: int
    losses: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _require_int("step", self.step, minimum=0)
        _require_int("epoch", self.epoch, minimum=0)
        _require_int("batch_in_epoch", self.batch_in_epoch, minimum=0)
        _require_int("batches_per_epoch", self.batches_per_epoch, minimum=1)
        if self.batch_in_epoch >= self.batches_per_epoch:
            raise ValueError("batch_in_epoch must be < batches_per_epoch")
        if not isinstance(self.losses, tuple):
            raise ValueError("losses must be a tuple")
        normalized = tuple(
            _require_float("losses[]", loss) for loss in self.losses
        )
        object.__setattr__(self, "losses", normalized)

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "batch_in_epoch": self.batch_in_epoch,
            "batches_per_epoch": self.batches_per_epoch,
            "losses": list(self.losses),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> TrainState:
        keys = set(payload)
        if keys != TRAIN_STATE_KEYS:
            missing = sorted(TRAIN_STATE_KEYS - keys)
            unknown = sorted(keys - TRAIN_STATE_KEYS)
            raise ValueError(
                "train_state keys mismatch: "
                f"missing={missing}, unknown={unknown}"
            )
        raw_losses = payload.get("losses")
        if not isinstance(raw_losses, list):
            raise ValueError("train_state.losses must be a list")
        return cls(
            step=_require_int("train_state.step", payload.get("step"), minimum=0),
            epoch=_require_int("train_state.epoch", payload.get("epoch"), minimum=0),
            batch_in_epoch=_require_int(
                "train_state.batch_in_epoch",
                payload.get("batch_in_epoch"),
                minimum=0,
            ),
            batches_per_epoch=_require_int(
                "train_state.batches_per_epoch",
                payload.get("batches_per_epoch"),
                minimum=1,
            ),
            losses=tuple(
                _require_float("train_state.losses[]", value)
                for value in raw_losses
            ),
        )

    def advance(self, loss: float) -> TrainState:
        value = _require_float("loss", loss)
        step = self.step + 1
        if self.batch_in_epoch + 1 >= self.batches_per_epoch:
            epoch = self.epoch + 1
            batch_in_epoch = 0
        else:
            epoch = self.epoch
            batch_in_epoch = self.batch_in_epoch + 1
        return TrainState(
            step=step,
            epoch=epoch,
            batch_in_epoch=batch_in_epoch,
            batches_per_epoch=self.batches_per_epoch,
            losses=self.losses + (value,),
        )


def _require_schema(
    payload: Mapping[str, object],
    expected: str,
    label: str,
) -> None:
    schema = payload.get("schema")
    if schema != expected:
        raise ValueError(
            f"{label} has unsupported schema {schema!r}; expected {expected!r}"
        )


def _validate_wall_saved_at(value: object) -> None:
    _require_float("wall_saved_at", value, minimum=0.0)


def _torch_load(path: Path) -> object:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint file not found: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"cannot load checkpoint {path}: {exc}") from exc


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    train_state: TrainState,
) -> dict[str, object]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_state": train_state.as_dict(),
        "rng": capture_rng_state(),
        "wall_saved_at": time.time(),
    }


def _apply_training_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    *,
    model_state: object,
    optimizer_state: object,
    scheduler_state: object,
    train_state_payload: object,
    rng_payload: object,
) -> TrainState:
    model_mapping = _require_mapping("model state", model_state)
    optimizer_mapping = _require_mapping("optimizer state", optimizer_state)
    scheduler_mapping = _require_mapping("scheduler state", scheduler_state)
    rng_mapping = _require_mapping("rng state", rng_payload)
    train_state = TrainState.from_dict(
        _require_mapping("train_state", train_state_payload)
    )

    try:
        model.load_state_dict(model_mapping)
    except RuntimeError as exc:
        raise ValueError(f"model state mismatch: {exc}") from exc
    try:
        optimizer.load_state_dict(dict(optimizer_mapping))
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ValueError(f"optimizer state mismatch: {exc}") from exc
    try:
        scheduler.load_state_dict(dict(scheduler_mapping))
    except (KeyError, RuntimeError, ValueError) as exc:
        raise ValueError(f"scheduler state mismatch: {exc}") from exc
    restore_rng_state(rng_mapping)
    return train_state


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    train_state: TrainState,
) -> None:
    atomic_save(
        path,
        _checkpoint_payload(model, optimizer, scheduler, train_state),
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
) -> TrainState:
    payload = _require_mapping("checkpoint", _torch_load(path))
    _require_schema(payload, CHECKPOINT_SCHEMA, "checkpoint")
    missing = sorted(CHECKPOINT_KEYS - set(payload))
    if missing:
        raise ValueError(f"checkpoint is missing keys: {', '.join(missing)}")
    _validate_wall_saved_at(payload.get("wall_saved_at"))
    return _apply_training_payload(
        model,
        optimizer,
        scheduler,
        model_state=payload.get("model"),
        optimizer_state=payload.get("optimizer"),
        scheduler_state=payload.get("scheduler"),
        train_state_payload=payload.get("train_state"),
        rng_payload=payload.get("rng"),
    )


META_KEYS = frozenset(
    {
        "schema",
        "shard_count",
        "optimizer",
        "scheduler",
        "train_state",
        "rng",
        "wall_saved_at",
    }
)


def _sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _resolve_in_directory(base: Path, name: str) -> Path:
    if not name or Path(name).is_absolute():
        raise ValueError(f"checkpoint path escapes directory: {name!r}")
    try:
        candidate = (base / name).resolve()
    except OSError as exc:
        raise ValueError(f"invalid checkpoint path {name!r}: {exc}") from exc
    if not candidate.is_relative_to(base.resolve()):
        raise ValueError(f"checkpoint path escapes directory: {name!r}")
    if not candidate.is_file():
        raise ValueError(f"missing checkpoint file: {name}")
    return candidate


def _verify_sha256(path: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str):
        raise ValueError(f"sha256 for {label} must be a string")
    actual = _sha256_file(path)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            f"sha256 mismatch for {label} {path.name}: "
            f"expected {expected}, got {actual}"
        )


def _parse_index_entry(value: object, label: str) -> tuple[str, str]:
    entry = _require_mapping(label, value)
    return (
        _require_str(f"{label}.file", entry.get("file")),
        _require_str(f"{label}.sha256", entry.get("sha256")),
    )


def save_sharded_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    train_state: TrainState,
    num_shards: int,
) -> Path:
    count = _require_int("num_shards", num_shards, minimum=1)
    state_dict = model.state_dict()
    if count > len(state_dict):
        raise ValueError(
            f"num_shards must be <= {len(state_dict)} model state entries"
        )
    directory.mkdir(parents=True, exist_ok=True)
    token = f"step{train_state.step:08d}-{uuid.uuid4().hex[:12]}"

    shards: list[dict[str, Tensor]] = [{} for _ in range(count)]
    for index, (key, value) in enumerate(state_dict.items()):
        shards[index % count][key] = value

    shard_entries: list[dict[str, str]] = []
    for index, shard in enumerate(shards):
        shard_path = directory / f"model.shard-{index:03d}.{token}.pt"
        atomic_save(shard_path, shard)
        shard_entries.append(
            {"file": shard_path.name, "sha256": _sha256_file(shard_path)}
        )

    meta_path = directory / f"meta.{token}.pt"
    meta_payload: dict[str, object] = {
        "schema": CHECKPOINT_SCHEMA,
        "shard_count": count,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_state": train_state.as_dict(),
        "rng": capture_rng_state(),
        "wall_saved_at": time.time(),
    }
    atomic_save(meta_path, meta_payload)

    index_payload = {
        "schema": INDEX_SCHEMA,
        "shard_count": count,
        "shards": shard_entries,
        "meta": {
            "file": meta_path.name,
            "sha256": _sha256_file(meta_path),
        },
    }
    index_path = directory / "index.json"
    atomic_write_json(index_path, index_payload)
    return index_path


def load_sharded_checkpoint(
    directory: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
) -> TrainState:
    index_path = directory / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"checkpoint index not found: {index_path}")
    try:
        raw_index = json.loads(index_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read checkpoint index {index_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in checkpoint index {index_path}: {exc}") from exc

    index_payload = _require_mapping("checkpoint index", raw_index)
    _require_schema(index_payload, INDEX_SCHEMA, "checkpoint index")
    shard_count = _require_int(
        "index.shard_count",
        index_payload.get("shard_count"),
        minimum=1,
    )
    raw_shards = index_payload.get("shards")
    if not isinstance(raw_shards, list):
        raise ValueError("index.shards must be a list")
    shard_entries = [
        _parse_index_entry(value, f"index.shards[{index}]")
        for index, value in enumerate(raw_shards)
    ]
    if len(shard_entries) != shard_count:
        raise ValueError(
            f"index.shard_count is {shard_count} but {len(shard_entries)} shards are listed"
        )
    meta_file, meta_sha = _parse_index_entry(index_payload.get("meta"), "index.meta")

    base = directory.resolve()
    shard_paths = [
        (name, digest, _resolve_in_directory(base, name))
        for name, digest in shard_entries
    ]
    meta_path = _resolve_in_directory(base, meta_file)

    for name, digest, path in shard_paths:
        _verify_sha256(path, digest, f"shard {name}")
    _verify_sha256(meta_path, meta_sha, f"meta {meta_file}")

    meta_payload = _require_mapping("checkpoint meta", _torch_load(meta_path))
    _require_schema(meta_payload, CHECKPOINT_SCHEMA, "checkpoint meta")
    missing = sorted(META_KEYS - set(meta_payload))
    if missing:
        raise ValueError(f"checkpoint meta is missing keys: {', '.join(missing)}")
    meta_shard_count = _require_int(
        "meta.shard_count",
        meta_payload.get("shard_count"),
        minimum=1,
    )
    if meta_shard_count != shard_count:
        raise ValueError(
            f"meta.shard_count is {meta_shard_count} but index has {shard_count}"
        )
    _validate_wall_saved_at(meta_payload.get("wall_saved_at"))

    merged: dict[str, Tensor] = {}
    for name, _, path in shard_paths:
        shard_payload = _require_mapping(f"shard {name}", _torch_load(path))
        for key, value in shard_payload.items():
            if not isinstance(value, Tensor):
                raise ValueError(f"shard {name} key {key!r} is not a tensor")
            if key in merged:
                raise ValueError(f"duplicate key across shards: {key!r}")
            merged[key] = value

    return _apply_training_payload(
        model,
        optimizer,
        scheduler,
        model_state=merged,
        optimizer_state=meta_payload.get("optimizer"),
        scheduler_state=meta_payload.get("scheduler"),
        train_state_payload=meta_payload.get("train_state"),
        rng_payload=meta_payload.get("rng"),
    )


DEMO_CONFIG_KEYS = frozenset(
    {
        "total_steps",
        "interrupt_at",
        "shards",
        "seed",
        "batch_size",
        "batches_per_epoch",
        "num_samples",
        "feature_dim",
        "hidden_dim",
        "lr",
        "dropout",
        "tolerance",
    }
)


@dataclass(frozen=True)
class DemoConfig:
    total_steps: int = 12
    interrupt_at: int = 7
    shards: int = 3
    seed: int = 7
    batch_size: int = 8
    batches_per_epoch: int = 4
    num_samples: int = 32
    feature_dim: int = 6
    hidden_dim: int = 12
    lr: float = 0.05
    dropout: float = 0.2
    tolerance: float = TOLERANCE

    def __post_init__(self) -> None:
        _require_int("total_steps", self.total_steps, minimum=1)
        _require_int("interrupt_at", self.interrupt_at, minimum=1)
        _require_int("shards", self.shards, minimum=1, maximum=4)
        _require_int("seed", self.seed, minimum=0, maximum=NUMPY_SEED_LIMIT)
        _require_int("batch_size", self.batch_size, minimum=1)
        _require_int("batches_per_epoch", self.batches_per_epoch, minimum=1)
        _require_int("num_samples", self.num_samples, minimum=1)
        _require_int("feature_dim", self.feature_dim, minimum=1)
        _require_int("hidden_dim", self.hidden_dim, minimum=1)
        _require_float("lr", self.lr, minimum=0.0, strict_min=True)
        _require_float("dropout", self.dropout, minimum=0.0, maximum=1.0, strict_max=True)
        _require_float("tolerance", self.tolerance, minimum=0.0, strict_min=True)
        if self.interrupt_at >= self.total_steps:
            raise ValueError("interrupt_at must be < total_steps")
        if self.num_samples != self.batch_size * self.batches_per_epoch:
            raise ValueError(
                "num_samples must equal batch_size * batches_per_epoch"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "total_steps": self.total_steps,
            "interrupt_at": self.interrupt_at,
            "shards": self.shards,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "batches_per_epoch": self.batches_per_epoch,
            "num_samples": self.num_samples,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "dropout": self.dropout,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> DemoConfig:
        keys = set(payload)
        if keys != DEMO_CONFIG_KEYS:
            missing = sorted(DEMO_CONFIG_KEYS - keys)
            unknown = sorted(keys - DEMO_CONFIG_KEYS)
            raise ValueError(
                f"demo config keys mismatch: missing={missing}, unknown={unknown}"
            )
        return cls(
            total_steps=_require_int(
                "total_steps", payload.get("total_steps"), minimum=1
            ),
            interrupt_at=_require_int(
                "interrupt_at", payload.get("interrupt_at"), minimum=1
            ),
            shards=_require_int(
                "shards", payload.get("shards"), minimum=1, maximum=4
            ),
            seed=_require_int(
                "seed",
                payload.get("seed"),
                minimum=0,
                maximum=NUMPY_SEED_LIMIT,
            ),
            batch_size=_require_int(
                "batch_size", payload.get("batch_size"), minimum=1
            ),
            batches_per_epoch=_require_int(
                "batches_per_epoch", payload.get("batches_per_epoch"), minimum=1
            ),
            num_samples=_require_int(
                "num_samples", payload.get("num_samples"), minimum=1
            ),
            feature_dim=_require_int(
                "feature_dim", payload.get("feature_dim"), minimum=1
            ),
            hidden_dim=_require_int(
                "hidden_dim", payload.get("hidden_dim"), minimum=1
            ),
            lr=_require_float("lr", payload.get("lr"), minimum=0.0, strict_min=True),
            dropout=_require_float(
                "dropout",
                payload.get("dropout"),
                minimum=0.0,
                maximum=1.0,
                strict_max=True,
            ),
            tolerance=_require_float(
                "tolerance", payload.get("tolerance"), minimum=0.0, strict_min=True
            ),
        )


def build_demo(
    config: DemoConfig,
) -> tuple[nn.Module, torch.optim.Optimizer, CosineAnnealingLR, TrainState]:
    torch.manual_seed(config.seed)
    model: nn.Module = nn.Sequential(
        nn.Linear(config.feature_dim, config.hidden_dim),
        nn.ReLU(),
        nn.Dropout(config.dropout),
        nn.Linear(config.hidden_dim, 2),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.total_steps)
    state = TrainState(
        step=0,
        epoch=0,
        batch_in_epoch=0,
        batches_per_epoch=config.batches_per_epoch,
        losses=(),
    )
    return model, optimizer, scheduler, state


def _demo_dataset(config: DemoConfig) -> tuple[Tensor, Tensor]:
    generator = torch.Generator().manual_seed(config.seed)
    targets = torch.arange(config.num_samples, dtype=torch.long) % 2
    centers = torch.zeros(2, config.feature_dim)
    centers[0, 0] = -1.5
    centers[1, 0] = 1.5
    if config.feature_dim > 1:
        centers[0, 1:] = 0.25
        centers[1, 1:] = -0.25
    inputs = centers[targets] + 0.5 * torch.randn(
        config.num_samples,
        config.feature_dim,
        generator=generator,
    )
    return inputs, targets


def _demo_epoch_indices(config: DemoConfig, epoch: int) -> Tensor:
    generator = torch.Generator().manual_seed(config.seed * 1_000 + epoch)
    return torch.randperm(config.num_samples, generator=generator)


def _demo_batch(
    config: DemoConfig,
    epoch: int,
    batch_in_epoch: int,
) -> tuple[Tensor, Tensor]:
    if not 0 <= batch_in_epoch < config.batches_per_epoch:
        raise ValueError("batch_in_epoch must be < batches_per_epoch")
    inputs, targets = _demo_dataset(config)
    indices = _demo_epoch_indices(config, epoch)
    start = batch_in_epoch * config.batch_size
    selected = indices[start : start + config.batch_size]
    return inputs[selected], targets[selected]


def _train_demo_step(
    config: DemoConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    state: TrainState,
) -> tuple[TrainState, float]:
    inputs, targets = _demo_batch(config, state.epoch, state.batch_in_epoch)
    noise = np.random.standard_normal((config.batch_size, config.feature_dim))
    scale = 0.95 + 0.05 * random.random()
    batch = (inputs + torch.as_tensor(noise, dtype=torch.float32)) * scale
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(batch), targets)
    torch.autograd.backward(loss)
    optimizer.step()
    scheduler.step()
    value = float(loss.item())
    return state.advance(value), value


def run_demo_steps(
    config: DemoConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LRScheduler,
    state: TrainState,
    steps: int,
) -> tuple[TrainState, tuple[float, ...]]:
    count = _require_int("steps", steps, minimum=0)
    losses: list[float] = []
    for _ in range(count):
        state, loss = _train_demo_step(config, model, optimizer, scheduler, state)
        losses.append(loss)
    return state, tuple(losses)


def _configure_demo_runtime() -> None:
    torch.set_num_threads(1)


def _max_abs_diff(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("loss trajectories must have the same length")
    if not left:
        return 0.0
    return max(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def _rng_sources() -> tuple[str, ...]:
    state = capture_rng_state()
    sources = ["python", "numpy", "torch_cpu"]
    cuda_state = state.get("torch_cuda")
    if isinstance(cuda_state, list) and cuda_state:
        sources.append("torch_cuda")
    return tuple(sources)


def _pollute_rng(seed: int) -> None:
    pollution = (seed + 101) % (NUMPY_SEED_LIMIT + 1)
    random.seed(pollution)
    np.random.seed(pollution)
    torch.manual_seed(pollution)
    _ = random.random()
    _ = np.random.standard_normal(17)
    _ = torch.rand(17)


def _resume_worker_case(
    config: DemoConfig,
    checkpoint_path: Path,
    loader: Callable[
        [Path, nn.Module, torch.optim.Optimizer, LRScheduler],
        TrainState,
    ],
) -> tuple[tuple[float, ...], int]:
    model, optimizer, scheduler, _ = build_demo(config)
    _pollute_rng(config.seed)
    state = loader(checkpoint_path, model, optimizer, scheduler)
    remaining = config.total_steps - state.step
    if remaining < 0:
        raise ValueError("checkpoint step exceeds total_steps")
    state, losses = run_demo_steps(
        config,
        model,
        optimizer,
        scheduler,
        state,
        remaining,
    )
    return losses, state.step


def load_demo_config(path: Path) -> DemoConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read demo config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in demo config {path}: {exc}") from exc
    return DemoConfig.from_dict(_require_mapping("demo config", raw))


def _worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="resume-worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--sharded", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_demo_config(cast(Path, args.config))
    _configure_demo_runtime()

    single_losses, single_step = _resume_worker_case(
        config,
        cast(Path, args.single),
        load_checkpoint,
    )
    sharded_losses, sharded_step = _resume_worker_case(
        config,
        cast(Path, args.sharded),
        load_sharded_checkpoint,
    )
    payload = {
        "single": {
            "losses": list(single_losses),
            "final_step": single_step,
        },
        "sharded": {
            "losses": list(sharded_losses),
            "final_step": sharded_step,
        },
    }
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


def _run_worker(
    config_path: Path,
    single_path: Path,
    sharded_dir: Path,
) -> Mapping[str, object]:
    script = Path(__file__).resolve()
    command = [
        sys.executable,
        str(script),
        "worker",
        "--config",
        str(config_path.resolve()),
        "--single",
        str(single_path.resolve()),
        "--sharded",
        str(sharded_dir.resolve()),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("resume worker timed out") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"resume worker failed with exit {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    lines = completed.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("resume worker produced no output")
    try:
        raw = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid resume worker output: {exc}") from exc
    return _require_mapping("worker output", raw)


def _parse_worker_case(
    payload: Mapping[str, object],
    name: str,
    expected_steps: int,
    expected_final_step: int,
) -> tuple[float, ...]:
    section = _require_mapping(f"worker.{name}", payload.get(name))
    raw_losses = section.get("losses")
    if not isinstance(raw_losses, list):
        raise RuntimeError(f"worker.{name}.losses must be a list")
    losses = tuple(
        _require_float(f"worker.{name}.losses[]", value) for value in raw_losses
    )
    final_step = _require_int(
        f"worker.{name}.final_step",
        section.get("final_step"),
        minimum=0,
    )
    if len(losses) != expected_steps:
        raise RuntimeError(
            f"worker.{name} returned {len(losses)} losses, expected {expected_steps}"
        )
    if final_step != expected_final_step:
        raise RuntimeError(
            f"worker.{name} finished at step {final_step}, expected {expected_final_step}"
        )
    return losses


@dataclass(frozen=True)
class ResumeCase:
    format: Literal["single", "sharded"]
    max_abs_diff: float
    tolerance: float
    passed: bool
    resumed_steps: int
    post_losses: tuple[float, ...]
    baseline_post_losses: tuple[float, ...]
    checkpoint_bytes: int
    shard_count: int
    index_sha256: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "max_abs_diff": self.max_abs_diff,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "resumed_steps": self.resumed_steps,
            "post_losses": list(self.post_losses),
            "baseline_post_losses": list(self.baseline_post_losses),
            "checkpoint_bytes": self.checkpoint_bytes,
            "shard_count": self.shard_count,
            "index_sha256": self.index_sha256,
        }


@dataclass(frozen=True)
class ResumeDemoReport:
    config: DemoConfig
    rng_sources: tuple[str, ...]
    pre_interrupt_max_abs_diff: float
    baseline_losses: tuple[float, ...]
    single: ResumeCase
    sharded: ResumeCase

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": self.config.as_dict(),
            "rng_sources": list(self.rng_sources),
            "pre_interrupt_max_abs_diff": self.pre_interrupt_max_abs_diff,
            "baseline_losses": list(self.baseline_losses),
            "single": self.single.as_dict(),
            "sharded": self.sharded.as_dict(),
        }


def _sharded_report(index_path: Path) -> tuple[int, int, str]:
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read sharded index {index_path}: {exc}") from exc
    index = _require_mapping("checkpoint index", raw)
    _require_schema(index, INDEX_SCHEMA, "checkpoint index")
    shard_count = _require_int(
        "index.shard_count", index.get("shard_count"), minimum=1
    )
    raw_shards = index.get("shards")
    if not isinstance(raw_shards, list):
        raise RuntimeError("checkpoint index shards must be a list")
    entries = [
        _parse_index_entry(value, f"index.shards[{position}]")
        for position, value in enumerate(raw_shards)
    ]
    meta_file, _ = _parse_index_entry(index.get("meta"), "index.meta")
    base = index_path.parent.resolve()
    shard_paths = [
        _resolve_in_directory(base, name) for name, _ in entries
    ]
    meta_path = _resolve_in_directory(base, meta_file)
    total = index_path.stat().st_size
    total += sum(path.stat().st_size for path in shard_paths)
    total += meta_path.stat().st_size
    return total, shard_count, _sha256_file(index_path)


def _make_case(
    config: DemoConfig,
    name: Literal["single", "sharded"],
    baseline_losses: tuple[float, ...],
    resumed_losses: tuple[float, ...],
    checkpoint_bytes: int,
    shard_count: int,
    index_sha256: str | None,
) -> ResumeCase:
    baseline_post = baseline_losses[config.interrupt_at :]
    diff = _max_abs_diff(baseline_post, resumed_losses)
    return ResumeCase(
        format=name,
        max_abs_diff=diff,
        tolerance=config.tolerance,
        passed=diff < config.tolerance,
        resumed_steps=len(resumed_losses),
        post_losses=resumed_losses,
        baseline_post_losses=baseline_post,
        checkpoint_bytes=checkpoint_bytes,
        shard_count=shard_count,
        index_sha256=index_sha256,
    )


def run_resume_demo(
    config: DemoConfig,
    workdir: Path = DEFAULT_WORKDIR,
) -> ResumeDemoReport:
    _configure_demo_runtime()
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    config_path = workdir / DEMO_CONFIG_FILENAME
    atomic_write_json(config_path, config.as_dict())

    baseline_model, baseline_optimizer, baseline_scheduler, baseline_state = (
        build_demo(config)
    )
    _seed_training_rng(config.seed)
    _, baseline_losses = run_demo_steps(
        config,
        baseline_model,
        baseline_optimizer,
        baseline_scheduler,
        baseline_state,
        config.total_steps,
    )

    model, optimizer, scheduler, state = build_demo(config)
    _seed_training_rng(config.seed)
    state, pre_losses = run_demo_steps(
        config,
        model,
        optimizer,
        scheduler,
        state,
        config.interrupt_at,
    )
    if len(pre_losses) != config.interrupt_at:
        raise RuntimeError("interrupted run did not execute the requested steps")
    pre_diff = _max_abs_diff(baseline_losses[: config.interrupt_at], pre_losses)
    if pre_diff >= 1e-12:
        raise RuntimeError(
            f"baseline and interrupted prefix diverged by {pre_diff:.3e}"
        )
    if state.step != config.interrupt_at:
        raise RuntimeError("interrupted run counter does not match interrupt_at")

    single_path = workdir / "checkpoint.pt"
    save_checkpoint(single_path, model, optimizer, scheduler, state)
    sharded_dir = workdir / "checkpoint-sharded"
    sharded_index = save_sharded_checkpoint(
        sharded_dir,
        model,
        optimizer,
        scheduler,
        state,
        config.shards,
    )

    worker_payload = _run_worker(config_path, single_path, sharded_dir)
    expected_steps = config.total_steps - config.interrupt_at
    single_losses = _parse_worker_case(
        worker_payload,
        "single",
        expected_steps,
        config.total_steps,
    )
    sharded_losses = _parse_worker_case(
        worker_payload,
        "sharded",
        expected_steps,
        config.total_steps,
    )

    sharded_bytes, sharded_count, sharded_sha = _sharded_report(sharded_index)
    single_case = _make_case(
        config,
        "single",
        baseline_losses,
        single_losses,
        single_path.stat().st_size,
        1,
        None,
    )
    sharded_case = _make_case(
        config,
        "sharded",
        baseline_losses,
        sharded_losses,
        sharded_bytes,
        sharded_count,
        sharded_sha,
    )
    if not single_case.passed or not sharded_case.passed:
        raise RuntimeError(
            "resume mismatch: "
            f"single={single_case.max_abs_diff:.3e}, "
            f"sharded={sharded_case.max_abs_diff:.3e}"
        )

    report = ResumeDemoReport(
        config=config,
        rng_sources=_rng_sources(),
        pre_interrupt_max_abs_diff=pre_diff,
        baseline_losses=baseline_losses,
        single=single_case,
        sharded=sharded_case,
    )
    atomic_write_json(workdir / REPORT_FILENAME, report.as_dict())
    _atomic_write_text(workdir / SKILL_FILENAME, render_skill_recipe(config))
    return report


def format_report(report: ResumeDemoReport) -> str:
    config = report.config
    baseline = ", ".join(f"{value:.6f}" for value in report.baseline_losses)
    single = report.single
    sharded = report.sharded
    return "\n".join(
        [
            (
                f"config: total_steps={config.total_steps} "
                f"interrupt_at={config.interrupt_at} "
                f"epoch={config.interrupt_at // config.batches_per_epoch} "
                f"batch_in_epoch={config.interrupt_at % config.batches_per_epoch} "
                f"shards={config.shards} seed={config.seed}"
            ),
            f"rng: {', '.join(report.rng_sources)}",
            f"baseline_losses: {baseline}",
            (
                f"single: max_abs_diff={single.max_abs_diff:.3e} "
                f"tolerance={single.tolerance:.1e} "
                f"passed={'yes' if single.passed else 'no'} "
                f"resumed_steps={single.resumed_steps} "
                f"bytes={single.checkpoint_bytes}"
            ),
            (
                f"sharded: max_abs_diff={sharded.max_abs_diff:.3e} "
                f"tolerance={sharded.tolerance:.1e} "
                f"passed={'yes' if sharded.passed else 'no'} "
                f"resumed_steps={sharded.resumed_steps} "
                f"bytes={sharded.checkpoint_bytes} "
                f"index_sha256={sharded.index_sha256}"
            ),
        ]
    )


def render_skill_recipe(config: DemoConfig) -> str:
    return f"""# Skill: Checkpoint Save and Resume

Use this recipe when a training run must survive a kill and continue mid-epoch.

## Payload shape

```python
{{
    "schema": "checkpoint/v1",
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict(),
    "train_state": {{
        "step": step,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "batches_per_epoch": batches_per_epoch,
        "losses": list(losses),
    }},
    "rng": capture_rng_state(),
    "wall_saved_at": time.time(),
}}
```

The schema string is the migration hook. Bump it when the layout changes and
dispatch on it in the loader.

## Atomic write

1. Write to a unique temp file in the target directory.
2. Flush and `os.fsync` the temp file.
3. `os.replace(temp, target)`.
4. `os.fsync` the directory descriptor.
5. On failure, unlink the temp; never write into the target in place.

```python
fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{{target.name}}.", suffix=".tmp")
with os.fdopen(fd, "wb") as handle:
    torch.save(payload, handle)
    handle.flush()
    os.fsync(handle.fileno())
os.replace(tmp, target)
```

## RNG capture and restore

Capture Python `random.getstate`, NumPy `np.random.get_state`, Torch CPU
`torch.get_rng_state`, and Torch CUDA `torch.cuda.get_rng_state_all`. Store
plain numbers and lists only: NumPy's key array becomes `tolist()`, Torch's
byte tensor becomes a list of ints. Restore all sources after the model,
optimizer, and scheduler are loaded and immediately before the next draw.

## Sharded layout

```text
checkpoint/
  index.json
  meta.<generation>.pt
  model.shard-000.<generation>.pt
  model.shard-001.<generation>.pt
```

Round-robin state-dict keys across shards. Put optimizer, scheduler, train
state, RNG, and `shard_count` in the meta payload. Write every shard and the
meta atomically, hash each file with SHA-256, and write `index.json` last:

```json
{{
  "schema": "checkpoint-index/v1",
  "shard_count": 2,
  "shards": [
    {{"file": "model.shard-000.<generation>.pt", "sha256": "..."}},
    {{"file": "model.shard-001.<generation>.pt", "sha256": "..."}}
  ],
  "meta": {{"file": "meta.<generation>.pt", "sha256": "..."}}
}}
```

The index is the commit point. New filenames per save keep the previous index
valid until the atomic replace. The loader resolves each path inside the
checkpoint directory, verifies every SHA-256, and only then merges the shards.

## Config used by this recipe

- total_steps: {config.total_steps}
- interrupt_at: {config.interrupt_at}
- shards: {config.shards}
- seed: {config.seed}
- batch_size: {config.batch_size}
- batches_per_epoch: {config.batches_per_epoch}
- num_samples: {config.num_samples}
- feature_dim: {config.feature_dim}
- hidden_dim: {config.hidden_dim}
- lr: {config.lr}
- dropout: {config.dropout}
- tolerance: {config.tolerance}

## Wire it into a trainer

At the periodic save site, pass the live model, optimizer, scheduler, counters,
and loss history to `save_checkpoint`; on startup, construct the same objects,
call `load_checkpoint`, and continue from `state.step` and
`(state.epoch, state.batch_in_epoch)`. Restore the RNG inside the loader; do not
replay training steps to reach the cursor.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomic checkpoint save and mid-epoch resume demo."
    )
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--interrupt-at", type=int, default=None)
    parser.add_argument("--shards", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _config_from_args(args: argparse.Namespace) -> DemoConfig:
    base = DemoConfig()
    total_steps = cast(int | None, args.total_steps)
    interrupt_at = cast(int | None, args.interrupt_at)
    shards = cast(int | None, args.shards)
    seed = cast(int | None, args.seed)
    return DemoConfig(
        total_steps=base.total_steps if total_steps is None else total_steps,
        interrupt_at=base.interrupt_at if interrupt_at is None else interrupt_at,
        shards=base.shards if shards is None else shards,
        seed=base.seed if seed is None else seed,
        batch_size=base.batch_size,
        batches_per_epoch=base.batches_per_epoch,
        num_samples=base.num_samples,
        feature_dim=base.feature_dim,
        hidden_dim=base.hidden_dim,
        lr=base.lr,
        dropout=base.dropout,
        tolerance=base.tolerance,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "worker":
        return _worker_main(arguments[1:])

    parser = _build_parser()
    args = parser.parse_args(arguments)
    try:
        config = _config_from_args(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    workdir = cast(Path, args.workdir)
    try:
        report = run_resume_demo(config, workdir)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_report(report))
    print(f"wrote {(workdir / REPORT_FILENAME).as_posix()}")
    print(f"wrote {(workdir / SKILL_FILENAME).as_posix()}")
    print(f"wrote {(workdir / DEMO_CONFIG_FILENAME).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
