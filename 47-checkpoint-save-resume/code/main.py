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
from torch.optim.lr_scheduler import LRScheduler

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
