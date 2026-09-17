"""Global L2 clipping and mixed-precision training on a real optimizer step.

The module exposes a clip wrapper that reports pre- and post-clip norms, a
non-finite gradient scan, an autocast + GradScaler training step that skips
non-finite loss/gradient updates, incremental CSV logging, a rolling skip-rate
monitor, and a deterministic 20-step demo that injects an Inf gradient.

It is self-contained: no network, GPU requirement, pretrained weights, or
sibling lesson code is used. The pinned CPU build of PyTorch 2.15 enables
``torch.amp.GradScaler("cpu")``; that build is required for the scale assertions
in the tests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.amp import GradScaler

FEATURES = 4
CLASSES = 3
SAMPLES_PER_CLASS = 16
BATCH_SIZE = 48
DEFAULT_WORKDIR = Path("outputs/demo")
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "train.json"
)
TRAINING_LOG_COLUMNS = (
    "step",
    "lr",
    "grad_l2_pre_clip",
    "grad_l2_post_clip",
    "loss",
    "skipped",
    "skip_reason",
    "scaler_scale",
)
_CONFIG_KEYS = (
    "lr",
    "max_grad_norm",
    "init_scale",
    "growth_interval",
    "backoff_factor",
    "steps",
    "seed",
    "inject_inf_step",
    "autocast_dtype",
)
_AUTOCAST_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
SkipReason = Literal["non_finite_loss", "non_finite_grad"]


def _require_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_float(
    name: str,
    value: object,
    *,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if converted < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return converted


def _dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.bfloat16:
        return "bfloat16"
    raise ValueError("autocast dtype must be torch.float16 or torch.bfloat16")


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Validated settings for the demo and its training step."""

    lr: float
    max_grad_norm: float
    init_scale: float
    growth_interval: int
    backoff_factor: float
    steps: int
    seed: int
    inject_inf_step: int | None
    autocast_dtype: str

    def __post_init__(self) -> None:
        lr = _require_float("lr", self.lr, minimum=0.0)
        max_grad_norm = _require_float(
            "max_grad_norm", self.max_grad_norm, minimum=0.0
        )
        if max_grad_norm == 0.0:
            raise ValueError("max_grad_norm must be > 0")
        init_scale = _require_float(
            "init_scale", self.init_scale, minimum=0.0
        )
        if init_scale == 0.0:
            raise ValueError("init_scale must be > 0")
        growth_interval = _require_int(
            "growth_interval", self.growth_interval, minimum=1
        )
        backoff_factor = _require_float(
            "backoff_factor", self.backoff_factor, minimum=0.0, maximum=1.0
        )
        if not 0.0 < backoff_factor < 1.0:
            raise ValueError("backoff_factor must be in (0, 1)")
        steps = _require_int("steps", self.steps, minimum=1)
        seed = _require_int("seed", self.seed, minimum=0)
        inject_inf_step = (
            None
            if self.inject_inf_step is None
            else _require_int(
                "inject_inf_step", self.inject_inf_step, minimum=0
            )
        )
        if not isinstance(self.autocast_dtype, str):
            raise TypeError("autocast_dtype must be a string")
        if self.autocast_dtype not in _AUTOCAST_DTYPES:
            raise ValueError(
                "autocast_dtype must be 'float16' or 'bfloat16'"
            )
        object.__setattr__(self, "lr", lr)
        object.__setattr__(self, "max_grad_norm", max_grad_norm)
        object.__setattr__(self, "init_scale", init_scale)
        object.__setattr__(self, "growth_interval", growth_interval)
        object.__setattr__(self, "backoff_factor", backoff_factor)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "inject_inf_step", inject_inf_step)

    def to_dict(self) -> dict[str, object]:
        return {
            "lr": self.lr,
            "max_grad_norm": self.max_grad_norm,
            "init_scale": self.init_scale,
            "growth_interval": self.growth_interval,
            "backoff_factor": self.backoff_factor,
            "steps": self.steps,
            "seed": self.seed,
            "inject_inf_step": self.inject_inf_step,
            "autocast_dtype": self.autocast_dtype,
        }

    def autocast_dtype_torch(self) -> torch.dtype:
        return _AUTOCAST_DTYPES[self.autocast_dtype]

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> TrainConfig:
        expected = set(_CONFIG_KEYS)
        present = set(data)
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        if missing:
            raise ValueError("missing train config keys: " + ", ".join(missing))
        if unexpected:
            raise ValueError(
                "unexpected train config keys: " + ", ".join(unexpected)
            )
        return cls(
            lr=cast(float, data["lr"]),
            max_grad_norm=cast(float, data["max_grad_norm"]),
            init_scale=cast(float, data["init_scale"]),
            growth_interval=cast(int, data["growth_interval"]),
            backoff_factor=cast(float, data["backoff_factor"]),
            steps=cast(int, data["steps"]),
            seed=cast(int, data["seed"]),
            inject_inf_step=cast(int | None, data["inject_inf_step"]),
            autocast_dtype=cast(str, data["autocast_dtype"]),
        )


def load_train_config(path: Path) -> TrainConfig:
    """Load the nine-key JSON config with contextual validation errors."""

    try:
        text = path.read_text(encoding="utf-8")
        payload = cast(object, json.loads(text))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid train config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"invalid train config {path}: expected a JSON object"
        )
    try:
        return TrainConfig.from_dict(cast(Mapping[str, object], payload))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid train config {path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class ClipReport:
    """Norms measured immediately before and after in-place clipping."""

    pre_clip_norm: float
    post_clip_norm: float
    clipped: bool


def global_grad_l2_norm(parameters: Iterable[Tensor]) -> float:
    """L2 norm of all parameter gradients concatenated into one vector.

    Gradients are cast to float32 before squaring so a large float16 gradient
    cannot overflow in the intermediate square.
    """

    total = 0.0
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None:
            total += float(gradient.detach().float().pow(2).sum().item())
    return math.sqrt(total)


def has_non_finite_grad(parameters: Iterable[Tensor]) -> bool:
    """True when at least one parameter gradient contains NaN or Inf."""

    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and not bool(
            torch.isfinite(gradient).all().item()
        ):
            return True
    return False


def clip_global_l2_norm(
    parameters: Iterable[Tensor], max_norm: float
) -> ClipReport:
    """Clip gradients by the global L2 norm and report pre/post norms.

    Call only after ``GradScaler.unscale_`` and after the finite-gradient check:
    clipping a non-finite gradient scales Inf/NaN by a finite coefficient and
    corrupts every parameter gradient.
    """

    max_norm_value = _require_float(
        "max_norm", max_norm, minimum=0.0
    )
    if max_norm_value == 0.0:
        raise ValueError("max_norm must be > 0")
    params = tuple(parameters)
    pre_clip_norm = float(
        torch.nn.utils.clip_grad_norm_(params, max_norm_value).item()
    )
    post_clip_norm = global_grad_l2_norm(params)
    return ClipReport(
        pre_clip_norm=pre_clip_norm,
        post_clip_norm=post_clip_norm,
        clipped=pre_clip_norm > max_norm_value,
    )


@dataclass(frozen=True, slots=True)
class StepLog:
    """A successful optimizer update and its clipping/scale diagnostics."""

    step: int
    lr: float
    loss: float
    grad_l2_pre_clip: float
    grad_l2_post_clip: float
    scaler_scale: float
    clipped: bool

    def as_csv_row(self) -> tuple[object, ...]:
        return (
            self.step,
            self.lr,
            self.grad_l2_pre_clip,
            self.grad_l2_post_clip,
            self.loss,
            0,
            "",
            self.scaler_scale,
        )


@dataclass(frozen=True, slots=True)
class SkipLog:
    """A refused optimizer update and the evidence that caused the skip."""

    step: int
    lr: float
    loss: float
    scaler_scale: float
    reason: SkipReason
    grad_l2_pre_clip: float | None

    def as_csv_row(self) -> tuple[object, ...]:
        pre_clip: float | str = (
            "" if self.grad_l2_pre_clip is None else self.grad_l2_pre_clip
        )
        return (
            self.step,
            self.lr,
            pre_clip,
            "",
            self.loss,
            1,
            self.reason,
            self.scaler_scale,
        )


@dataclass(slots=True)
class AmpTrainState:
    """Model, AdamW, scaler, autocast dtype, and one full training step."""

    model: nn.Module
    optimizer: torch.optim.AdamW
    scaler: GradScaler
    device: torch.device
    autocast_dtype: torch.dtype
    max_grad_norm: float
    lr: float
    backoff_factor: float
    inject_inf_step: int | None = None
    global_step: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be an nn.Module")
        if not isinstance(self.optimizer, torch.optim.AdamW):
            raise TypeError("optimizer must be a torch.optim.AdamW")
        if not isinstance(self.scaler, GradScaler):
            raise TypeError("scaler must be a torch.amp.GradScaler")
        if not isinstance(self.device, torch.device):
            raise TypeError("device must be a torch.device")
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("device must be CPU or CUDA")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if self.autocast_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                "autocast_dtype must be torch.float16 or torch.bfloat16"
            )
        max_grad_norm = _require_float(
            "max_grad_norm", self.max_grad_norm, minimum=0.0
        )
        if max_grad_norm == 0.0:
            raise ValueError("max_grad_norm must be > 0")
        lr = _require_float("lr", self.lr, minimum=0.0)
        backoff_factor = _require_float(
            "backoff_factor", self.backoff_factor, minimum=0.0, maximum=1.0
        )
        if not 0.0 < backoff_factor < 1.0:
            raise ValueError("backoff_factor must be in (0, 1)")
        global_step = _require_int(
            "global_step", self.global_step, minimum=0
        )
        inject_inf_step = (
            None
            if self.inject_inf_step is None
            else _require_int(
                "inject_inf_step", self.inject_inf_step, minimum=0
            )
        )
        for parameter in self.model.parameters():
            if parameter.device.type != self.device.type:
                raise ValueError(
                    "all model parameters must be on the autocast device"
                )
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.backoff_factor = backoff_factor
        self.global_step = global_step
        self.inject_inf_step = inject_inf_step

    def _validate_batch(self, inputs: Tensor, targets: Tensor) -> None:
        if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
            raise TypeError("inputs and targets must be tensors")
        if inputs.ndim != 2:
            raise ValueError("inputs must be a 2-D tensor")
        if targets.ndim != 1:
            raise ValueError("targets must be a 1-D tensor")
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("inputs and targets must have the same batch size")
        if targets.dtype != torch.long:
            raise ValueError("targets must have dtype torch.long")
        if inputs.device != self.device or targets.device != self.device:
            raise ValueError("inputs and targets must be on the autocast device")

    def step(self, inputs: Tensor, targets: Tensor) -> StepLog | SkipLog:
        self._validate_batch(inputs, targets)
        step_index = self.global_step
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr
        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=self.device.type, dtype=self.autocast_dtype
        ):
            logits = cast(Tensor, self.model(inputs))
            loss = F.cross_entropy(logits, targets)

        loss_value = float(loss.detach().float().item())
        scale_used = float(self.scaler.get_scale())

        if not math.isfinite(loss_value):
            # Initialize the lazily created scale, then halve it without a
            # backward pass: a non-finite loss has no useful gradient.
            _ = self.scaler.scale(loss)
            self.scaler.update(new_scale=scale_used * self.backoff_factor)
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            return SkipLog(
                step=step_index,
                lr=self.lr,
                loss=loss_value,
                scaler_scale=scale_used,
                reason="non_finite_loss",
                grad_l2_pre_clip=None,
            )

        scaled = self.scaler.scale(loss)
        if not isinstance(scaled, Tensor):
            raise TypeError("scaler.scale must return a Tensor")
        torch.autograd.backward(scaled)

        if (
            self.inject_inf_step is not None
            and step_index == self.inject_inf_step
        ):
            _inject_first_grad_inf(self.model)

        self.scaler.unscale_(self.optimizer)
        parameters = tuple(self.model.parameters())
        if has_non_finite_grad(parameters):
            pre_clip_norm = global_grad_l2_norm(parameters)
            # The unscaled gradient scan recorded found_inf; update() halves
            # the factor and resets the growth tracker.
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            return SkipLog(
                step=step_index,
                lr=self.lr,
                loss=loss_value,
                scaler_scale=scale_used,
                reason="non_finite_grad",
                grad_l2_pre_clip=pre_clip_norm,
            )

        report = clip_global_l2_norm(parameters, self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.global_step += 1
        return StepLog(
            step=step_index,
            lr=self.lr,
            loss=loss_value,
            grad_l2_pre_clip=report.pre_clip_norm,
            grad_l2_post_clip=report.post_clip_norm,
            scaler_scale=scale_used,
            clipped=report.clipped,
        )


def _inject_first_grad_inf(model: nn.Module) -> None:
    """Set the first element of the first available gradient to +Inf."""

    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is not None:
            with torch.no_grad():
                gradient.reshape(-1)[0] = float("inf")
            return
    raise RuntimeError("cannot inject Inf: model has no gradients")


class StepLogWriter:
    """Append-only CSV writer that flushes the header and every row."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._handle: Any = None
        self._writer: Any = None

    def __enter__(self) -> Self:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle, lineterminator="\n")
        self._writer.writerow(TRAINING_LOG_COLUMNS)
        self._handle.flush()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self._handle.close()
        self._handle = None
        self._writer = None

    def write(self, record: StepLog | SkipLog) -> None:
        if self._writer is None or self._handle is None:
            raise RuntimeError("StepLogWriter is not open")
        self._writer.writerow(record.as_csv_row())
        self._handle.flush()


@dataclass(slots=True)
class SkipRateMonitor:
    """Rolling skip-rate window with a latched, consecutive-step alert."""

    window: int = 1000
    threshold: float = 0.05
    fail_after: int = 100
    _events: deque[bool] = field(default_factory=deque, repr=False)
    _skips: int = field(default=0, repr=False)
    _consecutive_above: int = field(default=0, repr=False)
    _alerted: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self.window = _require_int("window", self.window, minimum=1)
        self.threshold = _require_float(
            "threshold", self.threshold, minimum=0.0, maximum=1.0
        )
        self.fail_after = _require_int(
            "fail_after", self.fail_after, minimum=1
        )

    @property
    def rate(self) -> float:
        if not self._events:
            return 0.0
        return self._skips / len(self._events)

    @property
    def alerted(self) -> bool:
        return self._alerted

    def record(self, skipped: bool) -> bool:
        skipped_value = bool(skipped)
        self._events.append(skipped_value)
        if skipped_value:
            self._skips += 1
        if len(self._events) > self.window:
            removed = self._events.popleft()
            if removed:
                self._skips -= 1
        if len(self._events) >= self.window:
            if self.rate > self.threshold:
                self._consecutive_above += 1
            else:
                self._consecutive_above = 0
            if self._consecutive_above >= self.fail_after:
                self._alerted = True
        return self._alerted


def make_demo_batch(seed: int) -> tuple[Tensor, Tensor]:
    """Build a fixed, linearly separable three-cluster classification batch."""

    seed_value = _require_int("seed", seed, minimum=0)
    centers = torch.tensor(
        [
            [4.0, 0.0, 0.0, 0.0],
            [-4.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.arange(CLASSES, dtype=torch.long).repeat_interleave(
        SAMPLES_PER_CLASS
    )
    generator = torch.Generator().manual_seed(seed_value)
    noise = torch.randn(BATCH_SIZE, FEATURES, generator=generator)
    return centers[labels] + noise, labels


@dataclass(frozen=True, slots=True)
class DemoResult:
    """Everything one deterministic demo run produced."""

    config: TrainConfig
    logs: tuple[StepLog | SkipLog, ...]
    workdir: Path
    csv_path: Path
    config_snapshot_path: Path
    parameter_count: int
    steps: int
    seed: int
    inject_inf_step: int | None
    skipped: int
    skip_rate: float
    final_scale: float
    alerted: bool
    device: torch.device
    autocast_dtype: torch.dtype


def _write_config_snapshot(
    config: TrainConfig,
    path: Path,
    *,
    steps: int,
    seed: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    payload["device"] = device.type
    payload["effective_autocast_dtype"] = _dtype_name(autocast_dtype)
    payload["effective_steps"] = steps
    payload["effective_seed"] = seed
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_demo(
    config: TrainConfig,
    *,
    steps: int | None = None,
    workdir: Path,
    seed: int | None = None,
    device: torch.device | str = "cpu",
    autocast_dtype: torch.dtype | None = None,
    inject_inf: bool = True,
    skip_rate_window: int = 1000,
    skip_rate_threshold: float = 0.05,
    skip_rate_fail_after: int = 100,
) -> DemoResult:
    """Train the tiny classifier and write the run-directory artifacts."""

    if not isinstance(config, TrainConfig):
        raise TypeError("config must be a TrainConfig")
    steps_value = (
        config.steps
        if steps is None
        else _require_int("steps", steps, minimum=1)
    )
    seed_value = (
        config.seed if seed is None else _require_int("seed", seed, minimum=0)
    )
    device_value = (
        device if isinstance(device, torch.device) else torch.device(device)
    )
    if device_value.type not in {"cpu", "cuda"}:
        raise ValueError("device must be CPU or CUDA")
    if device_value.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if (
            device_value.index is not None
            and device_value.index >= torch.cuda.device_count()
        ):
            raise ValueError(
                f"CUDA device index {device_value.index} is out of range"
            )
    device_value = _normalize_device(device_value)
    dtype_value = (
        config.autocast_dtype_torch()
        if autocast_dtype is None
        else autocast_dtype
    )
    if dtype_value not in (torch.float16, torch.bfloat16):
        raise ValueError(
            "autocast_dtype must be torch.float16 or torch.bfloat16"
        )
    monitor = SkipRateMonitor(
        window=skip_rate_window,
        threshold=skip_rate_threshold,
        fail_after=skip_rate_fail_after,
    )

    torch.manual_seed(seed_value)
    if device_value.type == "cuda":
        torch.cuda.manual_seed_all(seed_value)
    model = nn.Linear(FEATURES, CLASSES).to(device_value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scaler = GradScaler(
        device_value.type,
        init_scale=config.init_scale,
        growth_interval=config.growth_interval,
        backoff_factor=config.backoff_factor,
    )
    state = AmpTrainState(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        device=device_value,
        autocast_dtype=dtype_value,
        max_grad_norm=config.max_grad_norm,
        lr=config.lr,
        backoff_factor=config.backoff_factor,
        inject_inf_step=config.inject_inf_step if inject_inf else None,
    )

    inputs, targets = make_demo_batch(seed_value)
    inputs = inputs.to(device_value)
    targets = targets.to(device_value)

    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    csv_path = workdir_path / "training_log.csv"
    config_snapshot_path = workdir_path / "run_config.json"
    _write_config_snapshot(
        config,
        config_snapshot_path,
        steps=steps_value,
        seed=seed_value,
        device=device_value,
        autocast_dtype=dtype_value,
    )

    logs: list[StepLog | SkipLog] = []
    with StepLogWriter(csv_path) as writer:
        for _ in range(steps_value):
            record = state.step(inputs, targets)
            writer.write(record)
            logs.append(record)
            monitor.record(isinstance(record, SkipLog))

    skipped = sum(isinstance(record, SkipLog) for record in logs)
    return DemoResult(
        config=config,
        logs=tuple(logs),
        workdir=workdir_path,
        csv_path=csv_path,
        config_snapshot_path=config_snapshot_path,
        parameter_count=sum(
            parameter.numel() for parameter in model.parameters()
        ),
        steps=steps_value,
        seed=seed_value,
        inject_inf_step=config.inject_inf_step if inject_inf else None,
        skipped=skipped,
        skip_rate=monitor.rate,
        final_scale=float(state.scaler.get_scale()),
        alerted=monitor.alerted,
        device=device_value,
        autocast_dtype=dtype_value,
    )


def format_demo_report(result: DemoResult) -> str:
    """Format the CLI report with one tagged row per training step."""

    config = result.config
    precision = _dtype_name(result.autocast_dtype)
    inject_text = (
        "none"
        if result.inject_inf_step is None
        else str(result.inject_inf_step)
    )
    lines = [
        "gradient clipping + mixed precision",
        f"config: lr={config.lr:.6f} max_grad_norm={config.max_grad_norm:.6f} "
        f"steps={result.steps} seed={result.seed} "
        f"inject_inf_step={inject_text} precision={precision} "
        f"device={result.device.type}",
        f"model: parameters={result.parameter_count} "
        f"features={FEATURES} classes={CLASSES} batch={BATCH_SIZE}",
    ]
    for record in result.logs:
        if isinstance(record, StepLog):
            clipped = "yes" if record.clipped else "no"
            lines.append(
                f"STEP step={record.step:03d} lr={record.lr:.6f} "
                f"loss={record.loss:.6f} "
                f"grad_pre={record.grad_l2_pre_clip:.6f} "
                f"grad_post={record.grad_l2_post_clip:.6f} "
                f"scale={record.scaler_scale:.1f} clipped={clipped}"
            )
        else:
            pre_clip = (
                "n/a"
                if record.grad_l2_pre_clip is None
                else f"{record.grad_l2_pre_clip:.6f}"
            )
            lines.append(
                f"SKIP step={record.step:03d} lr={record.lr:.6f} "
                f"loss={record.loss:.6f} reason={record.reason} "
                f"grad_pre={pre_clip} scale={record.scaler_scale:.1f}"
            )
    max_pre_clip = max(
        (
            record.grad_l2_pre_clip
            for record in result.logs
            if record.grad_l2_pre_clip is not None
        ),
        default=0.0,
    )
    alert = "yes" if result.alerted else "no"
    lines.extend(
        [
            f"summary: steps={result.steps} skipped={result.skipped} "
            f"skip_rate={result.skip_rate:.4f} "
            f"max_pre_clip={max_pre_clip:.6f} "
            f"final_scale={result.final_scale:.1f} alert={alert}",
            f"artifacts: workdir={result.workdir} "
            f"training_log={result.csv_path} "
            f"config={result.config_snapshot_path}",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny classifier with global L2 clipping and an "
            "autocast + GradScaler step; write a per-step CSV."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="JSON training config (default: configs/train.json)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="training steps (default: steps from the config)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DEFAULT_WORKDIR,
        help="run directory for CSV/config artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="overrides the config seed",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="autocast and optimizer device (cpu or cuda)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="use torch.bfloat16 autocast instead of the config dtype",
    )
    parser.add_argument(
        "--no-inject",
        action="store_true",
        help="disable the demo's synthetic Inf gradient",
    )
    parser.add_argument(
        "--skip-rate-window",
        type=int,
        default=1000,
        help="rolling window size for the skip-rate monitor",
    )
    parser.add_argument(
        "--skip-rate-threshold",
        type=float,
        default=0.05,
        help="alert when the rolling skip rate is strictly above this",
    )
    parser.add_argument(
        "--skip-rate-fail-after",
        type=int,
        default=100,
        help="consecutive full-window steps above threshold before alerting",
    )
    parser.add_argument(
        "--fail-on-skip-rate",
        action="store_true",
        help="exit 1 when the rolling skip-rate alert is active",
    )
    return parser


def _normalize_device(device: torch.device) -> torch.device:
    """Return the canonical device, dropping an irrelevant CPU index."""

    if device.type == "cpu":
        return torch.device("cpu")
    return device


def _parse_device(value: str) -> torch.device:
    try:
        device = torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid device {value!r}: {exc}") from exc
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be CPU or CUDA")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if (
            device.index is not None
            and device.index >= torch.cuda.device_count()
        ):
            raise ValueError(
                f"CUDA device index {device.index} is out of range"
            )
    return _normalize_device(device)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = load_train_config(args.config)
        device = _parse_device(args.device)
        dtype = (
            torch.bfloat16
            if args.bf16
            else config.autocast_dtype_torch()
        )
        result = run_demo(
            config,
            steps=args.steps,
            workdir=args.workdir,
            seed=args.seed,
            device=device,
            autocast_dtype=dtype,
            inject_inf=not args.no_inject,
            skip_rate_window=args.skip_rate_window,
            skip_rate_threshold=args.skip_rate_threshold,
            skip_rate_fail_after=args.skip_rate_fail_after,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    print(format_demo_report(result), end="", flush=True)
    if args.fail_on_skip_rate and result.alerted:
        print(
            "error: rolling skip rate exceeded the configured threshold",
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
