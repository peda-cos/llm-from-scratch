"""Cosine learning-rate schedule with linear warmup on a real AdamW step.

The module exposes a stateless schedule, a training-step wrapper that reads the
global step, per-step gradient-L2 logging, CSV/ASCII artifacts, and a
deterministic offline demo on a tiny linear classifier. It is self-contained:
no network, GPU, pretrained weights, or sibling lesson code is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

FEATURES = 4
CLASSES = 3
SAMPLES_PER_CLASS = 16
BATCH_SIZE = 48
DEMO_SEED = 7
DEFAULT_WORKDIR = Path("outputs/demo")
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "schedule.json"
)
TRAINING_LOG_COLUMNS = ("step", "lr", "grad_l2_norm", "loss")
SCHEDULE_COLUMNS = ("step", "lr")
_SCHEDULE_KEYS = ("warmup_steps", "total_steps", "lr_max", "lr_min")


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


@dataclass(frozen=True, slots=True)
class ScheduleConfig:
    """Validated schedule parameters shared by the curve and the trainer."""

    warmup_steps: int
    total_steps: int
    lr_max: float
    lr_min: float

    def __post_init__(self) -> None:
        warmup_steps = _require_int("warmup_steps", self.warmup_steps, minimum=0)
        total_steps = _require_int("total_steps", self.total_steps, minimum=1)
        lr_max = _require_float("lr_max", self.lr_max, minimum=0.0)
        if lr_max == 0.0:
            raise ValueError("lr_max must be > 0")
        lr_min = _require_float(
            "lr_min", self.lr_min, minimum=0.0, maximum=lr_max
        )
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be less than "
                f"total_steps ({total_steps})"
            )
        object.__setattr__(self, "warmup_steps", warmup_steps)
        object.__setattr__(self, "total_steps", total_steps)
        object.__setattr__(self, "lr_max", lr_max)
        object.__setattr__(self, "lr_min", lr_min)

    def to_dict(self) -> dict[str, int | float]:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "lr_max": self.lr_max,
            "lr_min": self.lr_min,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ScheduleConfig:
        expected = set(_SCHEDULE_KEYS)
        present = set(data)
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        if missing:
            raise ValueError(
                "missing schedule config keys: " + ", ".join(missing)
            )
        if unexpected:
            raise ValueError(
                "unexpected schedule config keys: " + ", ".join(unexpected)
            )
        return cls(
            warmup_steps=cast(int, data["warmup_steps"]),
            total_steps=cast(int, data["total_steps"]),
            lr_max=cast(float, data["lr_max"]),
            lr_min=cast(float, data["lr_min"]),
        )


def load_schedule_config(path: Path) -> ScheduleConfig:
    """Load the four-key JSON schedule config with contextual errors."""

    try:
        text = path.read_text(encoding="utf-8")
        payload = cast(object, json.loads(text))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid schedule config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"invalid schedule config {path}: expected a JSON object"
        )
    try:
        return ScheduleConfig.from_dict(cast(Mapping[str, object], payload))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid schedule config {path}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CosineWithWarmup:
    """Stateless linear-warmup, cosine-decay schedule with a hard floor."""

    config: ScheduleConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, ScheduleConfig):
            raise TypeError("config must be a ScheduleConfig")

    def lr(self, step: int) -> float:
        step_value = _require_int("step", step, minimum=0)
        config = self.config
        if config.warmup_steps > 0 and step_value <= config.warmup_steps:
            if step_value == config.warmup_steps:
                return config.lr_max
            return config.lr_max * step_value / config.warmup_steps
        if step_value >= config.total_steps:
            return config.lr_min
        progress = (step_value - config.warmup_steps) / (
            config.total_steps - config.warmup_steps
        )
        value = config.lr_min + 0.5 * (config.lr_max - config.lr_min) * (
            1.0 + math.cos(math.pi * progress)
        )
        return min(config.lr_max, max(config.lr_min, value))


@dataclass(frozen=True, slots=True)
class StepLog:
    """One training step's durable log row."""

    step: int
    lr: float
    grad_l2_norm: float
    loss: float


def grad_l2_norm(model: nn.Module) -> float:
    """L2 norm of all parameter gradients concatenated into one vector."""

    total = 0.0
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is not None:
            total += float(gradient.detach().pow(2).sum().item())
    return math.sqrt(total)


@dataclass(slots=True)
class TrainState:
    """Model, AdamW optimizer, fixed batch, and schedule in one step function."""

    model: nn.Module
    optimizer: torch.optim.Optimizer
    schedule: CosineWithWarmup
    inputs: Tensor
    targets: Tensor
    global_step: int = 0

    def __post_init__(self) -> None:
        self.global_step = _require_int(
            "global_step", self.global_step, minimum=0
        )
        if not isinstance(self.schedule, CosineWithWarmup):
            raise TypeError("schedule must be a CosineWithWarmup")
        if self.inputs.ndim != 2:
            raise ValueError("inputs must be a 2-D tensor")
        if self.targets.ndim != 1:
            raise ValueError("targets must be a 1-D tensor")
        if self.inputs.shape[0] != self.targets.shape[0]:
            raise ValueError(
                "inputs and targets must have the same batch size"
            )
        if self.targets.dtype != torch.long:
            raise ValueError("targets must have dtype torch.long")

    def step(self) -> StepLog:
        step_index = self.global_step
        total_steps = self.schedule.config.total_steps
        if step_index > total_steps:
            warnings.warn(
                f"global_step {step_index} exceeds total_steps {total_steps}; "
                "learning rate pinned at lr_min",
                RuntimeWarning,
                stacklevel=2,
            )
        lr = self.schedule.lr(step_index)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.optimizer.zero_grad(set_to_none=True)
        logits = cast(Tensor, self.model(self.inputs))
        loss = F.cross_entropy(logits, self.targets)
        torch.autograd.backward(loss)
        norm = grad_l2_norm(self.model)
        self.optimizer.step()
        self.global_step += 1
        return StepLog(
            step=step_index,
            lr=lr,
            grad_l2_norm=norm,
            loss=float(loss.detach().item()),
        )


def make_demo_batch(seed: int) -> tuple[Tensor, Tensor]:
    """Build a fixed, linearly separable three-cluster classification batch."""

    seed_value = _require_int("seed", seed, minimum=0)
    centers = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.arange(CLASSES, dtype=torch.long).repeat_interleave(
        SAMPLES_PER_CLASS
    )
    generator = torch.Generator().manual_seed(seed_value)
    noise = torch.randn(BATCH_SIZE, FEATURES, generator=generator)
    return centers[labels] + 0.25 * noise, labels


def plot_schedule_ascii(
    schedule: CosineWithWarmup,
    *,
    width: int = 64,
    height: int = 16,
    end_step: int | None = None,
) -> str:
    """Render the schedule as a fixed-width text plot."""

    if not isinstance(schedule, CosineWithWarmup):
        raise TypeError("schedule must be a CosineWithWarmup")
    width_value = _require_int("width", width, minimum=8)
    height_value = _require_int("height", height, minimum=2)
    config = schedule.config
    if end_step is None:
        end_value = config.total_steps
    else:
        end_value = _require_int("end_step", end_step, minimum=0)
        if end_value < config.total_steps:
            raise ValueError(
                f"end_step ({end_value}) must be >= total_steps "
                f"({config.total_steps})"
            )

    grid = [[" "] * width_value for _ in range(height_value)]
    previous_row: int | None = None
    for column in range(width_value):
        step = round(column * end_value / (width_value - 1))
        lr = schedule.lr(step)
        row = round(
            (config.lr_max - lr) / config.lr_max * (height_value - 1)
        )
        row = min(height_value - 1, max(0, row))
        if previous_row is not None:
            for fill_row in range(
                min(previous_row, row), max(previous_row, row) + 1
            ):
                grid[fill_row][column] = "*"
        else:
            grid[row][column] = "*"
        previous_row = row

    peak_label = f"{config.lr_max:.6f}"
    floor_label = f"{config.lr_min:.6f}"
    label_width = max(len(peak_label), len("0.000000"))
    lines = [
        f"lr schedule: steps 0..{end_value} | "
        f"warmup={config.warmup_steps} | peak={peak_label} | "
        f"floor={floor_label}"
    ]
    for row in range(height_value):
        if row == 0:
            label = peak_label
        elif row == height_value - 1:
            label = "0.000000"
        else:
            label = ""
        lines.append(f"{label:>{label_width}} |{''.join(grid[row])}")

    axis_prefix = " " * label_width + " +"
    lines.append(axis_prefix + "-" * width_value)
    left_tick = "0"
    right_tick = str(end_value)
    gap = width_value - len(left_tick) - len(right_tick)
    if gap < 1:
        tick_line = " " * (label_width + 2) + left_tick + " " + right_tick
    else:
        tick_line = (
            " " * (label_width + 2)
            + left_tick
            + " " * gap
            + right_tick
        )
    lines.append(tick_line)
    return "\n".join(lines) + "\n"


def _write_csv(
    path: Path,
    header: Sequence[str],
    rows: Sequence[Sequence[object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def write_schedule_csv(schedule: CosineWithWarmup, path: Path) -> None:
    """Write one ``step,lr`` row for every step from 0 through total_steps."""

    rows = [
        (step, schedule.lr(step))
        for step in range(schedule.config.total_steps + 1)
    ]
    _write_csv(path, SCHEDULE_COLUMNS, rows)


def write_training_log(logs: Sequence[StepLog], path: Path) -> None:
    """Write the fixed ``step,lr,grad_l2_norm,loss`` training schema."""

    rows = [
        (log.step, log.lr, log.grad_l2_norm, log.loss) for log in logs
    ]
    _write_csv(path, TRAINING_LOG_COLUMNS, rows)


@dataclass(frozen=True, slots=True)
class DemoResult:
    """Everything one deterministic demo run produced."""

    config: ScheduleConfig
    logs: tuple[StepLog, ...]
    workdir: Path
    training_log_path: Path
    schedule_csv_path: Path
    plot_path: Path
    config_snapshot_path: Path
    plot_text: str
    parameter_count: int
    steps: int


def _write_config_snapshot(config: ScheduleConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(config.to_dict(), indent=2, sort_keys=True)
    path.write_text(payload + "\n", encoding="utf-8")


def run_demo(
    config: ScheduleConfig,
    *,
    steps: int | None = None,
    workdir: Path,
    seed: int = DEMO_SEED,
    plot_width: int = 64,
    plot_height: int = 16,
) -> DemoResult:
    """Train the tiny classifier and write the run-directory artifacts."""

    if not isinstance(config, ScheduleConfig):
        raise TypeError("config must be a ScheduleConfig")
    steps_value = (
        config.total_steps
        if steps is None
        else _require_int("steps", steps, minimum=1)
    )
    seed_value = _require_int("seed", seed, minimum=0)
    workdir_path = Path(workdir)
    schedule = CosineWithWarmup(config)
    plot_text = plot_schedule_ascii(
        schedule,
        width=plot_width,
        height=plot_height,
        end_step=config.total_steps + max(1, config.total_steps // 10),
    )

    torch.manual_seed(seed_value)
    model = nn.Linear(FEATURES, CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr_max)
    inputs, targets = make_demo_batch(seed_value)
    state = TrainState(
        model=model,
        optimizer=optimizer,
        schedule=schedule,
        inputs=inputs,
        targets=targets,
    )
    logs = tuple(state.step() for _ in range(steps_value))

    workdir_path.mkdir(parents=True, exist_ok=True)
    training_log_path = workdir_path / "training_log.csv"
    schedule_csv_path = workdir_path / "schedule.csv"
    plot_path = workdir_path / "lr_schedule.txt"
    config_snapshot_path = workdir_path / "run_config.json"

    write_training_log(logs, training_log_path)
    write_schedule_csv(schedule, schedule_csv_path)
    _write_config_snapshot(config, config_snapshot_path)
    plot_path.write_text(plot_text, encoding="utf-8")
    return DemoResult(
        config=config,
        logs=logs,
        workdir=workdir_path,
        training_log_path=training_log_path,
        schedule_csv_path=schedule_csv_path,
        plot_path=plot_path,
        config_snapshot_path=config_snapshot_path,
        plot_text=plot_text,
        parameter_count=sum(
            parameter.numel() for parameter in model.parameters()
        ),
        steps=steps_value,
    )


def format_demo_report(result: DemoResult) -> str:
    """Format the CLI report, including the schedule plot."""

    config = result.config
    schedule = CosineWithWarmup(config)
    if result.logs:
        first_loss = result.logs[0].loss
        final_loss = result.logs[-1].loss
        max_grad_norm = max(log.grad_l2_norm for log in result.logs)
    else:
        first_loss = final_loss = max_grad_norm = 0.0
    lines = [
        "cosine lr with linear warmup",
        f"config: warmup_steps={config.warmup_steps} "
        f"total_steps={config.total_steps} "
        f"lr_max={config.lr_max:.6f} lr_min={config.lr_min:.6f}",
        f"model: parameters={result.parameter_count} features={FEATURES} "
        f"classes={CLASSES} batch={BATCH_SIZE}",
        "boundaries: "
        f"lr(0)={schedule.lr(0):.6f} "
        f"lr(warmup={config.warmup_steps})="
        f"{schedule.lr(config.warmup_steps):.6f} "
        f"lr(total={config.total_steps})="
        f"{schedule.lr(config.total_steps):.6f} "
        f"lr(total+1)={schedule.lr(config.total_steps + 1):.6f}",
    ]
    for log in result.logs:
        lines.append(
            f"step={log.step:03d} lr={log.lr:.6f} "
            f"grad_l2_norm={log.grad_l2_norm:.6f} loss={log.loss:.6f}"
        )
    lines.extend(
        [
            f"summary: steps={result.steps} "
            f"first_loss={first_loss:.6f} final_loss={final_loss:.6f} "
            f"max_grad_l2_norm={max_grad_norm:.6f}",
            f"artifacts: workdir={result.workdir} "
            f"training_log={result.training_log_path} "
            f"schedule={result.schedule_csv_path} "
            f"plot={result.plot_path} config={result.config_snapshot_path}",
            "",
            result.plot_text.rstrip("\n"),
        ]
    )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a tiny linear classifier with a cosine-with-warmup "
            "learning-rate schedule and write a per-step log."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="JSON schedule config (default: configs/schedule.json)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="training updates (default: total_steps from the config)",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DEFAULT_WORKDIR,
        help="run directory for CSV/plot artifacts",
    )
    parser.add_argument("--seed", type=int, default=DEMO_SEED)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = load_schedule_config(args.config)
        steps = (
            config.total_steps
            if args.steps is None
            else _require_int("steps", args.steps, minimum=1)
        )
        result = run_demo(
            config,
            steps=steps,
            workdir=args.workdir,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    print(format_demo_report(result), end="", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
