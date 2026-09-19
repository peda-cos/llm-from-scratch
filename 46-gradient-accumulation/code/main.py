from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

OptimizerName = Literal["sgd", "adamw"]

DEFAULT_CONFIG_PATH = Path("configs/sweep.json")
DEFAULT_WORKDIR = Path("outputs")
CURVE_FILENAME = "accum-curve.json"
SKILL_FILENAME = "skill-gradient-accumulation.md"
SCHEMA_VERSION = 1
EQUIVALENCE_BATCH = 16
EQUIVALENCE_ACCUM = 4
EQUIVALENCE_TOLERANCE = 1e-4
WORLD_SIZE = 1
OPTIMIZER_NAMES: tuple[OptimizerName, ...] = ("sgd", "adamw")


def _require_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_float(
    name: str,
    value: object,
    *,
    minimum: float,
    strict_min: bool = False,
    maximum: float | None = None,
    strict_max: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < minimum or (strict_min and result == minimum):
        comparator = ">" if strict_min else ">="
        raise ValueError(f"{name} must be {comparator} {minimum}")
    if maximum is not None and (result > maximum or (strict_max and result == maximum)):
        comparator = "<" if strict_max else "<="
        raise ValueError(f"{name} must be {comparator} {maximum}")
    return result


@dataclass(frozen=True)
class SweepConfig:
    micro_batch: int
    accum_steps: tuple[int, ...]
    num_steps: int
    lr: float
    momentum: float
    optimizer: OptimizerName
    seed: int
    input_dim: int
    hidden_dim: int
    num_classes: int
    noise: float

    def __post_init__(self) -> None:
        _require_int("micro_batch", self.micro_batch, minimum=1)
        _require_int("num_steps", self.num_steps, minimum=1)
        _require_int("seed", self.seed, minimum=0)
        _require_int("input_dim", self.input_dim, minimum=1)
        _require_int("hidden_dim", self.hidden_dim, minimum=1)
        _require_int("num_classes", self.num_classes, minimum=1)
        _require_float("lr", self.lr, minimum=0.0, strict_min=True)
        _require_float(
            "momentum",
            self.momentum,
            minimum=0.0,
            maximum=1.0,
            strict_max=True,
        )
        _require_float("noise", self.noise, minimum=0.0, strict_min=True)
        if not isinstance(self.accum_steps, tuple):
            raise ValueError("accum_steps must be a tuple of integers")
        if not self.accum_steps:
            raise ValueError("accum_steps must not be empty")
        for index, value in enumerate(self.accum_steps):
            _require_int(f"accum_steps[{index}]", value, minimum=1)
        if any(
            left >= right
            for left, right in zip(self.accum_steps, self.accum_steps[1:])
        ):
            raise ValueError("accum_steps must be strictly increasing")
        if self.optimizer not in OPTIMIZER_NAMES:
            raise ValueError(
                f'optimizer must be one of {", ".join(OPTIMIZER_NAMES)}'
            )
        if self.input_dim < self.num_classes:
            raise ValueError("input_dim must be >= num_classes")

    def as_dict(self) -> dict[str, object]:
        return {
            "micro_batch": self.micro_batch,
            "accum_steps": list(self.accum_steps),
            "num_steps": self.num_steps,
            "lr": self.lr,
            "momentum": self.momentum,
            "optimizer": self.optimizer,
            "seed": self.seed,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "noise": self.noise,
        }


def load_config(path: Path) -> SweepConfig:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read config {path}: {exc}") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"config {path} must contain a JSON object")
    payload = cast(dict[str, object], raw)
    expected = {field.name for field in fields(SweepConfig)}
    missing = sorted(expected - payload.keys())
    unknown = sorted(payload.keys() - expected)
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing keys: {', '.join(missing)}")
        if unknown:
            details.append(f"unknown keys: {', '.join(unknown)}")
        raise ValueError(f"config {path} has {'; '.join(details)}")

    accum_raw = payload["accum_steps"]
    if not isinstance(accum_raw, list):
        raise ValueError("accum_steps must be a JSON array")
    accum_steps = tuple(
        _require_int("accum_steps[]", value, minimum=1) for value in accum_raw
    )
    optimizer_raw = payload["optimizer"]
    if not isinstance(optimizer_raw, str):
        raise ValueError("optimizer must be a string")
    if optimizer_raw not in OPTIMIZER_NAMES:
        raise ValueError(
            f'optimizer must be one of {", ".join(OPTIMIZER_NAMES)}'
        )
    optimizer = optimizer_raw

    return SweepConfig(
        micro_batch=_require_int("micro_batch", payload["micro_batch"], minimum=1),
        accum_steps=accum_steps,
        num_steps=_require_int("num_steps", payload["num_steps"], minimum=1),
        lr=_require_float("lr", payload["lr"], minimum=0.0, strict_min=True),
        momentum=_require_float(
            "momentum",
            payload["momentum"],
            minimum=0.0,
            maximum=1.0,
            strict_max=True,
        ),
        optimizer=optimizer,
        seed=_require_int("seed", payload["seed"], minimum=0),
        input_dim=_require_int("input_dim", payload["input_dim"], minimum=1),
        hidden_dim=_require_int("hidden_dim", payload["hidden_dim"], minimum=1),
        num_classes=_require_int("num_classes", payload["num_classes"], minimum=1),
        noise=_require_float("noise", payload["noise"], minimum=0.0, strict_min=True),
    )


def build_model(config: SweepConfig, seed: int | None = None) -> nn.Module:
    torch.manual_seed(config.seed if seed is None else seed)
    return nn.Sequential(
        nn.Linear(config.input_dim, config.hidden_dim),
        nn.Tanh(),
        nn.Linear(config.hidden_dim, config.num_classes),
    )


def make_optimizer(model: nn.Module, config: SweepConfig) -> torch.optim.Optimizer:
    parameters = list(model.parameters())
    if config.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=config.lr,
            momentum=config.momentum,
        )
    return torch.optim.AdamW(parameters, lr=config.lr)


def make_blobs(
    num_samples: int,
    *,
    seed: int,
    input_dim: int,
    num_classes: int,
    noise: float,
) -> tuple[Tensor, Tensor]:
    _require_int("num_samples", num_samples, minimum=1)
    _require_int("seed", seed, minimum=0)
    _require_int("input_dim", input_dim, minimum=1)
    _require_int("num_classes", num_classes, minimum=1)
    _require_float("noise", noise, minimum=0.0, strict_min=True)
    if input_dim < num_classes:
        raise ValueError("input_dim must be >= num_classes")

    generator = torch.Generator().manual_seed(seed)
    centers = torch.zeros(num_classes, input_dim)
    for index in range(num_classes):
        centers[index, index] = 3.0
    targets = torch.arange(num_samples, dtype=torch.long) % num_classes
    inputs = centers[targets] + noise * torch.randn(
        num_samples,
        input_dim,
        generator=generator,
    )
    return inputs, targets


def mean_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    return F.cross_entropy(logits, targets, reduction="mean")


def effective_batch_size(
    micro_batch: int,
    accum_steps: int,
    world_size: int = 1,
) -> int:
    return (
        _require_int("micro_batch", micro_batch, minimum=1)
        * _require_int("accum_steps", accum_steps, minimum=1)
        * _require_int("world_size", world_size, minimum=1)
    )


def chunk_micro_batches(
    inputs: Tensor,
    targets: Tensor,
    accum_steps: int,
) -> tuple[tuple[Tensor, Tensor], ...]:
    _require_int("accum_steps", accum_steps, minimum=1)
    if inputs.ndim != 2:
        raise ValueError("inputs must be a 2-D tensor")
    if targets.ndim != 1:
        raise ValueError("targets must be a 1-D tensor")
    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must have the same number of samples")
    batch_size = int(inputs.shape[0])
    if batch_size < 1:
        raise ValueError("the batch must not be empty")
    if batch_size % accum_steps != 0:
        raise ValueError(
            "batch size must be divisible by accum_steps; for an uneven tail, "
            "scale each micro loss by micro_size / total_size instead"
        )
    micro_size = batch_size // accum_steps
    chunks: list[tuple[Tensor, Tensor]] = []
    for start in range(0, batch_size, micro_size):
        stop = start + micro_size
        chunks.append((inputs[start:stop], targets[start:stop]))
    return tuple(chunks)


@contextmanager
def no_sync_context(model: nn.Module) -> Generator[None, None, None]:
    factory = getattr(model, "no_sync", None)
    if factory is None or not callable(factory):
        yield
        return
    with cast(Callable[[], AbstractContextManager[None]], factory)():
        yield


def grad_l2_norm(parameters: Iterable[Tensor]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        squared += float(
            parameter.grad.detach().to(torch.float64).pow(2).sum().item()
        )
    return math.sqrt(squared)


@dataclass(frozen=True)
class AccumulationResult:
    micro_batches: int
    sync_calls: int
    skipped_syncs: int
    mean_loss: float
    grad_norm: float


def _validate_micro_batches(
    micro_batches: Sequence[tuple[Tensor, Tensor]],
) -> int:
    count = len(micro_batches)
    if count == 0:
        raise ValueError("micro_batches must not be empty")
    first_size: int | None = None
    for index, pair in enumerate(micro_batches):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"micro_batches[{index}] must be an (inputs, targets) tuple")
        inputs, targets = pair
        if not isinstance(inputs, Tensor) or not isinstance(targets, Tensor):
            raise ValueError(f"micro_batches[{index}] must contain tensors")
        if inputs.ndim != 2:
            raise ValueError(f"micro_batches[{index}] inputs must be 2-D")
        if targets.ndim != 1:
            raise ValueError(f"micro_batches[{index}] targets must be 1-D")
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError(
                f"micro_batches[{index}] inputs and targets have different lengths"
            )
        size = int(inputs.shape[0])
        if size < 1:
            raise ValueError(f"micro_batches[{index}] must not be empty")
        if first_size is None:
            first_size = size
        elif size != first_size:
            raise ValueError(
                "micro-batches must have equal sample counts; for an uneven tail, "
                "scale each micro loss by micro_size / total_size"
            )
    return count


def accumulate_gradients(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    micro_batches: Sequence[tuple[Tensor, Tensor]],
) -> AccumulationResult:
    micro_count = _validate_micro_batches(micro_batches)
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    sync_counter = 0
    skipped_syncs = 0
    for index, (inputs, targets) in enumerate(micro_batches):
        is_last = index == micro_count - 1
        if is_last:
            context: AbstractContextManager[None] = nullcontext()
            sync_counter = 1
        else:
            context = no_sync_context(model)
            skipped_syncs += 1
        with context:
            raw_loss = mean_cross_entropy(model(inputs), targets)
            loss_total += float(raw_loss.item())
            torch.autograd.backward(raw_loss / micro_count)
    return AccumulationResult(
        micro_batches=micro_count,
        sync_calls=sync_counter,
        skipped_syncs=skipped_syncs,
        mean_loss=loss_total / micro_count,
        grad_norm=grad_l2_norm(model.parameters()),
    )


def train_one_optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    micro_batches: Sequence[tuple[Tensor, Tensor]],
) -> AccumulationResult:
    result = accumulate_gradients(model, optimizer, micro_batches)
    optimizer.step()
    return result


def _parameter_pairs(left: nn.Module, right: nn.Module) -> list[tuple[Tensor, Tensor]]:
    left_parameters = list(left.parameters())
    right_parameters = list(right.parameters())
    if len(left_parameters) != len(right_parameters):
        raise ValueError("models must have the same number of parameters")
    return list(zip(left_parameters, right_parameters, strict=True))


def max_abs_gradient_diff(left: nn.Module, right: nn.Module) -> float:
    differences: list[float] = []
    with torch.no_grad():
        for left_parameter, right_parameter in _parameter_pairs(left, right):
            if left_parameter.grad is None or right_parameter.grad is None:
                raise RuntimeError("both models must have gradients")
            differences.append(
                float(
                    (left_parameter.grad - right_parameter.grad).abs().max().item()
                )
            )
    return max(differences, default=0.0)


def max_abs_parameter_diff(left: nn.Module, right: nn.Module) -> float:
    differences: list[float] = []
    with torch.no_grad():
        for left_parameter, right_parameter in _parameter_pairs(left, right):
            differences.append(
                float((left_parameter - right_parameter).abs().max().item())
            )
    return max(differences, default=0.0)


@dataclass(frozen=True)
class EquivalenceReport:
    batch_size: int
    accum_steps: int
    tolerance: float
    grad_max_abs_diff: float
    param_max_abs_diff: float
    loss_abs_diff: float
    full_grad_norm: float

    @property
    def passed(self) -> bool:
        return (
            self.grad_max_abs_diff < self.tolerance
            and self.param_max_abs_diff < self.tolerance
            and self.loss_abs_diff < self.tolerance
        )


def equivalence_check(
    config: SweepConfig,
    *,
    batch_size: int = EQUIVALENCE_BATCH,
    accum_steps: int = EQUIVALENCE_ACCUM,
    tolerance: float = EQUIVALENCE_TOLERANCE,
) -> EquivalenceReport:
    _require_int("batch_size", batch_size, minimum=1)
    _require_int("accum_steps", accum_steps, minimum=1)
    _require_float("tolerance", tolerance, minimum=0.0, strict_min=True)
    if batch_size % accum_steps != 0:
        raise ValueError("batch_size must be divisible by accum_steps")

    inputs, targets = make_blobs(
        batch_size,
        seed=config.seed,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        noise=config.noise,
    )
    full_model = build_model(config)
    accumulation_model = copy.deepcopy(full_model)
    full_optimizer = make_optimizer(full_model, config)
    accumulation_optimizer = make_optimizer(accumulation_model, config)

    full_optimizer.zero_grad(set_to_none=True)
    full_loss = mean_cross_entropy(full_model(inputs), targets)
    torch.autograd.backward(full_loss)
    full_grad_norm = grad_l2_norm(full_model.parameters())

    micro_batches = chunk_micro_batches(inputs, targets, accum_steps)
    accumulation = accumulate_gradients(
        accumulation_model,
        accumulation_optimizer,
        micro_batches,
    )
    grad_diff = max_abs_gradient_diff(full_model, accumulation_model)
    loss_diff = abs(float(full_loss.item()) - accumulation.mean_loss)

    full_optimizer.step()
    accumulation_optimizer.step()
    parameter_diff = max_abs_parameter_diff(full_model, accumulation_model)

    return EquivalenceReport(
        batch_size=batch_size,
        accum_steps=accum_steps,
        tolerance=tolerance,
        grad_max_abs_diff=grad_diff,
        param_max_abs_diff=parameter_diff,
        loss_abs_diff=loss_diff,
        full_grad_norm=full_grad_norm,
    )


@dataclass(frozen=True)
class CurvePoint:
    accum_steps: int
    micro_batch: int
    effective_batch: int
    world_size: int
    num_steps: int
    micro_batches: int
    optimizer_steps: int
    sync_calls: int
    skipped_syncs: int
    samples: int
    wall_time_sec: float
    samples_per_sec: float
    median_step_ms: float
    first_loss: float
    avg_loss: float
    final_loss: float

    def as_dict(self) -> dict[str, object]:
        return {
            "accum_steps": self.accum_steps,
            "micro_batch": self.micro_batch,
            "effective_batch": self.effective_batch,
            "world_size": self.world_size,
            "num_steps": self.num_steps,
            "micro_batches": self.micro_batches,
            "optimizer_steps": self.optimizer_steps,
            "sync_calls": self.sync_calls,
            "skipped_syncs": self.skipped_syncs,
            "samples": self.samples,
            "wall_time_sec": self.wall_time_sec,
            "samples_per_sec": self.samples_per_sec,
            "median_step_ms": self.median_step_ms,
            "first_loss": self.first_loss,
            "avg_loss": self.avg_loss,
            "final_loss": self.final_loss,
        }


@dataclass(frozen=True)
class SweepReport:
    config: SweepConfig
    points: tuple[CurvePoint, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "config": self.config.as_dict(),
            "points": [point.as_dict() for point in self.points],
        }


def make_training_batches(
    config: SweepConfig,
    accum_steps: int,
) -> tuple[tuple[tuple[Tensor, Tensor], ...], ...]:
    _require_int("accum_steps", accum_steps, minimum=1)
    samples_per_step = config.micro_batch * accum_steps
    total_samples = samples_per_step * config.num_steps
    inputs, targets = make_blobs(
        total_samples,
        seed=config.seed,
        input_dim=config.input_dim,
        num_classes=config.num_classes,
        noise=config.noise,
    )
    batches: list[tuple[tuple[Tensor, Tensor], ...]] = []
    for step in range(config.num_steps):
        start = step * samples_per_step
        end = start + samples_per_step
        batches.append(
            chunk_micro_batches(inputs[start:end], targets[start:end], accum_steps)
        )
    return tuple(batches)


def run_point(config: SweepConfig, accum_steps: int) -> CurvePoint:
    _require_int("accum_steps", accum_steps, minimum=1)
    effective_batch = effective_batch_size(
        config.micro_batch,
        accum_steps,
        WORLD_SIZE,
    )
    model = build_model(config)
    optimizer = make_optimizer(model, config)
    training_batches = make_training_batches(config, accum_steps)

    step_times: list[float] = []
    outcomes: list[AccumulationResult] = []
    for micro_batches in training_batches:
        started = time.perf_counter()
        outcomes.append(
            train_one_optimizer_step(model, optimizer, micro_batches)
        )
        step_times.append(time.perf_counter() - started)

    wall_time = sum(step_times)
    if wall_time <= 0.0:
        raise RuntimeError("training time must be positive")
    losses = [outcome.mean_loss for outcome in outcomes]
    samples = config.micro_batch * accum_steps * config.num_steps
    return CurvePoint(
        accum_steps=accum_steps,
        micro_batch=config.micro_batch,
        effective_batch=effective_batch,
        world_size=WORLD_SIZE,
        num_steps=config.num_steps,
        micro_batches=sum(outcome.micro_batches for outcome in outcomes),
        optimizer_steps=len(outcomes),
        sync_calls=sum(outcome.sync_calls for outcome in outcomes),
        skipped_syncs=sum(outcome.skipped_syncs for outcome in outcomes),
        samples=samples,
        wall_time_sec=wall_time,
        samples_per_sec=samples / wall_time,
        median_step_ms=statistics.median(step_times) * 1000.0,
        first_loss=losses[0],
        avg_loss=sum(losses) / len(losses),
        final_loss=losses[-1],
    )


def sweep_effective_batches(config: SweepConfig) -> SweepReport:
    return SweepReport(
        config=config,
        points=tuple(
            run_point(config, accum_steps) for accum_steps in config.accum_steps
        ),
    )


def write_curve(path: Path, report: SweepReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            report.as_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def render_skill_recipe(config: SweepConfig) -> str:
    accum_steps = ", ".join(str(value) for value in config.accum_steps)
    return f"""# Skill: Gradient Accumulation

Use this recipe when the full effective batch does not fit in memory.

## Contract

```text
effective_batch = micro_batch * accum_steps * world_size
```

On this single-device example `world_size = 1`, so
`effective_batch = micro_batch * accum_steps`.

1. Zero gradients once at the start of the effective step.
2. Divide every mean-reduced micro-batch loss by `accum_steps` before `backward()`.
3. Run the forward and backward of the first `N - 1` micro-batches inside
   `no_sync()` so DDP skips the gradient all-reduce.
4. Run the final micro-batch outside `no_sync()`; the accumulated gradient is
   reduced exactly once.
5. Call `optimizer.step()` once; optimizer state advances once per effective batch.

## Loop

```python
optimizer.zero_grad(set_to_none=True)
for index, (inputs, targets) in enumerate(micro_batches):
    is_last = index == len(micro_batches) - 1
    context = (
        contextlib.nullcontext()
        if is_last or not hasattr(model, "no_sync")
        else model.no_sync()
    )
    with context:
        loss = criterion(model(inputs), targets)
        (loss / accum_steps).backward()
optimizer.step()
```

The loss must use mean reduction over samples. Unequal micro-batch sizes need
per-micro scaling by `micro_size / total_size`; dividing by `accum_steps` is only
correct for equal chunks.

## Config used by this run

- micro_batch: {config.micro_batch}
- accum_steps: {accum_steps}
- num_steps per curve point: {config.num_steps}
- optimizer: {config.optimizer}
- learning rate: {config.lr}
- world_size: 1

## Evidence

Run `uv run --frozen python code/main.py`; inspect
`outputs/accum-curve.json`. Each point reports `effective_batch`, `samples`,
`samples_per_sec`, `median_step_ms`, `sync_calls`, `skipped_syncs`, and loss
summaries so the memory-for-time trade is visible.
"""


def write_skill(path: Path, config: SweepConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_skill_recipe(config), encoding="utf-8")


def run_demo(
    config: SweepConfig,
    workdir: Path,
) -> tuple[Path, Path, SweepReport]:
    report = sweep_effective_batches(config)
    curve_path = workdir / CURVE_FILENAME
    skill_path = workdir / SKILL_FILENAME
    write_curve(curve_path, report)
    write_skill(skill_path, config)
    return curve_path, skill_path, report


def format_equivalence(report: EquivalenceReport) -> str:
    passed = "yes" if report.passed else "no"
    return (
        f"equivalence: batch={report.batch_size} accum={report.accum_steps} "
        f"grad_max_abs_diff={report.grad_max_abs_diff:.3e} "
        f"param_max_abs_diff={report.param_max_abs_diff:.3e} "
        f"loss_abs_diff={report.loss_abs_diff:.3e} "
        f"tolerance={report.tolerance:.0e} passed={passed}"
    )


def format_curve(report: SweepReport) -> str:
    lines = [
        "accum  micro  effective  samples_per_sec  median_step_ms  "
        "sync_calls  skipped_syncs  avg_loss"
    ]
    for point in report.points:
        lines.append(
            f"{point.accum_steps:<7}{point.micro_batch:<7}"
            f"{point.effective_batch:<11}{point.samples_per_sec:<17.1f}"
            f"{point.median_step_ms:<16.3f}{point.sync_calls:<12}"
            f"{point.skipped_syncs:<15}{point.avg_loss:.6f}"
        )
    return "\n".join(lines)


def _parse_accum_steps(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(part == "" for part in parts):
        raise argparse.ArgumentTypeError(
            "--accum-steps must be a comma-separated list of positive integers"
        )
    try:
        parsed = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--accum-steps must be a comma-separated list of positive integers"
        ) from exc
    if any(value < 1 for value in parsed):
        raise argparse.ArgumentTypeError(
            "--accum-steps entries must be positive integers"
        )
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gradient accumulation equivalence check and throughput sweep."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="path to the sweep JSON config",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=DEFAULT_WORKDIR,
        help="directory for accum-curve.json and the skill recipe",
    )
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--micro-batch", type=int, default=None)
    parser.add_argument("--accum-steps", type=_parse_accum_steps, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--optimizer", choices=OPTIMIZER_NAMES, default=None)
    return parser


def _apply_overrides(config: SweepConfig, args: argparse.Namespace) -> SweepConfig:
    return SweepConfig(
        micro_batch=(
            args.micro_batch if args.micro_batch is not None else config.micro_batch
        ),
        accum_steps=(
            args.accum_steps if args.accum_steps is not None else config.accum_steps
        ),
        num_steps=args.num_steps if args.num_steps is not None else config.num_steps,
        lr=args.lr if args.lr is not None else config.lr,
        momentum=config.momentum,
        optimizer=(
            cast(OptimizerName, args.optimizer)
            if args.optimizer is not None
            else config.optimizer
        ),
        seed=args.seed if args.seed is not None else config.seed,
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        noise=config.noise,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config_path = cast(Path, args.config)
    workdir = cast(Path, args.workdir)
    try:
        config = _apply_overrides(load_config(config_path), args)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    equivalence = equivalence_check(config)
    print(format_equivalence(equivalence))
    if not equivalence.passed:
        return 1

    try:
        curve_path, skill_path, report = run_demo(config, workdir)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_curve(report))
    print(f"wrote {curve_path.as_posix()}")
    print(f"wrote {skill_path.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
