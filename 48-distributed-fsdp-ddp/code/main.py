from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import socket
import statistics
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Literal, cast

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import ProcessContext, spawn as _spawn
import torch.nn.functional as F
from torch import Tensor, nn

ReduceMode = Literal["sum", "avg"]
REDUCE_MODES: tuple[ReduceMode, ...] = ("sum", "avg")
EQUIVALENCE_TOLERANCE = 1e-3
SYNC_TOLERANCE = 1e-6
DEFAULT_WORKDIR = Path("outputs")
REPORT_FILENAME = "ddp-demo.json"
SKILL_FILENAME = "skill-distributed-fsdp-ddp.md"
SCHEMA = "ddp-demo/v1"


def _require_int(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _require_float(
    name: str,
    value: object,
    *,
    minimum: float,
    strict_min: bool = False,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < minimum or (strict_min and result == minimum):
        comparator = ">" if strict_min else ">="
        raise ValueError(f"{name} must be {comparator} {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return result


@dataclass(frozen=True)
class DemoConfig:
    world_size: int = 2
    steps: int = 5
    batch_per_rank: int = 8
    input_dim: int = 8
    hidden_dim: int = 16
    num_classes: int = 4
    lr: float = 0.1
    momentum: float = 0.9
    seed: int = 7
    reduce_mode: ReduceMode = "sum"
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        _require_int("world_size", self.world_size, minimum=1, maximum=8)
        _require_int("steps", self.steps, minimum=1, maximum=100)
        _require_int("batch_per_rank", self.batch_per_rank, minimum=1, maximum=1024)
        _require_int("input_dim", self.input_dim, minimum=1, maximum=1024)
        _require_int("hidden_dim", self.hidden_dim, minimum=1, maximum=1024)
        _require_int("num_classes", self.num_classes, minimum=2, maximum=1024)
        _require_int("seed", self.seed, minimum=0, maximum=2**31 - 1)
        _require_int(
            "timeout_seconds", self.timeout_seconds, minimum=5, maximum=600
        )
        _require_float("lr", self.lr, minimum=0.0, strict_min=True)
        _require_float("momentum", self.momentum, minimum=0.0, maximum=1.0)
        if self.momentum >= 1.0:
            raise ValueError("momentum must be < 1.0")
        if self.reduce_mode not in REDUCE_MODES:
            raise ValueError(
                f"reduce_mode must be one of {', '.join(REDUCE_MODES)}"
            )

    @property
    def total_batch(self) -> int:
        return self.world_size * self.batch_per_rank

    def as_dict(self) -> dict[str, object]:
        return {
            "world_size": self.world_size,
            "steps": self.steps,
            "batch_per_rank": self.batch_per_rank,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "lr": self.lr,
            "momentum": self.momentum,
            "seed": self.seed,
            "reduce_mode": self.reduce_mode,
            "timeout_seconds": self.timeout_seconds,
        }


@contextmanager
def _seeded(seed: int) -> Iterator[None]:
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.random.set_rng_state(state)


class TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.register_buffer("input_scale", torch.ones(1))

    def forward(self, inputs: Tensor) -> Tensor:
        scale = cast(Tensor, self.input_scale)
        return cast(Tensor, self.fc2(torch.tanh(self.fc1(inputs * scale))))


def build_model(config: DemoConfig, seed: int) -> TinyMLP:
    with _seeded(seed):
        return TinyMLP(config.input_dim, config.hidden_dim, config.num_classes)


def make_batch(config: DemoConfig, step: int) -> tuple[Tensor, Tensor]:
    if step < 0:
        raise ValueError("step must be >= 0")
    generator = torch.Generator().manual_seed(config.seed * 1_000_003 + step * 97)
    centers = torch.randn(
        config.num_classes, config.input_dim, generator=generator
    )
    targets = torch.randint(
        0,
        config.num_classes,
        (config.total_batch,),
        generator=generator,
    )
    noise = torch.randn(config.total_batch, config.input_dim, generator=generator)
    return centers[targets] + 0.5 * noise, targets


def split_batch(
    inputs: Tensor,
    targets: Tensor,
    world_size: int,
    rank: int,
) -> tuple[Tensor, Tensor]:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    if inputs.shape[0] % world_size != 0:
        raise ValueError("batch size must be divisible by world_size")
    if targets.shape[0] != inputs.shape[0]:
        raise ValueError("inputs and targets must have the same batch size")
    local_size = inputs.shape[0] // world_size
    start = rank * local_size
    return inputs[start : start + local_size], targets[start : start + local_size]


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def init_gloo(
    rank: int,
    world_size: int,
    port: int,
    timeout_seconds: int,
) -> None:
    if dist.is_initialized():
        raise RuntimeError("default process group is already initialized")
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build")
    if not dist.is_gloo_available():
        raise RuntimeError("the gloo backend is not available in this build")
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    if not 0 < port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=timeout_seconds),
    )


def all_reduce_grads_(
    module: nn.Module,
    world_size: int,
    reduce_mode: ReduceMode = "sum",
) -> int:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if reduce_mode not in REDUCE_MODES:
        raise ValueError(f"reduce_mode must be one of {', '.join(REDUCE_MODES)}")
    reduced = 0
    for parameter in module.parameters():
        grad = parameter.grad
        if grad is None:
            grad = torch.zeros_like(parameter.data)
            parameter.grad = grad
        if reduce_mode == "sum":
            dist.all_reduce(grad.data, op=dist.ReduceOp.SUM)
            grad.data.div_(world_size)
        else:
            dist.all_reduce(grad.data, op=dist.ReduceOp.AVG)
        reduced += 1
    return reduced


class MinimalDDP(nn.Module):
    module: nn.Module
    world_size: int
    reduce_mode: ReduceMode

    def __init__(
        self,
        module: nn.Module,
        world_size: int | None = None,
        reduce_mode: ReduceMode = "sum",
    ) -> None:
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "initialize the default process group before constructing MinimalDDP"
            )
        resolved_world_size = (
            dist.get_world_size() if world_size is None else world_size
        )
        if resolved_world_size != dist.get_world_size():
            raise ValueError("world_size does not match the default process group")
        if resolved_world_size < 1:
            raise ValueError("world_size must be >= 1")
        if reduce_mode not in REDUCE_MODES:
            raise ValueError(f"reduce_mode must be one of {', '.join(REDUCE_MODES)}")
        self.module = module
        self.world_size = resolved_world_size
        self.reduce_mode = reduce_mode
        # Rank 0 is the canonical initialization; every other seed is irrelevant.
        for tensor in (*module.parameters(), *module.buffers()):
            dist.broadcast(tensor.data, src=0)

    def forward(self, inputs: Tensor) -> Tensor:
        return cast(Tensor, self.module(inputs))

    def all_reduce_grads(self) -> int:
        return all_reduce_grads_(self.module, self.world_size, self.reduce_mode)


def mean_loss_across_ranks(loss: Tensor, world_size: int) -> float:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    value = loss.detach().reshape(1).clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value.div_(world_size)
    return float(value.item())


def grad_l2_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            raise RuntimeError("every parameter must have a gradient")
        total += float(parameter.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def _model_state_tensors(module: nn.Module) -> Iterator[Tensor]:
    for parameter in module.parameters():
        yield parameter.detach()
    for buffer in module.buffers():
        yield buffer.detach()


def state_max_abs_diff(module: nn.Module) -> float:
    world_size = dist.get_world_size()
    worst = 0.0
    for tensor in _model_state_tensors(module):
        local = tensor.contiguous()
        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local)
        base = gathered[0]
        for other in gathered[1:]:
            worst = max(worst, float((other - base).abs().max().item()))
    return worst


def steps_agree(step_count: int, world_size: int) -> bool:
    local = torch.tensor([step_count], dtype=torch.int64)
    gathered = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(gathered, local)
    return all(int(value.item()) == step_count for value in gathered)


@dataclass(frozen=True)
class ParamShard:
    name: str
    shape: tuple[int, ...]
    numel: int
    pad: int
    local: Tensor


@dataclass(frozen=True)
class FsdpParamReport:
    name: str
    numel: int
    pad: int
    shard_numel: int
    bit_equal: bool
    reshard_bit_equal: bool
    max_abs_diff: float

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "numel": self.numel,
            "pad": self.pad,
            "shard_numel": self.shard_numel,
            "bit_equal": self.bit_equal,
            "reshard_bit_equal": self.reshard_bit_equal,
            "max_abs_diff": self.max_abs_diff,
        }

    @classmethod
    def from_dict(cls, payload: object) -> FsdpParamReport:
        data = _expect_dict(payload, "fsdp parameter")
        numel = _as_int(_field(data, "numel", "fsdp parameter"), "numel")
        pad = _as_int(_field(data, "pad", "fsdp parameter"), "pad")
        shard_numel = _as_int(
            _field(data, "shard_numel", "fsdp parameter"), "shard_numel"
        )
        if numel < 0 or pad < 0 or shard_numel < 0:
            raise ValueError("FSDP sizes must be non-negative")
        return cls(
            name=_as_str(_field(data, "name", "fsdp parameter"), "name"),
            numel=numel,
            pad=pad,
            shard_numel=shard_numel,
            bit_equal=_as_bool(
                _field(data, "bit_equal", "fsdp parameter"), "bit_equal"
            ),
            reshard_bit_equal=_as_bool(
                _field(data, "reshard_bit_equal", "fsdp parameter"),
                "reshard_bit_equal",
            ),
            max_abs_diff=_as_float(
                _field(data, "max_abs_diff", "fsdp parameter"), "max_abs_diff"
            ),
        )


@dataclass(frozen=True)
class FsdpReport:
    params: tuple[FsdpParamReport, ...]
    bit_equal: bool
    reshard_bit_equal: bool
    padded_params: int
    padding_numel: int
    resident_param_bytes: int
    full_param_bytes: int
    gathered_extra_bytes: int
    shard_fraction: float

    def as_dict(self) -> dict[str, object]:
        return {
            "params": [parameter.as_dict() for parameter in self.params],
            "bit_equal": self.bit_equal,
            "reshard_bit_equal": self.reshard_bit_equal,
            "padded_params": self.padded_params,
            "padding_numel": self.padding_numel,
            "resident_param_bytes": self.resident_param_bytes,
            "full_param_bytes": self.full_param_bytes,
            "gathered_extra_bytes": self.gathered_extra_bytes,
            "shard_fraction": self.shard_fraction,
        }

    @classmethod
    def from_dict(cls, payload: object) -> FsdpReport:
        data = _expect_dict(payload, "fsdp")
        raw_params = _as_list(_field(data, "params", "fsdp"), "fsdp.params")
        params = tuple(FsdpParamReport.from_dict(item) for item in raw_params)
        padded_params = _as_int(
            _field(data, "padded_params", "fsdp"), "padded_params"
        )
        padding_numel = _as_int(
            _field(data, "padding_numel", "fsdp"), "padding_numel"
        )
        resident = _as_int(
            _field(data, "resident_param_bytes", "fsdp"),
            "resident_param_bytes",
        )
        full = _as_int(_field(data, "full_param_bytes", "fsdp"), "full_param_bytes")
        gathered = _as_int(
            _field(data, "gathered_extra_bytes", "fsdp"),
            "gathered_extra_bytes",
        )
        fraction = _as_float(
            _field(data, "shard_fraction", "fsdp"), "shard_fraction"
        )
        if min(padded_params, padding_numel, resident, full, gathered) < 0:
            raise ValueError("FSDP counts and bytes must be non-negative")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("shard_fraction must be in [0, 1]")
        return cls(
            params=params,
            bit_equal=_as_bool(_field(data, "bit_equal", "fsdp"), "bit_equal"),
            reshard_bit_equal=_as_bool(
                _field(data, "reshard_bit_equal", "fsdp"), "reshard_bit_equal"
            ),
            padded_params=padded_params,
            padding_numel=padding_numel,
            resident_param_bytes=resident,
            full_param_bytes=full,
            gathered_extra_bytes=gathered,
            shard_fraction=fraction,
        )


@dataclass(frozen=True)
class StepRecord:
    step: int
    local_loss: float
    global_loss: float
    grad_norm: float
    step_ms: float
    reduce_ms: float

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "local_loss": self.local_loss,
            "global_loss": self.global_loss,
            "grad_norm": self.grad_norm,
            "step_ms": self.step_ms,
            "reduce_ms": self.reduce_ms,
        }

    @classmethod
    def from_dict(cls, payload: object) -> StepRecord:
        data = _expect_dict(payload, "step record")
        step = _as_int(_field(data, "step", "step record"), "step")
        if step < 0:
            raise ValueError("step must be >= 0")
        return cls(
            step=step,
            local_loss=_as_float(
                _field(data, "local_loss", "step record"), "local_loss"
            ),
            global_loss=_as_float(
                _field(data, "global_loss", "step record"), "global_loss"
            ),
            grad_norm=_as_float(
                _field(data, "grad_norm", "step record"), "grad_norm"
            ),
            step_ms=_as_float(_field(data, "step_ms", "step record"), "step_ms"),
            reduce_ms=_as_float(
                _field(data, "reduce_ms", "step record"), "reduce_ms"
            ),
        )


@dataclass(frozen=True)
class RankReport:
    rank: int
    world_size: int
    reduce_mode: ReduceMode
    seed: int
    buffer_after_broadcast: float
    parameter_sum_after: float
    parameter_l2_after: float
    mean_global_loss: float
    final_global_loss: float
    grad_norm_after_all_reduce: float
    state_max_abs_diff: float
    steps_agreed: bool
    steps: tuple[StepRecord, ...]
    fsdp: FsdpReport

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "reduce_mode": self.reduce_mode,
            "seed": self.seed,
            "buffer_after_broadcast": self.buffer_after_broadcast,
            "parameter_sum_after": self.parameter_sum_after,
            "parameter_l2_after": self.parameter_l2_after,
            "mean_global_loss": self.mean_global_loss,
            "final_global_loss": self.final_global_loss,
            "grad_norm_after_all_reduce": self.grad_norm_after_all_reduce,
            "state_max_abs_diff": self.state_max_abs_diff,
            "steps_agreed": self.steps_agreed,
            "steps": [record.as_dict() for record in self.steps],
            "fsdp": self.fsdp.as_dict(),
        }

    @classmethod
    def from_dict(cls, payload: object) -> RankReport:
        data = _expect_dict(payload, "rank report")
        rank = _as_int(_field(data, "rank", "rank report"), "rank")
        world_size = _as_int(
            _field(data, "world_size", "rank report"), "world_size"
        )
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        reduce_mode = _as_str(
            _field(data, "reduce_mode", "rank report"), "reduce_mode"
        )
        if reduce_mode not in REDUCE_MODES:
            raise ValueError(f"reduce_mode must be one of {', '.join(REDUCE_MODES)}")
        raw_steps = _as_list(_field(data, "steps", "rank report"), "steps")
        steps = tuple(StepRecord.from_dict(item) for item in raw_steps)
        if not steps:
            raise ValueError("rank report must contain at least one step")
        return cls(
            rank=rank,
            world_size=world_size,
            reduce_mode=reduce_mode,
            seed=_as_int(_field(data, "seed", "rank report"), "seed"),
            buffer_after_broadcast=_as_float(
                _field(data, "buffer_after_broadcast", "rank report"),
                "buffer_after_broadcast",
            ),
            parameter_sum_after=_as_float(
                _field(data, "parameter_sum_after", "rank report"),
                "parameter_sum_after",
            ),
            parameter_l2_after=_as_float(
                _field(data, "parameter_l2_after", "rank report"),
                "parameter_l2_after",
            ),
            mean_global_loss=_as_float(
                _field(data, "mean_global_loss", "rank report"),
                "mean_global_loss",
            ),
            final_global_loss=_as_float(
                _field(data, "final_global_loss", "rank report"),
                "final_global_loss",
            ),
            grad_norm_after_all_reduce=_as_float(
                _field(data, "grad_norm_after_all_reduce", "rank report"),
                "grad_norm_after_all_reduce",
            ),
            state_max_abs_diff=_as_float(
                _field(data, "state_max_abs_diff", "rank report"),
                "state_max_abs_diff",
            ),
            steps_agreed=_as_bool(
                _field(data, "steps_agreed", "rank report"), "steps_agreed"
            ),
            steps=steps,
            fsdp=FsdpReport.from_dict(_field(data, "fsdp", "rank report")),
        )


@dataclass(frozen=True)
class GradientEvidence:
    max_abs_diff: float
    reference_loss: float
    reference_grad_norm: float


@dataclass(frozen=True)
class TimingSummary:
    total_wall_ms: float
    median_step_ms: float
    median_grad_reduce_ms: float


@dataclass(frozen=True)
class DemoEvidence:
    gradient_max_abs_diff: float
    loss_max_abs_diff: float
    final_parameter_max_abs_diff: float
    reference_gradient_norm: float
    reference_final_loss: float


@dataclass(frozen=True)
class DemoReport:
    config: DemoConfig
    ranks: tuple[RankReport, ...]
    evidence: DemoEvidence
    checks: dict[str, bool]
    timings: TimingSummary
    versions: dict[str, object]

    @property
    def all_checks_passed(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def as_dict(self) -> dict[str, object]:
        checks = dict(self.checks)
        checks["all_checks_passed"] = self.all_checks_passed
        return {
            "schema": SCHEMA,
            "config": self.config.as_dict(),
            "ranks": [rank.as_dict() for rank in self.ranks],
            "evidence": {
                "gradient_max_abs_diff": self.evidence.gradient_max_abs_diff,
                "loss_max_abs_diff": self.evidence.loss_max_abs_diff,
                "final_parameter_max_abs_diff": (
                    self.evidence.final_parameter_max_abs_diff
                ),
                "reference_gradient_norm": self.evidence.reference_gradient_norm,
                "reference_final_loss": self.evidence.reference_final_loss,
            },
            "checks": checks,
            "timings": {
                "total_wall_ms": self.timings.total_wall_ms,
                "median_step_ms": self.timings.median_step_ms,
                "median_grad_reduce_ms": self.timings.median_grad_reduce_ms,
            },
            "versions": self.versions,
        }


def shard_parameter(
    name: str,
    tensor: Tensor,
    world_size: int,
    rank: int,
) -> ParamShard:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be in [0, world_size)")
    flat = tensor.detach().reshape(-1).contiguous()
    numel = int(flat.numel())
    pad = (-numel) % world_size
    if pad:
        flat = torch.cat((flat, flat.new_zeros(pad)))
    local = flat.chunk(world_size)[rank].clone()
    return ParamShard(
        name=name,
        shape=tuple(int(dimension) for dimension in tensor.shape),
        numel=numel,
        pad=pad,
        local=local,
    )


def unshard_parameter(shard: ParamShard, world_size: int) -> Tensor:
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if int(shard.local.numel()) * world_size < shard.numel:
        raise ValueError("shards do not contain enough values to unshard")
    gathered = [torch.empty_like(shard.local) for _ in range(world_size)]
    dist.all_gather(gathered, shard.local)
    return torch.cat(gathered)[: shard.numel].reshape(shard.shape)


def reshard_parameter(full: Tensor, world_size: int, rank: int) -> Tensor:
    return shard_parameter("reshard", full, world_size, rank).local


def fsdp_round_trip_sketch(
    module: nn.Module,
    world_size: int,
    rank: int,
) -> FsdpReport:
    parameters: list[FsdpParamReport] = []
    resident_bytes = 0
    full_bytes = 0
    gathered_extra_bytes = 0
    padded_params = 0
    padding_numel = 0
    for name, parameter in module.named_parameters():
        shard = shard_parameter(name, parameter.data, world_size, rank)
        reconstructed = unshard_parameter(shard, world_size)
        original = parameter.detach()
        max_abs_diff = (
            float((reconstructed - original).abs().max().item())
            if original.numel()
            else 0.0
        )
        bit_equal = torch.equal(reconstructed, original)
        resharded = reshard_parameter(reconstructed, world_size, rank)
        reshard_bit_equal = torch.equal(resharded, shard.local)
        parameter_bytes = int(original.numel() * original.element_size())
        shard_bytes = int(shard.local.numel() * shard.local.element_size())
        resident_bytes += shard_bytes
        full_bytes += parameter_bytes
        gathered_extra_bytes = max(gathered_extra_bytes, parameter_bytes)
        if shard.pad:
            padded_params += 1
            padding_numel += shard.pad
        parameters.append(
            FsdpParamReport(
                name=name,
                numel=shard.numel,
                pad=shard.pad,
                shard_numel=int(shard.local.numel()),
                bit_equal=bit_equal,
                reshard_bit_equal=reshard_bit_equal,
                max_abs_diff=max_abs_diff,
            )
        )
        # Drop the unsharded copy before moving to the next parameter.
        del reconstructed, resharded, original
    fraction = resident_bytes / full_bytes if full_bytes else 0.0
    return FsdpReport(
        params=tuple(parameters),
        bit_equal=all(item.bit_equal for item in parameters),
        reshard_bit_equal=all(item.reshard_bit_equal for item in parameters),
        padded_params=padded_params,
        padding_numel=padding_numel,
        resident_param_bytes=resident_bytes,
        full_param_bytes=full_bytes,
        gathered_extra_bytes=gathered_extra_bytes,
        shard_fraction=fraction,
    )


def _state_from_model(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }


def _load_state(path: Path) -> dict[str, Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    data = _expect_dict(payload, path.name)
    state: dict[str, Tensor] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not isinstance(value, Tensor):
            raise ValueError(f"{path.name} must contain a string-to-Tensor state")
        state[key] = value
    if not state:
        raise ValueError(f"{path.name} is empty")
    return state


def full_batch_gradient(
    config: DemoConfig,
    init_state: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], float]:
    model = build_model(config, config.seed)
    model.load_state_dict(init_state)
    inputs, targets = make_batch(config, 0)
    loss = F.cross_entropy(model(inputs), targets)
    loss.backward()  # type: ignore[no-untyped-call]
    gradients: dict[str, Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"reference parameter {name} has no gradient")
        gradients[name] = parameter.grad.detach().clone()
    return gradients, float(loss.detach().item())


def _grad_l2_norm(gradients: Mapping[str, Tensor]) -> float:
    return math.sqrt(
        sum(float(gradient.detach().pow(2).sum().item()) for gradient in gradients.values())
    )


def max_abs_grad_diff(
    left: Mapping[str, Tensor],
    right: Mapping[str, Tensor],
) -> float:
    if set(left) != set(right):
        raise ValueError("gradient maps must have identical parameter names")
    worst = 0.0
    for name in left:
        left_gradient = left[name]
        right_gradient = right[name]
        if left_gradient.shape != right_gradient.shape:
            raise ValueError(f"gradient shape mismatch for {name}")
        worst = max(
            worst,
            float((left_gradient - right_gradient).abs().max().item()),
        )
    return worst


def max_abs_state_diff(
    left: Mapping[str, Tensor],
    right: Mapping[str, Tensor],
) -> float:
    if set(left) != set(right):
        raise ValueError("state maps must have identical keys")
    worst = 0.0
    for name in left:
        left_value = left[name]
        right_value = right[name]
        if left_value.shape != right_value.shape:
            raise ValueError(f"state shape mismatch for {name}")
        if left_value.numel() == 0:
            continue
        worst = max(
            worst,
            float((left_value - right_value).abs().max().item()),
        )
    return worst


def manual_all_reduce_matches_single_process(
    config: DemoConfig,
    init_state: Mapping[str, Tensor],
    reduced_grads: Mapping[str, Tensor],
) -> GradientEvidence:
    reference_grads, reference_loss = full_batch_gradient(config, init_state)
    return GradientEvidence(
        max_abs_diff=max_abs_grad_diff(reduced_grads, reference_grads),
        reference_loss=reference_loss,
        reference_grad_norm=_grad_l2_norm(reference_grads),
    )


def train_single_process(
    config: DemoConfig,
    init_state: Mapping[str, Tensor],
) -> tuple[dict[str, Tensor], tuple[float, ...]]:
    model = build_model(config, config.seed)
    model.load_state_dict(init_state)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.lr,
        momentum=config.momentum,
    )
    losses: list[float] = []
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        inputs, targets = make_batch(config, step)
        loss = F.cross_entropy(model(inputs), targets)
        loss.backward()  # type: ignore[no-untyped-call]
        losses.append(float(loss.detach().item()))
        optimizer.step()
    return _state_from_model(model), tuple(losses)


def run_rank(rank: int, config: DemoConfig, artifact_dir: Path) -> RankReport:
    if not 0 <= rank < config.world_size:
        raise ValueError("rank must be in [0, world_size)")
    seed = config.seed if rank == 0 else config.seed + 1009 * rank
    model = build_model(config, seed)
    if rank != 0:
        # Deliberate pre-broadcast divergence: MinimalDDP must erase it.
        cast(Tensor, model.input_scale).fill_(1.0 + rank)
    ddp = MinimalDDP(model, config.world_size, config.reduce_mode)
    buffer_after_broadcast = float(cast(Tensor, model.input_scale).item())
    init_state = _state_from_model(model)
    if rank == 0:
        torch.save(init_state, artifact_dir / "init_state.pt")
    fsdp = fsdp_round_trip_sketch(model, config.world_size, rank)
    optimizer = torch.optim.SGD(
        ddp.parameters(),
        lr=config.lr,
        momentum=config.momentum,
    )
    records: list[StepRecord] = []
    for step in range(config.steps):
        full_inputs, full_targets = make_batch(config, step)
        inputs, targets = split_batch(
            full_inputs, full_targets, config.world_size, rank
        )
        optimizer.zero_grad(set_to_none=True)
        step_start = time.perf_counter()
        loss = F.cross_entropy(ddp(inputs), targets)
        loss.backward()  # type: ignore[no-untyped-call]
        reduce_start = time.perf_counter()
        ddp.all_reduce_grads()
        reduce_ms = (time.perf_counter() - reduce_start) * 1000.0
        local_loss = float(loss.detach().item())
        global_loss = mean_loss_across_ranks(loss, config.world_size)
        grad_norm = grad_l2_norm(model)
        step_ms = (time.perf_counter() - step_start) * 1000.0
        records.append(
            StepRecord(
                step=step,
                local_loss=local_loss,
                global_loss=global_loss,
                grad_norm=grad_norm,
                step_ms=step_ms,
                reduce_ms=reduce_ms,
            )
        )
        if step == 0 and rank == 0:
            step_grads: dict[str, Tensor] = {}
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    raise RuntimeError(f"parameter {name} has no gradient")
                step_grads[name] = parameter.grad.detach().clone()
            torch.save(step_grads, artifact_dir / "step0_grads.pt")
        optimizer.step()

    state_diff = state_max_abs_diff(model)
    agreed = steps_agree(config.steps, config.world_size)
    if rank == 0:
        torch.save(_state_from_model(model), artifact_dir / "final_state.pt")
    dist.barrier()

    parameter_sum = sum(
        float(parameter.detach().sum().item())
        for parameter in model.parameters()
    )
    parameter_l2 = math.sqrt(
        sum(
            float(parameter.detach().pow(2).sum().item())
            for parameter in model.parameters()
        )
    )
    mean_global_loss = sum(record.global_loss for record in records) / len(records)
    return RankReport(
        rank=rank,
        world_size=config.world_size,
        reduce_mode=config.reduce_mode,
        seed=config.seed,
        buffer_after_broadcast=buffer_after_broadcast,
        parameter_sum_after=parameter_sum,
        parameter_l2_after=parameter_l2,
        mean_global_loss=mean_global_loss,
        final_global_loss=records[-1].global_loss,
        grad_norm_after_all_reduce=records[-1].grad_norm,
        state_max_abs_diff=state_diff,
        steps_agreed=agreed,
        steps=tuple(records),
        fsdp=fsdp,
    )


def _rank_entry(
    rank: int,
    config: DemoConfig,
    port: int,
    artifact_dir: Path,
) -> None:
    try:
        init_gloo(rank, config.world_size, port, config.timeout_seconds)
        torch.set_num_threads(1)
        report = run_rank(rank, config, artifact_dir)
        write_json(artifact_dir / f"rank-{rank}.json", report.as_dict())
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _expect_dict(payload: object, name: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must be a JSON object")
    data: dict[str, object] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise ValueError(f"{name} keys must be strings")
        data[key] = value
    return data


def _field(data: Mapping[str, object], key: str, name: str) -> object:
    if key not in data:
        raise ValueError(f"{name} is missing {key}")
    return data[key]


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _as_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _as_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _as_list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _read_json(path: Path) -> object:
    try:
        return cast(object, json.loads(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name} is not valid JSON") from exc


def read_rank_report(
    path: Path,
    expected_rank: int,
    world_size: int,
) -> RankReport:
    report = RankReport.from_dict(_read_json(path))
    if report.rank != expected_rank:
        raise ValueError(
            f"{path.name} contains rank {report.rank}, expected {expected_rank}"
        )
    if report.world_size != world_size:
        raise ValueError(
            f"{path.name} contains world_size {report.world_size}, expected {world_size}"
        )
    return report


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2) + "\n")


def write_skill(path: Path, config: DemoConfig) -> None:
    _write_text_atomic(path, render_skill(config))


def run_demo(config: DemoConfig, workdir: Path) -> DemoReport:
    torch.set_num_threads(1)
    workdir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(
        tempfile.mkdtemp(prefix=".ddp-ranks-", dir=workdir)
    )
    started = time.perf_counter()
    try:
        port = find_free_port()
        context: ProcessContext = _spawn(  # type: ignore[no-untyped-call]
            _rank_entry,
            args=(config, port, artifact_dir),
            nprocs=config.world_size,
            join=False,
        )
        deadline = time.monotonic() + config.timeout_seconds + 30.0
        while True:
            joined = context.join(timeout=0.5)
            if joined:
                break
            if time.monotonic() >= deadline:
                for process in context.processes:
                    if process.is_alive():
                        process.kill()
                for process in context.processes:
                    process.join(timeout=5.0)
                raise RuntimeError("rank processes did not finish in time")

        ranks = tuple(
            read_rank_report(
                artifact_dir / f"rank-{rank}.json",
                rank,
                config.world_size,
            )
            for rank in range(config.world_size)
        )
        init_state = _load_state(artifact_dir / "init_state.pt")
        step0_grads = _load_state(artifact_dir / "step0_grads.pt")
        final_state = _load_state(artifact_dir / "final_state.pt")

        gradient_evidence = manual_all_reduce_matches_single_process(
            config, init_state, step0_grads
        )
        reference_state, reference_losses = train_single_process(config, init_state)
        if len(reference_losses) != config.steps:
            raise RuntimeError("reference loss count does not match steps")
        final_parameter_diff = max_abs_state_diff(final_state, reference_state)
        loss_diff = 0.0
        for rank in ranks:
            if len(rank.steps) != config.steps:
                raise ValueError(
                    f"rank {rank.rank} reported {len(rank.steps)} steps, "
                    f"expected {config.steps}"
                )
            for record, reference_loss in zip(rank.steps, reference_losses):
                loss_diff = max(
                    loss_diff, abs(record.global_loss - reference_loss)
                )

        state_diff = max(rank.state_max_abs_diff for rank in ranks)
        parameter_sums = [rank.parameter_sum_after for rank in ranks]
        parameter_sum_spread = max(parameter_sums) - min(parameter_sums)
        checks = {
            "gradient_equivalence": (
                gradient_evidence.max_abs_diff < EQUIVALENCE_TOLERANCE
            ),
            "loss_equivalence": loss_diff < EQUIVALENCE_TOLERANCE,
            "final_parameter_equivalence": (
                final_parameter_diff < EQUIVALENCE_TOLERANCE
            ),
            "parameter_sync": state_diff < SYNC_TOLERANCE,
            "parameter_sums_agree": parameter_sum_spread < SYNC_TOLERANCE,
            "buffer_sync": all(
                abs(rank.buffer_after_broadcast - 1.0) <= 1e-9
                for rank in ranks
            ),
            "fsdp_round_trip": all(
                rank.fsdp.bit_equal and rank.fsdp.reshard_bit_equal
                for rank in ranks
            ),
            "fsdp_memory_saving": config.world_size == 1
            or all(
                rank.fsdp.resident_param_bytes < rank.fsdp.full_param_bytes
                for rank in ranks
            ),
            "steps_agreed": all(
                rank.steps_agreed and len(rank.steps) == config.steps
                for rank in ranks
            ),
        }
        step_times = [
            record.step_ms for rank in ranks for record in rank.steps
        ]
        reduce_times = [
            record.reduce_ms for rank in ranks for record in rank.steps
        ]
        timings = TimingSummary(
            total_wall_ms=(time.perf_counter() - started) * 1000.0,
            median_step_ms=float(statistics.median(step_times)),
            median_grad_reduce_ms=float(statistics.median(reduce_times)),
        )
        versions: dict[str, object] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "backend": "gloo",
            "gloo_available": bool(dist.is_gloo_available()),
            "cuda_available": bool(torch.cuda.is_available()),
            "nccl_available": bool(dist.is_nccl_available()),
        }
        report = DemoReport(
            config=config,
            ranks=ranks,
            evidence=DemoEvidence(
                gradient_max_abs_diff=gradient_evidence.max_abs_diff,
                loss_max_abs_diff=loss_diff,
                final_parameter_max_abs_diff=final_parameter_diff,
                reference_gradient_norm=gradient_evidence.reference_grad_norm,
                reference_final_loss=reference_losses[-1],
            ),
            checks=checks,
            timings=timings,
            versions=versions,
        )
        write_json(workdir / REPORT_FILENAME, report.as_dict())
        write_skill(workdir / SKILL_FILENAME, config)
        return report
    finally:
        shutil.rmtree(artifact_dir, ignore_errors=True)


def render_skill(config: DemoConfig) -> str:
    lines = [
        "# Skill: Distributed Data Parallel and FSDP",
        "",
        "Use this recipe when one model must train on a batch split across ranks.",
        "The two DDP collectives are the whole contract: broadcast the initial state,",
        "then all-reduce and average gradients after every backward pass.",
        "",
        "## 1. Bring up a gloo process group",
        "",
        "```python",
        'os.environ["MASTER_ADDR"] = "127.0.0.1"',
        'os.environ["MASTER_PORT"] = str(port)  # bind-and-close free port',
        'dist.init_process_group(backend="gloo", rank=rank,',
        "                        world_size=world_size,",
        "                        timeout=timedelta(seconds=60))",
        "```",
        "",
        "Every rank must dial the same rendezvous. Use `nccl` and CUDA tensors on a",
        "multi-GPU node; the collective calls do not change.",
        "",
        "## 2. Broadcast at construction",
        "",
        "```python",
        "class MinimalDDP(nn.Module):",
        "    def __init__(self, module, world_size, reduce_mode='sum'):",
        "        super().__init__()",
        "        self.module = module",
        "        for tensor in (*module.parameters(), *module.buffers()):",
        "            dist.broadcast(tensor.data, src=0)",
        "```",
        "",
        "Without this broadcast every rank keeps its own seed and the run diverges",
        "from the first step. Parameters and buffers must use the same order on every",
        "rank or a collective will hang.",
        "",
        "## 3. Average gradients after backward",
        "",
        "```python",
        "for parameter in module.parameters():",
        "    if parameter.grad is None:",
        "        parameter.grad = torch.zeros_like(parameter.data)",
        "    dist.all_reduce(parameter.grad.data, op=dist.ReduceOp.SUM)",
        "    parameter.grad.data.div_(world_size)",
        "optimizer.step()",
        "```",
        "",
        "Each rank computes a mean loss over its local batch of B examples. Summing",
        "the local gradients and dividing by N gives the gradient of the mean loss",
        "over the concatenated N*B examples. For equal local batch sizes:",
        "",
        "```text",
        "(1/N) * sum_r grad(mean_loss_r) == grad(mean_loss over all N*B examples)",
        "```",
        "",
        "Unequal local batch sizes need sample-count weighting, not a plain 1/N.",
        "",
        "## 4. Average the shown loss too",
        "",
        "```python",
        "value = loss.detach().reshape(1).clone()",
        "dist.all_reduce(value, op=dist.ReduceOp.SUM)",
        "value.div_(world_size)",
        "global_loss = float(value.item())",
        "```",
        "",
        "The rank-local loss is not the dashboard loss. All-reduce the scalar with the",
        "same equal-batch rule as the gradients.",
        "",
        "## 5. FSDP shard, unshard, reshard",
        "",
        "```python",
        "flat = parameter.detach().reshape(-1).contiguous()",
        "pad = (-flat.numel()) % world_size",
        "if pad:",
        "    flat = torch.cat((flat, flat.new_zeros(pad)))",
        "local = flat.chunk(world_size)[rank].clone()",
        "",
        "# unshard before the layer forward",
        "gathered = [torch.empty_like(local) for _ in range(world_size)]",
        "dist.all_gather(gathered, local)",
        "full = torch.cat(gathered)[:parameter.numel()].reshape(parameter.shape)",
        "",
        "# reshard after the forward",
        "flat = full.reshape(-1)",
        "pad = (-flat.numel()) % world_size",
        "if pad:",
        "    flat = torch.cat((flat, flat.new_zeros(pad)))",
        "local = flat.chunk(world_size)[rank].clone()",
        "```",
        "",
        "The round trip must be bit-equal; padding is dropped after the gather. The",
        "memory win is about 1/N for the local shard, paid for by an all-gather on",
        "every forward. Production FSDP overlaps that gather with compute.",
        "",
        "## 6. Invariants",
        "",
        "- Broadcast parameters and buffers before the first optimizer step.",
        "- All-reduce gradients once after backward, before the step.",
        "- Average the loss shown to the dashboard, not only the gradients.",
        "- Every rank calls the same collectives in the same order.",
        "- Each rank runs exactly the same number of steps.",
        "- Shard only what fits the FSDP story; buffers stay replicated.",
        "",
        "## Configuration used by this module",
        "",
        f"- world_size: {config.world_size}",
        f"- steps: {config.steps}",
        f"- batch_per_rank: {config.batch_per_rank}",
        f"- total_batch: {config.total_batch}",
        f"- input_dim: {config.input_dim}",
        f"- hidden_dim: {config.hidden_dim}",
        f"- num_classes: {config.num_classes}",
        f"- lr: {config.lr}",
        f"- momentum: {config.momentum}",
        f"- seed: {config.seed}",
        f"- reduce_mode: {config.reduce_mode}",
        "",
        "## Evidence",
        "",
        "Run `uv run --frozen python code/main.py` and inspect",
        "`outputs/ddp-demo.json`: it contains per-rank parameter sums, the",
        "post-all-reduce gradient norm, the FSDP round trip, and the",
        "manual-versus-single-process gradient difference.",
        "",
    ]
    return "\n".join(lines)


def format_summary(report: DemoReport) -> str:
    state_spread = max(rank.state_max_abs_diff for rank in report.ranks)
    fsdp = report.ranks[0].fsdp
    passed = "yes" if report.all_checks_passed else "no"
    lines = [
        (
            f"world_size={report.config.world_size} backend=gloo "
            f"steps={report.config.steps} "
            f"batch_per_rank={report.config.batch_per_rank} "
            f"reduce={report.config.reduce_mode}"
        ),
        f"gradient max-abs-diff: {report.evidence.gradient_max_abs_diff:.3e}",
        f"loss max-abs-diff: {report.evidence.loss_max_abs_diff:.3e}",
        (
            "final parameter max-abs-diff: "
            f"{report.evidence.final_parameter_max_abs_diff:.3e}"
        ),
        f"rank state spread: {state_spread:.3e}",
        (
            f"fsdp round trip: bit-equal={str(fsdp.bit_equal).lower()} "
            f"params={len(fsdp.params)} "
            f"shard_fraction={fsdp.shard_fraction:.3f}"
        ),
        f"checks: passed={passed}",
    ]
    if not report.all_checks_passed:
        failed = [name for name, value in report.checks.items() if not value]
        lines.append(f"failed checks: {', '.join(failed)}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Spawn CPU gloo ranks, average gradients and losses, and prove the "
            "single-process and FSDP round-trip equivalences."
        )
    )
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--batch-per-rank", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--reduce",
        dest="reduce_mode",
        choices=REDUCE_MODES,
        default="sum",
        help="gradient reduction: SUM then divide, or ReduceOp.AVG",
    )
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORKDIR)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = DemoConfig(
            world_size=cast(int, args.world_size),
            steps=cast(int, args.steps),
            batch_per_rank=cast(int, args.batch_per_rank),
            input_dim=cast(int, args.input_dim),
            hidden_dim=cast(int, args.hidden_dim),
            num_classes=cast(int, args.num_classes),
            lr=cast(float, args.lr),
            momentum=cast(float, args.momentum),
            seed=cast(int, args.seed),
            reduce_mode=cast(ReduceMode, args.reduce_mode),
            timeout_seconds=cast(int, args.timeout_seconds),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    workdir = cast(Path, args.workdir)
    try:
        report = run_demo(config, workdir)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(format_summary(report))
    print(f"wrote {(workdir / REPORT_FILENAME).as_posix()}")
    print(f"wrote {(workdir / SKILL_FILENAME).as_posix()}")
    return 0 if report.all_checks_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
