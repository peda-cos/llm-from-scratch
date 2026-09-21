from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

import main
from main import CHECKPOINT_SCHEMA, TrainState, load_checkpoint, save_checkpoint
from support import assert_plain_python_value


class _CustomPayloadObject:
    pass


def _build_objects() -> tuple[
    nn.Module,
    torch.optim.Optimizer,
    CosineAnnealingLR,
    TrainState,
]:
    torch.manual_seed(11)
    model: nn.Module = nn.Sequential(
        nn.Linear(4, 6),
        nn.ReLU(),
        nn.Linear(6, 3),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    state = TrainState(
        step=0,
        epoch=0,
        batch_in_epoch=0,
        batches_per_epoch=2,
        losses=(),
    )
    return model, optimizer, scheduler, state


def _run_steps(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    state: TrainState,
    steps: int,
) -> TrainState:
    generator = torch.Generator().manual_seed(3)
    inputs = torch.randn(8, 4, generator=generator)
    targets = torch.arange(8) % 3
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(inputs), targets)
        torch.autograd.backward(loss)
        optimizer.step()
        scheduler.step()
        state = state.advance(float(loss.item()))
    return state


def test_single_file_round_trip_restores_all_training_state(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler, state = _build_objects()
    state = _run_steps(model, optimizer, scheduler, state, 2)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, state)

    parameter = next(model.parameters())
    saved_parameter = parameter.detach().clone()
    saved_exp_avg = optimizer.state[parameter]["exp_avg"].clone()
    saved_lr = scheduler.get_last_lr()[0]
    saved_losses = state.losses

    _run_steps(model, optimizer, scheduler, state, 2)
    loaded = load_checkpoint(path, model, optimizer, scheduler)

    assert loaded.step == 2
    assert loaded.losses == saved_losses
    assert torch.equal(parameter.detach(), saved_parameter)
    assert torch.equal(optimizer.state[parameter]["exp_avg"], saved_exp_avg)
    assert scheduler.get_last_lr()[0] == pytest.approx(saved_lr)


def test_checkpoint_payload_schema_and_rng_keys(tmp_path: Path) -> None:
    model, optimizer, scheduler, state = _build_objects()
    state = _run_steps(model, optimizer, scheduler, state, 1)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, state)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert set(payload["train_state"]) == {
        "step",
        "epoch",
        "batch_in_epoch",
        "batches_per_epoch",
        "losses",
    }
    assert set(payload["rng"]) == {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
    }
    assert payload["wall_saved_at"] > 0.0
    assert_plain_python_value(payload["rng"])


def test_load_rejects_unknown_schema(tmp_path: Path) -> None:
    model, optimizer, scheduler, state = _build_objects()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, state)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema"] = "checkpoint/v999"
    main.atomic_save(path, payload)

    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(path, model, optimizer, scheduler)


def test_load_rejects_missing_keys(tmp_path: Path) -> None:
    model, optimizer, scheduler, state = _build_objects()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, scheduler, state)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload["optimizer"]
    main.atomic_save(path, payload)

    with pytest.raises(ValueError, match="missing keys"):
        load_checkpoint(path, model, optimizer, scheduler)


def test_load_rejects_truncated_file(tmp_path: Path) -> None:
    model, optimizer, scheduler, _ = _build_objects()
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"not a checkpoint")

    with pytest.raises(ValueError, match="cannot load checkpoint"):
        load_checkpoint(path, model, optimizer, scheduler)


def test_load_rejects_pickled_custom_object(tmp_path: Path) -> None:
    model, optimizer, scheduler, _ = _build_objects()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"schema": CHECKPOINT_SCHEMA, "evil": _CustomPayloadObject()},
        path,
    )

    with pytest.raises(ValueError, match="cannot load checkpoint"):
        load_checkpoint(path, model, optimizer, scheduler)


def test_train_state_advance_mid_epoch_and_boundary() -> None:
    state = TrainState(
        step=3,
        epoch=0,
        batch_in_epoch=3,
        batches_per_epoch=4,
        losses=(1.0,),
    )

    advanced = state.advance(2.0)

    assert advanced == TrainState(
        step=4,
        epoch=1,
        batch_in_epoch=0,
        batches_per_epoch=4,
        losses=(1.0, 2.0),
    )
    assert TrainState.from_dict(advanced.as_dict()) == advanced


def test_train_state_rejects_invalid_counters() -> None:
    with pytest.raises(ValueError, match="step"):
        TrainState(step=-1, epoch=0, batch_in_epoch=0, batches_per_epoch=2)
    with pytest.raises(ValueError, match="batch_in_epoch"):
        TrainState(step=0, epoch=0, batch_in_epoch=2, batches_per_epoch=2)
    with pytest.raises(ValueError, match="batches_per_epoch"):
        TrainState(step=0, epoch=0, batch_in_epoch=0, batches_per_epoch=0)


def test_train_state_rejects_unknown_keys() -> None:
    payload = TrainState(
        step=0,
        epoch=0,
        batch_in_epoch=0,
        batches_per_epoch=2,
    ).as_dict()
    payload["unknown"] = 1

    with pytest.raises(ValueError, match="keys mismatch"):
        TrainState.from_dict(payload)
