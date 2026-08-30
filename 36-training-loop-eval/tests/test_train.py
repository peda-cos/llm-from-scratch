from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch import Tensor

from main import (
    GPTConfig,
    GPTModel,
    TrainConfig,
    cosine_with_warmup,
    make_batches,
    train,
)


def _model(seed: int = 11) -> GPTModel:
    return GPTModel(
        GPTConfig(
            vocab_size=5,
            context_length=8,
            d_model=32,
            num_heads=4,
            num_layers=2,
            mlp_expansion=4,
            dropout=0.0,
        ),
        seed=seed,
    )


def _data() -> (
    tuple[list[tuple[Tensor, Tensor]], list[tuple[Tensor, Tensor]]]
):
    phrase = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    train_batches = make_batches(phrase.repeat(200), 4, 8)
    val_batches = make_batches(phrase.repeat(50), 4, 8)
    return train_batches, val_batches


def _config(
    *,
    steps: int = 40,
    eval_every: int = 10,
    eval_batches: int = 2,
    warmup_steps: int = 4,
) -> TrainConfig:
    return TrainConfig(
        steps=steps,
        eval_every=eval_every,
        eval_batches=eval_batches,
        warmup_steps=warmup_steps,
        max_lr=3e-3,
        min_lr=3e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        seed=17,
        sample_prompt=(1, 2, 3, 4),
        sample_new_tokens=4,
    )


def test_train_writes_exact_jsonl_and_learns(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    train_batches, val_batches = _data()
    config = _config()
    result = train(
        _model(),
        train_batches,
        val_batches,
        config,
        tmp_path / "nested" / "losses.jsonl",
    )

    lines = result.log_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    assert len(rows) == config.steps
    assert [row["step"] for row in rows] == list(range(1, 41))
    assert all(set(row) == {"step", "train_loss", "lr"} for row in rows)
    assert all(math.isfinite(row["train_loss"]) for row in rows)
    assert all(math.isfinite(row["lr"]) and row["lr"] > 0 for row in rows)
    for row in rows:
        expected = cosine_with_warmup(
            row["step"],
            config.warmup_steps,
            config.steps,
            config.max_lr,
            config.min_lr,
        )
        assert row["lr"] == pytest.approx(expected)

    assert [step for step, _ in result.eval_losses] == [10, 20, 30, 40]
    assert [step for step, _ in result.samples] == [10, 20, 30, 40]
    assert all(
        len(ids) == len(config.sample_prompt) + config.sample_new_tokens
        for _, ids in result.samples
    )
    first_loss = sum(result.train_losses[:4]) / 4
    final_loss = sum(result.train_losses[-4:]) / 4
    assert final_loss < 0.5 * first_loss
    assert result.eval_losses[-1][1] < first_loss

    captured = capsys.readouterr()
    step_lines = [
        line for line in captured.out.splitlines() if line.startswith("step=")
    ]
    assert len(step_lines) == config.steps
    assert "eval step=040 loss=" in captured.out
    assert "sample step=040 ids=" in captured.out
    assert captured.err == ""


def test_train_is_deterministic_and_overwrites_logs(tmp_path: Path) -> None:
    train_batches, val_batches = _data()
    config = _config()
    first_log = tmp_path / "first.jsonl"
    second_log = tmp_path / "second.jsonl"

    first = train(_model(), train_batches, val_batches, config, first_log)
    second = train(_model(), train_batches, val_batches, config, second_log)

    assert first_log.read_bytes() == second_log.read_bytes()
    assert first.eval_losses == second.eval_losses
    assert first.samples == second.samples
    assert first.train_losses == second.train_losses

    train(_model(), train_batches, val_batches, config, first_log)
    assert len(first_log.read_text(encoding="utf-8").splitlines()) == 40


def test_train_probes_final_step_and_validates(tmp_path: Path) -> None:
    train_batches, val_batches = _data()
    config = _config(steps=10, eval_every=4, eval_batches=1)
    result = train(
        _model(),
        train_batches,
        val_batches,
        config,
        tmp_path / "final-probe.jsonl",
    )
    assert [step for step, _ in result.eval_losses] == [4, 8, 10]
    assert [step for step, _ in result.samples] == [4, 8, 10]
    assert len(result.log_path.read_text(encoding="utf-8").splitlines()) == 10

    missing_parent = tmp_path / "never" / "created.jsonl"
    with pytest.raises(ValueError, match="eval_batches"):
        train(
            _model(),
            train_batches,
            val_batches,
            _config(eval_batches=99),
            missing_parent,
        )
    assert not missing_parent.exists()

    with pytest.raises(ValueError, match="train_batches"):
        train(_model(), [], val_batches, config, tmp_path / "empty.jsonl")
    with pytest.raises(ValueError, match="train_batches"):
        empty_inputs = torch.zeros((0, 8), dtype=torch.long)
        train(
            _model(),
            [(empty_inputs, empty_inputs)],
            val_batches,
            config,
            tmp_path / "empty-batch.jsonl",
        )


def test_train_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="warmup_steps"):
        TrainConfig(steps=4, warmup_steps=4)
    with pytest.raises(ValueError, match="min_lr"):
        TrainConfig(min_lr=1.0, max_lr=0.1)
    with pytest.raises(ValueError, match="sample_prompt"):
        TrainConfig(sample_prompt=())
    with pytest.raises(TypeError, match="sample_prompt"):
        TrainConfig(sample_prompt=[1, 2])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="grad_clip"):
        TrainConfig(grad_clip=0.0)
