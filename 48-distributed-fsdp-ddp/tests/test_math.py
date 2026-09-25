from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from main import (
    DemoConfig,
    ReduceMode,
    build_model,
    make_batch,
    manual_all_reduce_matches_single_process,
    split_batch,
)
from support import tiny_config


def _local_gradients(config: DemoConfig) -> list[dict[str, Tensor]]:
    model = build_model(config, config.seed)
    init_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    inputs, targets = make_batch(config, 0)
    gradients: list[dict[str, Tensor]] = []
    for rank in range(config.world_size):
        local_model = build_model(config, config.seed)
        local_model.load_state_dict(init_state)
        local_inputs, local_targets = split_batch(
            inputs, targets, config.world_size, rank
        )
        loss = F.cross_entropy(local_model(local_inputs), local_targets)
        loss.backward()  # type: ignore[no-untyped-call]
        local: dict[str, Tensor] = {}
        for name, parameter in local_model.named_parameters():
            grad = parameter.grad
            assert grad is not None
            local[name] = grad.detach().clone()
        gradients.append(local)
    return gradients


def test_local_gradient_average_matches_full_batch() -> None:
    config = tiny_config(world_size=2, batch_per_rank=4)
    model = build_model(config, config.seed)
    init_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    gradients = _local_gradients(config)
    averaged = {
        name: torch.stack([rank_grads[name] for rank_grads in gradients]).mean(dim=0)
        for name in gradients[0]
    }

    evidence = manual_all_reduce_matches_single_process(
        config, init_state, averaged
    )

    assert evidence.max_abs_diff < 1e-6
    assert evidence.reference_grad_norm > 0.0


def test_split_batch_is_an_exact_partition() -> None:
    for world_size in (2, 3):
        config = tiny_config(world_size=world_size, batch_per_rank=2)
        inputs, targets = make_batch(config, 0)
        parts = [
            split_batch(inputs, targets, config.world_size, rank)
            for rank in range(config.world_size)
        ]

        assert all(part_inputs.shape[0] == 2 for part_inputs, _ in parts)
        assert torch.equal(
            torch.cat([part_inputs for part_inputs, _ in parts]), inputs
        )
        assert torch.equal(
            torch.cat([part_targets for _, part_targets in parts]), targets
        )

    with pytest.raises(ValueError, match="divisible"):
        split_batch(torch.randn(5, 4), torch.arange(5), 2, 0)
    with pytest.raises(ValueError, match="rank"):
        split_batch(inputs, targets, 2, 2)


def test_config_rejects_invalid_values() -> None:
    base = tiny_config()

    with pytest.raises(ValueError, match="world_size"):
        replace(base, world_size=0)
    with pytest.raises(ValueError, match="steps"):
        replace(base, steps=0)
    with pytest.raises(ValueError, match="reduce_mode"):
        replace(base, reduce_mode=cast(ReduceMode, "mean"))
    with pytest.raises(ValueError, match="lr"):
        replace(base, lr=0.0)
    with pytest.raises(ValueError, match="momentum"):
        replace(base, momentum=1.0)
    with pytest.raises(ValueError, match="num_classes"):
        replace(base, num_classes=1)


def test_make_batch_is_deterministic() -> None:
    config = tiny_config()

    first_inputs, first_targets = make_batch(config, 0)
    second_inputs, second_targets = make_batch(config, 0)
    other_inputs, _ = make_batch(config, 1)

    assert torch.equal(first_inputs, second_inputs)
    assert torch.equal(first_targets, second_targets)
    assert not torch.equal(first_inputs, other_inputs)
