from __future__ import annotations

import math

import pytest
import torch

from main import build_model, reshard_parameter, shard_parameter
from support import tiny_config


def test_shard_parameter_pads_and_reconstructs() -> None:
    tensor = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    shards = [
        shard_parameter("weight", tensor, 3, rank) for rank in range(3)
    ]

    assert all(shard.pad == 2 for shard in shards)
    assert all(shard.local.numel() == 4 for shard in shards)
    reconstructed = torch.cat([shard.local for shard in shards])[: tensor.numel()]
    assert torch.equal(reconstructed.reshape(tensor.shape), tensor)


def test_reshard_is_the_local_slice() -> None:
    tensor = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    for world_size in (2, 4):
        shard = shard_parameter("weight", tensor, world_size, 0)
        resharded = reshard_parameter(tensor, world_size, 0)

        assert torch.equal(resharded, shard.local)
        assert shard.local.numel() <= math.ceil(tensor.numel() / world_size)


def test_fsdp_logical_bytes_drop_per_rank() -> None:
    config = tiny_config(world_size=4)
    model = build_model(config, config.seed)
    full_bytes = 0
    resident_bytes = 0
    for name, parameter in model.named_parameters():
        shard = shard_parameter(name, parameter.data, config.world_size, 0)
        full_bytes += int(parameter.numel() * parameter.element_size())
        resident_bytes += int(shard.local.numel() * shard.local.element_size())

    assert resident_bytes < full_bytes
    assert resident_bytes * config.world_size >= full_bytes

    one_rank_bytes = 0
    for name, parameter in model.named_parameters():
        shard = shard_parameter(name, parameter.data, 1, 0)
        one_rank_bytes += int(shard.local.numel() * shard.local.element_size())
    assert one_rank_bytes == full_bytes
    assert one_rank_bytes / full_bytes == pytest.approx(1.0)
