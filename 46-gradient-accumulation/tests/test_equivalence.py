from __future__ import annotations

import pytest
import torch

from main import (
    EQUIVALENCE_ACCUM,
    EQUIVALENCE_BATCH,
    chunk_micro_batches,
    effective_batch_size,
    equivalence_check,
)
from support import tiny_config


def test_equivalence_matches_gradients_and_parameters() -> None:
    report = equivalence_check(tiny_config())

    assert report.passed
    assert report.batch_size == EQUIVALENCE_BATCH
    assert report.accum_steps == EQUIVALENCE_ACCUM
    assert report.tolerance == 1e-4
    assert report.grad_max_abs_diff < report.tolerance
    assert report.param_max_abs_diff < report.tolerance
    assert report.loss_abs_diff < report.tolerance
    assert report.full_grad_norm > 0.0


def test_equivalence_requires_divisible_batch() -> None:
    with pytest.raises(ValueError, match="divisible"):
        equivalence_check(tiny_config(), batch_size=10, accum_steps=4)


def test_equivalence_rejects_invalid_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        equivalence_check(tiny_config(), tolerance=0.0)


def test_chunking_preserves_order_and_sizes() -> None:
    inputs = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    targets = torch.arange(6, dtype=torch.long)
    chunks = chunk_micro_batches(inputs, targets, 3)

    assert len(chunks) == 3
    assert all(int(chunk_inputs.shape[0]) == 2 for chunk_inputs, _ in chunks)
    assert torch.equal(torch.cat([chunk_inputs for chunk_inputs, _ in chunks]), inputs)
    assert torch.equal(torch.cat([chunk_targets for _, chunk_targets in chunks]), targets)


def test_chunking_rejects_non_divisible_length() -> None:
    inputs = torch.randn(5, 4)
    targets = torch.arange(5, dtype=torch.long)

    with pytest.raises(ValueError, match="divisible"):
        chunk_micro_batches(inputs, targets, 2)


def test_effective_batch_identity() -> None:
    assert effective_batch_size(8, 4, 2) == 64
    assert effective_batch_size(8, 4) == 32

    with pytest.raises(ValueError, match="world_size"):
        effective_batch_size(8, 4, 0)
