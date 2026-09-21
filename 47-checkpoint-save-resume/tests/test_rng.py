from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from main import _seed_training_rng, capture_rng_state, restore_rng_state
from support import assert_plain_python_value


def test_capture_does_not_advance_rng() -> None:
    first = capture_rng_state()
    second = capture_rng_state()

    assert first == second


def test_rng_round_trip_preserves_python_numpy_and_torch_streams() -> None:
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    random.gauss(0.0, 1.0)
    np.random.randn(1)
    torch.rand(3)

    state = capture_rng_state()
    expected_python = random.gauss(0.0, 1.0)
    expected_numpy = float(np.random.randn())
    expected_torch = torch.rand(4)

    random.seed(9)
    np.random.seed(9)
    torch.manual_seed(9)
    random.gauss(0.0, 1.0)
    np.random.randn(5)
    torch.rand(7)

    restore_rng_state(state)

    assert random.gauss(0.0, 1.0) == expected_python
    assert float(np.random.randn()) == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)


def test_rng_payload_is_plain_python_values() -> None:
    state = capture_rng_state()

    assert_plain_python_value(state)
    assert isinstance(state["torch_cpu"], list)
    assert isinstance(state["numpy"], dict)


def test_seed_training_rng_reproduces_all_streams() -> None:
    _seed_training_rng(17)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    _seed_training_rng(17)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))

    assert first == second


def test_restore_rejects_missing_python_state() -> None:
    with pytest.raises(ValueError, match="rng.python"):
        restore_rng_state({})


def test_restore_rejects_out_of_range_torch_byte() -> None:
    state = copy.deepcopy(capture_rng_state())
    state["torch_cpu"] = [256]

    with pytest.raises(ValueError, match="torch_cpu"):
        restore_rng_state(state)


def test_restore_rejects_missing_numpy_position() -> None:
    state = copy.deepcopy(capture_rng_state())
    numpy_state = state["numpy"]
    assert isinstance(numpy_state, dict)
    del numpy_state["pos"]

    with pytest.raises(ValueError, match="rng.numpy.pos"):
        restore_rng_state(state)


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_cuda_state_is_empty_without_cuda() -> None:
    assert capture_rng_state()["torch_cuda"] == []
