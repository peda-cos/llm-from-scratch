from __future__ import annotations

import math

import pytest

from main import cosine_with_warmup


def test_cosine_with_warmup_boundaries() -> None:
    assert cosine_with_warmup(0, 4, 10, 1.0, 0.1) == 0.0
    assert cosine_with_warmup(4, 4, 10, 1.0, 0.1) == pytest.approx(1.0)
    assert cosine_with_warmup(10, 4, 10, 1.0, 0.1) == pytest.approx(0.1)
    assert cosine_with_warmup(99, 4, 10, 1.0, 0.1) == pytest.approx(0.1)
    assert cosine_with_warmup(0, 0, 10, 1.0, 0.1) == pytest.approx(1.0)
    assert cosine_with_warmup(5, 0, 10, 1.0, 0.1) == pytest.approx(0.55)
    assert cosine_with_warmup(30, 4, 100, 0.003, 0.0003) >= 0.0003


def test_cosine_with_warmup_is_monotonic_on_both_phases() -> None:
    warmup = [
        cosine_with_warmup(step, 4, 20, 1.0, 0.1) for step in range(5)
    ]
    decay = [
        cosine_with_warmup(step, 4, 20, 1.0, 0.1)
        for step in range(4, 21)
    ]
    assert all(
        warmup[index] <= warmup[index + 1]
        for index in range(len(warmup) - 1)
    )
    assert all(
        decay[index] >= decay[index + 1]
        for index in range(len(decay) - 1)
    )
    midpoint = cosine_with_warmup(10, 4, 16, 1.0, 0.0)
    assert midpoint == pytest.approx(0.5)
    assert math.isclose(decay[-1], 0.1)


def test_cosine_with_warmup_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="step"):
        cosine_with_warmup(-1, 4, 10, 1.0, 0.1)
    with pytest.raises(ValueError, match="warmup_steps"):
        cosine_with_warmup(0, 10, 10, 1.0, 0.1)
    with pytest.raises(ValueError, match="total_steps"):
        cosine_with_warmup(0, 0, 0, 1.0, 0.1)
    with pytest.raises(ValueError, match="max_lr"):
        cosine_with_warmup(0, 1, 10, 0.0, 0.0)
    with pytest.raises(ValueError, match="min_lr"):
        cosine_with_warmup(0, 1, 10, 1.0, 2.0)
    with pytest.raises(TypeError, match="step"):
        cosine_with_warmup(1.5, 0, 10, 1.0, 0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="total_steps"):
        cosine_with_warmup(0, 0, True, 1.0, 0.1)
