from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from main import (
    clip_global_l2_norm,
    global_grad_l2_norm,
    has_non_finite_grad,
)


class _GradModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(torch.zeros(2))
        self.second = nn.Parameter(torch.zeros(2))


def test_grad_norm_is_concatenated_l2_norm() -> None:
    module = _GradModule()
    module.first.grad = torch.tensor([3.0, 4.0])
    module.second.grad = torch.tensor([12.0, 16.0])
    assert global_grad_l2_norm(module.parameters()) == pytest.approx(
        math.sqrt(425.0)
    )

    module.second.grad = None
    assert global_grad_l2_norm(module.parameters()) == pytest.approx(5.0)

    assert global_grad_l2_norm(_GradModule().parameters()) == 0.0


def test_grad_norm_accumulates_large_float16_in_float32() -> None:
    parameter = nn.Parameter(torch.zeros(2, dtype=torch.float16))
    parameter.grad = torch.tensor(
        [40000.0, 0.0], dtype=torch.float16
    )
    assert global_grad_l2_norm([parameter]) == pytest.approx(40000.0)


def test_clipping_reports_pre_and_post_norms_and_preserves_direction() -> None:
    module = _GradModule()
    module.first.grad = torch.tensor([3.0, 4.0])
    module.second.grad = torch.tensor([12.0, 16.0])

    report = clip_global_l2_norm(module.parameters(), max_norm=0.5)

    assert report.pre_clip_norm == pytest.approx(math.sqrt(425.0))
    assert report.post_clip_norm == pytest.approx(0.5)
    assert report.clipped is True
    assert module.first.grad is not None
    assert module.second.grad is not None
    clipped_first = module.first.grad
    assert clipped_first[0] / clipped_first[1] == pytest.approx(3.0 / 4.0)


def test_no_clipping_returns_equal_norms_and_leaves_gradients_untouched() -> None:
    module = _GradModule()
    module.first.grad = torch.tensor([3.0, 4.0])
    module.second.grad = torch.tensor([12.0, 16.0])
    before: list[torch.Tensor] = []
    for parameter in module.parameters():
        assert parameter.grad is not None
        before.append(parameter.grad.clone())

    report = clip_global_l2_norm(module.parameters(), max_norm=100.0)

    assert report.clipped is False
    assert report.pre_clip_norm == pytest.approx(report.post_clip_norm)
    for parameter, expected in zip(
        module.parameters(), before, strict=True
    ):
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, expected)


def test_clip_rejects_nonpositive_or_nonfinite_max_norm() -> None:
    module = _GradModule()
    with pytest.raises(ValueError, match="max_norm must be > 0"):
        clip_global_l2_norm(module.parameters(), max_norm=0.0)
    with pytest.raises(ValueError, match="must be finite"):
        clip_global_l2_norm(module.parameters(), max_norm=float("inf"))


def test_non_finite_grad_scan_covers_nan_inf_and_none() -> None:
    module = _GradModule()
    assert has_non_finite_grad(module.parameters()) is False

    module.first.grad = torch.tensor([1.0, float("inf")])
    assert has_non_finite_grad(module.parameters()) is True

    module.first.grad = torch.tensor([1.0, float("nan")])
    assert has_non_finite_grad(module.parameters()) is True

    module.first.grad = torch.tensor([1.0, 2.0])
    assert has_non_finite_grad(module.parameters()) is False
