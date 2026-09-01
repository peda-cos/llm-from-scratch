from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from main import calc_loss_batch, make_batches


def test_make_batches_shifts_targets_by_one() -> None:
    tokens = torch.arange(32, dtype=torch.long)
    batches = make_batches(tokens, batch_size=2, context_length=4)

    assert len(batches) == 3
    first_inputs, first_targets = batches[0]
    assert first_inputs.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert first_targets.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]

    last_inputs, last_targets = batches[-1]
    assert last_inputs.tolist() == [[16, 17, 18, 19], [20, 21, 22, 23]]
    assert last_targets.tolist() == [[17, 18, 19, 20], [21, 22, 23, 24]]
    assert torch.equal(last_targets[:, :-1], last_inputs[:, 1:])


def test_make_batches_is_deterministic_and_drops_partial_tail() -> None:
    tokens = torch.arange(45, dtype=torch.long)
    batch_size, context_length = 3, 4
    batches = make_batches(tokens, batch_size, context_length)
    expected = ((45 - 1) // context_length) // batch_size

    assert len(batches) == expected == 3
    for inputs, targets in batches:
        assert inputs.shape == (batch_size, context_length)
        assert targets.shape == (batch_size, context_length)
        assert torch.equal(targets[:, :-1], inputs[:, 1:])
    again = make_batches(tokens, batch_size, context_length)
    assert all(
        torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        for a, b in zip(batches, again, strict=True)
    )


def test_make_batches_uses_minimum_window_at_boundary() -> None:
    tokens = torch.arange(9, dtype=torch.long)
    batches = make_batches(tokens, batch_size=2, context_length=4)
    assert len(batches) == 1
    inputs, targets = batches[0]
    assert inputs.tolist() == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert targets.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]


def test_make_batches_rejects_invalid_inputs() -> None:
    tokens = torch.arange(32, dtype=torch.long)
    with pytest.raises(ValueError, match="token_ids"):
        make_batches(tokens.unsqueeze(0), 1, 4)
    with pytest.raises(TypeError, match="integer"):
        make_batches(tokens.float(), 1, 4)
    with pytest.raises(TypeError, match="integer"):
        make_batches(tokens.bool(), 1, 4)
    with pytest.raises(ValueError, match="batch_size"):
        make_batches(tokens, 0, 4)
    with pytest.raises(ValueError, match="context_length"):
        make_batches(tokens, 1, 0)
    with pytest.raises(ValueError, match="at least"):
        make_batches(torch.arange(8, dtype=torch.long), 2, 4)


class FixedLogits(nn.Module):
    def __init__(self, logits: Tensor) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, inputs: Tensor) -> Tensor:
        return self.logits


def test_calc_loss_batch_matches_manual_cross_entropy() -> None:
    torch.manual_seed(4)
    logits = torch.randn(2, 3, 5)
    targets = torch.tensor([[1, 2, 3], [4, 0, 1]], dtype=torch.long)
    expected = F.cross_entropy(logits.reshape(-1, 5), targets.reshape(-1))

    actual = calc_loss_batch(FixedLogits(logits), torch.zeros_like(targets), targets)
    assert torch.allclose(actual, expected)
    different = calc_loss_batch(
        FixedLogits(logits), torch.zeros_like(targets), targets.flip(0)
    )
    assert not torch.allclose(actual, different)


class ParameterLogits(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.linspace(-1.0, 1.0, 5))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.bias.reshape(1, 1, 5).expand(
            inputs.shape[0], inputs.shape[1], 5
        )


def test_calc_loss_batch_preserves_gradient_and_validates() -> None:
    model = ParameterLogits()
    inputs = torch.zeros((2, 3), dtype=torch.long)
    targets = torch.tensor([[0, 1, 2], [3, 4, 0]], dtype=torch.long)
    loss = calc_loss_batch(model, inputs, targets)
    torch.autograd.backward(loss)

    assert model.bias.grad is not None
    assert bool(torch.isfinite(model.bias.grad).all())
    assert float(model.bias.grad.abs().sum()) > 0.0

    with pytest.raises(ValueError, match="does not match"):
        calc_loss_batch(model, inputs, targets[:, :2])
    with pytest.raises(TypeError, match="integer"):
        calc_loss_batch(model, inputs, targets.float())
    with pytest.raises(ValueError, match="logits"):
        calc_loss_batch(FixedLogits(torch.zeros(2, 3)), inputs, targets)
    with pytest.raises(ValueError, match="logits"):
        calc_loss_batch(FixedLogits(torch.zeros(1, 2, 5)), inputs, targets)
