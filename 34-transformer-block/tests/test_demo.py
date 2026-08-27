from __future__ import annotations

import math

import pytest

from main import gradient_probe, main


def _probe_kwargs() -> dict[str, int]:
    return {
        "d_model": 16,
        "num_heads": 2,
        "num_layers": 2,
        "max_context_length": 4,
        "batch_size": 2,
        "vocabulary_size": 32,
        "seed": 1,
    }


def _tiny_argv() -> list[str]:
    return [
        "--d-model",
        "16",
        "--heads",
        "2",
        "--layers",
        "2",
        "--context-length",
        "4",
        "--batch-size",
        "2",
        "--vocab-size",
        "32",
        "--seed",
        "1",
    ]


def test_gradient_probe_pre_ln_embedding_gradient_is_larger() -> None:
    probe = gradient_probe(**_probe_kwargs())

    assert probe.pre_output_shape == (2, 4, 16)
    assert probe.post_output_shape == (2, 4, 16)
    assert probe.embedding_grad_ratio > 10.0
    assert probe.max_output_difference > 0.0
    assert probe.pre_parameter_count == probe.post_parameter_count
    assert probe.causal_future_weight_max == 0.0
    assert probe.causal_row_sum_max_error <= 1e-5
    assert len(probe.pre_block_grad_norms) == 2
    assert len(probe.post_block_grad_norms) == 2
    values = (
        probe.pre_probe_loss,
        probe.post_probe_loss,
        probe.pre_embedding_grad_norm,
        probe.post_embedding_grad_norm,
        probe.embedding_grad_ratio,
        probe.max_output_difference,
        *probe.pre_block_grad_norms,
        *probe.post_block_grad_norms,
    )
    assert all(math.isfinite(value) for value in values)


def test_demo_prints_shapes_and_gradients(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_tiny_argv()) == 0
    captured = capsys.readouterr()
    assert "pre_ln: output_shape=(2, 4, 16)" in captured.out
    assert "post_ln: output_shape=(2, 4, 16)" in captured.out
    assert "embedding_grad_ratio=" in captured.out
    assert "parameter_counts:" in captured.out
    assert "causal_check:" in captured.out


def test_demo_is_deterministic() -> None:
    first = gradient_probe(**_probe_kwargs())
    second = gradient_probe(**_probe_kwargs())

    for field in (
        "pre_probe_loss",
        "post_probe_loss",
        "pre_embedding_grad_norm",
        "post_embedding_grad_norm",
        "embedding_grad_ratio",
        "max_output_difference",
        "causal_future_weight_max",
        "causal_row_sum_max_error",
    ):
        assert getattr(first, field) == pytest.approx(
            getattr(second, field), abs=1e-12
        )
    assert first.pre_block_grad_norms == pytest.approx(
        second.pre_block_grad_norms, abs=1e-12
    )
    assert first.post_block_grad_norms == pytest.approx(
        second.post_block_grad_norms, abs=1e-12
    )


def test_demo_rejects_invalid_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for argv in (
        ["--layers", "1"],
        ["--d-model", "15", "--heads", "4"],
        ["--heads", "0"],
    ):
        assert main(argv) == 2
        captured = capsys.readouterr()
        assert "error:" in captured.err
