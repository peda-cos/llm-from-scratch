from __future__ import annotations

import pytest

from main import (
    GPTConfig,
    GPTModel,
    build_assembly_report,
    count_parameters,
    main,
)


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        context_length=8,
        d_model=16,
        num_heads=4,
        num_layers=2,
        mlp_expansion=4,
        dropout=0.0,
        use_bias=True,
        weight_tying=True,
    )


def test_assembly_report_counts_and_forward() -> None:
    config = _tiny_config()
    report = build_assembly_report(config, seed=3)

    assert report.parameters == count_parameters(
        GPTModel(config, seed=3)
    )
    assert (
        report.untied_parameters - report.parameters
        == config.vocab_size * config.d_model
    )
    assert report.reference_parameters == 124_000_000
    assert report.lm_head_shared
    assert report.forward_shape == (1, 8, 32)
    assert report.forward_ms >= 0.0


def test_demo_tiny_only_output_is_deterministic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = [
        "--tiny-only",
        "--tiny-d-model",
        "32",
        "--tiny-heads",
        "4",
        "--tiny-layers",
        "2",
        "--tiny-context-length",
        "8",
        "--tiny-vocab-size",
        "64",
        "--max-new-tokens",
        "4",
        "--temperature",
        "0.8",
        "--top-k",
        "10",
        "--seed",
        "5",
    ]

    assert main(argv) == 0
    first = capsys.readouterr()
    assert main(argv) == 0
    second = capsys.readouterr()

    assert first.out == second.out
    assert first.err == ""
    assert "tiny: parameters=" in first.out
    assert "tiny: prompt_ids=[1, 2, 3, 4]" in first.out
    assert "tiny: generated_ids=" in first.out
    assert "tiny: repeat_match=True" in first.out


def test_demo_rejects_invalid_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for argv in (
        ["--tiny-d-model", "30", "--tiny-heads", "4"],
        ["--tiny-vocab-size", "4"],
        ["--max-new-tokens", "-1"],
        ["--top-k", "0"],
        ["--temperature", "-1"],
    ):
        assert main(argv) == 2
        captured = capsys.readouterr()
        assert "error:" in captured.err
