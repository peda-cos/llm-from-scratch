from __future__ import annotations

import os
from pathlib import Path

import pytest

from main import main


def _tiny_args(fixture_dir: Path) -> list[str]:
    return [
        "--fixture-dir",
        str(fixture_dir),
        "--vocab-size",
        "64",
        "--context-length",
        "16",
        "--d-model",
        "16",
        "--num-heads",
        "2",
        "--num-layers",
        "2",
        "--mlp-expansion",
        "2",
        "--max-new-tokens",
        "4",
        "--top-k",
        "8",
        "--seed",
        "3",
        "--stub-seed",
        "17",
        "--sample-seed",
        "5",
    ]


def test_main_runs_demo_and_prints_the_contract(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_dir = tmp_path / "fixtures"
    assert main(_tiny_args(fixture_dir)) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    assert f"fixture: path={fixture_dir / 'gpt2-stub.safetensors'}" in captured.out
    assert "tensors=28 parameters=5760" in captured.out
    assert "model: parameters=5760 d_model=16 num_heads=2 num_layers=2" in captured.out
    assert "prompt_ids=[1, 2, 3, 4]" in captured.out
    assert "load: h.0.attn.c_attn.weight -> blocks.0.attn.qkv.weight" in captured.out
    assert "transposed=True" in captured.out
    assert "report: loaded=28 missing=0 unexpected=0 ignored=0 shape_mismatch=0" in captured.out
    assert "continuation_before_ids=" in captured.out
    assert "continuation_after_ids=" in captured.out
    assert "logits: max_abs_delta=" in captured.out
    assert "check: tokens_changed=true" in captured.out
    assert "load_error: checkpoint validation failed:" in captured.out
    assert "report: loaded=0 missing=0 unexpected=0 ignored=0 shape_mismatch=1" in captured.out
    assert "mismatch: source=h.0.attn.c_attn.weight" in captured.out
    assert "source_shape=(16, 24)" in captured.out
    assert "expected_source_shape=(16, 48)" in captured.out
    assert "check: model_unchanged=true" in captured.out
    assert "summary: loaded=28 ignored=0 tokens_changed=true model_unchanged=true" in captured.out

    good = fixture_dir / "gpt2-stub.safetensors"
    bad = fixture_dir / "gpt2-stub-bad.safetensors"
    assert good.is_file()
    assert bad.is_file()
    assert good.stat().st_size > 0
    assert bad.stat().st_size > 0


def test_main_is_deterministic_and_reuses_fixtures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_dir = tmp_path / "fixtures"
    args = _tiny_args(fixture_dir)

    assert main(args) == 0
    first = capsys.readouterr()
    first_mtime = (fixture_dir / "gpt2-stub.safetensors").stat().st_mtime_ns

    assert main(args) == 0
    second = capsys.readouterr()
    second_mtime = (fixture_dir / "gpt2-stub.safetensors").stat().st_mtime_ns

    assert first.out == second.out
    assert first.err == second.err == ""
    assert first_mtime == second_mtime


def test_main_regenerate_rewrites_fixtures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture_dir = tmp_path / "fixtures"
    args = _tiny_args(fixture_dir)
    assert main(args) == 0
    capsys.readouterr()
    fixture = fixture_dir / "gpt2-stub.safetensors"
    old_ns = fixture.stat().st_mtime_ns - 5_000_000_000
    os.utime(fixture, ns=(old_ns, old_ns))
    assert main([*args, "--regenerate"]) == 0
    capsys.readouterr()
    assert fixture.stat().st_mtime_ns > old_ns


@pytest.mark.parametrize(
    "extra",
    [
        ["--d-model", "12", "--num-heads", "5"],
        ["--max-new-tokens", "-1"],
        ["--top-k", "0"],
        ["--temperature", "-0.5"],
        ["--vocab-size", "2"],
    ],
)
def test_main_rejects_invalid_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    args = [*_tiny_args(tmp_path / "never"), *extra]
    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err
    assert not (tmp_path / "never").exists()
