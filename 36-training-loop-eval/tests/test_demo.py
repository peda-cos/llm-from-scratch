from __future__ import annotations

import json
from pathlib import Path

import pytest

from main import TOY_PHRASE, TOY_WORDS, main, make_synthetic_corpus


def _demo_args(output: Path) -> list[str]:
    return [
        "--steps",
        "8",
        "--eval-every",
        "4",
        "--eval-batches",
        "2",
        "--batch-size",
        "4",
        "--max-new-tokens",
        "6",
        "--seed",
        "7",
        "--output",
        str(output),
    ]


def test_main_runs_demo_and_writes_jsonl(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "demo" / "losses.jsonl"
    assert main(_demo_args(output)) == 0
    captured = capsys.readouterr()

    assert "training loop and evaluation" in captured.out
    assert 'corpus: toy_phrase=[1, 2, 3, 4, 1, 5] words="the cat sat on the mat"' in captured.out
    assert "data: train_tokens=" in captured.out
    assert "model: parameters=" in captured.out
    assert "train: steps=8 " in captured.out
    assert "eval step=004 loss=" in captured.out
    assert "eval step=008 loss=" in captured.out
    assert "sample step=004 ids=" in captured.out
    assert "sample step=008 ids=" in captured.out
    assert "summary: first_train_loss=" in captured.out
    assert f"log: {output} records=8" in captured.out
    assert captured.err == ""

    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 8
    assert [row["step"] for row in rows] == list(range(1, 9))
    assert all(set(row) == {"step", "train_loss", "lr"} for row in rows)
    generated_line = next(
        line for line in captured.out.splitlines() if line.startswith("sample step=008")
    )
    generated = json.loads(generated_line.split(" ids=", 1)[1])
    assert all(0 <= token < len(TOY_WORDS) for token in generated)
    assert len(generated) == len(TOY_PHRASE[:4]) + 6


def test_main_demo_is_deterministic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first_output = tmp_path / "first.jsonl"
    second_output = tmp_path / "second.jsonl"
    first_args = _demo_args(first_output)
    second_args = _demo_args(second_output)

    assert main(first_args) == 0
    first = capsys.readouterr()
    assert main(second_args) == 0
    second = capsys.readouterr()

    assert first_output.read_bytes() == second_output.read_bytes()
    first_probe = [
        line
        for line in first.out.splitlines()
        if line.startswith(("eval ", "sample "))
    ]
    second_probe = [
        line
        for line in second.out.splitlines()
        if line.startswith(("eval ", "sample "))
    ]
    assert first_probe == second_probe
    assert first.err == second.err == ""


def test_main_rejects_invalid_arguments(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for argv in (
        ["--steps", "0"],
        ["--eval-every", "0"],
        ["--eval-batches", "0"],
        ["--batch-size", "0"],
        ["--max-new-tokens", "-1"],
    ):
        assert main(argv) == 2
        captured = capsys.readouterr()
        assert "error:" in captured.err
        assert captured.out == ""


def test_synthetic_corpus_is_periodic() -> None:
    corpus = make_synthetic_corpus(3)
    assert corpus.tolist() == list(TOY_PHRASE) * 3
    assert bool(((corpus >= 0) & (corpus < len(TOY_WORDS))).all())
    with pytest.raises(ValueError, match="repetitions"):
        make_synthetic_corpus(0)
    with pytest.raises(TypeError, match="repetitions"):
        make_synthetic_corpus(True)
