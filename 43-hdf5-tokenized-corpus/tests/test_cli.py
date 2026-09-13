from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

from main import load_shard_index, main, run_benchmark


def test_demo_is_repeatable_and_validates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    workdir = tmp_path / "demo"
    assert main(["demo", "--workdir", str(workdir)]) == 0
    output = capsys.readouterr().out
    assert sum(line.startswith("batch=") for line in output.splitlines()) == 10
    assert "validated:" in output
    index_path = workdir / "shards.json"
    assert index_path.is_file()
    assert main(["validate", "--index", str(index_path)]) == 0

    assert main(["demo", "--workdir", str(workdir)]) == 0
    assert main(["validate", "--index", str(index_path)]) == 0


def test_no_arguments_runs_the_demo_in_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main([]) == 0
    assert (tmp_path / "outputs" / "demo" / "shards.json").is_file()
    assert capsys.readouterr().out.count("batch=") == 10


def test_build_consumes_a_jsonl_shard_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name, documents in {
        "shard-a": [("a-1", "first document"), ("a-2", "second document")],
        "shard-b": [("b-1", "third document")],
    }.items():
        lines = [
            json.dumps({"id": doc_id, "text": text}, ensure_ascii=False)
            for doc_id, text in documents
        ]
        (input_dir / f"{name}.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )

    workdir = tmp_path / "corpus"
    assert (
        main(
            [
                "build",
                "--input-dir",
                str(input_dir),
                "--workdir",
                str(workdir),
                "--chunk-size",
                "8",
            ]
        )
        == 0
    )
    index_path = workdir / "shards.json"
    index = load_shard_index(index_path)
    assert [shard.shard_id for shard in index.shards] == ["shard-a", "shard-b"]
    assert sum(shard.document_count for shard in index.shards) == 3
    assert len(list(workdir.glob("*.h5"))) == 2
    assert main(["validate", "--index", str(index_path)]) == 0
    assert "built:" in capsys.readouterr().out


def test_build_and_validate_report_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["build", "--input-dir", str(empty)]) == 1
    assert "no *.jsonl shards" in capsys.readouterr().err

    demo_dir = tmp_path / "demo"
    assert main(["demo", "--workdir", str(demo_dir)]) == 0
    capsys.readouterr()
    index_path = demo_dir / "shards.json"
    record = load_shard_index(index_path).shards[0]
    shard_path = demo_dir / record.path
    with h5py.File(shard_path, "r+") as handle:
        dataset = handle["tokens"]
        dataset[0] = (int(dataset[0]) + 1) % 256
    assert main(["validate", "--index", str(index_path)]) == 1
    assert "expected" in capsys.readouterr().err


def test_benchmark_compares_three_chunk_sizes(tmp_path: Path) -> None:
    rows = run_benchmark(
        tmp_path / "bench",
        window_size=8,
        batch_size=2,
        samples=5,
        shards=2,
        documents_per_shard=4,
        seed=1,
    )
    assert len(rows) == 3
    assert [row.chunk_size for row in rows] == [4, 8, 16]
    assert len({row.token_count for row in rows}) == 1
    assert rows[0].token_count > 0
    assert all(row.byte_size > 0 for row in rows)
    assert all(row.build_seconds > 0 for row in rows)
    assert all(row.read_seconds > 0 for row in rows)
    for row in rows:
        assert (tmp_path / "bench" / f"chunk-{row.chunk_size}" / "shards.json").is_file()
