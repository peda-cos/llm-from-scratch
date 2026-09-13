from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import pytest

from main import (
    ByteTokenizer,
    ChecksumMismatch,
    CorpusDocument,
    DocumentFormatError,
    IndexFormatError,
    MmapTokenStore,
    ShardFormatError,
    ShardedTokenizationPipeline,
    ShardIndex,
    load_shard_index,
    read_jsonl_documents,
    verify_shards,
)
from support import build_corpus, shard_path


def _jsonl_payload(built: Any) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(built.index_path.read_text(encoding="utf-8"))
    return payload


def test_pipeline_writes_boundaries_and_a_consistent_index(
    tmp_path: Path,
) -> None:
    built = build_corpus(tmp_path)
    assert built.index.total_tokens == int(built.reference.size)
    assert built.index.vocab_size == 257
    assert built.index.boundary_token_id == 256
    assert built.index.chunk_size == 8
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob("*.tmp"))

    for record in built.index.shards:
        expected = built.shard_streams[record.shard_id]
        assert record.token_count == int(expected.size)
        assert record.document_count == len(built.documents[record.shard_id])
        assert expected[-1] == built.tokenizer.boundary_token_id
        with h5py.File(shard_path(built, record.shard_id), "r") as handle:
            dataset = handle["tokens"]
            assert dataset[:].tolist() == expected.tolist()
            assert int(dataset.attrs["token_count"]) == record.token_count
            assert dataset.attrs["sha256"] == record.sha256

    payload = _jsonl_payload(built)
    assert payload["schema_version"] == 1
    assert payload["generator"] == "hdf5-tokenized-corpus/0.1.0"
    assert payload["total_tokens"] == built.index.total_tokens
    assert [shard["shard_id"] for shard in payload["shards"]] == [
        shard.shard_id for shard in built.index.shards
    ]


def test_pipeline_preserves_an_empty_input_shard(tmp_path: Path) -> None:
    documents = {
        "shard-empty": (),
        "shard-data": (CorpusDocument("only", "one document"),),
    }
    built = build_corpus(tmp_path, documents=documents)
    assert built.index.shards[0].shard_id == "shard-empty"
    assert built.index.shards[0].token_count == 0
    assert built.index.shards[0].document_count == 0
    assert built.index.shards[1].token_count == int(
        built.shard_streams["shard-data"].size
    )

    with MmapTokenStore(built.index_path) as store:
        assert store.num_tokens == int(built.reference.size)
        assert store.get_slice(0, store.num_tokens).tolist() == (
            built.reference.tolist()
        )


def test_jsonl_reader_parses_documents_and_rejects_bad_lines(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.jsonl"
    good.write_text(
        '{"id": "doc-1", "text": "hello"}\n'
        "\n"
        '{"id": "doc-2", "text": "Olá 🚀"}\n',
        encoding="utf-8",
    )
    documents = list(read_jsonl_documents(good))
    assert [document.document_id for document in documents] == ["doc-1", "doc-2"]
    assert documents[1].text == "Olá 🚀"

    for line in (
        '{"id": 3, "text": "bad id"}\n',
        '{"id": "doc-3"}\n',
        "not json\n",
        '["not", "an", "object"]\n',
    ):
        path = tmp_path / "bad.jsonl"
        path.write_text(line, encoding="utf-8")
        with pytest.raises(DocumentFormatError, match="bad.jsonl"):
            list(read_jsonl_documents(path))

    with pytest.raises(DocumentFormatError, match="cannot read JSONL"):
        list(read_jsonl_documents(tmp_path / "missing.jsonl"))


def test_pipeline_rejects_unsafe_ids_and_invalid_boundaries(
    tmp_path: Path,
) -> None:
    pipeline = ShardedTokenizationPipeline(tmp_path, ByteTokenizer(), chunk_size=8)
    with pytest.raises(ValueError, match="unsafe shard id"):
        pipeline.run({"../escape": []})

    class BadBoundaryTokenizer(ByteTokenizer):
        @property
        def boundary_token_id(self) -> int:
            return 257

    with pytest.raises(ValueError, match="boundary_token_id"):
        ShardedTokenizationPipeline(tmp_path, BadBoundaryTokenizer())


def test_pipeline_removes_the_partial_shard_when_input_fails(
    tmp_path: Path,
) -> None:
    pipeline = ShardedTokenizationPipeline(tmp_path, ByteTokenizer(), chunk_size=8)

    def failing_documents() -> Iterator[CorpusDocument]:
        yield CorpusDocument("first", "this document is written")
        raise RuntimeError("simulated input failure")

    with pytest.raises(RuntimeError, match="simulated input failure"):
        pipeline.run({"shard-0000": failing_documents()})
    assert not list(tmp_path.glob("*.partial"))
    assert not (tmp_path / "shards.json").exists()


def test_verify_shards_reports_and_detects_corruption(tmp_path: Path) -> None:
    built = build_corpus(tmp_path)
    report = verify_shards(built.index_path)
    assert report.shards_checked == len(built.index.shards)
    assert report.tokens_checked == built.index.total_tokens
    assert report.bytes_checked == sum(
        shard.byte_size for shard in built.index.shards
    )

    path = shard_path(built, "shard-0000")
    with h5py.File(path, "r+") as handle:
        dataset = handle["tokens"]
        original = int(dataset[0])
        dataset[0] = (original + 1) % 256
    with pytest.raises(ChecksumMismatch, match="shard-0000"):
        verify_shards(built.index_path)


def test_verify_shards_detects_missing_and_wrong_sized_files(
    tmp_path: Path,
) -> None:
    built = build_corpus(tmp_path / "missing")
    shard_path(built, "shard-0000").unlink()
    with pytest.raises(ShardFormatError, match="missing"):
        verify_shards(built.index_path)

    built = build_corpus(tmp_path / "size")
    path = shard_path(built, "shard-0000")
    with path.open("ab") as handle:
        handle.write(b"\x00")
    with pytest.raises(ShardFormatError, match="byte size"):
        verify_shards(built.index_path)


def test_index_validation_rejects_each_contract_violation(
    tmp_path: Path,
) -> None:
    built = build_corpus(tmp_path)
    base = _jsonl_payload(built)

    wrong_version = copy.deepcopy(base)
    wrong_version["schema_version"] = 2
    with pytest.raises(IndexFormatError, match="schema_version"):
        ShardIndex.from_dict(wrong_version)

    wrong_total = copy.deepcopy(base)
    wrong_total["total_tokens"] += 1
    with pytest.raises(IndexFormatError, match="total_tokens"):
        ShardIndex.from_dict(wrong_total)

    upper_sha = copy.deepcopy(base)
    upper_sha["shards"][0]["sha256"] = "A" * 64
    with pytest.raises(IndexFormatError, match="sha256"):
        ShardIndex.from_dict(upper_sha)

    traversal = copy.deepcopy(base)
    traversal["shards"][0]["path"] = "../escape.h5"
    with pytest.raises(IndexFormatError, match="bare filename"):
        ShardIndex.from_dict(traversal)

    duplicate = copy.deepcopy(base)
    duplicate["shards"].append(copy.deepcopy(duplicate["shards"][0]))
    duplicate["total_tokens"] += duplicate["shards"][0]["token_count"]
    with pytest.raises(IndexFormatError, match="duplicate shard"):
        ShardIndex.from_dict(duplicate)

    with pytest.raises(IndexFormatError, match="cannot read shard index"):
        load_shard_index(tmp_path / "missing-index.json")
