from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
import zstandard

from main import (
    DEFAULT_MAX_LINE_BYTES,
    CorpusDocument,
    CorpusPipeline,
    Dedup,
    DedupConfig,
    DocumentFormatError,
    LSHCandidate,
    LSHIndex,
    ManifestError,
    MinHasher,
    PipelineResult,
    PlanError,
    ShardPlanner,
    StreamingDownloader,
    ZstdDocIterator,
    build_demo_corpus,
    main,
    run_demo,
    verify_manifest,
)


def _jsonl(docs: Sequence[tuple[str, str]]) -> bytes:
    lines = [
        json.dumps({"id": doc_id, "text": text}, ensure_ascii=False, sort_keys=True)
        for doc_id, text in docs
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _zstd(data: bytes) -> bytes:
    return zstandard.ZstdCompressor(level=3).compress(data)


def _write_shard(path: Path, docs: Sequence[tuple[str, str]]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = _zstd(_jsonl(docs))
    path.write_bytes(compressed)
    return compressed


def _doc(
    shard_id: str,
    index: int,
    text: str,
    *,
    doc_id: str,
    raw: bytes | None = None,
) -> CorpusDocument:
    raw_bytes = raw if raw is not None else json.dumps(
        {"id": doc_id, "text": text}, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return CorpusDocument(
        shard_id=shard_id,
        index=index,
        offset=0,
        length=len(raw_bytes),
        raw=raw_bytes,
        obj={"id": doc_id, "text": text},
        text=text,
        doc_id=doc_id,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


class _GuardedReader(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.max_request = 0
        self.read_all_calls = 0

    def read(self, size: int | None = -1) -> bytes:
        if size is None or size < 0:
            self.read_all_calls += 1
            raise AssertionError("read() without a size is not allowed")
        self.max_request = max(self.max_request, size)
        return super().read(size)


def _run_file_pipeline(tmp_path: Path) -> PipelineResult:
    remote = tmp_path / "remote"
    urls, sizes, hashes = build_demo_corpus(remote)
    pipeline = CorpusPipeline(workdir=tmp_path / "run")
    return pipeline.process(urls, expected_sizes=sizes, expected_sha256=hashes)


def _rewrite_manifest(manifest_path: Path, manifest: dict[str, object]) -> None:
    data = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    manifest_path.write_bytes(data)
    lock_path = manifest_path.with_suffix(".lock")
    lock_path.write_text(
        f"{hashlib.sha256(data).hexdigest()}  {manifest_path.name}\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Planner and document streaming
# ---------------------------------------------------------------------------


def test_planner_derives_ids_paths_and_public_urls(tmp_path: Path) -> None:
    planner = ShardPlanner(tmp_path / "cache", tmp_path / "corpus")
    urls = [
        "file:///data/shard-0001.jsonl.zst?token=hidden",
        "https://example.invalid/nested/part-02.ndjson.zst",
    ]
    plans = planner.plan(
        urls,
        expected_sizes={"shard-0001": 12},
        expected_sha256={"part-02": "A" * 64},
    )
    assert [plan.shard_id for plan in plans] == ["shard-0001", "part-02"]
    assert plans[0].cache_path == tmp_path / "cache" / "shard-0001.jsonl.zst"
    assert plans[0].partial_path == tmp_path / "cache" / "shard-0001.jsonl.zst.part"
    assert (
        plans[0].checkpoint_path
        == tmp_path / "cache" / "shard-0001.jsonl.zst.partial.json"
    )
    assert plans[1].output_path == tmp_path / "corpus" / "part-02.jsonl"
    assert plans[0].expected_size == 12
    assert plans[1].expected_sha256 == "a" * 64
    assert planner.derive_shard_id(urls[0]) == "shard-0001"
    # The public URL strips the query and fragment.
    from main import _public_url  # small helper with no state

    assert _public_url(urls[0]) == "file:///data/shard-0001.jsonl.zst"
    assert (
        _public_url("https://user:pass@example.invalid/corpus/shard.zst?token=redacted")
        == "https://example.invalid/corpus/shard.zst"
    )


def test_planner_rejects_invalid_plans(tmp_path: Path) -> None:
    planner = ShardPlanner(tmp_path / "cache", tmp_path / "corpus")
    with pytest.raises(PlanError, match="at least one"):
        planner.plan([])
    with pytest.raises(PlanError, match="unsupported URL scheme"):
        planner.plan(["ftp://example.invalid/shard.jsonl.zst"])
    with pytest.raises(PlanError, match="duplicate shard id"):
        planner.plan(
            [
                "file:///one/shard-0001.jsonl.zst",
                "file:///two/shard-0001.jsonl.zst",
            ]
        )
    with pytest.raises(PlanError, match="unsafe or invalid shard id"):
        planner.plan(["file:///data/a..b.jsonl.zst"])
    with pytest.raises(PlanError, match="unplanned shard"):
        planner.plan(["file:///data/shard-0001.jsonl.zst"], expected_sizes={"other": 3})
    with pytest.raises(PlanError, match="non-negative"):
        planner.plan(["file:///data/shard-0001.jsonl.zst"], expected_sizes={"shard-0001": -1})
    with pytest.raises(PlanError, match="64 hex"):
        planner.plan(
            ["file:///data/shard-0001.jsonl.zst"],
            expected_sha256={"shard-0001": "not-a-hash"},
        )


def test_zstd_iterator_streams_documents_and_tracks_reads(tmp_path: Path) -> None:
    docs = [
        ("d-001", "alpha beta gamma delta epsilon"),
        ("d-002", "zeta eta theta iota kappa lambda"),
        ("d-003", "mu nu xi omicron pi rho sigma"),
    ]
    raw = _jsonl(docs)
    reader = _GuardedReader(_zstd(raw))
    iterator = ZstdDocIterator(reader, shard_id="shard-x", read_size=7)
    parsed = list(iterator)
    assert [document.doc_id for document in parsed] == ["d-001", "d-002", "d-003"]
    assert [document.index for document in parsed] == [1, 2, 3]
    assert [document.length for document in parsed] == [
        len(document.raw) for document in parsed
    ]
    assert [document.offset for document in parsed] == [
        0,
        parsed[0].length + 1,
        parsed[0].length + parsed[1].length + 2,
    ]
    assert parsed[0].sha256 == hashlib.sha256(parsed[0].raw).hexdigest()
    assert parsed[1].text == docs[1][1]
    assert iterator.bytes_read == len(raw)
    assert reader.read_all_calls == 0
    assert reader.max_request <= 7


def test_zstd_iterator_decodes_concatenated_frames() -> None:
    first = _zstd(_jsonl([("a", "alpha beta gamma")]))
    second = _zstd(_jsonl([("b", "delta epsilon zeta")]))
    iterator = ZstdDocIterator(
        _GuardedReader(first + second), shard_id="multi", read_size=5
    )
    parsed = list(iterator)
    assert [document.doc_id for document in parsed] == ["a", "b"]
    assert [document.text for document in parsed] == ["alpha beta gamma", "delta epsilon zeta"]


@pytest.mark.parametrize(
    ("raw", "max_line"),
    [
        (b"\n", DEFAULT_MAX_LINE_BYTES),
        (b'{"text":\n', DEFAULT_MAX_LINE_BYTES),
        (b"[]\n", DEFAULT_MAX_LINE_BYTES),
        (b'{"id":"x"}\n', DEFAULT_MAX_LINE_BYTES),
        (b'{"text":123}\n', DEFAULT_MAX_LINE_BYTES),
        (b"\xff\n", DEFAULT_MAX_LINE_BYTES),
        (b'{"text":"' + b"a" * 80 + b'"}\n', 40),
    ],
)
def test_zstd_iterator_rejects_malformed_jsonl(raw: bytes, max_line: int) -> None:
    iterator = ZstdDocIterator(
        _GuardedReader(_zstd(raw)),
        shard_id="bad",
        read_size=9,
        max_line_bytes=max_line,
    )
    with pytest.raises(DocumentFormatError):
        list(iterator)


# ---------------------------------------------------------------------------
# MinHash, LSH, dedup
# ---------------------------------------------------------------------------


def test_minhasher_is_deterministic_and_discriminative() -> None:
    hasher = MinHasher()
    other = MinHasher()
    text = "one two three four five six seven eight nine ten"
    assert hasher.signature(text) == other.signature(text)
    assert len(hasher.signature(text)) == 128
    assert hasher.signature(text) != hasher.signature("completely different words here")
    assert hasher.signature("") == (0,) * 128
    assert hasher.tokens("Hello, WORLD!") == ("hello", "world")
    assert hasher.shingles("one two") == frozenset({("one", "two")})
    assert hasher.shingles("") == frozenset()
    with pytest.raises(ValueError):
        hasher.similarity((1, 2), (1, 2))


def test_minhash_estimates_jaccard_of_shingle_sets() -> None:
    hasher = MinHasher(shingle_width=5)
    left = " ".join(f"w{index}" for index in range(40))
    right = " ".join([f"w{index}" for index in range(20)] + [f"x{index}" for index in range(20)])
    left_shingles = hasher.shingles(left)
    right_shingles = hasher.shingles(right)
    exact = len(left_shingles & right_shingles) / len(left_shingles | right_shingles)
    estimate = hasher.similarity(hasher.signature(left), hasher.signature(right))
    assert abs(estimate - exact) < 0.10

    base = (
        "the archive keeps every ledger from the northern ports and each entry records "
        "the ship the cargo the harbour fee and the day the record was signed"
    )
    footer = base + " retrieved from the public mirror"
    assert hasher.similarity(hasher.signature(base), hasher.signature(footer)) > 0.8
    unrelated = "rocket orbit telemetry payload fairing nozzle propellant combustion chamber"
    assert hasher.similarity(hasher.signature(base), hasher.signature(unrelated)) < 0.2


def test_lsh_buckets_and_collisions() -> None:
    index = LSHIndex(num_permutations=8, bands=2, rows=4)
    candidate = LSHCandidate(shard_id="s1", doc_id="d1", signature=tuple(range(8)))
    index.add(candidate)
    assert len(index) == 1
    assert index.query(tuple(range(8))) == (candidate,)
    # One changed component still shares the second band.
    assert index.query((99, 1, 2, 3, 4, 5, 6, 7)) == (candidate,)
    # Changing every component in both bands removes the collision.
    assert index.query((98, 97, 96, 95, 94, 93, 92, 91)) == ()
    with pytest.raises(ValueError):
        index.query(tuple(range(7)))
    with pytest.raises(ValueError):
        LSHIndex(num_permutations=8, bands=3, rows=4)


def test_dedup_tombstones_exact_and_near_duplicates_across_shards() -> None:
    dedup = Dedup(DedupConfig())
    base_text = (
        "the archive keeps every ledger from the northern ports and each entry records "
        "the ship the cargo the harbour fee and the day the record was signed"
    )
    footer_text = base_text + " retrieved from the public mirror"
    unrelated = "rocket orbit telemetry payload fairing nozzle propellant combustion chamber"

    first = _doc("shard-0000", 1, base_text, doc_id="a-001")
    assert dedup.classify(first).verdict == "keep"
    text_duplicate = _doc("shard-0001", 1, base_text, doc_id="b-001")
    text_verdict = dedup.classify(text_duplicate)
    assert text_verdict.verdict == "near_duplicate"
    assert text_verdict.matched_shard_id == "shard-0000"
    assert text_verdict.matched_doc_id == "a-001"
    assert text_verdict.estimated_jaccard == 1.0
    footer = _doc("shard-0002", 1, footer_text, doc_id="c-001")
    footer_verdict = dedup.classify(footer)
    assert footer_verdict.verdict == "near_duplicate"
    assert footer_verdict.estimated_jaccard is not None
    assert footer_verdict.estimated_jaccard < 1.0
    kept = _doc("shard-0002", 2, unrelated, doc_id="c-002")
    assert dedup.classify(kept).verdict == "keep"

    # The exact-hash fast path applies to raw-identical no-id lines.
    exact_dedup = Dedup(DedupConfig())
    raw = b'{"text":"identical line"}'
    one = _doc("s", 1, "identical line", doc_id="s:000001", raw=raw)
    two = _doc("s", 2, "identical line", doc_id="s:000002", raw=raw)
    assert exact_dedup.classify(one).verdict == "keep"
    exact = exact_dedup.classify(two)
    assert exact.verdict == "near_duplicate"
    assert exact.estimated_jaccard == 1.0
    assert exact.matched_doc_id == "s:000001"


def test_shingle_width_changes_dedup_verdict() -> None:
    base = (
        "the archive keeps every ledger from the northern ports and each entry records "
        "the ship the cargo the harbour fee and the day the record was signed"
    )
    footer = base + " retrieved from the public mirror"
    for width, expected in ((3, "near_duplicate"), (5, "near_duplicate"), (9, "keep")):
        dedup = Dedup(DedupConfig(shingle_width=width))
        assert dedup.classify(_doc("s", 1, base, doc_id="d1")).verdict == "keep"
        verdict = dedup.classify(_doc("s", 2, footer, doc_id="d2"))
        assert verdict.verdict == expected
        if expected == "keep":
            assert verdict.estimated_jaccard is None
        else:
            assert verdict.estimated_jaccard is not None

    short = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"
    longer = short + " omicron pi"
    for width, expected in ((3, "near_duplicate"), (13, "keep")):
        dedup = Dedup(DedupConfig(shingle_width=width))
        assert dedup.classify(_doc("s", 1, short, doc_id="d1")).verdict == "keep"
        assert dedup.classify(_doc("s", 2, longer, doc_id="d2")).verdict == expected

    estimates = {
        width: MinHasher(shingle_width=width).similarity(
            MinHasher(shingle_width=width).signature(base),
            MinHasher(shingle_width=width).signature(footer),
        )
        for width in (3, 5, 9)
    }
    assert [round(estimates[width], 3) for width in (3, 5, 9)] == [0.852, 0.820, 0.766]


# ---------------------------------------------------------------------------
# Manifest and pipeline
# ---------------------------------------------------------------------------


def test_manifest_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    result = _run_file_pipeline(tmp_path)
    manifest_path = result.manifest_path
    assert verify_manifest(manifest_path) == result.manifest
    original = manifest_path.read_bytes()
    lock_path = manifest_path.with_suffix(".lock")
    original_lock = lock_path.read_text(encoding="utf-8")
    assert original_lock.split()[0] == result.manifest_sha256

    manifest_path.write_bytes(original[:-2] + b"X\n")
    with pytest.raises(ManifestError, match="sha256"):
        verify_manifest(manifest_path)
    manifest_path.write_bytes(original)
    lock_path.unlink()
    with pytest.raises(ManifestError, match="lock"):
        verify_manifest(manifest_path)
    lock_path.write_text(original_lock, encoding="utf-8")

    shard = tmp_path / "run" / "corpus" / "shard-0000.jsonl"
    shard_bytes = shard.read_bytes()
    shard.write_bytes(shard_bytes[:-1] + b"X")
    with pytest.raises(ManifestError, match="sha256"):
        verify_manifest(manifest_path)
    shard.write_bytes(shard_bytes)


def test_manifest_semantic_checks_catch_tampering(tmp_path: Path) -> None:
    result = _run_file_pipeline(tmp_path)
    manifest_path = result.manifest_path
    tampered = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(tampered, dict)
    totals = tampered["totals"]
    assert isinstance(totals, dict)
    totals["documents"] = int(totals["documents"]) + 1
    _rewrite_manifest(manifest_path, tampered)
    with pytest.raises(ManifestError, match="totals.documents"):
        verify_manifest(manifest_path)

    escaped = json.loads(
        (tmp_path / "run" / "manifest.json").read_text(encoding="utf-8")
    )
    assert isinstance(escaped, dict)
    shards = escaped["shards"]
    assert isinstance(shards, list) and isinstance(shards[0], dict)
    shards[0]["output_path"] = "../escape.jsonl"
    _rewrite_manifest(manifest_path, escaped)
    with pytest.raises(ManifestError, match="escapes"):
        verify_manifest(manifest_path)


def test_downloader_fresh_file_and_reuse(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    compressed = _write_shard(
        remote / "shard-0001.jsonl.zst",
        [("d1", "alpha beta gamma"), ("d2", "delta epsilon zeta")],
    )
    planner = ShardPlanner(tmp_path / "cache", tmp_path / "corpus")
    plan = planner.plan(
        [(remote / "shard-0001.jsonl.zst").as_uri()],
        expected_sizes={"shard-0001": len(compressed)},
        expected_sha256={"shard-0001": hashlib.sha256(compressed).hexdigest()},
    )[0]
    downloader = StreamingDownloader(chunk_size=17)
    result = downloader.download(plan)
    assert result.reused is False
    assert result.compressed_bytes == len(compressed)
    assert result.sha256 == hashlib.sha256(compressed).hexdigest()
    assert plan.cache_path.read_bytes() == compressed
    assert not plan.partial_path.exists()
    assert not plan.checkpoint_path.exists()
    reused = downloader.download(plan)
    assert reused.reused is True
    assert reused.sha256 == result.sha256
    assert plan.cache_path.read_bytes() == compressed


def test_pipeline_end_to_end_and_deterministic_rerun(tmp_path: Path) -> None:
    first = _run_file_pipeline(tmp_path)
    manifest = first.manifest
    totals = manifest["totals"]
    assert isinstance(totals, dict)
    assert totals["shards"] == 3
    assert totals["documents"] == 9
    assert totals["unique_documents"] == 6
    assert totals["duplicate_count"] == 3

    shards = manifest["shards"]
    assert isinstance(shards, list)
    by_id = {}
    for raw in shards:
        assert isinstance(raw, dict)
        by_id[raw["shard_id"]] = raw
    for shard_id in ("shard-0000", "shard-0001", "shard-0002"):
        assert by_id[shard_id]["document_count"] == 3
        assert by_id[shard_id]["unique_document_count"] == 2
        assert by_id[shard_id]["duplicate_count"] == 1

    documents = by_id["shard-0001"]["documents"]
    assert isinstance(documents, list)
    keep = [record for record in documents if record["verdict"] == "keep"]
    duplicates = [record for record in documents if record["verdict"] == "near_duplicate"]
    assert len(keep) == 2
    assert len(duplicates) == 1
    assert duplicates[0]["matched_shard_id"] == "shard-0000"
    assert duplicates[0]["matched_doc_id"] == "a-001"

    output = tmp_path / "run" / "corpus" / "shard-0001.jsonl"
    contents = output.read_bytes()
    for record in keep:
        offset = record["offset"]
        length = record["length"]
        assert isinstance(offset, int) and isinstance(length, int)
        assert hashlib.sha256(contents[offset : offset + length]).hexdigest() == record["sha256"]

    second = _run_file_pipeline(tmp_path)
    assert second.manifest_sha256 == first.manifest_sha256
    assert verify_manifest(first.manifest_path) == first.manifest


def test_pipeline_fails_closed_when_manifest_and_lock_disagree(tmp_path: Path) -> None:
    first = _run_file_pipeline(tmp_path)
    first.manifest_path.write_bytes(first.manifest_path.read_bytes() + b" \n")
    with pytest.raises(ManifestError, match="sha256"):
        _run_file_pipeline(tmp_path)


def test_demo_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    result = run_demo(tmp_path / "demo")
    captured = capsys.readouterr()
    assert "manifest sha256:" in captured.out
    assert "resume:" in captured.out
    assert result.resumed_from_bytes > 0
    totals = result.manifest["totals"]
    assert isinstance(totals, dict)
    assert totals["documents"] == 9
    assert totals["unique_documents"] == 6
    assert totals["duplicate_count"] == 3
    assert verify_manifest(result.manifest_path) == result.manifest


def test_cli_demo_fetch_and_verify(tmp_path: Path) -> None:
    demo_dir = tmp_path / "cli-demo"
    assert main(["demo", "--workdir", str(demo_dir)]) == 0
    manifest = demo_dir / "manifest.json"
    assert main(["verify-manifest", "--manifest", str(manifest)]) == 0

    remote = tmp_path / "remote"
    urls, sizes, hashes = build_demo_corpus(remote)
    fetch_dir = tmp_path / "fetch"
    fetch_args = ["fetch", "--workdir", str(fetch_dir)]
    for url in urls:
        fetch_args.extend(["--url", url])
    for shard_id, size in sizes.items():
        fetch_args.extend(["--expected-size", f"{shard_id}={size}"])
    for shard_id, digest in hashes.items():
        fetch_args.extend(["--expected-sha256", f"{shard_id}={digest}"])
    assert main(fetch_args) == 0
    assert verify_manifest(fetch_dir / "manifest.json")

    assert main(["fetch", "--workdir", str(tmp_path / "bad"), "--url", "ftp://invalid/x.zst"]) == 2
