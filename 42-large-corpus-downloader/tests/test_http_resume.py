from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any, cast

import pytest
import zstandard

from local_server import LocalRangeServer, ServedRequest
from main import (
    ChecksumMismatch,
    IncompleteDownloadError,
    ResumeUnavailable,
    ResponseLike,
    ShardPlan,
    ShardPlanner,
    StreamingDownloader,
)


def _payload(docs: list[tuple[str, str]]) -> bytes:
    lines = [
        json.dumps({"id": doc_id, "text": text}, ensure_ascii=False, sort_keys=True)
        for doc_id, text in docs
    ]
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    return zstandard.ZstdCompressor(level=3).compress(raw)


def _download(plan: ShardPlan, *, resume_only: bool = False) -> Any:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    downloader = StreamingDownloader(
        chunk_size=32,
        opener=lambda request: cast(ResponseLike, opener.open(request, timeout=5.0)),
    )
    return downloader.download(plan, resume_only=resume_only)


def _planner(tmp_path: Path) -> ShardPlanner:
    return ShardPlanner(tmp_path / "cache", tmp_path / "corpus")


def _plan(
    tmp_path: Path,
    url: str,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
) -> ShardPlan:
    sizes = {"shard": expected_size} if expected_size is not None else None
    hashes = {"shard": expected_sha256} if expected_sha256 is not None else None
    return _planner(tmp_path).plan(
        [url], expected_sizes=sizes, expected_sha256=hashes
    )[0]


def _checkpoint(plan: ShardPlan) -> dict[str, Any]:
    parsed = json.loads(plan.checkpoint_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    return parsed


def _verified(plan: ShardPlan) -> int:
    checkpoint = _checkpoint(plan)
    verified = checkpoint["verified_bytes"]
    assert isinstance(verified, int)
    return verified


DOCS = [
    ("d-001", "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"),
    ("d-002", "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega"),
    ("d-003", "the archive keeps a ledger for every harbour and every signed cargo"),
]


def test_resume_after_truncated_response(tmp_path: Path) -> None:
    payload = _payload(DOCS)
    digest = hashlib.sha256(payload).hexdigest()
    with LocalRangeServer(
        payload, fail_first_get_after=max(1, len(payload) // 2)
    ) as server:
        plan = _plan(tmp_path, server.url, expected_size=len(payload), expected_sha256=digest)
        with pytest.raises(IncompleteDownloadError):
            _download(plan)

        checkpoint = _checkpoint(plan)
        verified = checkpoint["verified_bytes"]
        assert isinstance(verified, int)
        assert 0 < verified < len(payload)
        partial = plan.partial_path.read_bytes()
        assert len(partial) == verified
        assert checkpoint["sha256_prefix"] == hashlib.sha256(partial[:verified]).hexdigest()
        assert checkpoint["expected_size"] == len(payload)

        result = _download(plan)
        assert result.resumed_from_bytes == verified
        assert result.bytes_downloaded == len(payload) - verified
        assert result.attempts == 2
        assert plan.cache_path.read_bytes() == payload
        assert not plan.partial_path.exists()
        assert not plan.checkpoint_path.exists()
        request = server.requests[-1]
        assert request.range_header == f"bytes={verified}-"
        assert request.if_range == '"v1"'


def test_corrupt_prefix_triggers_fresh_request(tmp_path: Path) -> None:
    payload = _payload(DOCS)
    with LocalRangeServer(
        payload, fail_first_get_after=max(1, len(payload) // 2)
    ) as server:
        plan = _plan(tmp_path, server.url, expected_size=len(payload))
        with pytest.raises(IncompleteDownloadError):
            _download(plan)
        partial = bytearray(plan.partial_path.read_bytes())
        partial[0] ^= 0xFF
        plan.partial_path.write_bytes(bytes(partial))

        result = _download(plan)
        assert result.reused is False
        assert plan.cache_path.read_bytes() == payload
        assert server.requests[-1].range_header is None


def test_data_ahead_of_checkpoint_is_truncated(tmp_path: Path) -> None:
    payload = _payload(DOCS)
    with LocalRangeServer(
        payload, fail_first_get_after=max(1, len(payload) // 2)
    ) as server:
        plan = _plan(tmp_path, server.url, expected_size=len(payload))
        with pytest.raises(IncompleteDownloadError):
            _download(plan)
        verified = _verified(plan)
        with plan.partial_path.open("ab") as partial:
            partial.write(b"unverified suffix")

        _download(plan)
        assert plan.cache_path.read_bytes() == payload
        assert server.requests[-1].range_header == f"bytes={verified}-"


def test_checkpoint_ahead_of_data_restarts_from_zero(tmp_path: Path) -> None:
    payload = _payload(DOCS)
    with LocalRangeServer(
        payload, fail_first_get_after=max(1, len(payload) // 2)
    ) as server:
        plan = _plan(tmp_path, server.url, expected_size=len(payload))
        with pytest.raises(IncompleteDownloadError):
            _download(plan)
        verified = _verified(plan)
        with plan.partial_path.open("r+b") as partial:
            partial.truncate(verified - 1)

        _download(plan)
        assert plan.cache_path.read_bytes() == payload
        assert server.requests[-1].range_header is None


def test_server_ignoring_range_restarts_cleanly(tmp_path: Path) -> None:
    payload = _payload(DOCS)
    with LocalRangeServer(
        payload,
        honor_range=False,
        fail_first_get_after=max(1, len(payload) // 2),
    ) as server:
        plan = _plan(tmp_path, server.url, expected_size=len(payload))
        with pytest.raises(IncompleteDownloadError):
            _download(plan)
        result = _download(plan)
        assert result.reused is False
        assert plan.cache_path.read_bytes() == payload
        assert server.requests[-1].range_header is not None
        assert len(server.requests) == 2


def test_changed_resource_restarts_with_full_body(tmp_path: Path) -> None:
    old_payload = _payload(DOCS)
    new_payload = _payload(
        [("n-001", "completely new representation with a different compressed payload")]
    )
    with LocalRangeServer(
        old_payload, fail_first_get_after=max(1, len(old_payload) // 2)
    ) as server:
        plan = _plan(tmp_path, server.url)
        with pytest.raises(IncompleteDownloadError):
            _download(plan)
        server.replace(new_payload, etag='"v2"')
        result = _download(plan)
        assert plan.cache_path.read_bytes() == new_payload
        assert result.sha256 == hashlib.sha256(new_payload).hexdigest()
        request = server.requests[-1]
        assert request.range_header is not None
        assert request.if_range == '"v1"'


def test_resume_only_refuses_fresh_and_accepts_checkpoint(tmp_path: Path) -> None:
    payload = _payload(DOCS)
    with LocalRangeServer(
        payload, fail_first_get_after=max(1, len(payload) // 2)
    ) as server:
        plan = _plan(tmp_path, server.url, expected_size=len(payload))
        with pytest.raises(ResumeUnavailable):
            _download(plan, resume_only=True)
        assert len(server.requests) == 0

        with pytest.raises(IncompleteDownloadError):
            _download(plan)
        assert _verified(plan) > 0
        result = _download(plan, resume_only=True)
        assert result.reused is False
        assert plan.cache_path.read_bytes() == payload
        requests = cast(tuple[ServedRequest, ...], server.requests)
        assert len(requests) == 2
        assert requests[-1].range_header is not None


def test_checksum_mismatch_removes_partial_state(tmp_path: Path) -> None:
    payload = _payload(DOCS)
    with LocalRangeServer(payload) as server:
        plan = _plan(
            tmp_path,
            server.url,
            expected_size=len(payload),
            expected_sha256="0" * 64,
        )
        with pytest.raises(ChecksumMismatch):
            _download(plan)
        assert not plan.cache_path.exists()
        assert not plan.partial_path.exists()
        assert not plan.checkpoint_path.exists()
