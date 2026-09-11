"""Streaming, resumable, deduplicating corpus downloader.

The module implements the capstone pipeline:

* plan compressed shard URLs;
* download them with a verified byte-offset checkpoint and HTTP ``Range``;
* stream-decompress JSONL documents with Zstandard;
* fingerprint documents with MinHash and bucket them with LSH;
* tombstone near-duplicates;
* emit a shard manifest whose content hash is recorded in ``manifest.lock``.

Everything is standard-library plus ``zstandard``. The demo, the loopback
server, and the tests never contact an external host.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import http.client
import json
import math
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Literal, Protocol, cast

import zstandard

from local_server import LocalRangeServer

SCHEMA_VERSION = 1
GENERATOR = "large-corpus-downloader/0.1.0"
DEFAULT_SHINGLE_WIDTH = 5
DEFAULT_NUM_PERMUTATIONS = 128
DEFAULT_BANDS = 32
DEFAULT_ROWS = 4
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_CHUNK_SIZE = 1_048_576
DEFAULT_READ_SIZE = 65_536
DEFAULT_MAX_LINE_BYTES = 16_777_216
UINT64_MASK = (1 << 64) - 1
HASH_FAMILY = "blake2b-64+splitmix64"
_READ_CHUNK = 1_048_576
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONTENT_RANGE_RE = re.compile(r"^bytes\s+(\d+)-(\d+)/(\d+|\*)$")
_UNSATISFIED_RANGE_RE = re.compile(r"^bytes\s+\*/(\d+)$")


class CorpusError(Exception):
    """Base class for expected corpus-pipeline failures."""


class PlanError(CorpusError):
    """Invalid shard plan or CLI expectation."""


class DownloadError(CorpusError):
    """The remote representation could not be downloaded."""


class IncompleteDownloadError(DownloadError):
    """A transfer ended before the verified expected size."""


class ResumeUnavailable(DownloadError):
    """Resume-only mode found no trustworthy checkpoint."""


class ChecksumMismatch(DownloadError):
    """A size or SHA-256 contract was violated."""


class DocumentFormatError(CorpusError):
    """A decompressed line is not a valid corpus document."""


class ManifestError(CorpusError):
    """The manifest is missing, malformed, or disagrees with the files."""


class _RestartRequired(Exception):
    """Internal signal: discard the checkpoint and issue a fresh request."""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_READ_CHUNK)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _sha256_prefix(path: Path, length: int) -> tuple[str, Any]:
    """Hash exactly ``length`` bytes and return the digest plus live hash state."""
    if isinstance(length, bool) or not isinstance(length, int) or length < 0:
        raise ValueError("length must be a non-negative int")
    hasher = hashlib.sha256()
    remaining = length
    with path.open("rb") as fh:
        while remaining > 0:
            chunk = fh.read(min(_READ_CHUNK, remaining))
            if not chunk:
                raise ValueError("file is shorter than the requested prefix")
            hasher.update(chunk)
            remaining -= len(chunk)
    return hasher.hexdigest(), hasher


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, path)
        _fsync_dir(path.parent)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _splitmix64(value: int) -> int:
    z = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (z ^ (z >> 31)) & UINT64_MASK


def _public_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _resolve_relative(base: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ManifestError(f"manifest path must be relative: {relative!r}")
    base_resolved = base.resolve()
    target = (base_resolved / candidate).resolve()
    try:
        target.relative_to(base_resolved)
    except ValueError as exc:
        raise ManifestError(f"manifest path escapes its directory: {relative!r}") from exc
    return target


def _relative_to(base: Path, path: Path) -> str:
    base_resolved = base.resolve()
    try:
        return path.resolve().relative_to(base_resolved).as_posix()
    except ValueError as exc:
        raise ManifestError(f"artifact is outside the manifest directory: {path}") from exc


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestError(f"{field} must be an int")
    if value < minimum:
        raise ManifestError(f"{field} must be >= {minimum}")
    return value


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{field} must be a non-empty string")
    return value


def _require_object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ManifestError(f"{field} must be an object")
    result: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ManifestError(f"{field} has a non-string key")
        result[key] = item
    return result


def _require_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise ManifestError(f"{field} must be a list")
    return value


def _require_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError(f"{field} must be a number")
    return float(value)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShardPlan:
    shard_id: str
    url: str
    cache_path: Path
    partial_path: Path
    checkpoint_path: Path
    output_path: Path
    expected_size: int | None = None
    expected_sha256: str | None = None


class ShardPlanner:
    def __init__(self, cache_dir: Path, corpus_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.corpus_dir = Path(corpus_dir)

    @staticmethod
    def derive_shard_id(url: str) -> str:
        parsed = urllib.parse.urlsplit(url)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https", "file"}:
            raise PlanError(f"unsupported URL scheme {scheme!r}: {url!r}")
        path = urllib.parse.unquote(parsed.path)
        name = Path(path).name
        if not name:
            raise PlanError(f"URL has no shard filename: {url!r}")
        identifier = name
        for suffix in (".jsonl.zst", ".ndjson.zst", ".zst", ".jsonl"):
            if identifier.endswith(suffix):
                identifier = identifier[: -len(suffix)]
                break
        else:
            identifier = Path(name).stem
        if (
            not identifier
            or identifier in {".", ".."}
            or ".." in identifier
            or _SHARD_ID_RE.fullmatch(identifier) is None
        ):
            raise PlanError(f"unsafe or invalid shard id {identifier!r} from {url!r}")
        return identifier

    def plan(
        self,
        urls: Sequence[str],
        *,
        expected_sizes: Mapping[str, int] | None = None,
        expected_sha256: Mapping[str, str] | None = None,
    ) -> list[ShardPlan]:
        if not urls:
            raise PlanError("at least one shard URL is required")

        sizes = dict(expected_sizes or {})
        hashes = dict(expected_sha256 or {})
        for shard_id, size in sizes.items():
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise PlanError(f"expected size for {shard_id!r} must be a non-negative int")
        for shard_id, hash_value in hashes.items():
            if not isinstance(hash_value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", hash_value) is None:
                raise PlanError(f"expected sha256 for {shard_id!r} must be 64 hex characters")

        plans: list[ShardPlan] = []
        seen: set[str] = set()
        for url in urls:
            if not isinstance(url, str):
                raise PlanError("shard URLs must be strings")
            shard_id = self.derive_shard_id(url)
            if shard_id in seen:
                raise PlanError(f"duplicate shard id {shard_id!r}")
            seen.add(shard_id)
            expected_size = sizes.get(shard_id)
            digest = hashes.get(shard_id)
            plans.append(
                ShardPlan(
                    shard_id=shard_id,
                    url=url,
                    cache_path=self.cache_dir / f"{shard_id}.jsonl.zst",
                    partial_path=self.cache_dir / f"{shard_id}.jsonl.zst.part",
                    checkpoint_path=self.cache_dir / f"{shard_id}.jsonl.zst.partial.json",
                    output_path=self.corpus_dir / f"{shard_id}.jsonl",
                    expected_size=expected_size,
                    expected_sha256=digest.lower() if digest is not None else None,
                )
            )

        unknown_sizes = set(sizes) - seen
        unknown_hashes = set(hashes) - seen
        if unknown_sizes:
            raise PlanError(f"expected size for unplanned shard(s): {sorted(unknown_sizes)!r}")
        if unknown_hashes:
            raise PlanError(f"expected sha256 for unplanned shard(s): {sorted(unknown_hashes)!r}")
        return plans


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------


class ResponseLike(Protocol):
    status: int | None
    headers: Message

    def read(self, amt: int = -1) -> bytes: ...
    def close(self) -> None: ...
    def __enter__(self) -> ResponseLike: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


@dataclass(frozen=True)
class DownloadResult:
    shard_id: str
    path: Path
    compressed_bytes: int
    sha256: str
    attempts: int
    resumed_from_bytes: int
    bytes_downloaded: int
    reused: bool


@dataclass(frozen=True)
class _Checkpoint:
    shard_id: str
    url: str
    verified_bytes: int
    expected_size: int | None
    sha256_prefix: str
    etag: str | None
    last_modified: str | None
    attempts: int


class StreamingDownloader:
    def __init__(
        self,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        timeout: float = 30.0,
        opener: Callable[[urllib.request.Request], ResponseLike] | None = None,
    ) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive int")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.chunk_size = chunk_size
        self.timeout = timeout
        self._opener = opener if opener is not None else self._default_open

    def _default_open(self, request: urllib.request.Request) -> ResponseLike:
        return cast(ResponseLike, urllib.request.urlopen(request, timeout=self.timeout))

    def download(self, plan: ShardPlan, *, resume_only: bool = False) -> DownloadResult:
        restarts = 0
        while True:
            try:
                return self._download_once(plan, resume_only=resume_only)
            except _RestartRequired as exc:
                if resume_only:
                    raise ResumeUnavailable(
                        f"{plan.shard_id}: representation changed; resume-only refuses a fresh download"
                    ) from exc
                restarts += 1
                if restarts > 1:
                    raise DownloadError(
                        f"{plan.shard_id}: server did not provide a stable representation"
                    ) from exc
                self._discard_partial(plan)

    def preflight(self, plan: ShardPlan) -> None:
        final_info = self._existing_final(plan)
        if final_info is not None:
            return
        state = self._load_verified(plan, resume_only=True)
        if state is None:
            raise ResumeUnavailable(f"{plan.shard_id}: no complete cache file or checkpoint")
        checkpoint = state[0]
        if checkpoint.expected_size is not None and checkpoint.verified_bytes > checkpoint.expected_size:
            raise ResumeUnavailable(f"{plan.shard_id}: checkpoint exceeds its expected size")

    def _download_once(self, plan: ShardPlan, *, resume_only: bool) -> DownloadResult:
        final_info = self._existing_final(plan)
        if final_info is not None:
            size, digest = final_info
            self._discard_partial(plan)
            return DownloadResult(
                shard_id=plan.shard_id,
                path=plan.cache_path,
                compressed_bytes=size,
                sha256=digest,
                attempts=0,
                resumed_from_bytes=0,
                bytes_downloaded=0,
                reused=True,
            )

        if plan.cache_path.exists():
            if not plan.cache_path.is_file():
                raise DownloadError(f"{plan.shard_id}: cache path is not a file: {plan.cache_path}")
            plan.cache_path.unlink()

        state = self._load_verified(plan, resume_only=resume_only)
        resumed_from_bytes = 0
        if state is None:
            if resume_only:
                raise ResumeUnavailable(f"{plan.shard_id}: no checkpoint found")
            offset = 0
            hasher = hashlib.sha256()
            expected_size = plan.expected_size
            etag: str | None = None
            last_modified: str | None = None
            attempts = 0
        else:
            checkpoint, hasher = state
            offset = checkpoint.verified_bytes
            resumed_from_bytes = offset
            expected_size = checkpoint.expected_size
            if expected_size is None:
                expected_size = plan.expected_size
            etag = checkpoint.etag
            last_modified = checkpoint.last_modified
            attempts = checkpoint.attempts

        if expected_size is not None and offset == expected_size:
            return self._finalize(
                plan,
                offset=offset,
                hasher=hasher,
                expected_size=expected_size,
                attempts=attempts,
                resumed_from_bytes=resumed_from_bytes,
                bytes_downloaded=0,
            )

        headers: dict[str, str] = {}
        if offset > 0:
            headers["Range"] = f"bytes={offset}-"
            if etag is not None and not etag.startswith("W/"):
                headers["If-Range"] = etag

        request = urllib.request.Request(plan.url, headers=headers)
        try:
            response = self._opener(request)
        except urllib.error.HTTPError as exc:
            if exc.code == 416:
                total = self._parse_unsatisfied_total(exc.headers.get("Content-Range"))
                exc.close()
                if total is not None and total == offset:
                    return self._finalize(
                        plan,
                        offset=offset,
                        hasher=hasher,
                        expected_size=offset,
                        attempts=max(attempts, 1),
                        resumed_from_bytes=resumed_from_bytes,
                        bytes_downloaded=0,
                    )
                raise DownloadError(
                    f"{plan.shard_id}: range not satisfiable at byte {offset}"
                ) from exc
            exc.close()
            raise DownloadError(
                f"{plan.shard_id}: HTTP {exc.code} {exc.reason} from {plan.url}"
            ) from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise DownloadError(f"{plan.shard_id}: cannot open {plan.url}: {exc}") from exc

        try:
            status = response.status
            if status is None:
                status = 200
            content_length = self._content_length(response.headers)

            if status == 206:
                content_range = response.headers.get("Content-Range")
                parsed_range = (
                    _CONTENT_RANGE_RE.fullmatch(content_range.strip())
                    if content_range is not None
                    else None
                )
                if parsed_range is None:
                    raise DownloadError(f"{plan.shard_id}: 206 response without a valid Content-Range")
                start = int(parsed_range.group(1))
                end = int(parsed_range.group(2))
                total_text = parsed_range.group(3)
                if start != offset:
                    raise DownloadError(
                        f"{plan.shard_id}: server resumed at {start}, expected {offset}"
                    )
                if content_length is not None and end - start + 1 != content_length:
                    raise DownloadError(
                        f"{plan.shard_id}: Content-Range length disagrees with Content-Length"
                    )
                total = int(total_text) if total_text != "*" else None
                if total is not None and plan.expected_size is not None and total != plan.expected_size:
                    raise ChecksumMismatch(
                        f"{plan.shard_id}: server size {total} != expected {plan.expected_size}"
                    )
                if total is not None and expected_size is not None and total != expected_size:
                    if plan.expected_size is not None:
                        raise ChecksumMismatch(
                            f"{plan.shard_id}: server size {total} != expected {expected_size}"
                        )
                    raise _RestartRequired(f"{plan.shard_id}: checkpoint size changed")
                if total is not None:
                    expected_size = total
                response_etag = self._header(response.headers, "ETag")
                response_last_modified = self._header(response.headers, "Last-Modified")
                if etag is not None and response_etag is not None and etag != response_etag:
                    raise _RestartRequired(f"{plan.shard_id}: ETag changed")
                if (
                    etag is None
                    and last_modified is not None
                    and response_last_modified is not None
                    and last_modified != response_last_modified
                ):
                    raise _RestartRequired(f"{plan.shard_id}: Last-Modified changed")
                etag = response_etag
                last_modified = response_last_modified
            elif status == 200:
                response_etag = self._header(response.headers, "ETag")
                response_last_modified = self._header(response.headers, "Last-Modified")
                if plan.expected_size is not None and content_length is not None:
                    if content_length != plan.expected_size:
                        raise ChecksumMismatch(
                            f"{plan.shard_id}: Content-Length {content_length} "
                            f"!= expected {plan.expected_size}"
                        )
                if offset > 0:
                    self._discard_partial(plan)
                    offset = 0
                    hasher = hashlib.sha256()
                if plan.expected_size is not None:
                    expected_size = plan.expected_size
                elif content_length is not None:
                    expected_size = content_length
                etag = response_etag
                last_modified = response_last_modified
            else:
                raise DownloadError(f"{plan.shard_id}: unexpected HTTP status {status}")

            attempts += 1
            plan.partial_path.parent.mkdir(parents=True, exist_ok=True)
            fh: BinaryIO
            if offset > 0:
                if not plan.partial_path.is_file():
                    raise DownloadError(f"{plan.shard_id}: partial file disappeared")
                fh = plan.partial_path.open("r+b")
                fh.truncate(offset)
                fh.seek(offset)
            else:
                fh = plan.partial_path.open("wb")

            downloaded = 0
            try:
                with fh:
                    while True:
                        chunk = response.read(self.chunk_size)
                        if not chunk:
                            break
                        next_offset = offset + len(chunk)
                        if expected_size is not None and next_offset > expected_size:
                            raise ChecksumMismatch(
                                f"{plan.shard_id}: server sent more than {expected_size} bytes"
                            )
                        next_hash = hasher.copy()
                        next_hash.update(chunk)
                        self._write_checkpoint(
                            plan,
                            verified_bytes=next_offset,
                            expected_size=expected_size,
                            sha256_prefix=next_hash.hexdigest(),
                            etag=etag,
                            last_modified=last_modified,
                            attempts=attempts,
                        )
                        fh.write(chunk)
                        fh.flush()
                        os.fsync(fh.fileno())
                        hasher = next_hash
                        offset = next_offset
                        downloaded += len(chunk)
            except ChecksumMismatch:
                self._discard_partial(plan)
                raise
            except (OSError, http.client.HTTPException) as exc:
                raise IncompleteDownloadError(
                    f"{plan.shard_id}: transfer stopped at byte {offset}: {exc}"
                ) from exc

            if expected_size is None:
                expected_size = offset
                self._write_checkpoint(
                    plan,
                    verified_bytes=offset,
                    expected_size=expected_size,
                    sha256_prefix=hasher.hexdigest(),
                    etag=etag,
                    last_modified=last_modified,
                    attempts=attempts,
                )
            if offset != expected_size:
                if offset > expected_size:
                    self._discard_partial(plan)
                    raise ChecksumMismatch(
                        f"{plan.shard_id}: received {offset} bytes, expected {expected_size}"
                    )
                raise IncompleteDownloadError(
                    f"{plan.shard_id}: received {offset} of {expected_size} bytes"
                )
            return self._finalize(
                plan,
                offset=offset,
                hasher=hasher,
                expected_size=expected_size,
                attempts=attempts,
                resumed_from_bytes=resumed_from_bytes,
                bytes_downloaded=downloaded,
            )
        finally:
            response.close()

    # -- checkpoint and cache helpers -------------------------------------------------

    def _existing_final(self, plan: ShardPlan) -> tuple[int, str] | None:
        if not plan.cache_path.is_file():
            return None
        size = plan.cache_path.stat().st_size
        if plan.expected_size is not None and size != plan.expected_size:
            return None
        digest = _sha256_file(plan.cache_path)
        if plan.expected_sha256 is not None and not hmac.compare_digest(
            digest, plan.expected_sha256
        ):
            return None
        return size, digest

    def _load_verified(
        self, plan: ShardPlan, *, resume_only: bool
    ) -> tuple[_Checkpoint, Any] | None:
        if not plan.checkpoint_path.exists():
            if plan.partial_path.exists():
                if resume_only:
                    raise ResumeUnavailable(
                        f"{plan.shard_id}: partial data has no checkpoint"
                    )
                self._discard_partial(plan)
            return None

        try:
            checkpoint = self._read_checkpoint(plan)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if resume_only:
                raise ResumeUnavailable(f"{plan.shard_id}: invalid checkpoint: {exc}") from exc
            self._discard_partial(plan)
            return None

        try:
            if checkpoint.verified_bytes == 0 and not plan.partial_path.exists():
                hasher = hashlib.sha256()
                digest = hasher.hexdigest()
            else:
                digest, hasher = _sha256_prefix(plan.partial_path, checkpoint.verified_bytes)
        except (OSError, ValueError) as exc:
            if resume_only:
                raise ResumeUnavailable(
                    f"{plan.shard_id}: cannot verify checkpoint prefix: {exc}"
                ) from exc
            self._discard_partial(plan)
            return None

        if not hmac.compare_digest(digest, checkpoint.sha256_prefix):
            if resume_only:
                raise ResumeUnavailable(f"{plan.shard_id}: checkpoint prefix hash mismatch")
            self._discard_partial(plan)
            return None
        return checkpoint, hasher

    def _read_checkpoint(self, plan: ShardPlan) -> _Checkpoint:
        raw = plan.checkpoint_path.read_text(encoding="utf-8")
        data: Any = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("checkpoint must be a JSON object")
        if data.get("version") != SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint version")
        shard_id = data.get("shard_id")
        url = data.get("url")
        verified_bytes = data.get("verified_bytes")
        expected_size = data.get("expected_size")
        sha256_prefix = data.get("sha256_prefix")
        etag = data.get("etag")
        last_modified = data.get("last_modified")
        attempts = data.get("attempts")
        if not isinstance(shard_id, str) or shard_id != plan.shard_id:
            raise ValueError("checkpoint shard id mismatch")
        if not isinstance(url, str) or url != plan.url:
            raise ValueError("checkpoint URL mismatch")
        if isinstance(verified_bytes, bool) or not isinstance(verified_bytes, int) or verified_bytes < 0:
            raise ValueError("invalid verified_bytes")
        if expected_size is not None and (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ValueError("invalid expected_size")
        if (
            not isinstance(sha256_prefix, str)
            or _SHA256_RE.fullmatch(sha256_prefix) is None
        ):
            raise ValueError("invalid sha256_prefix")
        if etag is not None and not isinstance(etag, str):
            raise ValueError("invalid etag")
        if last_modified is not None and not isinstance(last_modified, str):
            raise ValueError("invalid last_modified")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0:
            raise ValueError("invalid attempts")
        if expected_size is not None and verified_bytes > expected_size:
            raise ValueError("checkpoint is beyond its expected size")
        if plan.expected_size is not None:
            if expected_size is not None and expected_size != plan.expected_size:
                raise ValueError("checkpoint expected size disagrees with the plan")
            if verified_bytes > plan.expected_size:
                raise ValueError("checkpoint exceeds the plan size")
        return _Checkpoint(
            shard_id=shard_id,
            url=url,
            verified_bytes=verified_bytes,
            expected_size=expected_size,
            sha256_prefix=sha256_prefix,
            etag=etag,
            last_modified=last_modified,
            attempts=attempts,
        )

    def _write_checkpoint(
        self,
        plan: ShardPlan,
        *,
        verified_bytes: int,
        expected_size: int | None,
        sha256_prefix: str,
        etag: str | None,
        last_modified: str | None,
        attempts: int,
    ) -> None:
        payload = {
            "version": SCHEMA_VERSION,
            "shard_id": plan.shard_id,
            "url": plan.url,
            "verified_bytes": verified_bytes,
            "expected_size": expected_size,
            "sha256_prefix": sha256_prefix,
            "etag": etag,
            "last_modified": last_modified,
            "attempts": attempts,
        }
        data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
        _atomic_write_bytes(plan.checkpoint_path, data)

    def _discard_partial(self, plan: ShardPlan) -> None:
        plan.partial_path.unlink(missing_ok=True)
        plan.checkpoint_path.unlink(missing_ok=True)

    def _finalize(
        self,
        plan: ShardPlan,
        *,
        offset: int,
        hasher: Any,
        expected_size: int,
        attempts: int,
        resumed_from_bytes: int,
        bytes_downloaded: int,
    ) -> DownloadResult:
        if offset != expected_size:
            self._discard_partial(plan)
            raise ChecksumMismatch(
                f"{plan.shard_id}: final offset {offset} != expected {expected_size}"
            )
        if plan.expected_size is not None and expected_size != plan.expected_size:
            self._discard_partial(plan)
            raise ChecksumMismatch(
                f"{plan.shard_id}: final size {expected_size} != expected {plan.expected_size}"
            )
        if plan.expected_size is not None and offset != plan.expected_size:
            self._discard_partial(plan)
            raise ChecksumMismatch(
                f"{plan.shard_id}: final offset {offset} != expected {plan.expected_size}"
            )
        if not plan.partial_path.is_file():
            if offset != 0:
                raise DownloadError(f"{plan.shard_id}: partial file is missing")
            plan.partial_path.parent.mkdir(parents=True, exist_ok=True)
            plan.partial_path.write_bytes(b"")
        if plan.partial_path.stat().st_size < offset:
            self._discard_partial(plan)
            raise DownloadError(f"{plan.shard_id}: partial file is shorter than {offset} bytes")
        with plan.partial_path.open("r+b") as fh:
            fh.truncate(offset)
            fh.flush()
            os.fsync(fh.fileno())
        digest = hasher.hexdigest()
        if plan.expected_sha256 is not None and not hmac.compare_digest(
            digest, plan.expected_sha256
        ):
            self._discard_partial(plan)
            raise ChecksumMismatch(f"{plan.shard_id}: final sha256 mismatch")
        os.replace(plan.partial_path, plan.cache_path)
        plan.checkpoint_path.unlink(missing_ok=True)
        _fsync_dir(plan.cache_path.parent)
        return DownloadResult(
            shard_id=plan.shard_id,
            path=plan.cache_path,
            compressed_bytes=offset,
            sha256=digest,
            attempts=attempts,
            resumed_from_bytes=resumed_from_bytes,
            bytes_downloaded=bytes_downloaded,
            reused=False,
        )

    @staticmethod
    def _header(headers: Message, name: str) -> str | None:
        value = headers.get(name)
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @classmethod
    def _content_length(cls, headers: Message) -> int | None:
        value = cls._header(headers, "Content-Length")
        if value is None:
            return None
        try:
            length = int(value)
        except ValueError as exc:
            raise DownloadError(f"invalid Content-Length {value!r}") from exc
        if length < 0:
            raise DownloadError(f"invalid Content-Length {value!r}")
        return length

    @staticmethod
    def _parse_unsatisfied_total(value: str | None) -> int | None:
        if value is None:
            return None
        match = _UNSATISFIED_RANGE_RE.fullmatch(value.strip())
        return int(match.group(1)) if match is not None else None


# ---------------------------------------------------------------------------
# Streaming decompression and JSONL documents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorpusDocument:
    shard_id: str
    index: int
    offset: int
    length: int
    raw: bytes
    obj: dict[str, object]
    text: str
    doc_id: str
    sha256: str


class ZstdDocIterator:
    def __init__(
        self,
        source: BinaryIO,
        *,
        shard_id: str,
        read_size: int = DEFAULT_READ_SIZE,
        max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    ) -> None:
        if read_size <= 0:
            raise ValueError("read_size must be positive")
        if max_line_bytes <= 0:
            raise ValueError("max_line_bytes must be positive")
        self.source = source
        self.shard_id = shard_id
        self.read_size = read_size
        self.max_line_bytes = max_line_bytes
        self._bytes_read = 0

    @property
    def bytes_read(self) -> int:
        return self._bytes_read

    def __iter__(self) -> Iterator[CorpusDocument]:
        decompressor = zstandard.ZstdDecompressor()
        try:
            reader = decompressor.stream_reader(
                self.source,
                read_size=self.read_size,
                read_across_frames=True,
                closefd=False,
            )
        except zstandard.ZstdError as exc:
            raise DocumentFormatError(f"{self.shard_id}: cannot open zstd stream: {exc}") from exc

        buffer = bytearray()
        offset = 0
        index = 0
        try:
            with reader:
                while True:
                    try:
                        chunk = reader.read(self.read_size)
                    except zstandard.ZstdError as exc:
                        raise DocumentFormatError(
                            f"{self.shard_id}: invalid zstd stream: {exc}"
                        ) from exc
                    if not chunk:
                        break
                    self._bytes_read += len(chunk)
                    buffer.extend(chunk)
                    while True:
                        newline = buffer.find(b"\n")
                        if newline < 0:
                            break
                        if newline > self.max_line_bytes:
                            raise DocumentFormatError(
                                f"{self.shard_id}: line {index + 1} exceeds "
                                f"{self.max_line_bytes} bytes"
                            )
                        raw = bytes(buffer[:newline])
                        del buffer[: newline + 1]
                        index += 1
                        document = self._parse_line(raw, index, offset)
                        offset += newline + 1
                        yield document
                    if len(buffer) > self.max_line_bytes:
                        raise DocumentFormatError(
                            f"{self.shard_id}: line {index + 1} exceeds "
                            f"{self.max_line_bytes} bytes"
                        )
            if buffer:
                index += 1
                document = self._parse_line(bytes(buffer), index, offset)
                yield document
        except zstandard.ZstdError as exc:
            raise DocumentFormatError(f"{self.shard_id}: invalid zstd stream: {exc}") from exc

    def _parse_line(self, raw: bytes, index: int, offset: int) -> CorpusDocument:
        if not raw:
            raise DocumentFormatError(f"{self.shard_id}: line {index} is blank")
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentFormatError(
                f"{self.shard_id}: line {index} is not valid UTF-8"
            ) from exc
        try:
            parsed: Any = json.loads(decoded)
        except json.JSONDecodeError as exc:
            raise DocumentFormatError(
                f"{self.shard_id}: line {index} is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(parsed, dict):
            raise DocumentFormatError(f"{self.shard_id}: line {index} is not a JSON object")
        obj: dict[str, object] = {}
        for key, value in parsed.items():
            if not isinstance(key, str):
                raise DocumentFormatError(f"{self.shard_id}: line {index} has a non-string key")
            obj[key] = value
        text = obj.get("text")
        if not isinstance(text, str):
            raise DocumentFormatError(f"{self.shard_id}: line {index} has no string 'text'")
        raw_id = obj.get("id")
        if raw_id is None:
            doc_id = f"{self.shard_id}:{index:06d}"
        elif isinstance(raw_id, str) and raw_id:
            doc_id = raw_id
        else:
            raise DocumentFormatError(
                f"{self.shard_id}: line {index} has an invalid 'id'"
            )
        return CorpusDocument(
            shard_id=self.shard_id,
            index=index,
            offset=offset,
            length=len(raw),
            raw=raw,
            obj=obj,
            text=text,
            doc_id=doc_id,
            sha256=_sha256_bytes(raw),
        )


# ---------------------------------------------------------------------------
# MinHash, LSH, dedup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DedupConfig:
    shingle_width: int = DEFAULT_SHINGLE_WIDTH
    num_permutations: int = DEFAULT_NUM_PERMUTATIONS
    bands: int = DEFAULT_BANDS
    rows: int = DEFAULT_ROWS
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD

    def __post_init__(self) -> None:
        for field_name, value in (
            ("shingle_width", self.shingle_width),
            ("num_permutations", self.num_permutations),
            ("bands", self.bands),
            ("rows", self.rows),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive int")
        if self.num_permutations != self.bands * self.rows:
            raise ValueError("num_permutations must equal bands * rows")
        threshold = self.similarity_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError("similarity_threshold must be a number")
        if not math.isfinite(float(threshold)) or not 0.0 < float(threshold) <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")


class MinHasher:
    def __init__(
        self,
        num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
        *,
        shingle_width: int = DEFAULT_SHINGLE_WIDTH,
    ) -> None:
        if (
            isinstance(num_permutations, bool)
            or not isinstance(num_permutations, int)
            or num_permutations <= 0
        ):
            raise ValueError("num_permutations must be a positive int")
        if isinstance(shingle_width, bool) or not isinstance(shingle_width, int) or shingle_width <= 0:
            raise ValueError("shingle_width must be a positive int")
        self._num_permutations = num_permutations
        self._shingle_width = shingle_width
        self._seeds = tuple(
            _splitmix64(((index + 1) * 0x9E3779B97F4A7C15) & UINT64_MASK)
            for index in range(num_permutations)
        )

    @property
    def num_permutations(self) -> int:
        return self._num_permutations

    @property
    def shingle_width(self) -> int:
        return self._shingle_width

    @property
    def seeds(self) -> tuple[int, ...]:
        return self._seeds

    @staticmethod
    def tokens(text: str) -> tuple[str, ...]:
        return tuple(re.findall(r"\w+", text.casefold()))

    def shingles(self, text: str) -> frozenset[tuple[str, ...]]:
        tokens = self.tokens(text)
        width = self._shingle_width
        if not tokens:
            return frozenset()
        if len(tokens) < width:
            return frozenset({tokens})
        return frozenset(
            tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)
        )

    @staticmethod
    def _base_hash(shingle: tuple[str, ...]) -> int:
        payload = b"\x1f".join(token.encode("utf-8") for token in shingle)
        return int.from_bytes(
            hashlib.blake2b(payload, digest_size=8, person=b"corpus42").digest(), "big"
        )

    def signature(self, text: str) -> tuple[int, ...]:
        shingles = self.shingles(text)
        if not shingles:
            return (0,) * self._num_permutations
        base_hashes = tuple(self._base_hash(shingle) for shingle in shingles)
        return tuple(
            min(_splitmix64(base_hash ^ seed) for base_hash in base_hashes)
            for seed in self._seeds
        )

    def similarity(self, first: tuple[int, ...], second: tuple[int, ...]) -> float:
        if len(first) != self._num_permutations or len(second) != self._num_permutations:
            raise ValueError("signatures must have num_permutations components")
        matches = sum(1 for left, right in zip(first, second) if left == right)
        return matches / self._num_permutations


@dataclass(frozen=True)
class LSHCandidate:
    shard_id: str
    doc_id: str
    signature: tuple[int, ...]


class LSHIndex:
    def __init__(
        self,
        *,
        num_permutations: int = DEFAULT_NUM_PERMUTATIONS,
        bands: int = DEFAULT_BANDS,
        rows: int = DEFAULT_ROWS,
    ) -> None:
        if (
            isinstance(num_permutations, bool)
            or not isinstance(num_permutations, int)
            or num_permutations <= 0
        ):
            raise ValueError("num_permutations must be a positive int")
        if isinstance(bands, bool) or not isinstance(bands, int) or bands <= 0:
            raise ValueError("bands must be a positive int")
        if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
            raise ValueError("rows must be a positive int")
        if num_permutations != bands * rows:
            raise ValueError("num_permutations must equal bands * rows")
        self.num_permutations = num_permutations
        self.bands = bands
        self.rows = rows
        self._entries: list[LSHCandidate] = []
        self._buckets: dict[tuple[int, tuple[int, ...]], list[int]] = {}

    def add(self, candidate: LSHCandidate) -> None:
        if len(candidate.signature) != self.num_permutations:
            raise ValueError("candidate signature has the wrong length")
        entry_index = len(self._entries)
        self._entries.append(candidate)
        for band_index in range(self.bands):
            start = band_index * self.rows
            key = (band_index, candidate.signature[start : start + self.rows])
            self._buckets.setdefault(key, []).append(entry_index)

    def query(self, signature: tuple[int, ...]) -> tuple[LSHCandidate, ...]:
        if len(signature) != self.num_permutations:
            raise ValueError("signature has the wrong length")
        matched: set[int] = set()
        for band_index in range(self.bands):
            start = band_index * self.rows
            key = (band_index, signature[start : start + self.rows])
            matched.update(self._buckets.get(key, ()))
        return tuple(self._entries[index] for index in sorted(matched))

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(frozen=True)
class DedupVerdict:
    verdict: Literal["keep", "near_duplicate"]
    matched_shard_id: str | None
    matched_doc_id: str | None
    estimated_jaccard: float | None


class Dedup:
    def __init__(self, config: DedupConfig = DedupConfig()) -> None:
        self.config = config
        self.hasher = MinHasher(
            config.num_permutations, shingle_width=config.shingle_width
        )
        self.index = LSHIndex(
            num_permutations=config.num_permutations,
            bands=config.bands,
            rows=config.rows,
        )
        self._exact: dict[str, tuple[str, str]] = {}

    def classify(self, document: CorpusDocument) -> DedupVerdict:
        exact = self._exact.get(document.sha256)
        if exact is not None:
            return DedupVerdict(
                verdict="near_duplicate",
                matched_shard_id=exact[0],
                matched_doc_id=exact[1],
                estimated_jaccard=1.0,
            )

        signature = self.hasher.signature(document.text)
        best: LSHCandidate | None = None
        best_similarity = -1.0
        for candidate in self.index.query(signature):
            similarity = self.hasher.similarity(signature, candidate.signature)
            if similarity > best_similarity:
                best = candidate
                best_similarity = similarity
        if best is not None and best_similarity >= self.config.similarity_threshold:
            return DedupVerdict(
                verdict="near_duplicate",
                matched_shard_id=best.shard_id,
                matched_doc_id=best.doc_id,
                estimated_jaccard=round(best_similarity, 6),
            )

        self.index.add(
            LSHCandidate(
                shard_id=document.shard_id,
                doc_id=document.doc_id,
                signature=signature,
            )
        )
        self._exact[document.sha256] = (document.shard_id, document.doc_id)
        return DedupVerdict(
            verdict="keep",
            matched_shard_id=None,
            matched_doc_id=None,
            estimated_jaccard=None,
        )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentRecord:
    doc_id: str
    line: int
    sha256: str
    verdict: Literal["keep", "near_duplicate"]
    offset: int | None
    length: int | None
    matched_shard_id: str | None
    matched_doc_id: str | None
    estimated_jaccard: float | None


@dataclass(frozen=True)
class ShardResult:
    shard_id: str
    url: str
    compressed_path: Path
    output_path: Path
    compressed_bytes: int
    compressed_sha256: str
    decompressed_bytes: int
    output_bytes: int
    output_sha256: str
    documents: tuple[DocumentRecord, ...]


class ManifestWriter:
    def __init__(self, config: DedupConfig) -> None:
        self.config = config
        self._shards: list[ShardResult] = []
        self._shard_ids: set[str] = set()

    def add_shard(self, result: ShardResult) -> None:
        if result.shard_id in self._shard_ids:
            raise ManifestError(f"duplicate shard id in manifest: {result.shard_id}")
        self._shard_ids.add(result.shard_id)
        self._shards.append(result)

    def write(self, manifest_path: Path) -> str:
        manifest_path = Path(manifest_path)
        base = manifest_path.parent.resolve()
        shards: list[dict[str, object]] = []
        totals = {
            "shards": len(self._shards),
            "compressed_bytes": 0,
            "decompressed_bytes": 0,
            "output_bytes": 0,
            "documents": 0,
            "unique_documents": 0,
            "duplicate_count": 0,
        }
        for result in self._shards:
            documents: list[dict[str, object]] = []
            unique = 0
            duplicates = 0
            for record in result.documents:
                if record.verdict == "keep":
                    unique += 1
                else:
                    duplicates += 1
                documents.append(
                    {
                        "doc_id": record.doc_id,
                        "line": record.line,
                        "sha256": record.sha256,
                        "verdict": record.verdict,
                        "offset": record.offset,
                        "length": record.length,
                        "matched_shard_id": record.matched_shard_id,
                        "matched_doc_id": record.matched_doc_id,
                        "estimated_jaccard": record.estimated_jaccard,
                    }
                )
            if unique + duplicates != len(result.documents):
                raise ManifestError(f"{result.shard_id}: inconsistent document verdicts")
            compressed_relative = _relative_to(base, result.compressed_path)
            output_relative = _relative_to(base, result.output_path)
            shards.append(
                {
                    "shard_id": result.shard_id,
                    "url": _public_url(result.url),
                    "compressed_path": compressed_relative,
                    "compressed_bytes": result.compressed_bytes,
                    "compressed_sha256": result.compressed_sha256,
                    "decompressed_bytes": result.decompressed_bytes,
                    "output_path": output_relative,
                    "output_bytes": result.output_bytes,
                    "sha256": result.output_sha256,
                    "document_count": len(result.documents),
                    "unique_document_count": unique,
                    "duplicate_count": duplicates,
                    "documents": documents,
                }
            )
            totals["compressed_bytes"] += result.compressed_bytes
            totals["decompressed_bytes"] += result.decompressed_bytes
            totals["output_bytes"] += result.output_bytes
            totals["documents"] += len(result.documents)
            totals["unique_documents"] += unique
            totals["duplicate_count"] += duplicates

        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "generator": GENERATOR,
            "dedup": {
                "algorithm": "minhash-lsh",
                "shingle_unit": "word",
                "shingle_width": self.config.shingle_width,
                "num_permutations": self.config.num_permutations,
                "bands": self.config.bands,
                "rows": self.config.rows,
                "similarity_threshold": self.config.similarity_threshold,
                "hash_family": HASH_FAMILY,
            },
            "shards": shards,
            "totals": totals,
        }
        data = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
        _atomic_write_bytes(manifest_path, data)
        digest = _sha256_bytes(data)
        lock_path = manifest_path.with_suffix(".lock")
        lock_data = f"{digest}  {manifest_path.name}\n".encode("utf-8")
        _atomic_write_bytes(lock_path, lock_data)
        return digest


def verify_manifest(
    manifest_path: Path, *, verify_shards: bool = True
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    lock_path = manifest_path.with_suffix(".lock")
    try:
        data = manifest_path.read_bytes()
    except OSError as exc:
        raise ManifestError(f"cannot read manifest: {manifest_path}") from exc
    try:
        lock_text = lock_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ManifestError(f"cannot read manifest lock: {lock_path}") from exc
    parts = lock_text.split()
    if len(parts) != 2 or _SHA256_RE.fullmatch(parts[0]) is None:
        raise ManifestError("malformed manifest lock")
    if parts[1] != manifest_path.name:
        raise ManifestError("manifest lock names a different file")
    if not hmac.compare_digest(_sha256_bytes(data), parts[0]):
        raise ManifestError("manifest sha256 does not match manifest.lock")

    try:
        parsed: Any = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest is not valid JSON: {exc.msg}") from exc
    manifest = _require_object(parsed, "manifest")
    if _require_int(manifest.get("schema_version"), "schema_version", minimum=1) != SCHEMA_VERSION:
        raise ManifestError(f"unsupported manifest schema version")
    dedup = _require_object(manifest.get("dedup"), "dedup")
    for key in ("shingle_width", "num_permutations", "bands", "rows"):
        _require_int(dedup.get(key), f"dedup.{key}", minimum=1)
    _require_str(dedup.get("algorithm"), "dedup.algorithm")
    _require_float(dedup.get("similarity_threshold"), "dedup.similarity_threshold")

    shards_raw = _require_list(manifest.get("shards"), "shards")
    if not verify_shards:
        return manifest

    base = manifest_path.parent.resolve()
    keep_refs: set[tuple[str, str]] = set()
    duplicate_refs: list[tuple[str, str, str, str]] = []
    all_doc_ids: set[str] = set()
    seen_shard_ids: set[str] = set()
    totals = {
        "shards": 0,
        "compressed_bytes": 0,
        "decompressed_bytes": 0,
        "output_bytes": 0,
        "documents": 0,
        "unique_documents": 0,
        "duplicate_count": 0,
    }

    for shard_raw in shards_raw:
        shard = _require_object(shard_raw, "shard")
        shard_id = _require_str(shard.get("shard_id"), "shard.shard_id")
        if shard_id in seen_shard_ids:
            raise ManifestError(f"duplicate shard id {shard_id!r}")
        seen_shard_ids.add(shard_id)
        _require_str(shard.get("url"), f"{shard_id}.url")
        compressed_path = _resolve_relative(
            base, _require_str(shard.get("compressed_path"), f"{shard_id}.compressed_path")
        )
        output_path = _resolve_relative(
            base, _require_str(shard.get("output_path"), f"{shard_id}.output_path")
        )
        compressed_bytes = _require_int(
            shard.get("compressed_bytes"), f"{shard_id}.compressed_bytes"
        )
        compressed_sha = _require_str(
            shard.get("compressed_sha256"), f"{shard_id}.compressed_sha256"
        )
        decompressed_bytes = _require_int(
            shard.get("decompressed_bytes"), f"{shard_id}.decompressed_bytes"
        )
        output_bytes = _require_int(shard.get("output_bytes"), f"{shard_id}.output_bytes")
        output_sha = _require_str(shard.get("sha256"), f"{shard_id}.sha256")
        document_count = _require_int(
            shard.get("document_count"), f"{shard_id}.document_count"
        )
        unique_count = _require_int(
            shard.get("unique_document_count"), f"{shard_id}.unique_document_count"
        )
        duplicate_count = _require_int(
            shard.get("duplicate_count"), f"{shard_id}.duplicate_count"
        )
        if not output_path.is_file():
            raise ManifestError(f"{shard_id}: output file is missing: {output_path}")
        if output_path.stat().st_size != output_bytes:
            raise ManifestError(f"{shard_id}: output size mismatch")
        if not hmac.compare_digest(_sha256_file(output_path), output_sha):
            raise ManifestError(f"{shard_id}: output sha256 mismatch")
        if compressed_path.is_file():
            if compressed_path.stat().st_size != compressed_bytes:
                raise ManifestError(f"{shard_id}: compressed size mismatch")
            if not hmac.compare_digest(_sha256_file(compressed_path), compressed_sha):
                raise ManifestError(f"{shard_id}: compressed sha256 mismatch")

        documents_raw = _require_list(shard.get("documents"), f"{shard_id}.documents")
        if len(documents_raw) != document_count:
            raise ManifestError(f"{shard_id}: document_count mismatch")
        next_offset = 0
        kept_here = 0
        duplicated_here = 0
        with output_path.open("rb") as output:
            for document_raw in documents_raw:
                document = _require_object(document_raw, f"{shard_id}.document")
                doc_id = _require_str(document.get("doc_id"), f"{shard_id}.doc_id")
                if doc_id in all_doc_ids:
                    raise ManifestError(f"{shard_id}: duplicate document id {doc_id!r}")
                all_doc_ids.add(doc_id)
                _require_int(document.get("line"), f"{shard_id}.{doc_id}.line", minimum=1)
                document_sha = _require_str(
                    document.get("sha256"), f"{shard_id}.{doc_id}.sha256"
                )
                verdict = document.get("verdict")
                if verdict not in {"keep", "near_duplicate"}:
                    raise ManifestError(f"{shard_id}: {doc_id}: invalid verdict")
                if verdict == "keep":
                    offset = _require_int(
                        document.get("offset"), f"{shard_id}.{doc_id}.offset"
                    )
                    length = _require_int(
                        document.get("length"), f"{shard_id}.{doc_id}.length"
                    )
                    if offset != next_offset:
                        raise ManifestError(f"{shard_id}: {doc_id}: non-contiguous offset")
                    output.seek(offset)
                    raw = output.read(length)
                    if len(raw) != length:
                        raise ManifestError(f"{shard_id}: {doc_id}: truncated line")
                    if not hmac.compare_digest(_sha256_bytes(raw), document_sha):
                        raise ManifestError(f"{shard_id}: {doc_id}: document hash mismatch")
                    after = output.read(1)
                    if after == b"\n":
                        next_offset = offset + length + 1
                    elif after == b"":
                        next_offset = offset + length
                    else:
                        raise ManifestError(f"{shard_id}: {doc_id}: missing newline")
                    if document.get("matched_shard_id") is not None:
                        raise ManifestError(f"{shard_id}: {doc_id}: keep has a match shard")
                    if document.get("matched_doc_id") is not None:
                        raise ManifestError(f"{shard_id}: {doc_id}: keep has a match doc")
                    if document.get("estimated_jaccard") is not None:
                        raise ManifestError(f"{shard_id}: {doc_id}: keep has a similarity")
                    keep_refs.add((shard_id, doc_id))
                    kept_here += 1
                else:
                    if document.get("offset") is not None or document.get("length") is not None:
                        raise ManifestError(
                            f"{shard_id}: {doc_id}: duplicate has output offsets"
                        )
                    matched_shard = _require_str(
                        document.get("matched_shard_id"),
                        f"{shard_id}.{doc_id}.matched_shard_id",
                    )
                    matched_doc = _require_str(
                        document.get("matched_doc_id"),
                        f"{shard_id}.{doc_id}.matched_doc_id",
                    )
                    _require_float(
                        document.get("estimated_jaccard"),
                        f"{shard_id}.{doc_id}.estimated_jaccard",
                    )
                    duplicate_refs.append((shard_id, doc_id, matched_shard, matched_doc))
                    duplicated_here += 1
            if next_offset != output_path.stat().st_size:
                raise ManifestError(f"{shard_id}: output has trailing bytes")
        if kept_here != unique_count:
            raise ManifestError(f"{shard_id}: unique_document_count mismatch")
        if duplicated_here != duplicate_count:
            raise ManifestError(f"{shard_id}: duplicate_count mismatch")
        totals["shards"] += 1
        totals["compressed_bytes"] += compressed_bytes
        totals["decompressed_bytes"] += decompressed_bytes
        totals["output_bytes"] += output_bytes
        totals["documents"] += document_count
        totals["unique_documents"] += unique_count
        totals["duplicate_count"] += duplicate_count

    for shard_id, doc_id, matched_shard, matched_doc in duplicate_refs:
        if (matched_shard, matched_doc) not in keep_refs:
            raise ManifestError(
                f"{shard_id}: {doc_id}: tombstone points at unknown keeper "
                f"{matched_shard}/{matched_doc}"
            )

    declared_totals = _require_object(manifest.get("totals"), "totals")
    for key, value in totals.items():
        if _require_int(declared_totals.get(key), f"totals.{key}") != value:
            raise ManifestError(f"totals.{key} mismatch")
    return manifest


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResult:
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, object]


class CorpusPipeline:
    def __init__(
        self,
        *,
        workdir: Path,
        cache_dir: Path | None = None,
        corpus_dir: Path | None = None,
        manifest_path: Path | None = None,
        config: DedupConfig = DedupConfig(),
        resume_only: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.workdir = Path(workdir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else self.workdir / "cache"
        self.corpus_dir = Path(corpus_dir) if corpus_dir is not None else self.workdir / "corpus"
        self.manifest_path = (
            Path(manifest_path) if manifest_path is not None else self.workdir / "manifest.json"
        )
        base = self.manifest_path.parent.resolve()
        for name, path in (("cache_dir", self.cache_dir), ("corpus_dir", self.corpus_dir)):
            try:
                path.resolve().relative_to(base)
            except ValueError as exc:
                raise PlanError(
                    f"{name} must be inside the manifest directory {base}"
                ) from exc
        self.config = config
        self.resume_only = resume_only
        self.downloader = StreamingDownloader(chunk_size=chunk_size)

    def _verify_existing_manifest(self) -> None:
        lock_path = self.manifest_path.with_suffix(".lock")
        if not self.manifest_path.exists() and not lock_path.exists():
            return
        verify_manifest(self.manifest_path, verify_shards=False)

    def process(
        self,
        urls: Sequence[str],
        *,
        expected_sizes: Mapping[str, int] | None = None,
        expected_sha256: Mapping[str, str] | None = None,
        downloads: Mapping[str, DownloadResult] | None = None,
    ) -> PipelineResult:
        self._verify_existing_manifest()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        planner = ShardPlanner(self.cache_dir, self.corpus_dir)
        plans = planner.plan(
            urls, expected_sizes=expected_sizes, expected_sha256=expected_sha256
        )
        if downloads is not None:
            plan_ids = {plan.shard_id for plan in plans}
            unknown = set(downloads) - plan_ids
            if unknown:
                raise PlanError(f"downloads supplied for unplanned shard(s): {sorted(unknown)!r}")
        if self.resume_only:
            for plan in plans:
                self.downloader.preflight(plan)

        dedup = Dedup(self.config)
        writer = ManifestWriter(self.config)
        seen_doc_ids: set[str] = set()
        for plan in plans:
            if downloads is not None and plan.shard_id in downloads:
                result = downloads[plan.shard_id]
                if result.shard_id != plan.shard_id:
                    raise PlanError(f"download result shard id mismatch for {plan.shard_id}")
                if result.path.resolve() != plan.cache_path.resolve():
                    raise PlanError(f"download result path mismatch for {plan.shard_id}")
                if not plan.cache_path.is_file():
                    raise DownloadError(f"{plan.shard_id}: supplied cache file is missing")
            else:
                result = self.downloader.download(plan, resume_only=self.resume_only)

            output_part = plan.output_path.with_name(plan.output_path.name + ".part")
            documents: list[DocumentRecord] = []
            written_bytes = 0
            try:
                with plan.cache_path.open("rb") as compressed:
                    iterator = ZstdDocIterator(compressed, shard_id=plan.shard_id)
                    with output_part.open("wb") as output:
                        for document in iterator:
                            if document.doc_id in seen_doc_ids:
                                raise DocumentFormatError(
                                    f"{plan.shard_id}: duplicate document id {document.doc_id!r}"
                                )
                            seen_doc_ids.add(document.doc_id)
                            verdict = dedup.classify(document)
                            if verdict.verdict == "keep":
                                offset = output.tell()
                                output.write(document.raw + b"\n")
                                documents.append(
                                    DocumentRecord(
                                        doc_id=document.doc_id,
                                        line=document.index,
                                        sha256=document.sha256,
                                        verdict="keep",
                                        offset=offset,
                                        length=document.length,
                                        matched_shard_id=None,
                                        matched_doc_id=None,
                                        estimated_jaccard=None,
                                    )
                                )
                                written_bytes += document.length + 1
                            else:
                                documents.append(
                                    DocumentRecord(
                                        doc_id=document.doc_id,
                                        line=document.index,
                                        sha256=document.sha256,
                                        verdict="near_duplicate",
                                        offset=None,
                                        length=None,
                                        matched_shard_id=verdict.matched_shard_id,
                                        matched_doc_id=verdict.matched_doc_id,
                                        estimated_jaccard=verdict.estimated_jaccard,
                                    )
                                )
                        output.flush()
                        os.fsync(output.fileno())
                if output_part.stat().st_size != written_bytes:
                    raise CorpusError(f"{plan.shard_id}: output byte count mismatch")
                os.replace(output_part, plan.output_path)
                _fsync_dir(self.corpus_dir)
            except Exception:
                output_part.unlink(missing_ok=True)
                raise

            output_bytes = plan.output_path.stat().st_size
            writer.add_shard(
                ShardResult(
                    shard_id=plan.shard_id,
                    url=_public_url(plan.url),
                    compressed_path=plan.cache_path,
                    output_path=plan.output_path,
                    compressed_bytes=result.compressed_bytes,
                    compressed_sha256=result.sha256,
                    decompressed_bytes=iterator.bytes_read,
                    output_bytes=output_bytes,
                    output_sha256=_sha256_file(plan.output_path),
                    documents=tuple(documents),
                )
            )

        digest = writer.write(self.manifest_path)
        verified = verify_manifest(self.manifest_path)
        return PipelineResult(
            manifest_path=self.manifest_path,
            manifest_sha256=digest,
            manifest=verified,
        )


# ---------------------------------------------------------------------------
# Demo corpus and CLI
# ---------------------------------------------------------------------------

_DEMO_BASE = (
    "the archive keeps every ledger from the northern ports and each entry records "
    "the ship the cargo the harbour fee and the day the record was signed"
)
_DEMO_FOOTER = _DEMO_BASE + " retrieved from the public mirror"
_DEMO_TRACKING = _DEMO_BASE + " ?utm_source=mirror"
_DEMO_DOCUMENTS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    (
        "shard-0000",
        (
            ("a-001", _DEMO_BASE),
            (
                "a-002",
                "the orchard ledger tracks pear harvests irrigation hours cold storage "
                "batches and the names of the pickers who signed each crate at dawn",
            ),
            ("a-003", _DEMO_BASE),
        ),
    ),
    (
        "shard-0001",
        (
            ("b-001", _DEMO_FOOTER),
            (
                "b-002",
                "the reactor log records coolant pressure control rod depth turbine "
                "vibration and the technician who signed the evening inspection sheet",
            ),
            (
                "b-003",
                "the ferry timetable lists departure quay passenger counts freight "
                "tonnage and the weather delay noted by the harbour master",
            ),
        ),
    ),
    (
        "shard-0002",
        (
            ("c-001", _DEMO_TRACKING),
            (
                "c-002",
                "the seed bank catalogue stores germination rates collection dates "
                "soil notes and the collector who documented each accession",
            ),
            (
                "c-003",
                "the observatory notebook holds telescope focus sky transparency "
                "exposure times and the observer who closed the dome before rain",
            ),
        ),
    ),
)


@dataclass(frozen=True)
class DemoResult:
    workdir: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, object]
    resumed_from_bytes: int


def build_demo_corpus(remote_dir: Path) -> tuple[list[str], dict[str, int], dict[str, str]]:
    remote_dir = Path(remote_dir)
    remote_dir.mkdir(parents=True, exist_ok=True)
    urls: list[str] = []
    sizes: dict[str, int] = {}
    hashes: dict[str, str] = {}
    compressor = zstandard.ZstdCompressor(level=3)
    for shard_id, documents in _DEMO_DOCUMENTS:
        lines = [
            json.dumps(
                {"id": doc_id, "text": text},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for doc_id, text in documents
        ]
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        compressed = compressor.compress(raw)
        path = remote_dir / f"{shard_id}.jsonl.zst"
        path.write_bytes(compressed)
        urls.append(path.resolve().as_uri())
        sizes[shard_id] = len(compressed)
        hashes[shard_id] = _sha256_bytes(compressed)
    return urls, sizes, hashes


def _proxy_free_opener(request: urllib.request.Request) -> ResponseLike:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return cast(ResponseLike, opener.open(request, timeout=30.0))


def _print_manifest_summary(manifest: Mapping[str, object]) -> None:
    shards = _require_list(manifest.get("shards"), "shards")
    print(
        f"{'shard':<14} {'docs':>5} {'unique':>7} {'dup':>4} "
        f"{'compressed':>11} {'decompressed':>13} {'output':>8}"
    )
    for raw in shards:
        shard = _require_object(raw, "shard")
        print(
            f"{_require_str(shard.get('shard_id'), 'shard_id'):<14} "
            f"{_require_int(shard.get('document_count'), 'document_count'):>5} "
            f"{_require_int(shard.get('unique_document_count'), 'unique'):>7} "
            f"{_require_int(shard.get('duplicate_count'), 'duplicate'):>4} "
            f"{_require_int(shard.get('compressed_bytes'), 'compressed'):>11} "
            f"{_require_int(shard.get('decompressed_bytes'), 'decompressed'):>13} "
            f"{_require_int(shard.get('output_bytes'), 'output'):>8}"
        )
    totals = _require_object(manifest.get("totals"), "totals")
    print(
        f"{'total':<14} "
        f"{_require_int(totals.get('documents'), 'documents'):>5} "
        f"{_require_int(totals.get('unique_documents'), 'unique'):>7} "
        f"{_require_int(totals.get('duplicate_count'), 'duplicate'):>4} "
        f"{_require_int(totals.get('compressed_bytes'), 'compressed'):>11} "
        f"{_require_int(totals.get('decompressed_bytes'), 'decompressed'):>13} "
        f"{_require_int(totals.get('output_bytes'), 'output'):>8}"
    )


def run_demo(workdir: Path = Path("outputs/demo"), *, shingle_width: int = DEFAULT_SHINGLE_WIDTH) -> DemoResult:
    workdir = Path(workdir)
    remote_dir = workdir / "remote"
    cache_dir = workdir / "cache"
    corpus_dir = workdir / "corpus"
    manifest_path = workdir / "manifest.json"
    for directory in (remote_dir, cache_dir, corpus_dir):
        shutil.rmtree(directory, ignore_errors=True)
    manifest_path.unlink(missing_ok=True)
    manifest_path.with_suffix(".lock").unlink(missing_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    urls, sizes, hashes = build_demo_corpus(remote_dir)
    shard_one_payload = (remote_dir / "shard-0001.jsonl.zst").read_bytes()
    file_downloader = StreamingDownloader()
    http_downloader = StreamingDownloader(opener=_proxy_free_opener)

    with LocalRangeServer(
        shard_one_payload,
        path="/shard-0001.jsonl.zst",
        fail_first_get_after=max(1, len(shard_one_payload) // 2),
    ) as server:
        urls[1] = server.url
        planner = ShardPlanner(cache_dir, corpus_dir)
        plans = planner.plan(urls, expected_sizes=sizes, expected_sha256=hashes)
        by_id = {plan.shard_id: plan for plan in plans}

        downloaded: dict[str, DownloadResult] = {}
        downloaded["shard-0000"] = file_downloader.download(by_id["shard-0000"])
        try:
            http_downloader.download(by_id["shard-0001"])
        except IncompleteDownloadError:
            pass
        else:
            raise CorpusError("demo origin did not simulate a disconnect")
        resumed = http_downloader.download(by_id["shard-0001"])
        downloaded["shard-0001"] = resumed
        downloaded["shard-0002"] = file_downloader.download(by_id["shard-0002"])

        pipeline = CorpusPipeline(
            workdir=workdir,
            config=DedupConfig(shingle_width=shingle_width),
        )
        result = pipeline.process(
            urls,
            expected_sizes=sizes,
            expected_sha256=hashes,
            downloads=downloaded,
        )

    print(f"demo corpus: {workdir}")
    _print_manifest_summary(result.manifest)
    print(
        f"resume: shard-0001 resumed at {resumed.resumed_from_bytes} of "
        f"{resumed.compressed_bytes} compressed bytes after a simulated disconnect"
    )
    print(f"manifest: {result.manifest_path}")
    print(f"manifest sha256: {result.manifest_sha256}")
    return DemoResult(
        workdir=workdir,
        manifest_path=result.manifest_path,
        manifest_sha256=result.manifest_sha256,
        manifest=result.manifest,
        resumed_from_bytes=resumed.resumed_from_bytes,
    )


def _parse_assignments(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise PlanError(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key or not value:
            raise PlanError(f"expected non-empty KEY=VALUE, got {item!r}")
        if key in result:
            raise PlanError(f"duplicate expectation for {key!r}")
        result[key] = value
    return result


def _parse_size_assignments(values: Sequence[str]) -> dict[str, int]:
    parsed = _parse_assignments(values)
    result: dict[str, int] = {}
    for key, value in parsed.items():
        try:
            size = int(value)
        except ValueError as exc:
            raise PlanError(f"expected size for {key!r} is not an int") from exc
        if size < 0:
            raise PlanError(f"expected size for {key!r} must be non-negative")
        result[key] = size
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="large-corpus-downloader")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="run the deterministic offline demo")
    demo_parser.add_argument("--workdir", type=Path, default=Path("outputs/demo"))
    demo_parser.add_argument("--shingle-width", type=int, default=DEFAULT_SHINGLE_WIDTH)

    fetch_parser = subparsers.add_parser("fetch", help="download and deduplicate shard URLs")
    fetch_parser.add_argument("--url", action="append", required=True)
    fetch_parser.add_argument("--workdir", type=Path, default=Path("outputs/run"))
    fetch_parser.add_argument("--cache-dir", type=Path)
    fetch_parser.add_argument("--corpus-dir", type=Path)
    fetch_parser.add_argument("--manifest", type=Path)
    fetch_parser.add_argument("--shingle-width", type=int, default=DEFAULT_SHINGLE_WIDTH)
    fetch_parser.add_argument("--resume-only", action="store_true")
    fetch_parser.add_argument("--expected-size", action="append", default=[])
    fetch_parser.add_argument("--expected-sha256", action="append", default=[])

    verify_parser = subparsers.add_parser(
        "verify-manifest", help="verify a manifest and its shard files"
    )
    verify_parser.add_argument("--manifest", type=Path, default=Path("outputs/demo/manifest.json"))

    raw_args = list(sys.argv[1:] if argv is None else argv)
    if not raw_args or (
        raw_args[0].startswith("-") and raw_args[0] not in {"-h", "--help"}
    ):
        raw_args = ["demo", *raw_args]
    args = parser.parse_args(raw_args)
    command = args.command or "demo"
    try:
        if command == "demo":
            run_demo(Path(args.workdir), shingle_width=int(args.shingle_width))
            return 0
        if command == "fetch":
            config = DedupConfig(shingle_width=int(args.shingle_width))
            pipeline = CorpusPipeline(
                workdir=Path(args.workdir),
                cache_dir=Path(args.cache_dir) if args.cache_dir is not None else None,
                corpus_dir=Path(args.corpus_dir) if args.corpus_dir is not None else None,
                manifest_path=Path(args.manifest) if args.manifest is not None else None,
                config=config,
                resume_only=bool(args.resume_only),
            )
            result = pipeline.process(
                list(args.url),
                expected_sizes=_parse_size_assignments(list(args.expected_size)),
                expected_sha256=_parse_assignments(list(args.expected_sha256)),
            )
            _print_manifest_summary(result.manifest)
            print(f"manifest: {result.manifest_path}")
            print(f"manifest sha256: {result.manifest_sha256}")
            return 0
        if command == "verify-manifest":
            manifest = verify_manifest(Path(args.manifest))
            _print_manifest_summary(manifest)
            print(f"manifest: {args.manifest}")
            return 0
        parser.error(f"unknown command {command!r}")
    except (CorpusError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
