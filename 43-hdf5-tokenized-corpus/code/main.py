"""Streaming byte tokenization into sharded, resizable HDF5 datasets.

The pipeline tokenizes documents incrementally, buffers token ids to a fixed
chunk size, and extends one HDF5 dataset per input shard in chunk-sized
strides. A ``shards.json`` index records the token count, document count,
SHA-256, and byte size of every shard. The reader opens shard files through
h5py's chunk cache (it does not memory-map them), concatenates the selected
shards into one logical token stream, and serves fixed-length windows.

The module is intentionally self-contained: no sibling lesson code, no network
calls, and no GPU. The demo, the tests, and the benchmark all run offline.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import operator
import os
import re
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Protocol, cast

import h5py
import numpy as np
from numpy.typing import NDArray

SCHEMA_VERSION = 1
GENERATOR = "hdf5-tokenized-corpus/0.1.0"
DATASET_NAME = "tokens"
TOKENS_DTYPE = np.dtype("<u4")
BYTE_VOCAB_SIZE = 257
BOUNDARY_TOKEN_ID = 256
UINT32_MAX = 0xFFFF_FFFF
INDEX_FILENAME = "shards.json"
DEFAULT_CHUNK_SIZE = 8192
DEFAULT_COMPRESSION_LEVEL = 4
DEFAULT_RDCC_NBYTES = 1 << 20
DEFAULT_VERIFY_BLOCK_TOKENS = 1_048_576

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

TokenArray = NDArray[np.uint32]
TokenBatch = tuple[TokenArray, TokenArray]
Compression = Literal["gzip"]


class CorpusError(Exception):
    """Base class for expected corpus format and lifecycle errors."""


class IndexFormatError(CorpusError):
    """The shard index is missing, malformed, or internally inconsistent."""


class ShardFormatError(CorpusError):
    """A shard file does not match the contract recorded in the index."""


class ChecksumMismatch(CorpusError):
    """A shard's token bytes do not match the recorded SHA-256."""


class DocumentFormatError(CorpusError):
    """An input document is not valid JSONL with string id and text fields."""


class TokenValueError(CorpusError):
    """A token id is not a uint32 integer."""


class ClosedStoreError(CorpusError):
    """The token store was used after close."""


class Tokenizer(Protocol):
    """The minimal tokenizer contract needed by the pipeline."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def boundary_token_id(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...


class ByteTokenizer:
    """Deterministic UTF-8 byte tokenizer with one reserved boundary id.

    Token ids 0 through 255 are the UTF-8 bytes. Id 256 is reserved for the
    document separator and is never produced by :meth:`encode`.
    """

    @property
    def vocab_size(self) -> int:
        return BYTE_VOCAB_SIZE

    @property
    def boundary_token_id(self) -> int:
        return BOUNDARY_TOKEN_ID

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        raw = bytearray()
        for token in ids:
            if isinstance(token, bool) or not isinstance(token, int):
                raise ValueError(f"token id must be an int, got {token!r}")
            if not 0 <= token <= 0xFF:
                raise ValueError(f"token id outside the byte vocabulary: {token}")
            raw.append(token)
        return bytes(raw).decode("utf-8")


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One input document, already decoded from JSONL."""

    document_id: str
    text: str


@dataclass(frozen=True, slots=True)
class ShardStats:
    token_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ShardRecord:
    shard_id: str
    path: str
    token_count: int
    document_count: int
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class ShardIndex:
    chunk_size: int
    vocab_size: int
    boundary_token_id: int
    total_tokens: int
    shards: tuple[ShardRecord, ...]
    schema_version: int = SCHEMA_VERSION
    generator: str = GENERATOR

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generator": self.generator,
            "chunk_size": self.chunk_size,
            "vocab_size": self.vocab_size,
            "boundary_token_id": self.boundary_token_id,
            "total_tokens": self.total_tokens,
            "shards": [
                {
                    "shard_id": shard.shard_id,
                    "path": shard.path,
                    "token_count": shard.token_count,
                    "document_count": shard.document_count,
                    "sha256": shard.sha256,
                    "byte_size": shard.byte_size,
                }
                for shard in self.shards
            ],
        }

    @classmethod
    def from_dict(cls, payload: object) -> ShardIndex:
        data = _as_mapping(payload, "index")
        version = _require_int(data, "schema_version", "index", minimum=0)
        if version != SCHEMA_VERSION:
            raise IndexFormatError(
                f"index.schema_version is {version}, expected {SCHEMA_VERSION}"
            )
        chunk_size = _require_int(data, "chunk_size", "index", minimum=1)
        vocab_size = _require_int(data, "vocab_size", "index", minimum=1)
        boundary_token_id = _require_int(
            data, "boundary_token_id", "index", minimum=0
        )
        if boundary_token_id >= vocab_size:
            raise IndexFormatError(
                "index.boundary_token_id must be smaller than index.vocab_size"
            )
        total_tokens = _require_int(data, "total_tokens", "index", minimum=0)

        generator_value = data.get("generator", GENERATOR)
        if not isinstance(generator_value, str):
            raise IndexFormatError("index.generator must be a string")

        raw_shards = data.get("shards")
        if not isinstance(raw_shards, list):
            raise IndexFormatError("index.shards must be a list")

        shards: list[ShardRecord] = []
        seen_ids: set[str] = set()
        seen_paths: set[str] = set()
        for position, raw_shard in enumerate(raw_shards):
            where = f"index.shards[{position}]"
            shard_data = _as_mapping(raw_shard, where)
            shard_id = _require_str(shard_data, "shard_id", where)
            if not _SHARD_ID_RE.fullmatch(shard_id):
                raise IndexFormatError(f"{where}.shard_id is not a safe id")
            if shard_id in seen_ids:
                raise IndexFormatError(f"duplicate shard id: {shard_id}")
            seen_ids.add(shard_id)

            path = _require_str(shard_data, "path", where)
            _validate_bare_filename(path, f"{where}.path")
            if path in seen_paths:
                raise IndexFormatError(f"duplicate shard path: {path}")
            seen_paths.add(path)

            token_count = _require_int(shard_data, "token_count", where, minimum=0)
            document_count = _require_int(
                shard_data, "document_count", where, minimum=0
            )
            sha256 = _require_str(shard_data, "sha256", where)
            if not _SHA256_RE.fullmatch(sha256):
                raise IndexFormatError(f"{where}.sha256 must be 64 lowercase hex")
            byte_size = _require_int(shard_data, "byte_size", where, minimum=0)

            shards.append(
                ShardRecord(
                    shard_id=shard_id,
                    path=path,
                    token_count=token_count,
                    document_count=document_count,
                    sha256=sha256,
                    byte_size=byte_size,
                )
            )

        summed_tokens = sum(shard.token_count for shard in shards)
        if summed_tokens != total_tokens:
            raise IndexFormatError(
                "index.total_tokens does not equal the sum of shard token counts"
            )

        return cls(
            chunk_size=chunk_size,
            vocab_size=vocab_size,
            boundary_token_id=boundary_token_id,
            total_tokens=total_tokens,
            shards=tuple(shards),
            schema_version=version,
            generator=generator_value,
        )


def _as_mapping(value: object, where: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise IndexFormatError(f"{where} must be a JSON object")
    return cast(Mapping[str, object], value)


def _require_int(
    data: Mapping[str, object],
    key: str,
    where: str,
    *,
    minimum: int,
) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise IndexFormatError(f"{where}.{key} must be an integer")
    if value < minimum:
        raise IndexFormatError(f"{where}.{key} must be >= {minimum}")
    return value


def _require_str(data: Mapping[str, object], key: str, where: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise IndexFormatError(f"{where}.{key} must be a string")
    if not value:
        raise IndexFormatError(f"{where}.{key} must not be empty")
    return value


def _validate_bare_filename(path: str, where: str) -> None:
    if Path(path).name != path or path in {".", ".."} or "\x00" in path:
        raise IndexFormatError(f"{where} must be a bare filename")


def load_shard_index(path: Path) -> ShardIndex:
    index_path = Path(path)
    try:
        raw = index_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise IndexFormatError(f"{index_path}: cannot read shard index: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise IndexFormatError(f"{index_path}: invalid JSON: {exc}") from exc
    try:
        return ShardIndex.from_dict(payload)
    except IndexFormatError as exc:
        raise IndexFormatError(f"{index_path}: {exc}") from exc


def write_shard_index(path: Path, index: ShardIndex) -> None:
    index_path = Path(path)
    payload = json.dumps(index.to_dict(), indent=2, sort_keys=True) + "\n"
    temporary = index_path.with_name(index_path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, index_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _token_array(tokens: Sequence[int]) -> TokenArray:
    try:
        raw = np.asarray(tokens)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TokenValueError(f"tokens are not a sequence of integers: {exc}") from exc
    if raw.ndim != 1:
        raise TokenValueError(f"tokens must be one-dimensional, got shape {raw.shape}")
    if raw.size == 0:
        return np.empty(0, dtype=TOKENS_DTYPE)
    if raw.dtype.kind not in {"i", "u"}:
        raise TokenValueError(f"tokens must be integers, got dtype {raw.dtype}")
    minimum = int(raw.min())
    maximum = int(raw.max())
    if minimum < 0 or maximum > UINT32_MAX:
        raise TokenValueError(
            f"token ids must be in 0..{UINT32_MAX}, got {minimum}..{maximum}"
        )
    values: TokenArray = np.ascontiguousarray(raw, dtype=TOKENS_DTYPE)
    return values


class HDF5ShardWriter:
    """Append-only writer for one chunked, resizable token dataset.

    Tokens are buffered in a NumPy array of exactly ``chunk_size`` elements.
    A full buffer triggers one resize by ``chunk_size`` and one contiguous
    write; the residual buffer is written into the final partial range at
    close. The writer enables SWMR after creating the dataset, so a reader can
    observe committed chunks while the writer is still open.
    """

    def __init__(
        self,
        path: Path,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        compression: Compression | None = None,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if compression not in {None, "gzip"}:
            raise ValueError("compression must be None or 'gzip'")
        if not 0 <= compression_level <= 9:
            raise ValueError("compression_level must be in 0..9")

        self.path = Path(path)
        self._chunk_size = chunk_size
        self._file = h5py.File(self.path, "w", libver="latest")
        dataset_kwargs: dict[str, Any] = {
            "shape": (0,),
            "maxshape": (None,),
            "chunks": (chunk_size,),
            "dtype": TOKENS_DTYPE,
        }
        if compression == "gzip":
            dataset_kwargs["compression"] = "gzip"
            dataset_kwargs["compression_opts"] = compression_level
        self._dataset = self._file.create_dataset(DATASET_NAME, **dataset_kwargs)
        self._dataset.attrs["token_count"] = 0
        self._dataset.attrs["sha256"] = ""
        self._file.swmr_mode = True

        self._buffer: TokenArray = np.zeros(chunk_size, dtype=TOKENS_DTYPE)
        self._buffered = 0
        self._token_count = 0
        self._hasher = hashlib.sha256()
        self._closed = False
        self._stats: ShardStats | None = None

    @property
    def token_count(self) -> int:
        return self._token_count

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed HDF5ShardWriter")

    def _flush(self, count: int) -> None:
        start = int(self._dataset.shape[0])
        self._dataset.resize((start + count,))
        self._dataset[start : start + count] = self._buffer[:count]
        self._dataset.flush()
        self._hasher.update(self._buffer[:count])

    def append(self, tokens: Sequence[int]) -> None:
        self._ensure_open()
        values = _token_array(tokens)
        if values.size == 0:
            return
        self._token_count += int(values.size)
        offset = 0
        while offset < values.size:
            available = self._chunk_size - self._buffered
            take = min(available, int(values.size) - offset)
            self._buffer[self._buffered : self._buffered + take] = values[
                offset : offset + take
            ]
            self._buffered += take
            offset += take
            if self._buffered == self._chunk_size:
                self._flush(self._chunk_size)
                self._buffered = 0

    def close(self) -> ShardStats:
        if self._closed:
            if self._stats is None:
                raise RuntimeError("writer was closed after a failed finalization")
            return self._stats
        try:
            if self._buffered:
                self._flush(self._buffered)
                self._buffered = 0
            digest = self._hasher.hexdigest()
            self._dataset.attrs["token_count"] = self._token_count
            self._dataset.attrs["sha256"] = digest
            self._dataset.flush()
        except BaseException:
            try:
                self._file.close()
            finally:
                self._closed = True
            raise
        self._file.close()
        self._closed = True
        self._stats = ShardStats(token_count=self._token_count, sha256=digest)
        return self._stats

    def __enter__(self) -> HDF5ShardWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:
                pass


def read_jsonl_documents(path: Path) -> Iterator[CorpusDocument]:
    source = Path(path)
    try:
        handle = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise DocumentFormatError(f"{source}: cannot read JSONL: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DocumentFormatError(
                    f"{source}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(payload, dict):
                raise DocumentFormatError(
                    f"{source}:{line_number}: document must be a JSON object"
                )
            document_id = payload.get("id")
            text = payload.get("text")
            if not isinstance(document_id, str):
                raise DocumentFormatError(
                    f"{source}:{line_number}: field 'id' must be a string"
                )
            if not isinstance(text, str):
                raise DocumentFormatError(
                    f"{source}:{line_number}: field 'text' must be a string"
                )
            yield CorpusDocument(document_id=document_id, text=text)


class ShardedTokenizationPipeline:
    """Tokenize input shards into one HDF5 file each and write the index."""

    def __init__(
        self,
        output_dir: Path,
        tokenizer: Tokenizer,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        compression: Compression | None = None,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if compression not in {None, "gzip"}:
            raise ValueError("compression must be None or 'gzip'")
        if not 0 <= compression_level <= 9:
            raise ValueError("compression_level must be in 0..9")
        boundary = tokenizer.boundary_token_id
        vocab = tokenizer.vocab_size
        if not 0 <= boundary < vocab:
            raise ValueError(
                "tokenizer.boundary_token_id must be in [0, tokenizer.vocab_size)"
            )
        self._output_dir = Path(output_dir)
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._compression: Compression | None = compression
        self._compression_level = compression_level

    def run(self, shards: Mapping[str, Iterable[CorpusDocument]]) -> ShardIndex:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        records: list[ShardRecord] = []
        boundary = self._tokenizer.boundary_token_id

        for shard_id, documents in shards.items():
            if not _SHARD_ID_RE.fullmatch(shard_id):
                raise ValueError(f"unsafe shard id: {shard_id!r}")
            partial_path = self._output_dir / f"{shard_id}.h5.partial"
            final_path = self._output_dir / f"{shard_id}.h5"
            writer: HDF5ShardWriter | None = None
            document_count = 0
            try:
                writer = HDF5ShardWriter(
                    partial_path,
                    chunk_size=self._chunk_size,
                    compression=self._compression,
                    compression_level=self._compression_level,
                )
                for document in documents:
                    if not isinstance(document, CorpusDocument):
                        raise TypeError(
                            "input shards must yield CorpusDocument instances"
                        )
                    writer.append(self._tokenizer.encode(document.text))
                    writer.append([boundary])
                    document_count += 1
                stats = writer.close()
                os.replace(partial_path, final_path)
            except BaseException:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass
                partial_path.unlink(missing_ok=True)
                raise
            records.append(
                ShardRecord(
                    shard_id=shard_id,
                    path=final_path.name,
                    token_count=stats.token_count,
                    document_count=document_count,
                    sha256=stats.sha256,
                    byte_size=final_path.stat().st_size,
                )
            )

        index = ShardIndex(
            chunk_size=self._chunk_size,
            vocab_size=self._tokenizer.vocab_size,
            boundary_token_id=boundary,
            total_tokens=sum(record.token_count for record in records),
            shards=tuple(records),
        )
        write_shard_index(self._output_dir / INDEX_FILENAME, index)
        return index


def _validate_tokens_dataset(
    dataset: Any,
    record: ShardRecord,
    chunk_size: int,
) -> int:
    if dataset is None:
        raise ShardFormatError(f"{record.shard_id}: dataset is missing")
    dtype = dataset.dtype
    if dtype.kind != "u" or dtype.itemsize != 4:
        raise ShardFormatError(
            f"{record.shard_id}: tokens must be uint32, got {dtype}"
        )
    chunks = dataset.chunks
    if chunks is None or tuple(chunks) != (chunk_size,):
        raise ShardFormatError(
            f"{record.shard_id}: chunks are {chunks}, expected ({chunk_size},)"
        )
    shape = int(dataset.shape[0])
    try:
        token_count = int(dataset.attrs["token_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ShardFormatError(
            f"{record.shard_id}: token_count attribute is missing or invalid"
        ) from exc
    if token_count < 0 or token_count > shape:
        raise ShardFormatError(
            f"{record.shard_id}: token_count {token_count} is outside dataset "
            f"shape {shape}"
        )
    if token_count != record.token_count:
        raise ShardFormatError(
            f"{record.shard_id}: attribute token_count {token_count} does not "
            f"match index {record.token_count}"
        )
    return token_count


class MmapTokenStore:
    """Read-only view over selected HDF5 shards as one token stream.

    Despite the historical name, h5py does not memory-map the file. It reads
    whole chunks through the HDF5 chunk cache and the OS page cache, then
    copies the requested hyperslab into the output array.
    """

    def __init__(
        self,
        index_path: Path,
        *,
        shard_ids: Sequence[str] | None = None,
        rdcc_nbytes: int = DEFAULT_RDCC_NBYTES,
    ) -> None:
        if rdcc_nbytes < 1:
            raise ValueError("rdcc_nbytes must be >= 1")
        self._index_path = Path(index_path)
        self._index = load_shard_index(self._index_path)
        self._rdcc_nbytes = rdcc_nbytes

        if shard_ids is None:
            selected = list(self._index.shards)
        else:
            requested = list(shard_ids)
            if len(set(requested)) != len(requested):
                raise ValueError("shard_ids contains duplicates")
            known = {record.shard_id for record in self._index.shards}
            unknown = [shard_id for shard_id in requested if shard_id not in known]
            if unknown:
                raise ValueError(f"unknown shard id(s): {', '.join(unknown)}")
            selected_ids = set(requested)
            selected = [
                record
                for record in self._index.shards
                if record.shard_id in selected_ids
            ]

        self._records = tuple(selected)
        self._closed = False
        self._files: list[Any] = []
        self._datasets: list[Any] = []
        self._offsets = np.zeros(len(self._records) + 1, dtype=np.int64)
        cursor = 0
        for position, record in enumerate(self._records):
            self._offsets[position] = cursor
            cursor += record.token_count
        self._offsets[len(self._records)] = cursor
        self._num_tokens = cursor

        try:
            for record in self._records:
                self._open_shard(record)
        except BaseException:
            self.close()
            raise

    def _open_shard(self, record: ShardRecord) -> None:
        path = self._index_path.parent / record.path
        if not path.is_file():
            raise ShardFormatError(f"{record.shard_id}: shard file is missing: {path}")
        file_handle: Any = None
        try:
            file_handle = h5py.File(
                path,
                "r",
                libver="latest",
                swmr=True,
                rdcc_nbytes=self._rdcc_nbytes,
            )
            dataset = file_handle.get(DATASET_NAME)
            if dataset is None:
                raise ShardFormatError(
                    f"{record.shard_id}: dataset {DATASET_NAME!r} is missing"
                )
            _validate_tokens_dataset(dataset, record, self._index.chunk_size)
        except BaseException:
            if file_handle is not None:
                file_handle.close()
            raise
        self._files.append(file_handle)
        self._datasets.append(dataset)

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @property
    def num_shards(self) -> int:
        return len(self._records)

    @property
    def shard_ids(self) -> tuple[str, ...]:
        return tuple(record.shard_id for record in self._records)

    @property
    def chunk_size(self) -> int:
        return self._index.chunk_size

    @property
    def vocab_size(self) -> int:
        return self._index.vocab_size

    @property
    def boundary_token_id(self) -> int:
        return self._index.boundary_token_id

    def get_slice(self, start: int, stop: int) -> TokenArray:
        if self._closed:
            raise ClosedStoreError("cannot read from a closed MmapTokenStore")
        if isinstance(start, bool) or isinstance(stop, bool):
            raise TypeError("start and stop must be integers")
        try:
            begin = operator.index(start)
            end = operator.index(stop)
        except TypeError as exc:
            raise TypeError("start and stop must be integers") from exc
        if begin < 0 or end > self._num_tokens:
            raise IndexError(
                f"slice {begin}:{end} is outside 0:{self._num_tokens}"
            )
        if begin > end:
            raise ValueError("start must not be greater than stop")

        output: TokenArray = np.empty(end - begin, dtype=TOKENS_DTYPE)
        position = begin
        destination = 0
        while position < end:
            shard_index = int(
                np.searchsorted(self._offsets, position, side="right") - 1
            )
            if shard_index < 0 or shard_index >= len(self._records):
                raise RuntimeError(
                    f"token offset {position} did not resolve to a shard"
                )
            shard_start = int(self._offsets[shard_index])
            shard_end = int(self._offsets[shard_index + 1])
            take = min(end - position, shard_end - position)
            if take <= 0:
                raise RuntimeError(
                    f"zero-length shard segment at token offset {position}"
                )
            local_start = position - shard_start
            self._datasets[shard_index].read_direct(
                output[destination : destination + take],
                np.s_[local_start : local_start + take],
            )
            position += take
            destination += take
        return output

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for file_handle in self._files:
            try:
                file_handle.close()
            except Exception:
                pass
        self._files.clear()
        self._datasets.clear()

    def __enter__(self) -> MmapTokenStore:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class VerificationReport:
    shards_checked: int
    tokens_checked: int
    bytes_checked: int


def verify_shards(
    index_path: Path,
    *,
    block_tokens: int = DEFAULT_VERIFY_BLOCK_TOKENS,
) -> VerificationReport:
    if block_tokens < 1:
        raise ValueError("block_tokens must be >= 1")
    source = Path(index_path)
    index = load_shard_index(source)
    shards_checked = 0
    tokens_checked = 0
    bytes_checked = 0

    for record in index.shards:
        path = source.parent / record.path
        if not path.is_file():
            raise ShardFormatError(f"{record.shard_id}: shard file is missing: {path}")
        actual_size = path.stat().st_size
        if actual_size != record.byte_size:
            raise ShardFormatError(
                f"{record.shard_id}: byte size {actual_size} does not match "
                f"index {record.byte_size}"
            )
        try:
            with h5py.File(path, "r", libver="latest", swmr=True) as handle:
                dataset = handle.get(DATASET_NAME)
                if dataset is None:
                    raise ShardFormatError(
                        f"{record.shard_id}: dataset {DATASET_NAME!r} is missing"
                    )
                token_count = _validate_tokens_dataset(
                    dataset, record, index.chunk_size
                )
                hasher = hashlib.sha256()
                for start in range(0, token_count, block_tokens):
                    stop = min(start + block_tokens, token_count)
                    block = np.asarray(dataset[start:stop], dtype=TOKENS_DTYPE)
                    hasher.update(block)
                actual_hash = hasher.hexdigest()
        except OSError as exc:
            raise ShardFormatError(
                f"{record.shard_id}: cannot read shard: {exc}"
            ) from exc
        if not hmac.compare_digest(actual_hash, record.sha256):
            raise ChecksumMismatch(
                f"{record.shard_id}: expected {record.sha256}, got {actual_hash}"
            )
        shards_checked += 1
        tokens_checked += token_count
        bytes_checked += actual_size

    return VerificationReport(
        shards_checked=shards_checked,
        tokens_checked=tokens_checked,
        bytes_checked=bytes_checked,
    )


class SlidingWindowDataloader:
    """Deterministic sliding-window sampler over a token store."""

    def __init__(
        self,
        store: MmapTokenStore,
        *,
        window_size: int,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if store.num_tokens < window_size + 1:
            raise ValueError(
                f"store has {store.num_tokens} tokens, need at least "
                f"{window_size + 1} for one window"
            )
        self._store = store
        self._window_size = window_size
        self._batch_size = batch_size
        self._max_start = store.num_tokens - (window_size + 1)
        self._rng = np.random.Generator(np.random.PCG64(seed))

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def boundary_token_id(self) -> int:
        return self._store.boundary_token_id

    def sample_batch(self) -> TokenBatch:
        starts = self._rng.integers(
            0,
            self._max_start + 1,
            size=self._batch_size,
            dtype=np.int64,
        )
        inputs: TokenArray = np.empty(
            (self._batch_size, self._window_size), dtype=TOKENS_DTYPE
        )
        targets: TokenArray = np.empty(
            (self._batch_size, self._window_size), dtype=TOKENS_DTYPE
        )
        for row, raw_start in enumerate(starts):
            start = int(raw_start)
            window = self._store.get_slice(
                start, start + self._window_size + 1
            )
            inputs[row] = window[:-1]
            targets[row] = window[1:]
        return inputs, targets

    def iter_batches(self, num_batches: int) -> Iterator[TokenBatch]:
        if isinstance(num_batches, bool) or not isinstance(num_batches, int):
            raise ValueError("num_batches must be a non-negative integer")
        if num_batches < 0:
            raise ValueError("num_batches must be a non-negative integer")
        for _ in range(num_batches):
            yield self.sample_batch()


_DEMO_SHARDS: dict[str, tuple[tuple[str, str], ...]] = {
    "shard-0000": (
        ("demo-0000", "HDF5 stores a token stream as chunked integer data."),
        ("demo-0001", "Olá, mundo! UTF-8 bytes survive the round trip."),
        ("demo-0002", "Windows can straddle documents at the boundary token."),
    ),
    "shard-0001": (
        ("demo-0003", "The writer buffers tokens before extending the dataset."),
        ("demo-0004", "Reader workers open their own shard files with SWMR."),
        ("demo-0005", "A sliding window reads one extra id for its target."),
    ),
}

_BENCHMARK_TEXT = (
    "The token stream is a contiguous sequence of integer ids used for "
    "language-model training and chunked storage measurements. "
)


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    chunk_size: int
    token_count: int
    byte_size: int
    build_seconds: float
    read_seconds: float
    samples: int


def _demo_documents() -> dict[str, list[CorpusDocument]]:
    return {
        shard_id: [
            CorpusDocument(document_id=document_id, text=text)
            for document_id, text in documents
        ]
        for shard_id, documents in _DEMO_SHARDS.items()
    }


def _benchmark_documents(
    shards: int, documents_per_shard: int
) -> dict[str, list[CorpusDocument]]:
    return {
        f"bench-{shard:04d}": [
            CorpusDocument(
                document_id=f"bench-{shard:04d}-{number:05d}",
                text=f"{_BENCHMARK_TEXT}{number}",
            )
            for number in range(documents_per_shard)
        ]
        for shard in range(shards)
    }


def _new_pipeline(
    workdir: Path,
    *,
    chunk_size: int,
    compression: Compression | None,
    compression_level: int,
) -> ShardedTokenizationPipeline:
    return ShardedTokenizationPipeline(
        workdir,
        ByteTokenizer(),
        chunk_size=chunk_size,
        compression=compression,
        compression_level=compression_level,
    )


def run_demo(
    workdir: Path,
    *,
    chunk_size: int = 64,
    window_size: int = 16,
    batch_size: int = 4,
    batches: int = 10,
    seed: int = 0,
    compression: Compression | None = None,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
) -> int:
    output_dir = Path(workdir)
    pipeline = _new_pipeline(
        output_dir,
        chunk_size=chunk_size,
        compression=compression,
        compression_level=compression_level,
    )
    build_started = time.perf_counter()
    index = pipeline.run(_demo_documents())
    build_seconds = time.perf_counter() - build_started

    print(
        f"corpus: shards={len(index.shards)} documents="
        f"{sum(shard.document_count for shard in index.shards)} "
        f"tokens={index.total_tokens} chunk_size={index.chunk_size}"
    )
    for shard in index.shards:
        print(
            f"  {shard.shard_id}: tokens={shard.token_count} "
            f"docs={shard.document_count} bytes={shard.byte_size} "
            f"sha256={shard.sha256[:16]}..."
        )
    print(f"build: {build_seconds:.4f}s")

    index_path = output_dir / INDEX_FILENAME
    read_started = time.perf_counter()
    with MmapTokenStore(index_path) as store:
        loader = SlidingWindowDataloader(
            store,
            window_size=window_size,
            batch_size=batch_size,
            seed=seed,
        )
        for batch_number, (input_ids, target_ids) in enumerate(
            loader.iter_batches(batches)
        ):
            if not np.array_equal(input_ids[:, 1:], target_ids[:, :-1]):
                raise RuntimeError("window shift invariant violated")
            digest = hashlib.sha256(
                input_ids.tobytes() + target_ids.tobytes()
            ).hexdigest()[:16]
            print(
                f"batch={batch_number:02d} input={input_ids.shape} "
                f"target={target_ids.shape} checksum={digest}"
            )
        read_seconds = time.perf_counter() - read_started
        print(
            f"read: {read_seconds:.4f}s samples={batches} "
            f"samples_per_second={batches / read_seconds:.1f} "
            f"tokens_per_second={batches * batch_size * window_size / read_seconds:.1f}"
        )

    report = verify_shards(index_path)
    print(
        f"validated: shards={report.shards_checked} "
        f"tokens={report.tokens_checked} bytes={report.bytes_checked}"
    )
    print(f"index={index_path}")
    return 0


def run_build(
    input_dir: Path,
    workdir: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    compression: Compression | None = None,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
) -> int:
    source_dir = Path(input_dir)
    paths = sorted(
        path for path in source_dir.glob("*.jsonl") if path.is_file()
    )
    if not paths:
        raise DocumentFormatError(f"{source_dir}: no *.jsonl shards found")
    shards = {path.stem: read_jsonl_documents(path) for path in paths}
    pipeline = _new_pipeline(
        workdir,
        chunk_size=chunk_size,
        compression=compression,
        compression_level=compression_level,
    )
    index = pipeline.run(shards)
    documents = sum(shard.document_count for shard in index.shards)
    print(
        f"built: shards={len(index.shards)} documents={documents} "
        f"tokens={index.total_tokens} chunk_size={index.chunk_size}"
    )
    for shard in index.shards:
        print(
            f"  {shard.shard_id}: tokens={shard.token_count} "
            f"docs={shard.document_count} bytes={shard.byte_size}"
        )
    print(f"index={Path(workdir) / INDEX_FILENAME}")
    return 0


def run_validate(index_path: Path) -> int:
    report = verify_shards(index_path)
    print(
        f"validated: shards={report.shards_checked} "
        f"tokens={report.tokens_checked} bytes={report.bytes_checked}"
    )
    return 0


def run_benchmark(
    workdir: Path,
    *,
    window_size: int = 128,
    batch_size: int = 16,
    samples: int = 500,
    shards: int = 4,
    documents_per_shard: int = 1024,
    seed: int = 0,
    compression: Compression | None = None,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
) -> list[BenchmarkRow]:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if shards < 1:
        raise ValueError("shards must be >= 1")
    if documents_per_shard < 1:
        raise ValueError("documents_per_shard must be >= 1")

    output_root = Path(workdir)
    documents = _benchmark_documents(shards, documents_per_shard)
    chunk_sizes = sorted(
        {max(1, window_size // 2), window_size, window_size * 2}
    )
    rows: list[BenchmarkRow] = []
    for chunk_size in chunk_sizes:
        run_dir = output_root / f"chunk-{chunk_size}"
        pipeline = _new_pipeline(
            run_dir,
            chunk_size=chunk_size,
            compression=compression,
            compression_level=compression_level,
        )
        build_started = time.perf_counter()
        index = pipeline.run(documents)
        build_seconds = time.perf_counter() - build_started

        with MmapTokenStore(run_dir / INDEX_FILENAME) as store:
            loader = SlidingWindowDataloader(
                store,
                window_size=window_size,
                batch_size=batch_size,
                seed=seed,
            )
            loader.sample_batch()
            read_started = time.perf_counter()
            for _ in range(samples):
                input_ids, target_ids = loader.sample_batch()
                _ = int(input_ids[0, 0]) + int(target_ids[0, 0])
            read_seconds = time.perf_counter() - read_started
        rows.append(
            BenchmarkRow(
                chunk_size=chunk_size,
                token_count=index.total_tokens,
                byte_size=sum(shard.byte_size for shard in index.shards),
                build_seconds=build_seconds,
                read_seconds=read_seconds,
                samples=samples,
            )
        )

    print(
        f"{'chunk':>6} {'tokens':>10} {'bytes':>10} {'build_s':>9} "
        f"{'read_s':>8} {'samples/s':>11} {'Mtoken/s':>10}"
    )
    for row in rows:
        samples_per_second = row.samples / row.read_seconds
        tokens_per_second = row.samples * window_size / row.read_seconds
        print(
            f"{row.chunk_size:>6} {row.token_count:>10} {row.byte_size:>10} "
            f"{row.build_seconds:>9.4f} {row.read_seconds:>8.4f} "
            f"{samples_per_second:>11.1f} {tokens_per_second / 1e6:>10.2f}"
        )
    return rows


def _parse_compression(value: str) -> Compression | None:
    if value == "none":
        return None
    if value == "gzip":
        return "gzip"
    raise ValueError("compression must be 'none' or 'gzip'")


def _add_compression_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compression",
        choices=("none", "gzip"),
        default="none",
        help="HDF5 filter for the token dataset (default: none)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=DEFAULT_COMPRESSION_LEVEL,
        help="gzip level 0..9 (default: 4)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hdf5-tokenized-corpus",
        description=(
            "Tokenize JSONL into chunked HDF5 shards, validate the shard "
            "index, or sample sliding-window batches."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="build and stream the demo corpus")
    demo.add_argument("--workdir", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--chunk-size", type=int, default=64)
    demo.add_argument("--window-size", type=int, default=16)
    demo.add_argument("--batch-size", type=int, default=4)
    demo.add_argument("--batches", type=int, default=10)
    demo.add_argument("--seed", type=int, default=0)
    _add_compression_options(demo)

    build = subparsers.add_parser(
        "build", help="tokenize a directory of JSONL shards into HDF5"
    )
    build.add_argument("--input-dir", type=Path, required=True)
    build.add_argument("--workdir", type=Path, default=Path("outputs/corpus"))
    build.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    _add_compression_options(build)

    validate = subparsers.add_parser(
        "validate", help="recompute every shard SHA-256 and check the index"
    )
    validate.add_argument("--index", type=Path, required=True)

    benchmark = subparsers.add_parser(
        "benchmark", help="compare half, equal, and double chunk sizes"
    )
    benchmark.add_argument("--workdir", type=Path, default=Path("outputs/benchmark"))
    benchmark.add_argument("--window", type=int, default=128)
    benchmark.add_argument("--batch-size", type=int, default=16)
    benchmark.add_argument("--samples", type=int, default=500)
    benchmark.add_argument("--shards", type=int, default=4)
    benchmark.add_argument("--docs-per-shard", type=int, default=1024)
    benchmark.add_argument("--seed", type=int, default=0)
    _add_compression_options(benchmark)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        return run_demo(Path("outputs/demo"))

    parser = _build_parser()
    args = parser.parse_args(arguments)
    try:
        if args.command == "demo":
            return run_demo(
                Path(args.workdir),
                chunk_size=args.chunk_size,
                window_size=args.window_size,
                batch_size=args.batch_size,
                batches=args.batches,
                seed=args.seed,
                compression=_parse_compression(args.compression),
                compression_level=args.compression_level,
            )
        if args.command == "build":
            return run_build(
                Path(args.input_dir),
                Path(args.workdir),
                chunk_size=args.chunk_size,
                compression=_parse_compression(args.compression),
                compression_level=args.compression_level,
            )
        if args.command == "validate":
            return run_validate(Path(args.index))
        if args.command == "benchmark":
            run_benchmark(
                Path(args.workdir),
                window_size=args.window,
                batch_size=args.batch_size,
                samples=args.samples,
                shards=args.shards,
                documents_per_shard=args.docs_per_shard,
                seed=args.seed,
                compression=_parse_compression(args.compression),
                compression_level=args.compression_level,
            )
            return 0
        parser.error("unknown command")
    except (CorpusError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
