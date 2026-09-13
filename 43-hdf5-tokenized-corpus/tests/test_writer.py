from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pytest

from main import (
    HDF5ShardWriter,
    TOKENS_DTYPE,
    TokenValueError,
)


def _canonical_hash(tokens: list[int]) -> str:
    values = np.asarray(tokens, dtype=TOKENS_DTYPE)
    return hashlib.sha256(values.tobytes()).hexdigest()


def test_writer_buffers_in_chunk_strides_and_records_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "shard.h5"
    tokens = list(range(20))
    writer = HDF5ShardWriter(path, chunk_size=8)
    writer.append(tokens[:5])
    writer.append(tokens[5:17])
    writer.append(tokens[17:])
    stats = writer.close()

    assert stats.token_count == 20
    assert stats.sha256 == _canonical_hash(tokens)
    with h5py.File(path, "r") as handle:
        dataset = handle["tokens"]
        assert dataset.shape == (20,)
        assert dataset.chunks == (8,)
        assert dataset.maxshape == (None,)
        assert dataset.dtype == TOKENS_DTYPE
        assert int(dataset.attrs["token_count"]) == 20
        assert dataset.attrs["sha256"] == stats.sha256
        assert dataset[:].tolist() == tokens


@pytest.mark.parametrize("compression", [None, "gzip"])
def test_writer_handles_an_empty_stream(
    tmp_path: Path, compression: Literal["gzip"] | None
) -> None:
    path = tmp_path / "empty.h5"
    writer = HDF5ShardWriter(path, chunk_size=8, compression=compression)
    stats = writer.close()

    assert stats.token_count == 0
    assert stats.sha256 == hashlib.sha256(b"").hexdigest()
    with h5py.File(path, "r") as handle:
        dataset = handle["tokens"]
        assert dataset.shape == (0,)
        assert int(dataset.attrs["token_count"]) == 0
        assert dataset.attrs["sha256"] == stats.sha256
        assert dataset.compression == compression


def test_writer_rejects_invalid_tokens_and_late_appends(tmp_path: Path) -> None:
    writer = HDF5ShardWriter(tmp_path / "bad.h5", chunk_size=4)
    with pytest.raises(TokenValueError, match="0..4294967295"):
        writer.append([-1])
    with pytest.raises(TokenValueError, match="0..4294967295"):
        writer.append([2**32])
    with pytest.raises(TokenValueError, match="must be integers"):
        writer.append([1.5])  # type: ignore[list-item]
    writer.append([1, 2, 3, 4])
    writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer.append([5])
    assert writer.close().token_count == 4


def test_writer_rejects_invalid_options(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        HDF5ShardWriter(tmp_path / "a.h5", chunk_size=0)
    with pytest.raises(ValueError, match="compression_level"):
        HDF5ShardWriter(tmp_path / "b.h5", compression_level=10)


def test_writer_is_live_readable_in_swmr_mode(tmp_path: Path) -> None:
    path = tmp_path / "live.h5"
    writer = HDF5ShardWriter(path, chunk_size=4)
    writer.append([1, 2, 3, 4])

    reader = h5py.File(path, "r", libver="latest", swmr=True)
    try:
        dataset = reader["tokens"]
        dataset.refresh()
        assert dataset.shape == (4,)
        assert dataset[:].tolist() == [1, 2, 3, 4]

        writer.append([5, 6, 7, 8])
        dataset.refresh()
        assert dataset.shape == (8,)
        assert dataset[:].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]

        stats = writer.close()
        dataset.refresh()
        assert int(dataset.attrs["token_count"]) == 8
        assert dataset.attrs["sha256"] == stats.sha256
    finally:
        reader.close()


def test_writer_gzip_round_trips_a_partial_final_chunk(tmp_path: Path) -> None:
    path = tmp_path / "gzip.h5"
    tokens = list(range(19))
    writer = HDF5ShardWriter(path, chunk_size=8, compression="gzip")
    writer.append(tokens)
    stats = writer.close()

    with h5py.File(path, "r") as handle:
        dataset = handle["tokens"]
        assert dataset.compression == "gzip"
        assert dataset[:].tolist() == tokens
        assert int(dataset.attrs["token_count"]) == 19
        assert dataset.attrs["sha256"] == stats.sha256
