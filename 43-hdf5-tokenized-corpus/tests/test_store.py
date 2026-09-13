from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from main import (
    ClosedStoreError,
    MmapTokenStore,
    ShardFormatError,
    TOKENS_DTYPE,
)
from support import build_corpus, shard_path


def test_store_reads_across_shard_boundaries(tmp_path: Path) -> None:
    built = build_corpus(tmp_path)
    boundary = built.index.shards[0].token_count
    total = built.index.total_tokens
    with MmapTokenStore(built.index_path) as store:
        assert store.num_tokens == total
        assert store.num_shards == 2
        for start, stop in (
            (0, 5),
            (boundary - 3, boundary + 3),
            (0, total),
            (total, total),
            (boundary, boundary),
        ):
            observed = store.get_slice(start, stop)
            assert observed.dtype == TOKENS_DTYPE
            np.testing.assert_array_equal(observed, built.reference[start:stop])


def test_store_truncates_reads_at_the_recorded_token_count(
    tmp_path: Path,
) -> None:
    built = build_corpus(tmp_path)
    path = shard_path(built, "shard-0000")
    with h5py.File(path, "r+") as handle:
        dataset = handle["tokens"]
        original_shape = int(dataset.shape[0])
        dataset.resize((original_shape + 4,))
        dataset[original_shape:] = 0
        assert int(dataset.attrs["token_count"]) == original_shape

    with MmapTokenStore(built.index_path) as store:
        assert store.num_tokens == int(built.reference.size)
        np.testing.assert_array_equal(
            store.get_slice(0, store.num_tokens), built.reference
        )


def test_store_subset_preserves_index_order(tmp_path: Path) -> None:
    built = build_corpus(tmp_path)
    with MmapTokenStore(
        built.index_path, shard_ids=["shard-0001", "shard-0000"]
    ) as store:
        assert store.shard_ids == ("shard-0000", "shard-0001")
        expected = np.concatenate(
            (
                built.shard_streams["shard-0000"],
                built.shard_streams["shard-0001"],
            )
        )
        np.testing.assert_array_equal(store.get_slice(0, store.num_tokens), expected)

    with MmapTokenStore(built.index_path, shard_ids=["shard-0001"]) as store:
        np.testing.assert_array_equal(
            store.get_slice(0, store.num_tokens),
            built.shard_streams["shard-0001"],
        )

    with MmapTokenStore(built.index_path, shard_ids=[]) as store:
        assert store.num_tokens == 0
        assert store.get_slice(0, 0).size == 0

    with pytest.raises(ValueError, match="unknown shard"):
        MmapTokenStore(built.index_path, shard_ids=["nope"])
    with pytest.raises(ValueError, match="duplicates"):
        MmapTokenStore(
            built.index_path, shard_ids=["shard-0000", "shard-0000"]
        )


def test_store_rejects_invalid_ranges_and_closed_use(tmp_path: Path) -> None:
    built = build_corpus(tmp_path)
    store = MmapTokenStore(built.index_path)
    total = store.num_tokens
    with pytest.raises(IndexError):
        store.get_slice(-1, 1)
    with pytest.raises(IndexError):
        store.get_slice(0, total + 1)
    with pytest.raises(ValueError, match="greater than stop"):
        store.get_slice(2, 1)
    with pytest.raises(TypeError, match="integers"):
        store.get_slice("0", 1)  # type: ignore[arg-type]
    store.close()
    store.close()
    with pytest.raises(ClosedStoreError, match="closed"):
        store.get_slice(0, 1)


def test_store_rejects_a_token_count_that_disagrees_with_the_index(
    tmp_path: Path,
) -> None:
    built = build_corpus(tmp_path)
    path = shard_path(built, "shard-0000")
    with h5py.File(path, "r+") as handle:
        dataset = handle["tokens"]
        dataset.attrs["token_count"] = int(dataset.attrs["token_count"]) - 1
    with pytest.raises(ShardFormatError, match="does not match index"):
        MmapTokenStore(built.index_path)


def test_store_rejects_a_missing_shard(tmp_path: Path) -> None:
    built = build_corpus(tmp_path)
    shard_path(built, "shard-0000").unlink()
    with pytest.raises(ShardFormatError, match="missing"):
        MmapTokenStore(built.index_path)
