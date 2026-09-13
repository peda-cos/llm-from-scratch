from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from main import MmapTokenStore, SlidingWindowDataloader, TOKENS_DTYPE
from support import build_corpus


def test_batches_shift_the_window_and_draw_from_the_stream(
    tmp_path: Path,
) -> None:
    built = build_corpus(tmp_path)
    reference = built.reference
    window_size = 8
    batch_size = 4
    with MmapTokenStore(built.index_path) as store:
        loader = SlidingWindowDataloader(
            store,
            window_size=window_size,
            batch_size=batch_size,
            seed=3,
        )
        inputs, targets = loader.sample_batch()

    assert inputs.shape == (batch_size, window_size)
    assert targets.shape == (batch_size, window_size)
    assert inputs.dtype == TOKENS_DTYPE
    assert targets.dtype == TOKENS_DTYPE
    np.testing.assert_array_equal(inputs[:, 1:], targets[:, :-1])
    assert int(inputs.min()) >= 0
    assert int(inputs.max()) <= 256

    for row in range(batch_size):
        assert any(
            np.array_equal(inputs[row], reference[start : start + window_size])
            and np.array_equal(
                targets[row], reference[start + 1 : start + window_size + 1]
            )
            for start in range(0, reference.size - window_size)
        ), f"row {row} does not match any stream window"


def test_same_seed_is_reproducible_and_a_different_seed_diverges(
    tmp_path: Path,
) -> None:
    built = build_corpus(tmp_path)
    with MmapTokenStore(built.index_path) as store:
        first = SlidingWindowDataloader(
            store, window_size=8, batch_size=4, seed=11
        )
        replay = SlidingWindowDataloader(
            store, window_size=8, batch_size=4, seed=11
        )
        other = SlidingWindowDataloader(
            store, window_size=8, batch_size=4, seed=12
        )

        for _ in range(3):
            first_inputs, first_targets = first.sample_batch()
            replay_inputs, replay_targets = replay.sample_batch()
            np.testing.assert_array_equal(first_inputs, replay_inputs)
            np.testing.assert_array_equal(first_targets, replay_targets)

        differs = False
        for _ in range(3):
            first_inputs, _ = first.sample_batch()
            other_inputs, _ = other.sample_batch()
            differs = differs or not np.array_equal(first_inputs, other_inputs)
        assert differs


def test_full_stream_window_contains_the_boundary_token(tmp_path: Path) -> None:
    built = build_corpus(tmp_path)
    with MmapTokenStore(built.index_path) as store:
        loader = SlidingWindowDataloader(
            store,
            window_size=store.num_tokens - 1,
            batch_size=2,
            seed=0,
        )
        assert loader.boundary_token_id == 256
        inputs, targets = loader.sample_batch()
    assert inputs.shape[1] == built.index.total_tokens - 1
    assert np.any(inputs == 256) or np.any(targets == 256)


def test_iter_batches_yields_the_requested_count(tmp_path: Path) -> None:
    built = build_corpus(tmp_path)
    with MmapTokenStore(built.index_path) as store:
        loader = SlidingWindowDataloader(
            store, window_size=8, batch_size=2, seed=5
        )
        assert list(loader.iter_batches(0)) == []
        batches = list(loader.iter_batches(3))
    assert len(batches) == 3
    for inputs, targets in batches:
        assert inputs.shape == (2, 8)
        assert targets.shape == (2, 8)


def test_dataloader_rejects_invalid_parameters(tmp_path: Path) -> None:
    built = build_corpus(tmp_path)
    with MmapTokenStore(built.index_path) as store:
        with pytest.raises(ValueError, match="window_size"):
            SlidingWindowDataloader(store, window_size=0, batch_size=1)
        with pytest.raises(ValueError, match="batch_size"):
            SlidingWindowDataloader(store, window_size=8, batch_size=0)
        with pytest.raises(ValueError, match="seed"):
            SlidingWindowDataloader(store, window_size=8, batch_size=1, seed=-1)
        with pytest.raises(ValueError, match="need at least"):
            SlidingWindowDataloader(
                store, window_size=store.num_tokens, batch_size=1
            )
        loader = SlidingWindowDataloader(
            store, window_size=8, batch_size=1, seed=0
        )
        with pytest.raises(ValueError, match="num_batches"):
            list(loader.iter_batches(-1))
        with pytest.raises(ValueError, match="num_batches"):
            list(loader.iter_batches(1.5))  # type: ignore[arg-type]
