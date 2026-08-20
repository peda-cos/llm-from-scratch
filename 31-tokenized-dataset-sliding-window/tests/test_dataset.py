from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import torch

from main import (
    ByteTokenizer,
    SlidingWindowDataset,
    _encode_corpus_to_ids,
    epoch_seed,
    make_dataloader,
)

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "corpus.txt"


def make_dataset(n: int = 64, t: int = 4, s: int = 1) -> SlidingWindowDataset:
    return SlidingWindowDataset(torch.arange(n, dtype=torch.long), t, s)


def collect_starts(
    dataset: SlidingWindowDataset,
    *,
    base_seed: int = 7,
    epoch_index: int = 0,
    batch_size: int = 8,
    shuffle: bool = True,
) -> list[int]:
    loader = make_dataloader(
        dataset,
        batch_size,
        base_seed=base_seed,
        epoch_index=epoch_index,
        shuffle=shuffle,
    )
    starts: list[int] = []
    for input_ids, _ in loader:
        starts.extend(int(token) for token in input_ids[:, 0].tolist())
    return starts


class CountingTokenizer:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids
        self.calls: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.calls.append(text)
        return list(self.ids)


@pytest.mark.parametrize(
    "text",
    ["plain ASCII", "Olá, mundo!", "你好，世界", "emoji 🚀🔥", ""],
)
def test_byte_tokenizer_round_trip(text: str) -> None:
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(text)
    assert all(0 <= token_id < 256 for token_id in ids)
    assert tokenizer.decode(ids) == text


def test_encode_corpus_calls_tokenizer_once() -> None:
    tokenizer = CountingTokenizer([3, 1, 4, 1, 5])
    ids = _encode_corpus_to_ids(tokenizer, "the corpus text")
    assert tokenizer.calls == ["the corpus text"]
    assert ids.tolist() == [3, 1, 4, 1, 5]


def test_encode_corpus_returns_long_cpu_tensor() -> None:
    ids = _encode_corpus_to_ids(ByteTokenizer(), "abc")
    assert ids.ndim == 1
    assert ids.dtype == torch.long
    assert ids.device.type == "cpu"
    assert ids.tolist() == [97, 98, 99]
    assert _encode_corpus_to_ids(ByteTokenizer(), "").shape == (0,)


@pytest.mark.parametrize(
    ("num_ids", "context_length", "stride", "expected"),
    [
        (0, 4, 1, 0),
        (4, 4, 1, 0),
        (5, 4, 1, 1),
        (10, 4, 2, 3),
        (100, 10, 5, 18),
        (1000, 16, 16, 62),
        (1000, 16, 8, 123),
        (1000, 16, 4, 246),
        (1000, 16, 1, 984),
    ],
)
def test_num_examples_hand_computed(
    num_ids: int, context_length: int, stride: int, expected: int
) -> None:
    assert (
        SlidingWindowDataset.num_examples(num_ids, context_length, stride) == expected
    )


def test_num_examples_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="num_ids"):
        SlidingWindowDataset.num_examples(-1, 4, 1)
    with pytest.raises(ValueError, match="context_length"):
        SlidingWindowDataset.num_examples(5, 0, 1)
    with pytest.raises(ValueError, match="stride"):
        SlidingWindowDataset.num_examples(5, 4, 0)
    with pytest.raises(TypeError, match="context_length"):
        SlidingWindowDataset.num_examples(5, cast(Any, True), 1)
    with pytest.raises(TypeError, match="stride"):
        SlidingWindowDataset.num_examples(5, 4, cast(Any, "1"))


@pytest.mark.parametrize("stride", [1, 2, 5, 6, 10])
def test_len_matches_window_count(stride: int) -> None:
    ids = torch.arange(37, dtype=torch.long)
    dataset = SlidingWindowDataset(ids, 5, stride)
    assert len(dataset) == SlidingWindowDataset.num_examples(37, 5, stride)
    assert SlidingWindowDataset(ids, 5).stride == 5


def test_window_contents_and_shift_by_one() -> None:
    ids = torch.arange(20, dtype=torch.long)
    dataset = SlidingWindowDataset(ids, 4, 3)
    assert len(dataset) == 6

    for index in range(len(dataset)):
        start = index * 3
        input_ids, target_ids = dataset[index]
        assert input_ids.tolist() == ids[start : start + 4].tolist()
        assert target_ids.tolist() == ids[start + 1 : start + 5].tolist()
        assert torch.equal(input_ids[1:], target_ids[:-1])


def test_negative_index_and_out_of_range() -> None:
    dataset = make_dataset(20, 4, 3)
    assert torch.equal(dataset[-1][0], dataset[len(dataset) - 1][0])
    with pytest.raises(IndexError, match="out of range"):
        _ = dataset[len(dataset)]
    with pytest.raises(IndexError, match="out of range"):
        _ = dataset[-len(dataset) - 1]


def test_tail_window_is_dropped() -> None:
    ids = torch.arange(23, dtype=torch.long)
    dataset = SlidingWindowDataset(ids, 4, 5)
    assert len(dataset) == 4

    input_ids, target_ids = dataset[3]
    assert input_ids.tolist() == list(range(15, 19))
    assert target_ids.tolist() == list(range(16, 20))

    seen: set[int] = set()
    for index in range(len(dataset)):
        sample_input, sample_target = dataset[index]
        seen.update(sample_input.tolist())
        seen.update(sample_target.tolist())
    assert seen.isdisjoint({20, 21, 22})


def test_stride_increases_effective_dataset_size() -> None:
    counts = {
        stride: SlidingWindowDataset.num_examples(1000, 16, stride)
        for stride in (16, 8, 4, 1)
    }
    assert counts == {16: 62, 8: 123, 4: 246, 1: 984}
    assert counts[1] > counts[4] > counts[8] > counts[16]


@pytest.mark.parametrize("num_ids", [0, 3])
def test_empty_and_short_corpus_yield_no_examples(num_ids: int) -> None:
    dataset = make_dataset(num_ids, 4, 1)
    assert len(dataset) == 0
    loader = make_dataloader(dataset, 2, base_seed=1)
    assert list(loader) == []


def test_dataset_rejects_invalid_ids_and_config() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        SlidingWindowDataset(torch.zeros((2, 3), dtype=torch.long), 4)
    with pytest.raises(TypeError, match="integer dtype"):
        SlidingWindowDataset(torch.tensor([0.0, 1.0]), 4)
    with pytest.raises(TypeError, match="integer dtype"):
        SlidingWindowDataset(torch.tensor([True, False]), 4)
    with pytest.raises(ValueError, match="non-negative"):
        SlidingWindowDataset(torch.tensor([0, -1]), 4)
    with pytest.raises(ValueError, match="context_length"):
        SlidingWindowDataset(torch.arange(10), 0)
    with pytest.raises(ValueError, match="stride"):
        SlidingWindowDataset(torch.arange(10), 4, 0)

    from_list = SlidingWindowDataset([1, 2, 3, 4, 5], 2, 1)
    assert from_list.token_ids.dtype == torch.long
    assert from_list[0][0].tolist() == [1, 2]


def test_same_seed_and_epoch_reproduce_order() -> None:
    dataset = make_dataset(64, 4, 1)
    first = collect_starts(dataset, base_seed=99, epoch_index=2)
    second = collect_starts(dataset, base_seed=99, epoch_index=2)
    assert first == second
    assert sorted(first) == list(range(len(dataset)))


def test_different_epoch_changes_order_without_loss() -> None:
    dataset = make_dataset(64, 4, 1)
    epoch_zero = collect_starts(dataset, base_seed=41, epoch_index=0)
    epoch_one = collect_starts(dataset, base_seed=41, epoch_index=1)
    assert epoch_zero != epoch_one
    assert sorted(epoch_zero) == list(range(len(dataset)))
    assert sorted(epoch_one) == list(range(len(dataset)))


def test_base_seed_plus_epoch_equivalence() -> None:
    dataset = make_dataset(64, 4, 1)
    assert epoch_seed(100, 3) == 103
    assert collect_starts(dataset, base_seed=100, epoch_index=3) == collect_starts(
        dataset, base_seed=103, epoch_index=0
    )


def test_batches_have_shifted_shape_contract() -> None:
    dataset = SlidingWindowDataset(torch.arange(40, dtype=torch.long), 4, 2)
    loader = make_dataloader(dataset, 3, base_seed=5)
    for input_ids, target_ids in loader:
        assert input_ids.shape == target_ids.shape
        assert input_ids.shape[1] == 4
        assert input_ids.dtype == torch.long
        assert target_ids.dtype == torch.long
        assert torch.equal(input_ids[:, 1:], target_ids[:, :-1])


def test_drop_last_controls_partial_batch() -> None:
    dataset = SlidingWindowDataset(torch.arange(14, dtype=torch.long), 4, 1)
    assert len(dataset) == 10

    full = make_dataloader(dataset, 4, base_seed=1, drop_last=False)
    assert [input_ids.shape[0] for input_ids, _ in full] == [4, 4, 2]

    trimmed = make_dataloader(dataset, 4, base_seed=1, drop_last=True)
    assert [input_ids.shape[0] for input_ids, _ in trimmed] == [4, 4]


def test_sequential_loader_has_no_shuffle() -> None:
    dataset = make_dataset(30, 4, 1)
    starts = collect_starts(dataset, batch_size=4, shuffle=False)
    assert starts == list(range(len(dataset)))


def test_dataloader_rejects_invalid_arguments() -> None:
    dataset = make_dataset(20, 4, 1)
    with pytest.raises(ValueError, match="batch_size"):
        make_dataloader(dataset, 0, base_seed=1)
    with pytest.raises(ValueError, match="base_seed"):
        make_dataloader(dataset, 2, base_seed=-1)
    with pytest.raises(ValueError, match="epoch_index"):
        make_dataloader(dataset, 2, base_seed=1, epoch_index=-1)
    with pytest.raises(TypeError, match="batch_size"):
        make_dataloader(dataset, cast(Any, True), base_seed=1)


def test_committed_corpus_pipeline() -> None:
    raw = CORPUS_PATH.read_bytes()
    assert raw.isascii()
    text = CORPUS_PATH.read_text(encoding="utf-8")
    tokenizer = ByteTokenizer()
    ids = _encode_corpus_to_ids(tokenizer, text)
    assert ids.numel() == len(raw)
    assert int(ids.min()) >= 0
    assert int(ids.max()) < tokenizer.vocab_size

    dataset = SlidingWindowDataset(ids, 16, 16)
    assert len(dataset) == SlidingWindowDataset.num_examples(ids.numel(), 16, 16)
    assert len(dataset) > 0

    loader = make_dataloader(dataset, 8, base_seed=1234, drop_last=True)
    input_ids, target_ids = next(iter(loader))
    assert input_ids.shape == target_ids.shape == (8, 16)
    assert torch.equal(input_ids[:, 1:], target_ids[:, :-1])
