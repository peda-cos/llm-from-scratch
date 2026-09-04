from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from main import (
    BYTE_VOCAB_SIZE,
    PAD_ID,
    VOCAB_SIZE,
    ByteTokenizer,
    ClassificationDataset,
    Classifier,
    ConfusionMatrix,
    DemoConfig,
    LMBody,
    MeanPool,
    ModelConfig,
    count_parameters,
    evaluate,
    format_metrics,
    freeze_body,
    main,
    make_sms_fixture,
    pretrain_quick,
    run_demo,
    stratified_split,
    train_classifier,
    unfreeze_body,
)


def _tiny_config(
    context_length: int = 16,
    d_model: int = 16,
    num_heads: int = 2,
    num_layers: int = 1,
    dropout: float = 0.0,
) -> ModelConfig:
    return ModelConfig(
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )


def _tiny_dataset() -> ClassificationDataset:
    tokenizer = ByteTokenizer()
    texts = (
        "win cash now",
        "free prize",
        "claim gift",
        "win 500",
        "lunch at noon",
        "call sam 6pm",
        "see jo later",
        "meet ana 8",
    )
    labels = (1, 1, 1, 1, 0, 0, 0, 0)
    return ClassificationDataset(texts, labels, tokenizer, 16)


def _loader(
    dataset: ClassificationDataset, batch_size: int = 4
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class _FixedLogits(nn.Module):
    logits: Tensor

    def __init__(self, logits: Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(
        self, token_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        return self.logits


def _metric_batches(
    labels: Sequence[int],
) -> list[tuple[Tensor, Tensor, Tensor]]:
    label_tensor = torch.tensor(list(labels), dtype=torch.long)
    ids = torch.zeros((len(labels), 1), dtype=torch.long)
    mask = torch.ones((len(labels), 1), dtype=torch.bool)
    return [(ids, mask, label_tensor)]


def _tiny_args() -> list[str]:
    return [
        "--messages",
        "64",
        "--d-model",
        "16",
        "--num-heads",
        "2",
        "--num-layers",
        "1",
        "--pretrain-epochs",
        "1",
        "--head-epochs",
        "2",
        "--full-epochs",
        "3",
        "--batch-size",
        "16",
        "--dropout",
        "0",
        "--head-lr",
        "0.001",
        "--full-lr",
        "0.001",
        "--seed",
        "0",
    ]


def _tiny_demo_config() -> DemoConfig:
    return DemoConfig(
        messages=64,
        d_model=16,
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        pretrain_epochs=1,
        head_epochs=2,
        full_epochs=3,
        batch_size=16,
        head_learning_rate=1e-3,
        full_learning_rate=1e-3,
    )


def test_byte_tokenizer_round_trips_utf8_and_reserves_pad() -> None:
    tokenizer = ByteTokenizer()
    assert tokenizer.vocab_size == VOCAB_SIZE == 260
    assert tokenizer.pad_id == PAD_ID == 256

    text = "café"
    ids = tokenizer.encode(text)
    assert ids == list(text.encode("utf-8"))
    assert all(0 <= value < BYTE_VOCAB_SIZE for value in ids)
    assert tokenizer.decode(ids) == text

    with pytest.raises(ValueError):
        tokenizer.decode([PAD_ID])
    with pytest.raises(ValueError):
        tokenizer.decode([VOCAB_SIZE - 1])
    with pytest.raises(TypeError):
        tokenizer.decode([True])


def test_byte_tokenizer_pads_truncates_and_rejects_specials() -> None:
    tokenizer = ByteTokenizer()
    ids, mask = tokenizer.pad_batch(("ab", "abcdef"), 4)
    assert ids.tolist() == [[97, 98, PAD_ID, PAD_ID], [97, 98, 99, 100]]
    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert ids.dtype == torch.long
    assert mask.dtype == torch.bool

    with pytest.raises(ValueError):
        tokenizer.pad_batch(("a",), 0)
    with pytest.raises(ValueError):
        tokenizer.pad_batch((), 4)
    with pytest.raises(TypeError):
        tokenizer.pad_batch(cast(Sequence[str], "a"), 4)


def test_sms_fixture_is_balanced_unique_and_within_byte_bounds() -> None:
    first = make_sms_fixture()
    second = make_sms_fixture()
    assert first == second
    assert len(first.messages) == 800
    assert first.labels.count(1) == 400
    assert first.labels.count(0) == 400
    assert len(set(first.messages)) == 800
    for message in first.messages:
        assert 5 <= len(message.encode("utf-8")) <= 25
    assert make_sms_fixture(seed=1).messages != first.messages


def test_stratified_split_keeps_balance_and_disjoint_sets() -> None:
    fixture = make_sms_fixture()
    split = stratified_split(fixture.messages, fixture.labels)
    assert (len(split.train_texts), len(split.test_texts)) == (640, 160)
    assert split.train_labels.count(1) == 320
    assert split.train_labels.count(0) == 320
    assert split.test_labels.count(1) == 80
    assert split.test_labels.count(0) == 80
    assert set(split.train_texts).isdisjoint(split.test_texts)
    assert set(split.train_texts) | set(split.test_texts) == set(
        fixture.messages
    )

    small = make_sms_fixture(64)
    small_split = stratified_split(small.messages, small.labels)
    assert (len(small_split.train_texts), len(small_split.test_texts)) == (
        52,
        12,
    )
    assert small_split.train_labels.count(1) == 26
    assert small_split.train_labels.count(0) == 26
    assert small_split.test_labels.count(1) == 6
    assert small_split.test_labels.count(0) == 6

    with pytest.raises(ValueError):
        stratified_split(fixture.messages, fixture.labels, train_fraction=1.0)


def test_classification_dataset_pads_and_masks() -> None:
    tokenizer = ByteTokenizer()
    dataset = ClassificationDataset(
        ("abcd", "ab", "abcdef"), (0, 1, 0), tokenizer, 4
    )
    assert len(dataset) == 3

    ids, mask, label = dataset[0]
    assert ids.tolist() == [97, 98, 99, 100]
    assert mask.tolist() == [True, True, True, True]
    assert label.item() == 0
    assert label.dtype == torch.long

    ids, mask, _ = dataset[1]
    assert ids.tolist() == [97, 98, PAD_ID, PAD_ID]
    assert mask.tolist() == [True, True, False, False]

    ids, mask, _ = dataset[2]
    assert ids.tolist() == [97, 98, 99, 100]
    assert mask.tolist() == [True, True, True, True]

    with pytest.raises(ValueError):
        ClassificationDataset(("",), (0,), tokenizer, 4)
    with pytest.raises(ValueError):
        ClassificationDataset(("a",), (0, 1), tokenizer, 4)


def test_body_shape_causal_invariance_and_validation() -> None:
    config = _tiny_config()
    body = LMBody(config, seed=3)
    body.eval()
    assert body.vocab_size == VOCAB_SIZE
    assert body.pad_id == PAD_ID
    assert body.context_length == config.context_length

    padded_ids = torch.tensor([[1, 2, 3, PAD_ID, PAD_ID]])
    short_ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        padded = body(padded_ids)
        short = body(short_ids)
    assert padded.shape == (1, 5, config.d_model)
    drift = float((padded[:, :3, :] - short).abs().max().item())
    assert drift < 1e-6
    assert torch.allclose(padded[:, :3, :], short, atol=1e-6, rtol=1e-5)

    with pytest.raises(ValueError):
        body(torch.tensor([[VOCAB_SIZE]]))
    with pytest.raises(ValueError):
        body(torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError):
        body(torch.ones(1, config.context_length + 1, dtype=torch.long))


def test_mean_pool_weights_by_mask_and_rejects_all_pad() -> None:
    pool = MeanPool()
    hidden = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])
    pooled = pool(hidden, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0, 3.0]]))

    with pytest.raises(ValueError):
        pool(hidden, torch.zeros(1, 3, dtype=torch.bool))
    with pytest.raises(ValueError):
        pool(hidden, torch.ones(1, 2, dtype=torch.bool))


def test_classifier_shares_body_and_swaps_head() -> None:
    body = LMBody(_tiny_config(), seed=1)
    classifier = Classifier(body, seed=2)
    body.eval()
    classifier.eval()
    assert classifier.body is body

    ids = torch.tensor([[72, 105, 33, PAD_ID]])
    explicit_mask = torch.tensor([[True, True, True, False]])
    with torch.no_grad():
        hidden = body(ids)
        logits = classifier(ids)
        explicit_logits = classifier(ids, explicit_mask)
    assert hidden.shape == (1, 4, 16)
    assert logits.shape == (1, 2)
    assert classifier.head.in_features == 16
    assert classifier.head.out_features == 2
    assert torch.allclose(logits, explicit_logits)


def test_freeze_and_unfreeze_toggle_only_body() -> None:
    body = LMBody(_tiny_config(), seed=0)
    classifier = Classifier(body, seed=0)
    body_count = count_parameters(body)
    head_count = count_parameters(classifier.head)

    assert freeze_body(classifier) == body_count
    assert not any(parameter.requires_grad for parameter in body.parameters())
    assert all(parameter.requires_grad for parameter in classifier.head.parameters())
    assert count_parameters(classifier, trainable_only=True) == head_count

    assert unfreeze_body(classifier) == body_count
    assert all(parameter.requires_grad for parameter in body.parameters())
    assert all(parameter.requires_grad for parameter in classifier.head.parameters())

    with pytest.raises(TypeError):
        freeze_body(cast(Classifier, nn.Linear(2, 2)))


def test_head_only_training_changes_head_and_freezes_body() -> None:
    torch.manual_seed(0)
    dataset = _tiny_dataset()
    loader = _loader(dataset)
    body = LMBody(_tiny_config(), seed=0)
    classifier = Classifier(body, seed=0)
    freeze_body(classifier)

    before_body = tuple(
        parameter.detach().clone() for parameter in body.parameters()
    )
    before_head = tuple(
        parameter.detach().clone()
        for parameter in classifier.head.parameters()
    )
    optimizer = torch.optim.Adam(
        [parameter for parameter in classifier.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    history = train_classifier(classifier, loader, optimizer, 2)
    assert len(history.losses) == 2
    assert len(history.accuracies) == 2
    assert all(
        math.isfinite(value)
        for value in history.losses + history.accuracies
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_body, body.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            before_head, classifier.head.parameters(), strict=True
        )
    )


def test_full_fine_tuning_changes_body() -> None:
    torch.manual_seed(0)
    dataset = _tiny_dataset()
    loader = _loader(dataset)
    body = LMBody(_tiny_config(), seed=0)
    classifier = Classifier(body, seed=0)

    freeze_body(classifier)
    head_optimizer = torch.optim.Adam(
        [parameter for parameter in classifier.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    train_classifier(classifier, loader, head_optimizer, 1)

    before_full = tuple(
        parameter.detach().clone() for parameter in body.parameters()
    )
    unfreeze_body(classifier)
    full_optimizer = torch.optim.Adam(
        [parameter for parameter in classifier.parameters() if parameter.requires_grad],
        lr=1e-2,
    )
    train_classifier(classifier, loader, full_optimizer, 3)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            before_full, body.parameters(), strict=True
        )
    )


@pytest.mark.parametrize(
    ("labels", "logits", "expected"),
    [
        (
            [0, 1],
            [[2.0, 0.0], [0.0, 2.0]],
            (1.0, 1.0, 1.0, (1, 0, 0, 1)),
        ),
        (
            [0, 1, 1, 0, 1],
            [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            (1.0, 2.0 / 3.0, 0.8, (2, 0, 1, 2)),
        ),
        (
            [0, 1],
            [[2.0, 0.0], [2.0, 0.0]],
            (0.0, 0.0, 0.0, (1, 0, 1, 0)),
        ),
        (
            [0, 1],
            [[0.0, 2.0], [0.0, 2.0]],
            (0.5, 1.0, 2.0 / 3.0, (0, 1, 0, 1)),
        ),
    ],
)
def test_evaluate_reports_precision_recall_f1_and_confusion(
    labels: list[int],
    logits: list[list[float]],
    expected: tuple[float, float, float, tuple[int, int, int, int]],
) -> None:
    model = _FixedLogits(torch.tensor(logits))
    model.train()
    metrics = evaluate(model, _metric_batches(labels))
    precision, recall, f1, counts = expected
    assert metrics.precision == pytest.approx(precision)
    assert metrics.recall == pytest.approx(recall)
    assert metrics.f1 == pytest.approx(f1)
    assert metrics.confusion == ConfusionMatrix(
        tn=counts[0], fp=counts[1], fn=counts[2], tp=counts[3]
    )
    assert metrics.confusion.as_rows() == ((counts[0], counts[1]), (counts[2], counts[3]))
    assert model.training is True

    text = format_metrics("fixed", metrics)
    assert "fixed: precision=" in text
    assert "fixed confusion (rows=true, cols=pred):" in text


def test_pretrain_quick_changes_body_and_returns_history() -> None:
    torch.manual_seed(0)
    body = LMBody(_tiny_config(), seed=0)
    tokenizer = ByteTokenizer()
    texts = (
        "win cash now",
        "free prize",
        "lunch at noon",
        "call sam 6pm",
        "see jo later",
        "meet ana 8",
        "claim gift",
        "you won 500",
    )
    before = tuple(
        parameter.detach().clone() for parameter in body.parameters()
    )
    parameter_count = count_parameters(body)
    losses = pretrain_quick(
        body,
        tokenizer,
        texts,
        epochs=1,
        batch_size=4,
        learning_rate=1e-2,
        seed=5,
    )
    assert len(losses) == 1
    assert math.isfinite(losses[0])
    assert count_parameters(body) == parameter_count
    assert any(
        not torch.equal(before_value, after_value)
        for before_value, after_value in zip(
            before, body.parameters(), strict=True
        )
    )

    after_training = tuple(
        parameter.detach().clone() for parameter in body.parameters()
    )
    assert pretrain_quick(body, tokenizer, texts, epochs=0) == ()
    assert all(
        torch.equal(before_value, after_value)
        for before_value, after_value in zip(
            after_training, body.parameters(), strict=True
        )
    )


def test_cli_demo_runs_both_regimes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(_tiny_args()) == 0
    captured = capsys.readouterr()
    output = captured.out

    assert "demo: seed=0 messages=64 train=52 test=12" in output
    assert "fixture: spam=32 ham=32 byte_lengths=" in output
    assert (
        "split: train=52 test=12 train_spam=26 train_ham=26 "
        "test_spam=6 test_ham=6"
    ) in output
    assert "tokenizer: vocab=260 pad_id=256" in output
    assert "model: body_parameters=" in output
    assert "regime: head-only trainable_parameters=" in output
    assert "regime: full trainable_parameters=" in output
    assert "head-only: precision=" in output
    assert "full: precision=" in output
    assert "head-only confusion (rows=true, cols=pred):" in output
    assert "full confusion (rows=true, cols=pred):" in output
    assert (
        "check: body_unchanged_in_head_only=true "
        "body_changed_in_full=true"
    ) in output
    assert "summary:" in output
    assert "error:" not in captured.err


def test_cli_demo_is_deterministic(
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _tiny_demo_config()
    first = run_demo(config)
    capsys.readouterr()
    second = run_demo(config)
    capsys.readouterr()

    assert (
        first.head_metrics.confusion.as_rows()
        == second.head_metrics.confusion.as_rows()
    )
    assert (
        first.full_metrics.confusion.as_rows()
        == second.full_metrics.confusion.as_rows()
    )
    assert first.head_metrics.precision == pytest.approx(
        second.head_metrics.precision
    )
    assert first.full_metrics.precision == pytest.approx(
        second.full_metrics.precision
    )
    assert first.head_history.losses == pytest.approx(
        second.head_history.losses
    )
    assert first.full_history.losses == pytest.approx(
        second.full_history.losses
    )
    assert first.body_unchanged_in_head_only is True
    assert first.body_changed_in_full is True
    assert second.body_unchanged_in_head_only is True
    assert second.body_changed_in_full is True


@pytest.mark.parametrize(
    "args",
    [
        ["--messages", "7"],
        ["--d-model", "15", "--num-heads", "4"],
        ["--train-fraction", "1.0"],
    ],
)
def test_cli_rejects_invalid_configuration(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args) == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
