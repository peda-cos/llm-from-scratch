from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

from main import (
    BYTE_VOCAB_SIZE,
    CATEGORY_ORDER,
    IGNORE_INDEX,
    INST_ID,
    MAX_SEQUENCE_LENGTH,
    PAD_ID,
    RESPONSE_TERMINATOR_ID,
    RESP_ID,
    VOCAB_SIZE,
    CategoryScore,
    DataSplit,
    DemoConfig,
    EvalResult,
    InstructionPair,
    InstructionTokenizer,
    ModelConfig,
    SFTDataset,
    SFTReport,
    TinyGPT,
    count_parameters,
    evaluate,
    exact_match,
    generate,
    make_dataset,
    main,
    normalize_text,
    run_demo,
    sft_collate,
    sft_loss,
    split_dataset,
    train_sft,
)


def _tiny_config(max_len: int = 16) -> ModelConfig:
    return ModelConfig(
        hidden=16,
        num_heads=2,
        num_layers=1,
        max_len=max_len,
        dropout=0.0,
    )


def _ignore(_: str) -> None:
    pass


class _ScriptedModel(nn.Module):
    """Return a scripted next token for each generated position."""

    context_length: int
    script: tuple[int, ...]
    start_length: int

    def __init__(
        self,
        script: Sequence[int],
        *,
        start_length: int,
        context_length: int = 64,
    ) -> None:
        super().__init__()
        if not script:
            raise ValueError("script must not be empty")
        self.script = tuple(script)
        self.start_length = start_length
        self.context_length = context_length

    def forward(self, token_ids: Tensor) -> Tensor:
        batch, length = token_ids.shape
        index = max(0, length - self.start_length)
        next_id = self.script[min(index, len(self.script) - 1)]
        logits = torch.full((batch, length, VOCAB_SIZE), -1000.0)
        logits[:, -1, next_id] = 1000.0
        return logits


class _UniformModel(nn.Module):
    context_length: int

    def __init__(self, context_length: int = 64) -> None:
        super().__init__()
        self.context_length = context_length

    def forward(self, token_ids: Tensor) -> Tensor:
        batch, length = token_ids.shape
        return torch.zeros((batch, length, VOCAB_SIZE))


def _pair(
    instruction: str,
    response: str,
    category: str = "capitals",
    base: str = "France",
    template: int = 0,
) -> InstructionPair:
    return InstructionPair(instruction, response, category, base, template)


def test_tokenizer_round_trips_and_inserts_specials() -> None:
    tokenizer = InstructionTokenizer()
    assert tokenizer.max_len == MAX_SEQUENCE_LENGTH == 64
    assert tokenizer.vocab_size == VOCAB_SIZE == 259
    assert tokenizer.inst_id == INST_ID == 256
    assert tokenizer.resp_id == RESP_ID == 257
    assert tokenizer.pad_id == PAD_ID == 258

    text = "café"
    prefix = tokenizer.encode_prefix(text)
    assert prefix == [INST_ID] + list(text.encode("utf-8")) + [RESP_ID]

    ids, response_start = tokenizer.encode_pair(text, "ok")
    assert ids[0] == INST_ID
    assert ids[1 : 1 + len(text.encode("utf-8"))] == list(text.encode("utf-8"))
    assert ids[response_start - 1] == RESP_ID
    assert ids[response_start:] == list(b"ok") + [RESPONSE_TERMINATOR_ID]
    assert response_start == 2 + len(text.encode("utf-8"))
    assert tokenizer.decode_response(ids[response_start:-1]) == "ok"
    assert tokenizer.decode_response(list("café".encode("utf-8"))) == "café"


def test_tokenizer_rejects_invalid_values() -> None:
    tokenizer = InstructionTokenizer()
    with pytest.raises(ValueError):
        tokenizer.encode_prefix("   ")
    with pytest.raises(ValueError):
        tokenizer.encode_pair("", "x")
    with pytest.raises(ValueError):
        tokenizer.encode_pair("x", "")
    with pytest.raises(TypeError):
        tokenizer.encode_prefix(cast(str, 3))
    with pytest.raises(TypeError):
        tokenizer.encode_pair(cast(str, None), "x")
    with pytest.raises(ValueError):
        InstructionTokenizer(3)

    short = InstructionTokenizer(max_len=5)
    with pytest.raises(ValueError):
        short.encode_prefix("abc")
    with pytest.raises(ValueError):
        short.encode_pair("ab", "cd")

    with pytest.raises(TypeError):
        tokenizer.decode_response(cast(Sequence[int], "ab"))
    with pytest.raises(TypeError):
        tokenizer.decode_response([True])
    with pytest.raises(TypeError):
        tokenizer.decode_response(["a"])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        tokenizer.decode_response([INST_ID])
    with pytest.raises(ValueError):
        tokenizer.decode_response([BYTE_VOCAB_SIZE])


def test_make_dataset_is_deterministic_and_balanced() -> None:
    first = make_dataset(0)
    second = make_dataset(0)
    assert first == second
    assert len(first) == 200
    counts = {category: sum(1 for pair in first if pair.category == category) for category in CATEGORY_ORDER}
    assert counts == {
        "capitals": 40,
        "arithmetic": 30,
        "lists": 30,
        "summaries": 30,
        "code": 30,
        "definitions": 40,
    }
    assert len({pair.instruction for pair in first}) == 200
    tokenizer = InstructionTokenizer()
    for pair in first:
        ids, _ = tokenizer.encode_pair(pair.instruction, pair.response)
        assert len(ids) <= MAX_SEQUENCE_LENGTH

    other = make_dataset(1)
    assert [pair.instruction for pair in other] != [pair.instruction for pair in first]
    assert set(other) == set(first)


def test_split_dataset_is_stratified_and_disjoint() -> None:
    pairs = make_dataset(0)
    split = split_dataset(pairs, 0.2, 0)
    assert isinstance(split, DataSplit)
    assert (len(split.train), len(split.test)) == (160, 40)
    assert {pair.category for pair in split.train} == set(CATEGORY_ORDER)
    assert {pair.category for pair in split.test} == set(CATEGORY_ORDER)
    assert {pair.instruction for pair in split.train}.isdisjoint(
        {pair.instruction for pair in split.test}
    )
    test_counts = {
        category: sum(1 for pair in split.test if pair.category == category)
        for category in CATEGORY_ORDER
    }
    assert test_counts == {
        "capitals": 8,
        "arithmetic": 6,
        "lists": 6,
        "summaries": 6,
        "code": 6,
        "definitions": 8,
    }
    train_bases = {(pair.category, pair.base) for pair in split.train}
    assert all((pair.category, pair.base) in train_bases for pair in split.test)

    with pytest.raises(ValueError):
        split_dataset(pairs, 0.0, 0)
    with pytest.raises(ValueError):
        split_dataset(pairs, 1.0, 0)
    with pytest.raises(ValueError):
        split_dataset((), 0.2, 0)


def test_sft_dataset_masks_instruction_and_boundary() -> None:
    tokenizer = InstructionTokenizer()
    dataset = SFTDataset((_pair("hi", "yo"),), tokenizer)
    assert len(dataset) == 1
    ids, labels = dataset[0]
    assert ids.tolist() == [
        INST_ID,
        ord("h"),
        ord("i"),
        RESP_ID,
        ord("y"),
        ord("o"),
        RESPONSE_TERMINATOR_ID,
    ]
    assert labels.tolist() == [
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        IGNORE_INDEX,
        ord("y"),
        ord("o"),
        RESPONSE_TERMINATOR_ID,
    ]
    assert int((labels != IGNORE_INDEX).sum().item()) == 3

    with pytest.raises(ValueError):
        SFTDataset((), tokenizer)


def test_sft_collate_pads_and_masks() -> None:
    tokenizer = InstructionTokenizer()
    dataset = SFTDataset(
        (_pair("hi", "yo"), _pair("hello there", "yes")),
        tokenizer,
    )
    ids, labels = sft_collate([dataset[0], dataset[1]])
    assert ids.shape == labels.shape == (2, 17)
    assert ids.dtype == torch.long
    assert labels.dtype == torch.long
    assert ids[0, -1].item() == PAD_ID
    assert labels[0, -1].item() == IGNORE_INDEX
    assert labels[0, 4].item() == ord("y")
    assert ids[1, 0].item() == INST_ID
    assert labels[1, 0].item() == IGNORE_INDEX
    assert int((labels[1] != IGNORE_INDEX).sum().item()) == 4

    with pytest.raises(ValueError):
        sft_collate([])
    with pytest.raises(ValueError):
        sft_collate([(torch.tensor([1]), torch.tensor([1, 2]))])
    with pytest.raises(ValueError):
        sft_collate([(torch.tensor([[1]]), torch.tensor([[1]]))])
    with pytest.raises(TypeError):
        sft_collate(
            cast(
                Sequence[tuple[Tensor, Tensor]],
                [(torch.tensor([1]), torch.tensor([1.0]))],
            )
        )
    with pytest.raises(ValueError):
        sft_collate([(torch.tensor([1]), torch.tensor([VOCAB_SIZE]))])


def test_sft_loss_matches_manual_shift() -> None:
    torch.manual_seed(0)
    logits = torch.randn(1, 4, VOCAB_SIZE)
    labels = torch.tensor([[-100, -100, 5, 6]], dtype=torch.long)
    expected = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, VOCAB_SIZE),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )
    actual = sft_loss(logits, labels)
    assert torch.allclose(actual, expected)
    assert math.isfinite(float(actual.item()))

    with pytest.raises(ValueError):
        sft_loss(logits, torch.full((1, 4), IGNORE_INDEX))
    with pytest.raises(ValueError):
        sft_loss(torch.randn(1, 3, VOCAB_SIZE), labels)
    with pytest.raises(TypeError):
        sft_loss(torch.randn(1, 4, VOCAB_SIZE).to(torch.int64), labels)


def test_sft_loss_ignores_masked_gradient_positions() -> None:
    logits = torch.randn(1, 4, VOCAB_SIZE, requires_grad=True)
    labels = torch.tensor([[-100, -100, 5, 6]], dtype=torch.long)
    loss = sft_loss(logits, labels)
    torch.autograd.backward(loss)
    assert logits.grad is not None
    grad = logits.grad
    assert torch.allclose(
        grad[0, 0], torch.zeros(VOCAB_SIZE), atol=1e-7
    )
    assert float(grad[0, 1].abs().sum().item()) > 0.0


def test_tiny_gpt_shape_causality_and_validation() -> None:
    config = _tiny_config()
    model = TinyGPT(config, seed=0)
    model.eval()
    assert model.vocab_size == VOCAB_SIZE
    assert model.pad_id == PAD_ID
    assert model.context_length == config.max_len

    ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (1, 4, VOCAB_SIZE)

    with torch.no_grad():
        first = model(torch.tensor([[1, 2, 3, 4]]))
        second = model(torch.tensor([[1, 2, 3, 5]]))
        short = model(torch.tensor([[1, 2, 3]]))
        padded = model(torch.tensor([[1, 2, 3, PAD_ID, PAD_ID]]))
    assert torch.allclose(first[:, :3], second[:, :3], atol=1e-6)
    assert torch.allclose(short, padded[:, :3], atol=1e-6)

    with pytest.raises(ValueError):
        model(torch.tensor([[VOCAB_SIZE]]))
    with pytest.raises(ValueError):
        model(torch.tensor([1, 2, 3]))
    with pytest.raises(ValueError):
        model(torch.ones(1, config.max_len + 1, dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.tensor([[-1]]))


def test_tiny_gpt_is_tied_and_counts_parameters() -> None:
    model = TinyGPT(seed=0)
    assert model.head.weight is model.token_embedding.weight
    assert count_parameters(model) == 253_152
    assert count_parameters(model, trainable_only=True) == 253_152
    assert torch.count_nonzero(model.token_embedding.weight[PAD_ID]).item() == 0

    with pytest.raises(ValueError):
        ModelConfig(hidden=15, num_heads=4)
    with pytest.raises(ValueError):
        ModelConfig(dropout=1.0)


def test_generate_greedy_stop_rules_and_context_limit() -> None:
    tokenizer = InstructionTokenizer()
    start = len(tokenizer.encode_prefix("hi"))
    assert start == 4

    newline_model = _ScriptedModel(
        [ord("a"), ord("b"), RESPONSE_TERMINATOR_ID, ord("x")],
        start_length=start,
    )
    assert generate(newline_model, tokenizer, "hi", max_new_tokens=10) == "ab"

    control_model = _ScriptedModel([ord("h"), RESP_ID, ord("x")], start_length=start)
    assert generate(control_model, tokenizer, "hi", max_new_tokens=10) == "h"

    punctuation_model = _ScriptedModel(
        [ord("h"), ord("i"), ord("."), ord("."), ord("x")],
        start_length=start,
    )
    assert generate(punctuation_model, tokenizer, "hi", max_new_tokens=10) == "hi.."

    length_model = _ScriptedModel([ord("a")] * 10, start_length=start)
    assert generate(length_model, tokenizer, "hi", max_new_tokens=2) == "aa"

    context_model = _ScriptedModel(
        [ord("a")] * 10, start_length=start, context_length=start + 2
    )
    assert generate(context_model, tokenizer, "hi", max_new_tokens=10) == "aa"

    mode_model = _ScriptedModel([ord("a")], start_length=start)
    mode_model.train()
    generate(mode_model, tokenizer, "hi", max_new_tokens=1)
    assert mode_model.training is True
    mode_model.eval()
    generate(mode_model, tokenizer, "hi", max_new_tokens=1)
    assert mode_model.training is False


def test_generate_temperature_is_seeded_and_validated() -> None:
    tokenizer = InstructionTokenizer()
    model = _UniformModel()
    first = generate(
        model, tokenizer, "hi", temperature=1.5, seed=7, max_new_tokens=6
    )
    second = generate(
        model, tokenizer, "hi", temperature=1.5, seed=7, max_new_tokens=6
    )
    assert first == second

    with pytest.raises(ValueError):
        generate(model, tokenizer, "hi", temperature=-1.0)
    with pytest.raises(ValueError):
        generate(model, tokenizer, "hi", temperature=float("nan"))
    with pytest.raises(ValueError):
        generate(model, tokenizer, "hi", max_new_tokens=0)
    with pytest.raises(ValueError):
        generate(model, tokenizer, "hi", seed=-1)


def test_exact_match_normalizes_only_case_and_whitespace() -> None:
    assert normalize_text("  Hello   World ") == "hello world"
    assert exact_match(" Paris. ", "paris.") == 1.0
    assert exact_match("Paris!", "Paris.") == 0.0
    assert exact_match("The capital of France is Paris.", "Paris.") == 0.0
    with pytest.raises(TypeError):
        exact_match(cast(str, 3), "x")


def test_evaluate_reports_overall_and_per_category() -> None:
    tokenizer = InstructionTokenizer()
    model = _ScriptedModel(
        [ord("a"), ord("b"), RESPONSE_TERMINATOR_ID],
        start_length=len(tokenizer.encode_prefix("hi")),
    )
    pairs = (
        _pair("hi", "ab", "capitals", "France"),
        _pair("hi", "cd", "lists", "colors"),
    )
    model.train()
    result = evaluate(model, tokenizer, pairs)
    assert result.correct == 1
    assert result.total == 2
    assert result.exact_match == 0.5
    assert result.predictions == ("ab", "ab")
    assert [category.category for category in result.categories] == [
        "capitals",
        "lists",
    ]
    assert result.categories[0] == CategoryScore("capitals", 1, 1, 1.0)
    assert result.categories[1] == CategoryScore("lists", 0, 1, 0.0)
    assert model.training is True

    with pytest.raises(ValueError):
        evaluate(model, tokenizer, ())


def test_train_sft_reduces_loss_and_calls_eval() -> None:
    tokenizer = InstructionTokenizer()
    pairs = make_dataset(0)[:16]
    dataset = SFTDataset(pairs, tokenizer)
    loader: DataLoader[tuple[Tensor, Tensor]] = DataLoader(
        dataset, batch_size=4, shuffle=False, collate_fn=sft_collate
    )
    model = TinyGPT(_tiny_config(max_len=64), seed=0)
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    calls: list[int] = []

    def eval_fn() -> EvalResult:
        calls.append(1)
        return EvalResult(0, 1, 0.0, (), ())

    report = train_sft(
        model,
        loader,
        optimizer,
        epochs=4,
        eval_every=2,
        eval_fn=eval_fn,
        log=_ignore,
    )
    assert isinstance(report, SFTReport)
    assert len(report.losses) == 4
    assert all(math.isfinite(value) for value in report.losses)
    assert report.losses[-1] < report.losses[0]
    assert [epoch for epoch, _ in report.evals] == [2, 4]
    assert report.final_eval == report.evals[-1][1]
    assert len(calls) == 2
    assert any(
        not torch.equal(before_value, after_value)
        for before_value, after_value in zip(
            before, model.parameters(), strict=True
        )
    )

    bad_loader = [
        (
            torch.tensor([[INST_ID, ord("h"), RESP_ID, ord("i")]]),
            torch.full((1, 4), IGNORE_INDEX),
        )
    ]
    with pytest.raises(ValueError):
        train_sft(
            model,
            bad_loader,
            optimizer,
            epochs=1,
            log=_ignore,
        )
    with pytest.raises(ValueError):
        train_sft(model, loader, optimizer, epochs=0, log=_ignore)


def test_demo_cli_prints_contract(capsys: pytest.CaptureFixture[str]) -> None:
    args = [
        "--epochs",
        "2",
        "--hidden",
        "16",
        "--num-heads",
        "2",
        "--num-layers",
        "1",
        "--batch-size",
        "8",
        "--eval-every",
        "1",
    ]
    assert main(args) == 0
    captured = capsys.readouterr()
    output = captured.out
    assert "instruction-tuning-sft: seed=0 train=160 test=40 max_len=64" in output
    assert "dataset: pairs=200 capitals=40 arithmetic=30 lists=30 summaries=30 code=30 definitions=40" in output
    assert "tokenizer: vocab=259 inst=256 resp=257 pad=258 terminator=10" in output
    assert "model: parameters=8336 context_length=64 hidden=16 heads=2 layers=1" in output
    assert "mask: targets=" in output
    assert "baseline: exact_match=" in output
    assert "epoch   1: loss=" in output
    assert "epoch   2: loss=" in output
    assert "final: exact_match=" in output
    assert "per-category: capitals=" in output
    assert "sample: [MATCH]" in output or "sample: [MISS]" in output
    assert "sample:         gold=" in output
    assert "sample:         prediction=" in output
    assert "check: loss_finite=true mask_only_response=true" in output
    assert "error:" not in captured.err


def test_demo_is_deterministic(capsys: pytest.CaptureFixture[str]) -> None:
    config = DemoConfig(
        hidden=16,
        num_heads=2,
        num_layers=1,
        epochs=1,
        batch_size=8,
        eval_every=1,
    )
    first = run_demo(config)
    capsys.readouterr()
    second = run_demo(config)
    capsys.readouterr()
    assert first.losses == pytest.approx(second.losses)
    assert first.baseline.exact_match == pytest.approx(second.baseline.exact_match)
    assert first.final.exact_match == pytest.approx(second.final.exact_match)
    assert first.final.predictions == second.final.predictions
    assert first.mask_ok is True
    assert second.mask_ok is True


@pytest.mark.parametrize(
    "args",
    [
        ["--hidden", "15", "--num-heads", "4"],
        ["--epochs", "0"],
        ["--test-fraction", "1.0"],
    ],
)
def test_cli_rejects_invalid_configuration(
    args: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(args) == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
