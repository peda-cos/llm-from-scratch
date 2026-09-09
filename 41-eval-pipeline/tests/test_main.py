from __future__ import annotations

from collections import Counter

import pytest
import torch

from main import (
    BYTE_VOCAB_SIZE,
    INST_ID,
    MAX_SEQUENCE_LENGTH,
    NEWLINE_ID,
    PAD_ID,
    RESP_ID,
    VOCAB_SIZE,
    InstructionTokenizer,
    JudgeVerdict,
    ModelConfig,
    QaExample,
    TinyGPT,
    count_parameters,
    exact_match,
    make_em_fixture,
    make_f1_fixture,
    make_judge_fixture,
    make_lm_fixture,
    make_lm_training_corpus,
    make_training_pairs,
    mock_judge,
    normalise_text,
    token_f1,
)


def _tiny_config() -> ModelConfig:
    return ModelConfig(hidden=16, num_heads=2, num_layers=1)


def test_tokenizer_specials_and_example_round_trip() -> None:
    tokenizer = InstructionTokenizer()
    prefix = tokenizer.encode_prefix("What is 2 + 3?")
    assert prefix[0] == INST_ID
    assert prefix[-1] == RESP_ID
    assert tokenizer.decode(prefix[1:-1]) == "What is 2 + 3?"

    ids, completion_start = tokenizer.encode_example("Q", "café")
    assert ids[:completion_start] == tokenizer.encode_prefix("Q")
    assert ids[completion_start:-1] == list("café".encode("utf-8"))
    assert ids[-1] == NEWLINE_ID
    assert tokenizer.decode(ids[completion_start:-1]) == "café"
    assert tokenizer.vocab_size == VOCAB_SIZE
    assert tokenizer.inst_id == INST_ID
    assert tokenizer.resp_id == RESP_ID
    assert tokenizer.pad_id == PAD_ID
    assert tokenizer.max_len == MAX_SEQUENCE_LENGTH
    assert BYTE_VOCAB_SIZE == 256


def test_tokenizer_rejects_invalid_input() -> None:
    with pytest.raises(ValueError):
        InstructionTokenizer(max_len=3)
    tokenizer = InstructionTokenizer(max_len=12)
    with pytest.raises(ValueError):
        tokenizer.encode_prefix("x" * 20)
    with pytest.raises(ValueError):
        tokenizer.encode_example("Q", "line one\nline two")
    with pytest.raises(TypeError):
        tokenizer.encode_prefix(3)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        tokenizer.decode("not ids")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        tokenizer.decode([256])


def test_model_config_validation() -> None:
    with pytest.raises(ValueError):
        ModelConfig(hidden=15, num_heads=4)
    with pytest.raises(ValueError):
        ModelConfig(num_layers=0)
    with pytest.raises(ValueError):
        ModelConfig(dropout=1.0)
    with pytest.raises(TypeError):
        ModelConfig(hidden=True)


def test_model_forward_shape_tied_head_and_padding() -> None:
    config = _tiny_config()
    model = TinyGPT(config, seed=0)
    token_ids = torch.tensor([[INST_ID, 65, 66, RESP_ID, 67, NEWLINE_ID]])
    logits = model(token_ids)
    assert logits.shape == (1, token_ids.shape[1], VOCAB_SIZE)
    assert torch.isfinite(logits).all()
    assert model.head.weight is model.token_embedding.weight
    assert bool(torch.count_nonzero(model.token_embedding.weight[PAD_ID]) == 0)
    assert model.context_length == config.max_len
    expected = (
        VOCAB_SIZE * config.hidden
        + config.max_len * config.hidden
        + config.num_layers
        * (
            4 * config.hidden * config.hidden
            + 2 * 4 * config.hidden * config.hidden
            + 4 * config.hidden
        )
        + 2 * config.hidden
    )
    assert count_parameters(model) == expected


def test_causal_prefix_is_invariant() -> None:
    model = TinyGPT(_tiny_config(), seed=1)
    model.eval()
    token_ids = torch.tensor([[INST_ID, 65, 66, 67, RESP_ID, 68, 69, 70]])
    with torch.no_grad():
        full = model(token_ids)
        prefix = model(token_ids[:, :4])
    assert torch.allclose(full[:, :4], prefix, atol=1e-6)


def test_model_rejects_invalid_inputs() -> None:
    model = TinyGPT(_tiny_config(), seed=0)
    with pytest.raises(TypeError):
        model(torch.zeros((1, 4), dtype=torch.float32))
    with pytest.raises(ValueError):
        model(torch.zeros((0, 4), dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.zeros((1, 0), dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.full((1, 4), VOCAB_SIZE, dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.zeros((1, MAX_SEQUENCE_LENGTH + 1), dtype=torch.long))


def test_seeded_models_are_deterministic() -> None:
    first = TinyGPT(_tiny_config(), seed=7)
    second = TinyGPT(_tiny_config(), seed=7)
    third = TinyGPT(_tiny_config(), seed=8)
    token_ids = torch.tensor([[INST_ID, 65, RESP_ID]])
    with torch.no_grad():
        first_logits = first(token_ids)
        second_logits = second(token_ids)
        third_logits = third(token_ids)
    assert torch.equal(first_logits, second_logits)
    assert not torch.equal(first_logits, third_logits)


def test_fixtures_are_deterministic_held_out_and_balanced() -> None:
    training = make_training_pairs()
    assert len(training) == 150
    assert len({pair.instruction for pair in training}) == len(training)
    assert Counter(pair.category for pair in training) == {
        "capitals": 24,
        "arithmetic": 18,
        "lists": 42,
        "definitions": 36,
        "summaries": 30,
    }

    em = make_em_fixture()
    f1 = make_f1_fixture()
    judge = make_judge_fixture()
    lm = make_lm_fixture()

    assert len(em) == len(f1) == len(judge) == len(lm) == 20
    assert make_em_fixture() == em
    assert make_f1_fixture() == f1
    assert make_judge_fixture() == judge
    assert make_lm_fixture() == lm
    assert len(make_lm_training_corpus()) == 30

    all_eval_prompts = {
        example.instruction for example in (*em, *f1, *judge)
    }
    training_prompts = {example.instruction for example in training}
    assert all_eval_prompts.isdisjoint(training_prompts)
    assert len(all_eval_prompts) == 60

    assert Counter(example.category for example in em) == {
        "capitals": 8,
        "arithmetic": 6,
        "lists": 6,
    }
    assert Counter(example.category for example in f1) == {
        "lists": 8,
        "definitions": 12,
    }
    assert Counter(example.category for example in judge) == {
        "definitions": 10,
        "summaries": 10,
    }

    tokenizer = InstructionTokenizer()
    for example in (*training, *em, *f1, *judge):
        tokenizer.encode_example(example.instruction, example.reference)
    for document in (*make_lm_training_corpus(), *lm):
        assert "\n" not in document
        assert 2 <= len(document.encode("utf-8")) < MAX_SEQUENCE_LENGTH


def test_qa_example_validation() -> None:
    with pytest.raises(ValueError):
        QaExample("", "answer", "test")
    with pytest.raises(ValueError):
        QaExample("prompt", "line one\nline two", "test")
    with pytest.raises(TypeError):
        QaExample("prompt", 3, "test")  # type: ignore[arg-type]


def test_normalise_text_contract() -> None:
    assert normalise_text("  Hello   WORLD.  ") == "hello world"
    assert normalise_text("Paris!!!") == "paris"
    assert normalise_text("What?!.") == "what"
    assert normalise_text("   ") == ""
    with pytest.raises(TypeError):
        normalise_text(3)  # type: ignore[arg-type]


def test_exact_match_uses_normalisation() -> None:
    assert exact_match("Paris", " paris. ")
    assert exact_match("Red,  Blue!", "red, blue")
    assert not exact_match("Paris", "Lyon")


def test_token_f1_multiset_and_empty_rules() -> None:
    assert token_f1("red blue green", "red green yellow") == pytest.approx(2 / 3)
    assert token_f1("a a b", "a b b") == pytest.approx(2 / 3)
    assert token_f1("blue red", "red blue") == 1.0
    assert token_f1("", "") == 1.0
    assert token_f1("something", "") == 0.0
    assert token_f1("", "something") == 0.0
    assert token_f1("alpha beta", "gamma delta") == 0.0


def test_mock_judge_covers_every_band() -> None:
    assert mock_judge("q", "Paris.", " paris ").score == 5
    assert mock_judge("q", "a b c", "a b").score == 4
    assert mock_judge("q", "a c", "a b").score == 3
    assert mock_judge("q", "a f g h i", "a b c d e").score == 2
    assert mock_judge("q", "x y z w v", "a b c d e").score == 1
    for prediction, reference in (
        ("Paris", "Paris"),
        ("a b c", "a b"),
        ("a c", "a b"),
        ("a f g h i", "a b c d e"),
        ("x y z w v", "a b c d e"),
    ):
        verdict = mock_judge("q", prediction, reference)
        assert 1 <= verdict.score <= 5
        assert verdict.rationale
    with pytest.raises(ValueError):
        JudgeVerdict(0, "bad")
    with pytest.raises(ValueError):
        JudgeVerdict(6, "bad")
    with pytest.raises(ValueError):
        JudgeVerdict(3, "")
