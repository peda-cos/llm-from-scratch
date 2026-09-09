from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from main import (
    BYTE_VOCAB_SIZE,
    INST_ID,
    MAX_SEQUENCE_LENGTH,
    NEWLINE_ID,
    PAD_ID,
    RESP_ID,
    VOCAB_SIZE,
    Aggregator,
    EvalDatasets,
    EvalResult,
    EvalSuite,
    HistogramBin,
    InstructionTokenizer,
    JudgeVerdict,
    ModelConfig,
    PerplexityExampleResult,
    QaExample,
    TextExampleResult,
    TinyGPT,
    count_parameters,
    exact_match,
    exact_match_eval,
    judge_eval,
    make_em_fixture,
    make_f1_fixture,
    make_judge_fixture,
    make_lm_fixture,
    make_lm_training_corpus,
    make_training_pairs,
    mock_judge,
    normalise_text,
    perplexity_eval,
    token_f1,
    token_f1_eval,
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


class ScriptedPredictor:
    def __init__(self, predictions: Sequence[str]) -> None:
        self._predictions = tuple(predictions)
        self._index = 0

    def predict(self, instruction: str) -> str:
        if not instruction:
            raise AssertionError("instruction must not be empty")
        if self._index >= len(self._predictions):
            raise AssertionError("predictor received too many requests")
        value = self._predictions[self._index]
        self._index += 1
        return value


class ZeroLogitsModel(torch.nn.Module):
    def forward(self, token_ids: Tensor) -> Tensor:
        batch, length = token_ids.shape
        return torch.zeros((batch, length, VOCAB_SIZE))


class NextTokenCopyModel(torch.nn.Module):
    def forward(self, token_ids: Tensor) -> Tensor:
        batch, length = token_ids.shape
        logits = torch.zeros((batch, length, VOCAB_SIZE))
        for row in range(batch):
            for position in range(length - 1):
                target = int(token_ids[row, position + 1].item())
                logits[row, position, target] = 20.0
        return logits


def _examples() -> tuple[QaExample, ...]:
    return (
        QaExample("capital?", "Paris", "test"),
        QaExample("capital again?", "Paris", "test"),
        QaExample("capital last?", "Rome", "test"),
    )


def test_exact_match_eval_aggregates_predictions() -> None:
    predictor = ScriptedPredictor([" paris. ", "Lyon", "rome"])
    result = exact_match_eval(predictor, _examples())
    assert result.name == "exact_match"
    assert result.metric == "accuracy"
    assert result.value == pytest.approx(2 / 3)
    assert len(result.examples) == 3
    first, second, third = result.examples
    assert isinstance(first, TextExampleResult)
    assert isinstance(second, TextExampleResult)
    assert isinstance(third, TextExampleResult)
    assert first.score == 1.0
    assert second.score == 0.0
    assert third.normalised_prediction == "rome"


def test_token_f1_eval_aggregates_predictions() -> None:
    predictor = ScriptedPredictor(["red blue green", "x"])
    examples = (
        QaExample("colors?", "red green yellow", "test"),
        QaExample("other?", "a b", "test"),
    )
    result = token_f1_eval(predictor, examples)
    assert result.name == "token_f1"
    assert result.metric == "mean_token_f1"
    assert result.value == pytest.approx(1 / 3)
    scores = [
        example.score
        for example in result.examples
        if isinstance(example, TextExampleResult)
    ]
    assert scores == pytest.approx([2 / 3, 0.0])


def test_judge_eval_aggregates_scores_and_rationales() -> None:
    predictor = ScriptedPredictor(["Paris", "a b c", "x"])
    examples = (
        QaExample("q1", "paris", "test"),
        QaExample("q2", "a b", "test"),
        QaExample("q3", "a b", "test"),
    )
    result = judge_eval(predictor, examples)
    assert result.name == "judge"
    assert result.metric == "mean_judge_score"
    assert result.value == pytest.approx((5 + 4 + 1) / 3)
    text_examples = [
        example
        for example in result.examples
        if isinstance(example, TextExampleResult)
    ]
    assert [example.score for example in text_examples] == [5.0, 4.0, 1.0]
    assert all(example.rationale for example in text_examples)


def test_perplexity_uniform_model_counts_only_real_targets() -> None:
    tokenizer = InstructionTokenizer()
    model = ZeroLogitsModel()
    result = perplexity_eval(model, tokenizer, ("ab", "hello"))
    assert result.value == pytest.approx(float(VOCAB_SIZE))
    perplexity_examples = [
        example
        for example in result.examples
        if isinstance(example, PerplexityExampleResult)
    ]
    assert [example.tokens for example in perplexity_examples] == [1, 4]
    assert sum(bin.count for bin in result.histogram) == 5
    assert all(bin.count >= 0 for bin in result.histogram)


def test_perplexity_uses_next_token_alignment() -> None:
    tokenizer = InstructionTokenizer()
    model = NextTokenCopyModel()
    model.train()
    result = perplexity_eval(model, tokenizer, ("abc", "hello"))
    assert result.value == pytest.approx(1.0, abs=1e-5)
    assert model.training is True
    assert sum(bin.count for bin in result.histogram) == 6


def test_perplexity_rejects_invalid_documents() -> None:
    tokenizer = InstructionTokenizer(max_len=8)
    model = ZeroLogitsModel()
    with pytest.raises(ValueError):
        perplexity_eval(model, tokenizer, ())
    with pytest.raises(ValueError):
        perplexity_eval(model, tokenizer, ("a",))
    with pytest.raises(ValueError):
        perplexity_eval(model, tokenizer, ("x" * 9,))


def test_eval_suite_runs_all_evals_in_spec_order() -> None:
    tokenizer = InstructionTokenizer()
    datasets = EvalDatasets(
        lm_corpus=make_lm_fixture(),
        em=make_em_fixture(),
        f1=make_f1_fixture(),
        judge=make_judge_fixture(),
    )
    model = TinyGPT(ModelConfig(hidden=16, num_heads=2, num_layers=1), seed=0)
    suite = EvalSuite(
        model,
        tokenizer,
        datasets,
        max_new_tokens=1,
        enabled=("judge", "perplexity", "exact_match", "token_f1"),
    )
    results = suite.run()
    assert [result.name for result in results] == [
        "perplexity",
        "exact_match",
        "token_f1",
        "judge",
    ]
    assert all(len(result.examples) == 20 for result in results)

    subset = EvalSuite(
        model,
        tokenizer,
        datasets,
        max_new_tokens=1,
        enabled=("exact_match", "token_f1"),
    )
    assert [result.name for result in subset.run()] == [
        "exact_match",
        "token_f1",
    ]
    with pytest.raises(ValueError):
        EvalSuite(
            model,
            tokenizer,
            datasets,
            max_new_tokens=1,
            enabled=("bogus",),
        )


def _text_result(score: float) -> TextExampleResult:
    return TextExampleResult(
        instruction="q",
        reference="r",
        prediction="p",
        score=score,
        normalised_prediction="p",
        normalised_reference="r",
        rationale="test",
    )


def _result(name: str, value: float) -> EvalResult:
    if name == "perplexity":
        example = PerplexityExampleResult(
            document="ab",
            tokens=1,
            mean_nll=math.log(value),
            perplexity=value,
        )
        return EvalResult(
            name=name,
            metric="perplexity",
            value=value,
            examples=(example,),
            histogram=(HistogramBin("[0,0.5)", 1),),
        )
    metric = {
        "exact_match": "accuracy",
        "token_f1": "mean_token_f1",
        "judge": "mean_judge_score",
    }[name]
    return EvalResult(
        name=name,
        metric=metric,
        value=value,
        examples=(_text_result(value),),
    )


def test_aggregator_normalisation_contract() -> None:
    assert Aggregator.normalise("perplexity", 1.0) == 1.0
    assert Aggregator.normalise("perplexity", math.e) == pytest.approx(0.5)
    assert Aggregator.normalise("exact_match", 0.25) == 0.25
    assert Aggregator.normalise("token_f1", 1.0) == 1.0
    assert Aggregator.normalise("judge", 5.0) == 1.0
    for name, value in (
        ("perplexity", 0.5),
        ("exact_match", 1.1),
        ("token_f1", -0.1),
        ("judge", 0.0),
        ("judge", 6.0),
        ("bogus", 1.0),
    ):
        with pytest.raises(ValueError):
            Aggregator.normalise(name, value)


def test_aggregator_weighted_mean_and_contributions() -> None:
    aggregator = Aggregator({"exact_match": 1.0, "token_f1": 3.0})
    results = (_result("exact_match", 0.5), _result("token_f1", 1.0))
    report = aggregator.aggregate(results)
    assert report.aggregate == pytest.approx(0.875)
    assert [scored.contribution for scored in report.evals] == pytest.approx(
        [0.125, 0.75]
    )
    assert sum(scored.contribution for scored in report.evals) == pytest.approx(
        report.aggregate
    )
    assert report.weights == {"exact_match": 1.0, "token_f1": 3.0}


def test_aggregator_rejects_invalid_configuration_and_results() -> None:
    with pytest.raises(ValueError):
        Aggregator({"bogus": 1.0})
    with pytest.raises(ValueError):
        Aggregator({"exact_match": -1.0})
    with pytest.raises(ValueError):
        Aggregator({"exact_match": float("nan")})
    with pytest.raises(ValueError):
        Aggregator({})
    with pytest.raises(ValueError):
        Aggregator({"exact_match": 0.0})
    with pytest.raises(ValueError):
        Aggregator().aggregate([])
    duplicate = (_result("exact_match", 1.0), _result("exact_match", 1.0))
    with pytest.raises(ValueError):
        Aggregator({"exact_match": 1.0}).aggregate(duplicate)
    with pytest.raises(ValueError):
        Aggregator({"exact_match": 1.0}).aggregate(
            (_result("exact_match", 1.0), _result("token_f1", 1.0))
        )


def test_report_json_has_stable_keys_and_recomputable_aggregate() -> None:
    results = (
        _result("perplexity", math.e),
        _result("exact_match", 0.5),
        _result("token_f1", 1.0),
        _result("judge", 2.5),
    )
    report = Aggregator().aggregate(results, metadata={"seed": 0})
    assert report.aggregate == pytest.approx(0.65)
    payload = report.to_json_dict()
    assert payload["schema_version"] == 1
    assert payload["aggregate"] == pytest.approx(0.65)
    evals = payload["evals"]
    assert isinstance(evals, dict)
    assert set(evals) == {"perplexity", "exact_match", "token_f1", "judge"}
    perplexity_payload = evals["perplexity"]
    assert isinstance(perplexity_payload, dict)
    assert perplexity_payload["scored_tokens"] == 1
    assert perplexity_payload["num_examples"] == 1
    parsed = json.loads(report.to_json())
    assert parsed["schema_version"] == 1
    assert parsed["metadata"] == {"seed": 0}
    assert parsed["aggregate"] == pytest.approx(0.65)
