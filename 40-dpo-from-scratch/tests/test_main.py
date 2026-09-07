from __future__ import annotations

import math
from collections.abc import Sequence
from typing import cast

import pytest
import torch
import torch.nn.functional as F
from main import (
    BYTE_VOCAB_SIZE,
    CATEGORY_ORDER,
    INST_ID,
    MAX_SEQUENCE_LENGTH,
    PAD_ID,
    RESP_ID,
    VOCAB_SIZE,
    DemoConfig,
    InstructionTokenizer,
    ModelConfig,
    Preference,
    TinyGPT,
    build_reference_and_policy,
    count_parameters,
    dpo_loss,
    evaluate_margins,
    main,
    make_preferences,
    run_demo,
    sequence_log_prob,
    train_dpo,
    warmup_pretrain,
)


def _tiny_config(max_len: int = MAX_SEQUENCE_LENGTH) -> ModelConfig:
    return ModelConfig(
        hidden=16,
        num_heads=2,
        num_layers=1,
        max_len=max_len,
    )


def _tiny_demo() -> DemoConfig:
    return DemoConfig(
        warmup_epochs=2,
        steps=2,
        hidden=16,
        num_heads=2,
        num_layers=1,
    )


def test_tokenizer_encodes_prefix_completion_and_round_trips() -> None:
    tokenizer = InstructionTokenizer()
    prefix = tokenizer.encode_prefix("What is 2 + 3?")
    assert prefix[0] == INST_ID
    assert prefix[-1] == RESP_ID
    assert tokenizer.decode(prefix[1:-1]) == "What is 2 + 3?"
    assert tokenizer.vocab_size == VOCAB_SIZE
    assert tokenizer.inst_id == INST_ID
    assert tokenizer.resp_id == RESP_ID
    assert tokenizer.pad_id == PAD_ID
    assert tokenizer.max_len == MAX_SEQUENCE_LENGTH
    assert BYTE_VOCAB_SIZE == 256

    ids, completion_start = tokenizer.encode_completion("Q", "café")
    assert ids[:completion_start] == tokenizer.encode_prefix("Q")
    assert tokenizer.decode(ids[completion_start:]) == "café"
    assert completion_start == len(tokenizer.encode_prefix("Q"))


def test_tokenizer_rejects_invalid_input() -> None:
    tokenizer = InstructionTokenizer(max_len=8)
    with pytest.raises(ValueError):
        tokenizer.encode_prefix("x" * 10)
    with pytest.raises(ValueError):
        tokenizer.encode_completion("Q", "")
    with pytest.raises(TypeError):
        tokenizer.encode_prefix(cast(str, 3))
    with pytest.raises(ValueError):
        tokenizer.encode_completion("Q", "abcdefgh")
    with pytest.raises(ValueError):
        tokenizer.decode([INST_ID])
    with pytest.raises(ValueError):
        tokenizer.decode([-1])
    with pytest.raises(TypeError):
        tokenizer.decode([True])
    with pytest.raises(TypeError):
        tokenizer.decode(cast(Sequence[int], "x"))


def test_make_preferences_contract() -> None:
    preferences = make_preferences()
    assert len(preferences) == 12
    assert preferences == make_preferences()
    counts = {
        category: sum(
            1 for preference in preferences if preference.category == category
        )
        for category in CATEGORY_ORDER
    }
    assert counts == {"capital": 4, "arithmetic": 4, "list": 4}
    assert len({preference.prompt for preference in preferences}) == 12
    tokenizer = InstructionTokenizer()
    for preference in preferences:
        assert preference.chosen != preference.rejected
        assert len(preference.chosen.encode("utf-8")) < len(
            preference.rejected.encode("utf-8")
        )
        chosen_ids, completion_start = tokenizer.encode_completion(
            preference.prompt, preference.chosen
        )
        rejected_ids, _ = tokenizer.encode_completion(
            preference.prompt, preference.rejected
        )
        assert 0 < completion_start < len(chosen_ids) <= MAX_SEQUENCE_LENGTH
        assert len(rejected_ids) <= MAX_SEQUENCE_LENGTH


def test_preference_validation() -> None:
    with pytest.raises(ValueError):
        Preference(
            prompt="Q",
            chosen="a",
            rejected="b",
            category="unknown",
        )
    with pytest.raises(ValueError):
        Preference(
            prompt=" ",
            chosen="a",
            rejected="b",
            category="capital",
        )
    with pytest.raises(ValueError):
        Preference(
            prompt="Q",
            chosen="same",
            rejected="same",
            category="list",
        )
    with pytest.raises(TypeError):
        Preference(
            prompt=cast(str, 1),
            chosen="a",
            rejected="b",
            category="capital",
        )


def test_tiny_gpt_shape_tied_head_and_causality() -> None:
    config = ModelConfig(
        hidden=16,
        num_heads=2,
        num_layers=1,
        max_len=16,
    )
    model = TinyGPT(config, seed=0)
    assert model.head.weight is model.token_embedding.weight
    assert model.context_length == 16
    assert model.pad_id == PAD_ID
    assert count_parameters(model) > 0

    token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    logits = model(token_ids)
    assert logits.shape == (2, 4, VOCAB_SIZE)
    changed_future = torch.tensor([[1, 2, 9, 9], [5, 6, 9, 9]])
    assert torch.allclose(
        logits[:, :2], model(changed_future)[:, :2], atol=1e-6
    )

    with pytest.raises(ValueError):
        ModelConfig(hidden=15, num_heads=2)
    with pytest.raises(ValueError):
        model(torch.tensor([[-1]]))
    with pytest.raises(ValueError):
        model(torch.tensor([[VOCAB_SIZE]]))
    with pytest.raises(TypeError):
        model(torch.tensor([[1.0]]))


def test_sequence_log_prob_matches_manual_completion_sum() -> None:
    tokenizer = InstructionTokenizer()
    model = TinyGPT(_tiny_config(), seed=3)
    model.eval()
    prompt = "What is 2 + 3?"
    completion = "5."
    ids, completion_start = tokenizer.encode_completion(prompt, completion)
    logits = model(torch.tensor([ids]))
    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    expected = torch.stack(
        [log_probs[position - 1, ids[position]] for position in range(
            completion_start, len(ids)
        )]
    ).sum()
    actual = sequence_log_prob(model, tokenizer, prompt, completion)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert float(actual.item()) < 0.0


def test_sequence_log_prob_backpropagates_and_rejects_bad_input() -> None:
    tokenizer = InstructionTokenizer()
    model = TinyGPT(_tiny_config(), seed=4)
    value = sequence_log_prob(model, tokenizer, "Q", "Paris.")
    assert value.requires_grad is True
    torch.autograd.backward(value)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert any(float(gradient.abs().sum().item()) > 0.0 for gradient in gradients)

    with pytest.raises(ValueError):
        sequence_log_prob(model, tokenizer, "Q", "")
    with pytest.raises(ValueError):
        sequence_log_prob(
            model, InstructionTokenizer(max_len=8), "Q", "abcdefgh"
        )


def test_dpo_loss_matches_manual_formula() -> None:
    policy_chosen = torch.tensor([-1.0, -2.0])
    policy_rejected = torch.tensor([-3.0, -4.0])
    reference_chosen = torch.tensor([-1.5, -2.5])
    reference_rejected = torch.tensor([-2.5, -3.5])
    losses, reward_delta = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=0.1,
    )
    expected_delta = 0.1 * (
        (policy_chosen - reference_chosen)
        - (policy_rejected - reference_rejected)
    )
    assert torch.allclose(reward_delta, expected_delta)
    assert torch.allclose(losses, -F.logsigmoid(expected_delta))

    zero_loss, zero_delta = dpo_loss(
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        beta=0.1,
    )
    assert float(zero_delta.item()) == 0.0
    assert float(zero_loss.item()) == pytest.approx(math.log(2.0), abs=1e-7)


def test_dpo_loss_gradient_signs_and_reference_detachment() -> None:
    policy_chosen = torch.tensor([-1.0], requires_grad=True)
    policy_rejected = torch.tensor([-2.0], requires_grad=True)
    reference_chosen = torch.tensor([-1.5])
    reference_rejected = torch.tensor([-1.0])
    losses, reward_delta = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=0.1,
    )
    torch.autograd.backward(losses.sum())
    assert policy_chosen.grad is not None
    assert policy_rejected.grad is not None
    expected_slope = 0.1 * (1.0 - torch.sigmoid(reward_delta.detach()))
    assert torch.allclose(
        policy_chosen.grad, -expected_slope, atol=1e-6
    )
    assert torch.allclose(policy_rejected.grad, expected_slope, atol=1e-6)
    assert float(policy_chosen.grad.item()) < 0.0
    assert float(policy_rejected.grad.item()) > 0.0
    assert reference_chosen.grad is None
    assert reference_rejected.grad is None
    assert reward_delta.shape == (1,)


def test_dpo_loss_numerical_stability_and_validation() -> None:
    high_loss, high_delta = dpo_loss(
        torch.tensor([1000.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        beta=1.0,
    )
    low_loss, low_delta = dpo_loss(
        torch.tensor([-1000.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        beta=1.0,
    )
    assert math.isfinite(float(high_loss.item()))
    assert math.isfinite(float(low_loss.item()))
    assert float(high_delta.item()) == 1000.0
    assert float(low_delta.item()) == -1000.0

    zero = torch.zeros(1)
    with pytest.raises(ValueError):
        dpo_loss(torch.zeros(1), torch.zeros(2), zero, zero)
    with pytest.raises(ValueError):
        dpo_loss(torch.zeros(0), zero, zero, zero)
    with pytest.raises(ValueError):
        dpo_loss(torch.zeros(1, 1), zero, zero, zero)
    with pytest.raises(ValueError):
        dpo_loss(zero, zero, zero, zero, beta=0.0)
    with pytest.raises(TypeError):
        dpo_loss(zero, zero, zero, zero, beta=cast(float, True))
    with pytest.raises(ValueError):
        dpo_loss(zero, zero, zero, zero, beta=-0.1)


def test_warmup_pretrain_reduces_loss_and_raises_chosen_logprob() -> None:
    tokenizer = InstructionTokenizer()
    preferences = make_preferences()
    config = ModelConfig(
        hidden=32,
        num_heads=2,
        num_layers=1,
        max_len=MAX_SEQUENCE_LENGTH,
    )
    model = TinyGPT(config, seed=0)
    with torch.no_grad():
        before = torch.stack(
            [
                sequence_log_prob(
                    model, tokenizer, preference.prompt, preference.chosen
                )
                for preference in preferences
            ]
        ).mean()
    losses = warmup_pretrain(
        model,
        tokenizer,
        preferences,
        epochs=20,
    )
    with torch.no_grad():
        after = torch.stack(
            [
                sequence_log_prob(
                    model, tokenizer, preference.prompt, preference.chosen
                )
                for preference in preferences
            ]
        ).mean()
    assert len(losses) == 20
    assert all(math.isfinite(loss) for loss in losses)
    assert losses[-1] < losses[0]
    assert float(after.item()) > float(before.item())

    with pytest.raises(ValueError):
        warmup_pretrain(model, tokenizer, (), epochs=1)
    with pytest.raises(ValueError):
        warmup_pretrain(model, tokenizer, preferences, epochs=0)


def test_build_reference_and_policy_copies_and_freezes() -> None:
    tokenizer = InstructionTokenizer()
    preferences = make_preferences()
    pair = build_reference_and_policy(
        _tiny_config(),
        tokenizer,
        preferences,
        seed=0,
        warmup_epochs=2,
    )
    assert len(pair.warmup_losses) == 2
    assert pair.reference.training is False
    assert pair.policy.training is True
    assert all(
        not parameter.requires_grad
        for parameter in pair.reference.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in pair.policy.parameters()
    )
    reference_state = pair.reference.state_dict()
    policy_state = pair.policy.state_dict()
    assert reference_state.keys() == policy_state.keys()
    for key in reference_state:
        assert torch.equal(reference_state[key], policy_state[key])


def test_reference_log_probs_stay_fixed_during_dpo() -> None:
    tokenizer = InstructionTokenizer()
    preferences = make_preferences()
    pair = build_reference_and_policy(
        _tiny_config(),
        tokenizer,
        preferences,
        seed=0,
        warmup_epochs=2,
    )
    policy_before = [
        parameter.detach().clone()
        for parameter in pair.policy.parameters()
    ]
    with torch.no_grad():
        reference_before = [
            sequence_log_prob(
                pair.reference, tokenizer, preference.prompt, preference.chosen
            )
            for preference in preferences
        ]
        reference_before_rejected = [
            sequence_log_prob(
                pair.reference, tokenizer, preference.prompt, preference.rejected
            )
            for preference in preferences
        ]

    report = train_dpo(
        pair.policy,
        pair.reference,
        tokenizer,
        preferences,
        beta=0.1,
        lr=1e-4,
        steps=2,
    )
    assert report.losses[0] == pytest.approx(math.log(2.0), abs=1e-6)
    assert report.losses[-1] < report.losses[0]
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            policy_before, pair.policy.parameters(), strict=True
        )
    )
    with torch.no_grad():
        reference_after = [
            sequence_log_prob(
                pair.reference, tokenizer, preference.prompt, preference.chosen
            )
            for preference in preferences
        ]
        reference_after_rejected = [
            sequence_log_prob(
                pair.reference, tokenizer, preference.prompt, preference.rejected
            )
            for preference in preferences
        ]
    assert all(
        torch.allclose(before, after, atol=1e-6)
        for before, after in zip(
            reference_before, reference_after, strict=True
        )
    )
    assert all(
        torch.allclose(before, after, atol=1e-6)
        for before, after in zip(
            reference_before_rejected, reference_after_rejected, strict=True
        )
    )


def test_evaluate_margins_matches_manual_and_restores_mode() -> None:
    tokenizer = InstructionTokenizer()
    preferences = make_preferences()
    model = TinyGPT(_tiny_config(), seed=0)
    model.train()
    with torch.no_grad():
        expected = (
            torch.stack(
                [
                    sequence_log_prob(
                        model,
                        tokenizer,
                        preference.prompt,
                        preference.chosen,
                    )
                    for preference in preferences
                ]
            )
            - torch.stack(
                [
                    sequence_log_prob(
                        model,
                        tokenizer,
                        preference.prompt,
                        preference.rejected,
                    )
                    for preference in preferences
                ]
            )
        ).mean()
    actual = evaluate_margins(model, tokenizer, preferences)
    assert actual == pytest.approx(float(expected.item()))
    assert model.training is True

    model.eval()
    evaluate_margins(model, tokenizer, preferences)
    assert model.training is False
    with pytest.raises(ValueError):
        evaluate_margins(model, tokenizer, ())


def test_default_demo_trains_dpo_and_self_checks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = run_demo()
    captured = capsys.readouterr()
    assert report.success is True
    assert report.reference_unchanged is True
    assert report.loss_decreased is True
    assert report.margin_increased is True
    assert report.dpo.losses[-1] < report.dpo.losses[0]
    assert report.final_margin > report.initial_margin
    assert report.dpo.losses[0] == pytest.approx(math.log(2.0), abs=1e-6)
    assert report.parameters > 0
    assert "step   1: loss=0.69315" in captured.out
    assert "step  30:" in captured.out
    assert "final:" in captured.out
    assert (
        "check: ref_unchanged=true loss_decreased=true "
        "margin_increased=true"
    ) in captured.out
    assert "error:" not in captured.err


def test_demo_cli_contract_and_determinism(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = [
        "--warmup-epochs",
        "2",
        "--steps",
        "2",
        "--hidden",
        "16",
        "--num-heads",
        "2",
        "--num-layers",
        "1",
    ]
    assert main(args) == 0
    first = capsys.readouterr()
    assert "step   1:" in first.out
    assert "step   2:" in first.out
    assert "final:" in first.out
    assert "check:" in first.out
    assert "Traceback" not in first.err

    tiny = _tiny_demo()
    first_report = run_demo(tiny)
    first_output = capsys.readouterr().out
    second_report = run_demo(tiny)
    second_output = capsys.readouterr().out
    assert first_output == second_output
    assert first_report.dpo.losses == second_report.dpo.losses


@pytest.mark.parametrize(
    "args",
    [
        ["--beta", "0"],
        ["--steps", "0"],
        ["--warmup-epochs", "0"],
        ["--lr", "0"],
        ["--hidden", "15"],
        ["--num-heads", "0"],
        ["--num-layers", "0"],
        ["--seed", "-1"],
    ],
)
def test_cli_rejects_invalid_configuration(
    args: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(args) == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
