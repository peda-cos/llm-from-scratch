from __future__ import annotations

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
    ModelConfig,
    TinyGPT,
    count_parameters,
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
