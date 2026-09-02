from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from main import GPTConfig, GPTModel, count_parameters, generate, sample_next_token


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=64,
        context_length=16,
        d_model=16,
        num_heads=2,
        num_layers=2,
        mlp_expansion=2,
        dropout=0.0,
        use_bias=True,
        weight_tying=True,
    )


def _expected_parameters(config: GPTConfig, *, tied: bool) -> int:
    per_block = (
        (4 + 2 * config.mlp_expansion) * config.d_model**2
        + 4 * config.d_model
    )
    if config.use_bias:
        per_block += (config.mlp_expansion + 5) * config.d_model
    total = (
        config.vocab_size * config.d_model
        + config.context_length * config.d_model
        + 2 * config.d_model
        + config.num_layers * per_block
    )
    if not tied:
        total += config.vocab_size * config.d_model
    return total


def test_real_config_parameter_count_matches_gpt2_small() -> None:
    with torch.device("meta"):
        model = GPTModel(GPTConfig())
    assert count_parameters(model) == 124_439_808


def test_tiny_parameter_count_matches_formula() -> None:
    config = _tiny_config()
    tied = GPTModel(config, seed=0)
    untied = GPTModel(replace(config, weight_tying=False), seed=0)

    assert count_parameters(tied) == _expected_parameters(config, tied=True)
    assert count_parameters(untied) == _expected_parameters(
        config, tied=False
    )
    assert count_parameters(untied) - count_parameters(tied) == 64 * 16


def test_local_parameter_names_follow_lesson_table() -> None:
    names = set(dict(GPTModel(_tiny_config(), seed=0).named_parameters()))

    assert "tok_embed.weight" in names
    assert "pos_embed.weight" in names
    assert "blocks.0.ln1.scale" in names
    assert "blocks.0.ln1.shift" in names
    assert "blocks.0.attn.qkv.weight" in names
    assert "blocks.0.attn.out_proj.weight" in names
    assert "blocks.0.mlp.fc1.weight" in names
    assert "blocks.0.mlp.fc2.weight" in names
    assert "final_ln.scale" in names
    assert "final_ln.shift" in names
    assert "token_embedding.weight" not in names
    assert "lm_head.weight" not in names


def test_weight_tying_shares_parameter_object() -> None:
    model = GPTModel(_tiny_config(), seed=0)
    assert model.lm_head.weight is model.tok_embed.weight
    assert model.lm_head.weight.data_ptr() == model.tok_embed.weight.data_ptr()


def test_generate_preserves_prompt_and_is_deterministic() -> None:
    model = GPTModel(_tiny_config(), seed=3)
    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    def run() -> torch.Tensor:
        generator = torch.Generator().manual_seed(11)
        return generate(
            model,
            prompt,
            max_new_tokens=4,
            temperature=1.0,
            top_k=8,
            generator=generator,
        )

    first = run()
    second = run()
    assert torch.equal(first, second)
    assert first.shape == (1, 8)
    assert torch.equal(first[:, :4], prompt)
    assert first[0, 4:].tolist() == second[0, 4:].tolist()


def test_generate_slides_context_and_handles_zero_tokens() -> None:
    model = GPTModel(_tiny_config(), seed=3)
    prompt = torch.tensor([list(range(1, 17))], dtype=torch.long)
    out = generate(model, prompt, max_new_tokens=2, greedy=True)
    assert out.shape == (1, 18)
    assert torch.equal(out[:, :16], prompt)

    clone = generate(model, prompt, max_new_tokens=0)
    assert torch.equal(clone, prompt)
    assert clone is not prompt


def test_generate_rejects_invalid_arguments() -> None:
    model = GPTModel(_tiny_config(), seed=3)
    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate(model, prompt, max_new_tokens=-1)
    with pytest.raises(ValueError, match="top_k"):
        generate(model, prompt, max_new_tokens=1, top_k=0)
    with pytest.raises(ValueError, match="temperature"):
        generate(model, prompt, max_new_tokens=1, temperature=-0.1)
    with pytest.raises(ValueError, match="idx"):
        generate(model, torch.tensor([1, 2]), max_new_tokens=1)


def test_sample_next_token_greedy_and_validation() -> None:
    logits = torch.tensor([[0.0, 2.0, 1.0]], dtype=torch.float32)
    assert sample_next_token(logits, greedy=True).tolist() == [[1]]
    with pytest.raises(ValueError, match="finite"):
        sample_next_token(torch.tensor([[0.0, math.nan]]), greedy=True)
    with pytest.raises(ValueError, match="shape"):
        sample_next_token(torch.tensor([0.0, 1.0]), greedy=True)
