from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from main import (
    GPTConfig,
    GPTModel,
    count_parameters,
    generate,
    generate_and_print_sample,
)


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=16,
        context_length=8,
        d_model=16,
        num_heads=4,
        num_layers=2,
        mlp_expansion=4,
        dropout=0.0,
    )


def _expected_parameters(config: GPTConfig) -> int:
    per_block = (
        (4 + 2 * config.mlp_expansion) * config.d_model**2
        + (config.mlp_expansion + 9) * config.d_model
    )
    return (
        config.vocab_size * config.d_model
        + config.context_length * config.d_model
        + 2 * config.d_model
        + config.num_layers * per_block
    )


def test_forward_shape_causality_and_tied_storage() -> None:
    config = _tiny_config()
    model = GPTModel(config, seed=3)
    model.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    with torch.no_grad():
        logits = model(ids)

    assert logits.shape == (1, 5, config.vocab_size)
    assert model.lm_head.weight is model.token_embedding.weight
    assert count_parameters(model) == _expected_parameters(config)

    changed = ids.clone()
    changed[0, -1] = 2
    with torch.no_grad():
        changed_logits = model(changed)
    assert torch.equal(logits[:, :-1], changed_logits[:, :-1])
    assert not torch.equal(logits[:, -1], changed_logits[:, -1])

    with pytest.raises(ValueError, match="context_length"):
        model(torch.ones(1, config.context_length + 1, dtype=torch.long))
    with pytest.raises(TypeError, match="integer"):
        model(torch.ones(1, 2))


def test_generate_greedy_is_deterministic_and_restores_mode() -> None:
    model = GPTModel(_tiny_config(), seed=5)
    model.train()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    first = generate(model, prompt, max_new_tokens=4)
    second = generate(model, prompt, max_new_tokens=4)

    assert first.shape == (1, 7)
    assert torch.equal(first[:, :3], prompt)
    assert torch.equal(first, second)
    assert bool(((first >= 0) & (first < model.config.vocab_size)).all())
    assert model.training
    assert generate(model, prompt, max_new_tokens=0).shape == prompt.shape


def test_generate_sampling_is_reproducible_and_validates() -> None:
    model = GPTModel(_tiny_config(), seed=9)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    first = generate(
        model,
        prompt,
        max_new_tokens=3,
        temperature=0.8,
        top_k=5,
        generator=torch.Generator().manual_seed(7),
    )
    second = generate(
        model,
        prompt,
        max_new_tokens=3,
        temperature=0.8,
        top_k=5,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(first, second)

    with pytest.raises(ValueError, match="idx"):
        generate(model, torch.tensor([1, 2]), max_new_tokens=1)
    with pytest.raises(TypeError, match="integer"):
        generate(model, prompt.float(), max_new_tokens=1)
    with pytest.raises(ValueError, match="max_new_tokens"):
        generate(model, prompt, max_new_tokens=-1)
    with pytest.raises(ValueError, match="temperature"):
        generate(model, prompt, max_new_tokens=1, temperature=-0.1)
    with pytest.raises(ValueError, match="top_k"):
        generate(model, prompt, max_new_tokens=1, top_k=0)

    class NoContext(nn.Module):
        def forward(self, x: Tensor) -> Tensor:
            return torch.zeros((x.shape[0], x.shape[1], 4))

    with pytest.raises(TypeError, match="context_length"):
        generate(NoContext(), prompt, max_new_tokens=1)


def test_sample_printer_emits_generated_ids(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = GPTModel(_tiny_config(), seed=1)
    model.train()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generated = generate_and_print_sample(
        model, prompt, 3, prefix="sample step=004"
    )
    captured = capsys.readouterr()

    assert generated.shape == (1, 6)
    assert "sample step=004 ids=[" in captured.out
    assert captured.err == ""
    assert model.training
