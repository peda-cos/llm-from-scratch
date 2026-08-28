from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from main import GPTConfig, GPTModel, generate, sample_next_token


class FixedLogits:
    def __init__(self, context_length: int, logits: Tensor) -> None:
        self._context_length = context_length
        self._logits = logits
        self.input_lengths: list[int] = []

    @property
    def context_length(self) -> int:
        return self._context_length

    def __call__(self, idx: Tensor) -> Tensor:
        self.input_lengths.append(idx.shape[1])
        return self._logits.expand(idx.shape[0], idx.shape[1], -1)


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        context_length=8,
        d_model=16,
        num_heads=4,
        num_layers=2,
        mlp_expansion=4,
        dropout=0.0,
        use_bias=True,
        weight_tying=True,
    )


def _tiny_model() -> GPTModel:
    model = GPTModel(_tiny_config(), seed=0)
    model.eval()
    return model


def test_greedy_matches_manual_argmax() -> None:
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generated = generate(
        model, prompt, max_new_tokens=4, greedy=True
    )

    assert generated.shape == (1, 7)
    assert torch.equal(generated[:, :3], prompt)

    context = prompt
    for step in range(4):
        with torch.no_grad():
            expected = model(context)[:, -1, :].argmax(
                dim=-1, keepdim=True
            )
        position = prompt.shape[1] + step
        assert torch.equal(
            generated[:, position : position + 1], expected
        )
        context = generated[:, : position + 1]


def test_temperature_zero_and_top_k_one_match_greedy() -> None:
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    greedy = generate(
        model, prompt, max_new_tokens=4, greedy=True
    )
    zero = generate(
        model, prompt, max_new_tokens=4, temperature=0.0
    )
    top_one = generate(
        model, prompt, max_new_tokens=4, temperature=1.0, top_k=1
    )

    assert torch.equal(greedy, zero)
    assert torch.equal(greedy, top_one)


def test_top_k_and_temperature_change_sampling() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    generator = torch.Generator().manual_seed(0)
    top_two = torch.cat(
        [
            sample_next_token(
                logits, temperature=1.0, top_k=2, generator=generator
            )
            for _ in range(100)
        ],
        dim=1,
    )
    assert set(top_two.flatten().tolist()) <= {3, 4}

    generator = torch.Generator().manual_seed(1)
    top_one = torch.cat(
        [
            sample_next_token(
                logits, temperature=1.0, top_k=1, generator=generator
            )
            for _ in range(100)
        ],
        dim=1,
    )
    assert bool((top_one == 4).all())

    sharp = torch.tensor([[2.0, 0.0]])
    generator = torch.Generator().manual_seed(2)
    low_temperature = torch.cat(
        [
            sample_next_token(
                sharp, temperature=0.01, generator=generator
            )
            for _ in range(100)
        ],
        dim=1,
    )
    assert bool((low_temperature == 0).all())

    generator = torch.Generator().manual_seed(3)
    high_temperature = torch.cat(
        [
            sample_next_token(
                sharp, temperature=10.0, generator=generator
            )
            for _ in range(100)
        ],
        dim=1,
    )
    assert set(high_temperature.flatten().tolist()) == {0, 1}


def test_sliding_window_crops_context() -> None:
    fake = FixedLogits(4, torch.tensor([0.0, 1.0, 2.0]))
    prompt = torch.tensor([[0, 1, 2]], dtype=torch.long)
    generated = generate(
        fake, prompt, max_new_tokens=4, greedy=True
    )

    assert fake.input_lengths == [3, 4, 4, 4]
    assert generated.shape == (1, 7)

    long_fake = FixedLogits(4, torch.tensor([0.0, 1.0, 2.0]))
    long_prompt = torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long)
    long_generated = generate(
        long_fake, long_prompt, max_new_tokens=1, greedy=True
    )
    assert long_fake.input_lengths == [4]
    assert long_generated.shape == (1, 7)


def test_generate_restores_mode_and_is_reproducible() -> None:
    config = replace(_tiny_config(), dropout=0.5)
    model = GPTModel(config, seed=0)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
    model.train()

    first = generate(
        model,
        prompt,
        max_new_tokens=4,
        temperature=1.0,
        top_k=5,
        generator=torch.Generator().manual_seed(11),
    )
    second = generate(
        model,
        prompt,
        max_new_tokens=4,
        temperature=1.0,
        top_k=5,
        generator=torch.Generator().manual_seed(11),
    )
    assert torch.equal(first, second)
    assert model.training

    model.eval()
    generate(model, prompt, max_new_tokens=1, greedy=True)
    assert not model.training


def test_generation_validates_inputs() -> None:
    model = _tiny_model()
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match="max_new_tokens"):
        generate(model, prompt, max_new_tokens=-1)
    with pytest.raises(ValueError, match="temperature"):
        generate(model, prompt, max_new_tokens=1, temperature=-1.0)
    with pytest.raises(ValueError, match="temperature"):
        generate(
            model,
            prompt,
            max_new_tokens=1,
            temperature=float("nan"),
        )
    with pytest.raises(ValueError, match="top_k"):
        generate(model, prompt, max_new_tokens=1, top_k=0)
    with pytest.raises(ValueError, match="shape"):
        generate(
            model,
            torch.tensor([1, 2, 3], dtype=torch.long),
            max_new_tokens=1,
        )
    with pytest.raises(TypeError, match="integer"):
        generate(
            model,
            torch.ones((1, 3), dtype=torch.float32),
            max_new_tokens=1,
        )
    with pytest.raises(ValueError, match="shape"):
        sample_next_token(torch.ones(5))
    with pytest.raises(ValueError, match="finite"):
        sample_next_token(torch.tensor([[float("nan"), 1.0]]))
    with pytest.raises(TypeError, match="top_k"):
        sample_next_token(
            torch.ones((1, 3)), top_k=cast(Any, True)
        )

    empty = generate(model, prompt, max_new_tokens=0)
    assert torch.equal(empty, prompt)
    assert empty is not prompt
