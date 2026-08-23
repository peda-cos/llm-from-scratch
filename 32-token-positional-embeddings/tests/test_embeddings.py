from __future__ import annotations

import math
from typing import Any, cast

import pytest
import torch
from torch import Tensor

from main import (
    EmbeddingComposer,
    LearnedPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TokenEmbedding,
    main,
    neighbour_cosine,
    parameter_count,
)


def reference_sinusoidal(length: int, dim: int) -> Tensor:
    """Float64 reference for the spec formula, computed feature by feature."""
    table = torch.empty((length, dim), dtype=torch.float64)
    for position in range(length):
        for feature in range(dim):
            angle = position / (10000.0 ** (2 * (feature // 2) / dim))
            table[position, feature] = (
                math.sin(angle) if feature % 2 == 0 else math.cos(angle)
            )
    return table


def test_token_embedding_shape_and_dtype() -> None:
    token = TokenEmbedding(10, 4, seed=0)
    ids = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    output = token(ids)
    assert output.shape == (2, 3, 4)
    assert output.dtype == torch.float32
    assert output.requires_grad


def test_token_embedding_parameter_count() -> None:
    token = TokenEmbedding(10, 4, seed=0)
    assert parameter_count(token) == 40
    assert token.embedding.weight.numel() == 40


def test_token_embedding_seeded_init_is_reproducible() -> None:
    first = TokenEmbedding(128, 64, seed=11)
    replay = TokenEmbedding(128, 64, seed=11)
    other = TokenEmbedding(128, 64, seed=12)
    assert torch.equal(first.embedding.weight, replay.embedding.weight)
    assert not torch.equal(first.embedding.weight, other.embedding.weight)

    weights = first.embedding.weight.detach()
    assert 0.018 <= float(weights.std()) <= 0.022
    assert abs(float(weights.mean())) <= 0.002


def test_token_embedding_gradients_only_reach_used_rows() -> None:
    token = TokenEmbedding(5, 3, seed=0)
    token(torch.tensor([[0, 2, 2]], dtype=torch.long)).sum().backward()

    grad = token.embedding.weight.grad
    assert grad is not None
    for unused in (1, 3, 4):
        assert float(grad[unused].abs().sum()) == 0.0
    assert torch.allclose(grad[2], 2 * grad[0])


def test_token_embedding_rejects_bad_ids() -> None:
    token = TokenEmbedding(10, 4, seed=0)
    with pytest.raises(ValueError, match="shape"):
        token(torch.tensor([1, 2], dtype=torch.long))
    with pytest.raises(ValueError, match="shape"):
        token(torch.zeros((1, 2, 3), dtype=torch.long))
    with pytest.raises(TypeError, match="int32 or int64"):
        token(torch.tensor([[1.0, 2.0]]))
    with pytest.raises(IndexError):
        token(torch.tensor([[10]], dtype=torch.long))
    with pytest.raises(IndexError):
        token(torch.tensor([[-1]], dtype=torch.long))

    ids32 = torch.tensor([[0, 2]], dtype=torch.int32)
    assert torch.equal(token(ids32), token(ids32.long()))


def test_learned_positional_shape_and_parameters() -> None:
    positional = LearnedPositionalEmbedding(6, 8, seed=1)
    output = positional(3)
    assert output.shape == (1, 3, 8)
    assert torch.equal(output[0], positional.embedding.weight[:3])
    assert parameter_count(positional) == 48


def test_learned_positional_rejects_out_of_range() -> None:
    positional = LearnedPositionalEmbedding(6, 4, seed=1)
    for length in (0, -1, 7):
        with pytest.raises(ValueError, match="seq_len"):
            positional(length)
    with pytest.raises(TypeError, match="seq_len"):
        positional(cast(Any, 2.0))


def test_learned_positional_seeded_init_is_reproducible() -> None:
    first = LearnedPositionalEmbedding(6, 8, seed=7)
    replay = LearnedPositionalEmbedding(6, 8, seed=7)
    other = LearnedPositionalEmbedding(6, 8, seed=8)
    assert torch.equal(first.embedding.weight, replay.embedding.weight)
    assert not torch.equal(first.embedding.weight, other.embedding.weight)
    assert first.embedding.weight.requires_grad


def test_sinusoidal_is_parameter_free_buffer() -> None:
    positional = SinusoidalPositionalEmbedding(8, 6)
    buffers = dict(positional.named_buffers())
    assert "table" in buffers
    assert positional.table.shape == (8, 6)
    assert dict(positional.named_parameters()) == {}
    assert parameter_count(positional) == 0

    output = positional(3)
    assert output.shape == (1, 3, 6)
    assert not output.requires_grad


@pytest.mark.parametrize("dim", [1, 2, 5, 6, 8])
def test_sinusoidal_matches_formula(dim: int) -> None:
    positional = SinusoidalPositionalEmbedding(4, dim)
    expected = reference_sinusoidal(4, dim).to(torch.float32)
    assert torch.allclose(positional.table, expected, rtol=0, atol=1e-6)


def test_sinusoidal_rows_are_distinct() -> None:
    positional = SinusoidalPositionalEmbedding(64, 16)
    assert torch.unique(positional.table, dim=0).shape[0] == 64


def test_sinusoidal_pairs_have_constant_norm() -> None:
    positional = SinusoidalPositionalEmbedding(32, 64)
    norms = positional.table.norm(dim=1)
    expected = math.sqrt(32.0)
    assert torch.allclose(norms, torch.full_like(norms, expected), atol=1e-5)


def test_sinusoidal_wavelengths() -> None:
    positional = SinusoidalPositionalEmbedding(8, 64)
    wavelengths = positional.wavelengths()
    assert wavelengths.shape == (32,)
    assert float(wavelengths[0]) == pytest.approx(2 * math.pi, rel=1e-6)

    ratios = wavelengths[1:] / wavelengths[:-1]
    expected_ratio = 10000.0 ** (2 / 64)
    assert torch.allclose(
        ratios, torch.full_like(ratios, expected_ratio), rtol=1e-5
    )
    assert bool((wavelengths[1:] > wavelengths[:-1]).all())

    narrower = SinusoidalPositionalEmbedding(8, 32).wavelengths()
    assert narrower.shape == (16,)
    assert float(narrower[1]) > float(wavelengths[1])


def test_sinusoidal_rejects_out_of_range() -> None:
    positional = SinusoidalPositionalEmbedding(6, 4)
    for length in (0, 7):
        with pytest.raises(ValueError, match="seq_len"):
            positional(length)


def test_neighbour_cosine_known_rows() -> None:
    rows = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert torch.allclose(neighbour_cosine(rows), torch.tensor([1.0, 0.0]))
    assert torch.allclose(neighbour_cosine(rows, 2), torch.tensor([0.0]))

    with pytest.raises(ValueError, match="gap"):
        neighbour_cosine(rows, 0)
    with pytest.raises(ValueError, match="rows"):
        neighbour_cosine(rows, 3)
    with pytest.raises(ValueError, match="two-dimensional"):
        neighbour_cosine(torch.ones(3), 1)
    with pytest.raises(TypeError, match="gap"):
        neighbour_cosine(rows, cast(Any, 1.0))


def test_sinusoidal_neighbour_cosine_is_position_invariant() -> None:
    positional = SinusoidalPositionalEmbedding(16, 64)
    rows = positional.table[:16]
    gap1 = neighbour_cosine(rows, 1)
    gap2 = neighbour_cosine(rows, 2)
    assert float(gap1.std()) < 1e-5
    assert float(gap1.mean()) > 0.95
    assert float(gap2.mean()) < float(gap1.mean())


def test_learned_init_neighbour_cosine_is_near_random() -> None:
    positional = LearnedPositionalEmbedding(16, 64, seed=1235)
    cosine = neighbour_cosine(positional.embedding.weight.detach(), 1)
    assert abs(float(cosine.mean())) < 0.25
    assert float(cosine.std()) > 0.05


def test_composer_shape_sum_and_parameter_count() -> None:
    token = TokenEmbedding(10, 8, seed=1)
    positional = LearnedPositionalEmbedding(6, 8, seed=2)
    composer = EmbeddingComposer(token, positional)
    ids = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]], dtype=torch.long
    )

    output = composer(ids)
    assert output.shape == (3, 4, 8)
    assert torch.allclose(output, token(ids) + positional(4))
    assert parameter_count(composer) == parameter_count(token) + parameter_count(
        positional
    )


def test_composer_broadcasts_positions_across_batch() -> None:
    token = TokenEmbedding(10, 8, seed=1)
    positional = LearnedPositionalEmbedding(6, 8, seed=2)
    composer = EmbeddingComposer(token, positional)
    ids = torch.tensor(
        [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]], dtype=torch.long
    )

    output = composer(ids)
    tokens = token(ids)
    delta = output - tokens
    assert torch.allclose(delta, positional(4).expand_as(delta))
    assert torch.allclose(delta[0], delta[1])
    assert torch.allclose(delta[0], delta[2])


def test_composer_length_contrast() -> None:
    token = TokenEmbedding(10, 8, seed=1)
    ids = torch.zeros((1, 17), dtype=torch.long)

    learned = LearnedPositionalEmbedding(16, 8, seed=2)
    with pytest.raises(ValueError, match="max_context_length"):
        EmbeddingComposer(token, learned)(ids)

    sinusoidal = SinusoidalPositionalEmbedding(64, 8)
    output = EmbeddingComposer(token, sinusoidal)(ids)
    assert output.shape == (1, 17, 8)
    expected = reference_sinusoidal(17, 8).to(torch.float32)
    assert torch.allclose(sinusoidal.table[16], expected[16], atol=1e-6)


def test_composer_rejects_dim_mismatch_and_overlong_context() -> None:
    token = TokenEmbedding(10, 8, seed=1)
    wrong_dim = LearnedPositionalEmbedding(6, 4, seed=2)
    with pytest.raises(ValueError, match="dim mismatch"):
        EmbeddingComposer(token, wrong_dim)

    positional = LearnedPositionalEmbedding(4, 8, seed=3)
    composer = EmbeddingComposer(token, positional)
    with pytest.raises(ValueError, match="max_context_length"):
        composer(torch.zeros((1, 5), dtype=torch.long))


def test_demo_smoke(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(
        [
            "demo",
            "--vocab-size",
            "16",
            "--dim",
            "8",
            "--context-length",
            "4",
            "--batch-size",
            "2",
            "--seed",
            "3",
        ]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    for expected in (
        "parameter_counts:",
        "composition_sum=ok",
        "batch_broadcast=ok",
        "learned_positional: rejected at T+1",
        "sinusoidal_positional: accepted at T+1",
    ):
        assert expected in output


def test_demo_rejects_invalid_config(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["demo", "--vocab-size", "0"]) == 2
    error = capsys.readouterr().err
    assert error.startswith("error:")
