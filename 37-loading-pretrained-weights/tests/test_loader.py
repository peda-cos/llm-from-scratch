from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from main import (
    GPTConfig,
    GPTModel,
    CheckpointLoadError,
    load_safetensors,
    make_stub_safetensors,
)


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=64,
        context_length=16,
        d_model=16,
        num_heads=2,
        num_layers=2,
        mlp_expansion=2,
        dropout=0.0,
    )


def _stub(tmp_path: Path, name: str = "stub.safetensors") -> Path:
    path = tmp_path / name
    make_stub_safetensors(path, _tiny_config(), seed=17)
    return path


def _snapshot(model: GPTModel) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _assert_unchanged(
    model: GPTModel, snapshot: dict[str, torch.Tensor]
) -> None:
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, snapshot[name]), name


def test_load_copies_and_transposes_every_tensor(tmp_path: Path) -> None:
    path = _stub(tmp_path)
    model = GPTModel(_tiny_config(), seed=23)
    report = load_safetensors(model, path)

    assert report.ok
    assert report.summary() == (
        "loaded=28 missing=0 unexpected=0 ignored=0 shape_mismatch=0"
    )
    assert len(report.loaded) == 28
    transposed = {
        entry.target for entry in report.loaded if entry.transposed
    }
    assert transposed == {
        "blocks.0.attn.qkv.weight",
        "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.fc1.weight",
        "blocks.0.mlp.fc2.weight",
        "blocks.1.attn.qkv.weight",
        "blocks.1.attn.out_proj.weight",
        "blocks.1.mlp.fc1.weight",
        "blocks.1.mlp.fc2.weight",
    }

    sources = load_file(str(path))
    parameters = dict(model.named_parameters())
    for entry in report.loaded:
        expected = sources[entry.source]
        if entry.transposed:
            expected = expected.t().contiguous()
        assert torch.equal(parameters[entry.target], expected), entry.target

    assert model.lm_head.weight is model.tok_embed.weight
    assert torch.equal(model.lm_head.weight, sources["wte.weight"])


def test_shape_mismatch_aborts_before_any_assignment(tmp_path: Path) -> None:
    path = tmp_path / "bad.safetensors"
    make_stub_safetensors(
        path, _tiny_config(), seed=17, corrupt_name="h.0.attn.c_attn.weight"
    )
    model = GPTModel(_tiny_config(), seed=23)
    before = _snapshot(model)

    with pytest.raises(CheckpointLoadError) as captured:
        load_safetensors(model, path)

    report = captured.value.report
    assert not report.ok
    assert report.loaded == []
    assert report.missing == []
    assert report.unexpected == []
    assert len(report.shape_mismatch) == 1
    mismatch = report.shape_mismatch[0]
    assert mismatch.source == "h.0.attn.c_attn.weight"
    assert mismatch.target == "blocks.0.attn.qkv.weight"
    assert mismatch.source_shape == (16, 24)
    assert mismatch.expected_source_shape == (16, 48)
    assert mismatch.target_shape == (48, 16)
    assert "checkpoint validation failed:" in str(captured.value)
    _assert_unchanged(model, before)


def test_missing_tensor_is_reported_and_not_assigned(tmp_path: Path) -> None:
    path = _stub(tmp_path)
    tensors = load_file(str(path))
    del tensors["h.0.mlp.c_proj.bias"]
    save_file(tensors, str(path), metadata={"format": "pt"})
    model = GPTModel(_tiny_config(), seed=23)
    before = _snapshot(model)

    with pytest.raises(CheckpointLoadError) as captured:
        load_safetensors(model, path)

    report = captured.value.report
    assert report.missing == ["blocks.0.mlp.fc2.bias"]
    assert report.loaded == []
    assert "missing: blocks.0.mlp.fc2.bias" in str(captured.value)
    _assert_unchanged(model, before)


def test_unexpected_tensor_is_strict(tmp_path: Path) -> None:
    path = _stub(tmp_path)
    tensors = load_file(str(path))
    tensors["mystery.weight"] = torch.zeros((2,), dtype=torch.float32)
    save_file(tensors, str(path), metadata={"format": "pt"})
    model = GPTModel(_tiny_config(), seed=23)
    before = _snapshot(model)

    with pytest.raises(CheckpointLoadError) as captured:
        load_safetensors(model, path)

    report = captured.value.report
    assert report.unexpected == ["mystery.weight"]
    assert report.missing == []
    assert report.loaded == []
    _assert_unchanged(model, before)


def test_prefix_and_attention_mask_buffers_are_ignored(tmp_path: Path) -> None:
    path = _stub(tmp_path)
    tensors = load_file(str(path))
    prefixed = {f"transformer.{name}": tensor for name, tensor in tensors.items()}
    prefixed["transformer.h.0.attn.bias"] = torch.zeros(
        (1, 1, 16, 16), dtype=torch.float32
    )
    prefixed["transformer.lm_head.weight"] = prefixed["transformer.wte.weight"].clone()
    save_file(prefixed, str(path), metadata={"format": "pt"})
    model = GPTModel(_tiny_config(), seed=23)

    report = load_safetensors(model, path)

    assert report.ok
    assert report.ignored == [
        "transformer.h.0.attn.bias",
        "transformer.lm_head.weight",
    ]
    assert len(report.loaded) == 28
    assert all(entry.source.startswith("transformer.") for entry in report.loaded)


def test_unsupported_configs_and_missing_file(tmp_path: Path) -> None:
    path = _stub(tmp_path)
    untied = GPTModel(
        GPTConfig(
            vocab_size=64,
            context_length=16,
            d_model=16,
            num_heads=2,
            num_layers=2,
            mlp_expansion=2,
            dropout=0.0,
            weight_tying=False,
        ),
        seed=0,
    )
    with pytest.raises(ValueError, match="weight_tying"):
        load_safetensors(untied, path)

    no_bias = GPTModel(
        GPTConfig(
            vocab_size=64,
            context_length=16,
            d_model=16,
            num_heads=2,
            num_layers=2,
            mlp_expansion=2,
            dropout=0.0,
            use_bias=False,
        ),
        seed=0,
    )
    with pytest.raises(ValueError, match="use_bias"):
        load_safetensors(no_bias, path)

    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        load_safetensors(GPTModel(_tiny_config(), seed=0), tmp_path / "absent")
