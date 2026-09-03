from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from main import (
    GPTConfig,
    make_pretrained_to_local,
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


def _demo_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=50257,
        context_length=1024,
        d_model=192,
        num_heads=6,
        num_layers=12,
        mlp_expansion=4,
        dropout=0.0,
    )


def test_stub_names_match_map_and_report(tmp_path: Path) -> None:
    config = _tiny_config()
    path = tmp_path / "nested" / "stub.safetensors"
    report = make_stub_safetensors(path, config, seed=5)

    tensors = load_file(str(path))
    assert path.is_file()
    assert set(tensors) == set(make_pretrained_to_local(config.num_layers))
    assert len(tensors) == 28
    assert report.tensor_count == 28
    assert report.parameter_count == 5760
    assert report.size_bytes == path.stat().st_size
    assert report.seed == 5
    assert report.corrupt_name is None

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        assert handle.metadata()["format"] == "pt"
        assert handle.metadata()["seed"] == "5"


def test_stub_uses_pretrained_linear_layout(tmp_path: Path) -> None:
    config = _tiny_config()
    path = tmp_path / "stub.safetensors"
    make_stub_safetensors(path, config, seed=7)
    tensors = load_file(str(path))

    assert tensors["h.0.attn.c_attn.weight"].shape == (16, 48)
    assert tensors["h.0.attn.c_proj.weight"].shape == (16, 16)
    assert tensors["h.0.mlp.c_fc.weight"].shape == (16, 32)
    assert tensors["h.0.mlp.c_proj.weight"].shape == (32, 16)
    assert tensors["wte.weight"].shape == (64, 16)
    assert tensors["wpe.weight"].shape == (16, 16)
    assert tensors["h.1.ln_1.weight"].shape == (16,)
    assert all(tensor.dtype == torch.float32 for tensor in tensors.values())


def test_stub_is_deterministic(tmp_path: Path) -> None:
    config = _tiny_config()
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    make_stub_safetensors(first, config, seed=11)
    make_stub_safetensors(second, config, seed=11)

    first_tensors = load_file(str(first))
    second_tensors = load_file(str(second))
    assert set(first_tensors) == set(second_tensors)
    for name, tensor in first_tensors.items():
        assert torch.equal(tensor, second_tensors[name]), name


def test_stub_corrupts_exactly_one_shape(tmp_path: Path) -> None:
    config = _tiny_config()
    path = tmp_path / "bad.safetensors"
    clean_path = tmp_path / "clean.safetensors"
    report = make_stub_safetensors(
        path,
        config,
        seed=13,
        corrupt_name="h.0.attn.c_attn.weight",
    )
    make_stub_safetensors(clean_path, config, seed=13)
    tensors = load_file(str(path))
    clean = load_file(str(clean_path))
    assert set(tensors) == set(make_pretrained_to_local(config.num_layers))
    assert tensors["h.0.attn.c_attn.weight"].shape == (16, 24)
    differing = [
        name
        for name in clean
        if tensors[name].shape != clean[name].shape
    ]
    assert differing == ["h.0.attn.c_attn.weight"]
    for name in clean:
        if name != "h.0.attn.c_attn.weight":
            assert torch.equal(tensors[name], clean[name]), name
    assert report.corrupt_name == "h.0.attn.c_attn.weight"


def test_demo_map_has_148_entries() -> None:
    assert len(make_pretrained_to_local(_demo_config().num_layers)) == 148


def test_stub_rejects_invalid_arguments(tmp_path: Path) -> None:
    config = _tiny_config()
    with pytest.raises(ValueError, match="corrupt_name"):
        make_stub_safetensors(
            tmp_path / "x.safetensors", config, corrupt_name="missing.weight"
        )
    with pytest.raises(TypeError, match="seed"):
        make_stub_safetensors(
            tmp_path / "x.safetensors", config, seed=True
        )
    with pytest.raises(TypeError, match="config"):
        make_stub_safetensors(
            tmp_path / "x.safetensors",
            object(),  # type: ignore[arg-type]
        )
