from __future__ import annotations

from typing import cast

import pytest

from main import GPTConfig, GPTModel, make_pretrained_to_local


def _tiny_config() -> GPTConfig:
    return GPTConfig(
        vocab_size=32,
        context_length=8,
        d_model=16,
        num_heads=2,
        num_layers=2,
        mlp_expansion=2,
        dropout=0.0,
    )


def test_map_expands_and_covers_every_model_parameter() -> None:
    config = _tiny_config()
    mapping = make_pretrained_to_local(config.num_layers)
    model = GPTModel(config, seed=0)

    assert len(mapping) == 12 * config.num_layers + 4 == 28
    assert set(mapping.values()) == set(dict(model.named_parameters()))
    assert len(set(mapping.values())) == len(mapping)


def test_map_spot_checks_each_parameter_family() -> None:
    mapping = make_pretrained_to_local(3)

    assert mapping["wte.weight"] == "tok_embed.weight"
    assert mapping["wpe.weight"] == "pos_embed.weight"
    assert mapping["ln_f.weight"] == "final_ln.scale"
    assert mapping["ln_f.bias"] == "final_ln.shift"
    assert mapping["h.0.ln_1.weight"] == "blocks.0.ln1.scale"
    assert mapping["h.0.ln_2.bias"] == "blocks.0.ln2.shift"
    assert mapping["h.1.attn.c_attn.weight"] == "blocks.1.attn.qkv.weight"
    assert mapping["h.1.attn.c_proj.weight"] == "blocks.1.attn.out_proj.weight"
    assert mapping["h.2.mlp.c_fc.weight"] == "blocks.2.mlp.fc1.weight"
    assert mapping["h.2.mlp.c_proj.weight"] == "blocks.2.mlp.fc2.weight"


def test_map_is_fresh_per_call() -> None:
    first = make_pretrained_to_local(1)
    second = make_pretrained_to_local(1)
    assert first == second
    assert first is not second
    first["wte.weight"] = "mutated"
    assert second["wte.weight"] == "tok_embed.weight"


def test_map_rejects_invalid_layer_counts() -> None:
    for value in (0, -1):
        with pytest.raises(ValueError, match="num_layers"):
            make_pretrained_to_local(value)
    for bad_value in (True, "2", 2.0):
        with pytest.raises(TypeError, match="num_layers"):
            make_pretrained_to_local(cast(int, bad_value))
