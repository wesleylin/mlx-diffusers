"""Phase 2: registry lookup and SD1.5 weight layouts. No module classes."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from safetensors.numpy import save_file

from mlx_diffusers.configuration import (
    ARCH_REGISTRY,
    CLASS_NAME_ALIASES,
    UnknownClassName,
    load_model,
    resolve_class_name,
)
from mlx_diffusers.utils.convert import CLIP_POSITION_IDS, CONVERTERS
from mlx_diffusers.utils.loading import read_safetensors

SD15_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def _write_component(directory: Path, config: dict, tensors: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(config))
    save_file(tensors, directory / "model.safetensors")


def _snapshot() -> Path:
    repo = "models--" + SD15_ID.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / repo / "snapshots"
    snaps = [path for path in sorted(root.iterdir()) if (path / "unet" / "config.json").is_file()]
    assert snaps, f"Phase 1 cache for {SD15_ID} is not on disk"
    return snaps[-1]


def test_unknown_class_name_raises():
    with pytest.raises(UnknownClassName, match="NotAModel"):
        resolve_class_name("NotAModel")


def test_unknown_class_name_on_load(tmp_path: Path):
    _write_component(
        tmp_path,
        {"_class_name": "NotAModel"},
        {"weight": np.ones((2, 2), dtype=np.float32)},
    )
    with pytest.raises(UnknownClassName, match="NotAModel"):
        load_model(tmp_path)


def test_alias_is_not_a_second_registry():
    assert set(CLASS_NAME_ALIASES.values()) <= set(ARCH_REGISTRY)
    assert set(CONVERTERS) == set(ARCH_REGISTRY)
    assert ARCH_REGISTRY["CLIPTextModel"].cls is not None
    assert ARCH_REGISTRY["UNet2DConditionModel"].cls is None
    assert ARCH_REGISTRY["AutoencoderKL"].cls is None


def test_known_class_returns_weights_without_a_module(tmp_path: Path):
    conv = np.arange(8 * 4 * 3 * 3, dtype=np.float32).reshape(8, 4, 3, 3)
    linear = np.arange(6 * 5, dtype=np.float32).reshape(6, 5)
    _write_component(
        tmp_path,
        {"_class_name": "UNet2DConditionModel"},
        {"conv.weight": conv, "proj.weight": linear, "proj.bias": np.zeros(6, dtype=np.float32)},
    )
    result = load_model(tmp_path)
    assert result.class_name == "UNet2DConditionModel"
    assert result.module_registered is False
    assert result.message == "module is not registered"
    assert result.unmapped == ()
    conv_out = result.weights["conv.weight"]
    linear_out = result.weights["proj.weight"]
    mx.eval(conv_out, linear_out)
    assert tuple(conv_out.shape) == (3, 3, 4, 8)
    assert tuple(linear_out.shape) == (5, 6)
    assert np.allclose(np.array(conv_out), np.transpose(conv, (2, 3, 1, 0)))
    assert np.allclose(np.array(linear_out), linear.T)


def test_sharded_safetensors_round_trip(tmp_path: Path):
    first = {"a.weight": np.ones((2, 3), dtype=np.float32)}
    second = {"b.bias": np.ones(4, dtype=np.float32)}
    save_file(first, tmp_path / "model-00001-of-00002.safetensors")
    save_file(second, tmp_path / "model-00002-of-00002.safetensors")
    index = {
        "weight_map": {
            "a.weight": "model-00001-of-00002.safetensors",
            "b.bias": "model-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    loaded = read_safetensors(tmp_path)
    assert set(loaded) == {"a.weight", "b.bias"}
    assert loaded["a.weight"].shape == (2, 3)


def test_unmapped_key_raises(tmp_path: Path):
    _write_component(
        tmp_path,
        {"_class_name": "AutoencoderKL"},
        {"mystery": np.ones((2, 2, 2), dtype=np.float32)},
    )
    with pytest.raises(RuntimeError, match="unmapped keys"):
        load_model(tmp_path)


def test_sd15_components_have_no_unmapped_keys_and_transposed_shapes():
    snapshot = _snapshot()
    expected = {
        "unet": {
            "class_name": "UNet2DConditionModel",
            "conv": "conv_in.weight",
            "conv_before": (320, 4, 3, 3),
            "conv_after": (3, 3, 4, 320),
            "linear": "time_embedding.linear_1.weight",
            "linear_before": (1280, 320),
            "linear_after": (320, 1280),
        },
        "vae": {
            "class_name": "AutoencoderKL",
            "conv": "encoder.conv_in.weight",
            "conv_before": (128, 3, 3, 3),
            "conv_after": (3, 3, 3, 128),
            "linear": "encoder.mid_block.attentions.0.query.weight",
            "linear_before": (512, 512),
            "linear_after": (512, 512),
        },
        "text_encoder": {
            "class_name": "CLIPTextModel",
            "conv": None,
            "linear": "text_model.encoder.layers.0.mlp.fc1.weight",
            "linear_before": (3072, 768),
            "linear_after": (768, 3072),
        },
    }
    for name, spec in expected.items():
        original = read_safetensors(snapshot / name)
        result = load_model(snapshot / name)
        assert result.class_name == spec["class_name"]
        assert result.unmapped == ()
        assert result.module_registered is (name == "text_encoder")
        assert set(result.weights) == set(original)
        linear = result.weights[spec["linear"]]
        mx.eval(linear)
        assert tuple(original[spec["linear"]].shape) == spec["linear_before"]
        assert tuple(linear.shape) == spec["linear_after"]
        assert np.allclose(np.array(linear), original[spec["linear"]].T)
        if spec["conv"] is None:
            assert not any(array.ndim == 4 for array in original.values())
            assert CLIP_POSITION_IDS in result.whitelisted
            ids = result.weights[CLIP_POSITION_IDS]
            mx.eval(ids)
            assert np.array_equal(np.array(ids), original[CLIP_POSITION_IDS])
            continue
        conv = result.weights[spec["conv"]]
        mx.eval(conv)
        assert tuple(original[spec["conv"]].shape) == spec["conv_before"]
        assert tuple(conv.shape) == spec["conv_after"]
        assert np.allclose(np.array(conv), np.transpose(original[spec["conv"]], (2, 3, 1, 0)))


def test_saved_conversion_keeps_transpose(tmp_path: Path):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "convert_weights.py"
    spec = importlib.util.spec_from_file_location("convert_weights", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    linear = np.arange(6, dtype=np.float32).reshape(2, 3)
    source = tmp_path / "unet"
    _write_component(source, {"_class_name": "UNet2DConditionModel"}, {"proj.weight": linear})
    output = tmp_path / "unet.safetensors"
    module.convert_directory(source, output)
    from safetensors import safe_open

    with safe_open(output, framework="numpy") as handle:
        saved = handle.get_tensor("proj.weight")
    assert np.array_equal(saved, linear.T)
