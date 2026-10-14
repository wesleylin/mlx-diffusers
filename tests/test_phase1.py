"""Smoke-check the SD1.5 golden npz files. Numeric gates start in later phases."""

from pathlib import Path

import numpy as np

from mlx_diffusers.utils.golden import compare, load_golden

GOLDEN_DIR = Path(__file__).parent / "goldens" / "sd15"
MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CAPTION = "a photograph of an astronaut riding a horse"

FLOAT32_SHAPES = {
    "sd15_text_embeddings.npz": {"last_hidden_state": (2, 77, 768)},
    "sd15_latents.npz": {"latents": (2, 4, 64, 64)},
    "sd15_unet.npz": {
        "noise_pred_999": (2, 4, 64, 64),
        "noise_pred_500": (2, 4, 64, 64),
        "noise_pred_1": (2, 4, 64, 64),
    },
    "sd15_vae.npz": {
        "decoded": (2, 3, 512, 512),
        "posterior_mean": (2, 4, 64, 64),
    },
    "sd15_modules.npz": {
        "timesteps_embedding": (2, 320),
        "down_block": (2, 320, 32, 32),
        "attention": (2, 320, 64, 64),
        "resnet": (2, 320, 64, 64),
    },
}


def _assert_model_id(arrays: dict[str, np.ndarray]) -> None:
    assert arrays["model_id"].shape == ()
    assert arrays["model_id"].dtype.kind == "U"
    assert str(arrays["model_id"].item()) == MODEL_ID


def test_sd15_golden_files_present():
    old = Path(__file__).parent / "golden"
    assert list(old.glob("*.npz")) == []
    found = sorted(path.name for path in GOLDEN_DIR.glob("sd15_*.npz"))
    assert found == [
        "sd15_ddim.npz",
        "sd15_latents.npz",
        "sd15_modules.npz",
        "sd15_text_embeddings.npz",
        "sd15_tokens.npz",
        "sd15_unet.npz",
        "sd15_vae.npz",
    ]


def test_sd15_tokens():
    arrays = load_golden(GOLDEN_DIR / "sd15_tokens.npz")
    input_ids = arrays["input_ids"]
    assert input_ids.shape == (2, 77)
    assert input_ids.dtype == np.int64
    assert arrays["prompts"].tolist() == [CAPTION, ""]
    assert not np.array_equal(input_ids[0], input_ids[1])
    _assert_model_id(arrays)


def test_sd15_float_goldens_shapes_and_dtypes():
    for name, expected in FLOAT32_SHAPES.items():
        arrays = load_golden(GOLDEN_DIR / name)
        _assert_model_id(arrays)
        for key, shape in expected.items():
            array = arrays[key]
            assert array.shape == shape, (name, key, array.shape)
            assert array.dtype == np.float32, (name, key, array.dtype)
            assert np.isfinite(array).all()


def test_sd15_unet_timesteps_and_module_names():
    unet = load_golden(GOLDEN_DIR / "sd15_unet.npz")
    assert unet["timesteps"].dtype == np.int64
    assert unet["timesteps"].tolist() == [999, 500, 1]

    modules = load_golden(GOLDEN_DIR / "sd15_modules.npz")
    assert int(modules["hook_timestep"]) == 999
    assert str(modules["timesteps_module"].item()) == "time_proj"
    assert str(modules["down_block_module"].item()) == "down_blocks.0"
    assert str(modules["attention_module"].item()) == "down_blocks.0.attentions.0"
    assert str(modules["resnet_module"].item()) == "down_blocks.0.resnets.0"

    latents = load_golden(GOLDEN_DIR / "sd15_latents.npz")
    assert int(latents["seed"]) == 0


def test_compare_identical_golden_is_zero():
    latents = load_golden(GOLDEN_DIR / "sd15_latents.npz")["latents"]
    mean_abs, max_abs = compare(latents, latents)
    assert mean_abs == 0.0
    assert max_abs == 0.0
