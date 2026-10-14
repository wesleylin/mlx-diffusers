"""Phase 9: SDXL text embeddings, module hooks, one UNet step, and a 25-step image."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mlx_diffusers.configuration import load_model
from mlx_diffusers.utils.golden import compare, load_golden

GOLDEN_DIR = Path(__file__).parent / "goldens" / "sdxl"
MODULE_INPUTS = GOLDEN_DIR / "module_inputs" / "sdxl_module_inputs.npz"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CAPTION = "a photograph of an astronaut riding a horse"
MEAN_FULL = 1e-5
MAX_FULL = 1e-3
MEAN_MODULE = 1e-5
MAX_MODULE = 5e-4
PIXEL_LIMIT = 2.5 / 255
MEDIA = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Cursor"
    / "AgentStores"
    / "cursor_agent_stores"
    / "bc-c831f663-f1f0-4ce6-85d7-6549b6e82a9f"
    / "files"
    / "media"
    / "phase9"
)


def _snapshot() -> Path:
    repo = Path.home() / ".cache" / "huggingface" / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
    snaps = [path for path in sorted(repo.iterdir()) if (path / "unet" / "config.json").is_file()]
    assert snaps, "SDXL snapshot is not on disk"
    return snaps[-1]


def _gate(name: str, actual, expected, mean_limit: float, max_limit: float) -> tuple[str, float, float, bool]:
    mx.eval(actual)
    mean_abs, max_abs = compare(actual, expected)
    return name, mean_abs, max_abs, mean_abs < mean_limit and max_abs < max_limit


def test_sdxl_text_embeddings():
    mx.set_default_device(mx.cpu)
    golden = load_golden(GOLDEN_DIR / "sdxl_text_embeddings.npz")
    snapshot = _snapshot()
    from transformers import CLIPTokenizer

    from mlx_diffusers.pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

    tokenizer = CLIPTokenizer.from_pretrained(snapshot / "tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(snapshot / "tokenizer_2")
    text_encoder = load_model(snapshot / "text_encoder").module
    text_encoder_2 = load_model(snapshot / "text_encoder_2").module
    pipe = StableDiffusionXLPipeline(
        tokenizer,
        tokenizer_2,
        text_encoder,
        text_encoder_2,
        None,
        None,
        None,
        {},
        4,
        128,
        8,
        True,
    )
    rows = []
    for index, prompt in enumerate((CAPTION, "")):
        hidden, pooled = pipe.encode_prompt(prompt)
        rows.append(_gate(f"hidden_{index}", hidden, golden["prompt_embeds"][index : index + 1], MEAN_FULL, MAX_FULL))
        rows.append(_gate(f"pooled_{index}", pooled, golden["pooled"][index : index + 1], MEAN_FULL, MAX_FULL))
    lines = [f"{name} {mean_abs:.3e} {max_abs:.3e} {'ok' if ok else 'FAIL'}" for name, mean_abs, max_abs, ok in rows]
    print("\n".join(lines))
    assert all(row[3] for row in rows), lines


def test_sdxl_module_gates():
    mx.set_default_device(mx.cpu)
    if not MODULE_INPUTS.is_file():
        pytest.skip(
            "SDXL module input activations are gitignored and were not generated; "
            f"missing {MODULE_INPUTS}. Run scripts/make_sdxl_goldens.py to write them."
        )
    snapshot = _snapshot()
    unet = load_model(snapshot / "unet").module
    golden = {**load_golden(MODULE_INPUTS), **load_golden(GOLDEN_DIR / "sdxl_modules.npz")}
    block = unet.down_blocks[1]
    rows = [
        _gate(
            "timesteps",
            unet.time_proj(mx.array(golden["timesteps_embedding_arg0"])),
            golden["timesteps_embedding"],
            MEAN_MODULE,
            MAX_MODULE,
        ),
        _gate(
            "resnet",
            block.resnets[0](mx.array(golden["resnet_arg0"]), mx.array(golden["resnet_arg1"])),
            golden["resnet"],
            MEAN_MODULE,
            MAX_MODULE,
        ),
        _gate(
            "attention",
            block.attentions[0](
                mx.array(golden["attention_arg0"]),
                encoder_hidden_states=mx.array(golden["attention_encoder_hidden_states"]),
            ),
            golden["attention"],
            MEAN_MODULE,
            MAX_MODULE,
        ),
        _gate(
            "down_block",
            block(
                mx.array(golden["down_block_hidden_states"]),
                mx.array(golden["down_block_temb"]),
                mx.array(golden["down_block_encoder_hidden_states"]),
            )[0],
            golden["down_block"],
            MEAN_MODULE,
            MAX_MODULE,
        ),
    ]
    lines = [f"{name} {mean_abs:.3e} {max_abs:.3e} {'ok' if ok else 'FAIL'}" for name, mean_abs, max_abs, ok in rows]
    print("\n".join(lines))
    assert all(row[3] for row in rows), lines


def test_sdxl_unet_steps():
    mx.set_default_device(mx.cpu)
    snapshot = _snapshot()
    unet = load_model(snapshot / "unet").module
    unet_golden = load_golden(GOLDEN_DIR / "sdxl_unet.npz")
    text = load_golden(GOLDEN_DIR / "sdxl_text_embeddings.npz")
    latents = mx.array(unet_golden["latents"])
    embeds = mx.array(text["prompt_embeds"][:1])
    pooled = mx.array(text["pooled"][:1])
    time_ids = mx.array(unet_golden["time_ids"])
    rows = []
    for timestep in (999, 500, 1):
        pred = unet(latents, timestep, embeds, {"text_embeds": pooled, "time_ids": time_ids})
        rows.append(_gate(str(timestep), pred, unet_golden[f"noise_pred_{timestep}"], MEAN_FULL, MAX_FULL))
    lines = [f"{name} {mean_abs:.3e} {max_abs:.3e} {'ok' if ok else 'FAIL'}" for name, mean_abs, max_abs, ok in rows]
    print("\n".join(lines))
    assert all(row[3] for row in rows), lines


def test_sdxl_25_step_image_matches_reference():
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate.py"
    output = GOLDEN_DIR / "sdxl_euler_mlx.png"
    snapshot = _snapshot()
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--prompt",
            CAPTION,
            "--seed",
            "0",
            "--steps",
            "25",
            "--guidance-scale",
            "5",
            "--scheduler",
            "euler",
            "--height",
            "768",
            "--width",
            "768",
            "--model",
            str(snapshot),
            "--output",
            str(output),
        ],
        check=True,
    )
    actual = np.load(output.with_suffix(".npy"))
    expected = np.load(GOLDEN_DIR / "sdxl_euler.npz")["image"]
    mean_abs, max_abs = compare(actual, expected)
    print(f"sdxl image mean_abs {mean_abs:.6f} max_abs {max_abs:.6f} limit {PIXEL_LIMIT:.6f}")
    torch_png = Image.open(GOLDEN_DIR / "sdxl_euler_torch.png").convert("RGB")
    mlx_png = Image.open(output).convert("RGB")
    side = Image.new("RGB", (torch_png.width + mlx_png.width, max(torch_png.height, mlx_png.height)))
    side.paste(torch_png, (0, 0))
    side.paste(mlx_png, (torch_png.width, 0))
    side_path = GOLDEN_DIR / "sdxl_euler_side_by_side.png"
    side.save(side_path)
    MEDIA.mkdir(parents=True, exist_ok=True)
    for path in (GOLDEN_DIR / "sdxl_euler_torch.png", output, side_path):
        shutil.copy(path, MEDIA / path.name)
    output.with_suffix(".npy").unlink()
    assert mean_abs < PIXEL_LIMIT, (mean_abs, max_abs)
