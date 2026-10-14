"""Write SDXL fp32 goldens and the 25-step reference image.

Module inputs go to ``tests/goldens/sdxl/module_inputs/`` and are gitignored.
Committed hook outputs are the timestep embedding, the first cross-attention
down block, its first ResNet, and its first attention. The image is 768 on
each side because this machine has 16GB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "tests" / "goldens" / "sdxl"
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
CAPTION = "a photograph of an astronaut riding a horse"
PROMPTS = (CAPTION, "")
HEIGHT = 768
WIDTH = 768
TIMESTEPS = (999, 500, 1)
SEED = 0
NAMED_OUTPUTS = {"timesteps_embedding", "timestep_embedding", "resnet", "attention", "down_block"}
OTHER_OUTPUT_MAX_BYTES = 3_000_000


def _snapshot() -> Path:
    repo = Path.home() / ".cache" / "huggingface" / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
    snaps = [path for path in sorted(repo.iterdir()) if (path / "unet" / "config.json").is_file()]
    if not snaps:
        raise SystemExit("SDXL snapshot is not on disk")
    return snaps[-1]


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def _encode(tokenizer, text_encoder, prompts: list[str]):
    tokens = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        output = text_encoder(tokens.input_ids, output_hidden_states=True)
    penultimate = output.hidden_states[-2]
    pooled = output[0] if output[0].ndim == 2 else None
    return _to_numpy(penultimate), None if pooled is None else _to_numpy(pooled)


def _commit(key: str, array: np.ndarray) -> bool:
    if key.endswith("_module") or key in {"hook_timestep", "model_id"}:
        return True
    if key in NAMED_OUTPUTS:
        return True
    if key.endswith("_arg0") or key.endswith("_arg1") or "hidden_states" in key or key.endswith("_temb") or "encoder" in key:
        return False
    return array.nbytes < OTHER_OUTPUT_MAX_BYTES and not key.endswith(tuple(f"_arg{i}" for i in range(8)))


def write_numeric(snapshot: Path) -> None:
    tokenizer = CLIPTokenizer.from_pretrained(snapshot / "tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(snapshot / "tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(snapshot / "text_encoder", torch_dtype=torch.float32).to("cpu").eval()
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(snapshot / "text_encoder_2", torch_dtype=torch.float32).to("cpu").eval()
    hidden_1, _ = _encode(tokenizer, text_encoder, list(PROMPTS))
    hidden_2, pooled = _encode(tokenizer_2, text_encoder_2, list(PROMPTS))
    prompt_embeds = np.concatenate([hidden_1, hidden_2], axis=-1)
    del text_encoder, text_encoder_2
    unet = UNet2DConditionModel.from_pretrained(snapshot / "unet", torch_dtype=torch.float32).to("cpu").eval()
    torch.manual_seed(SEED)
    latents = torch.randn((1, 4, HEIGHT // 8, WIDTH // 8), dtype=torch.float32)
    embeds = torch.from_numpy(prompt_embeds[:1])
    pooled_1 = torch.from_numpy(pooled[:1])
    time_ids = torch.tensor([[HEIGHT, WIDTH, 0, 0, HEIGHT, WIDTH]], dtype=torch.float32)
    added = {"text_embeds": pooled_1, "time_ids": time_ids}

    captured: dict[str, np.ndarray] = {}

    def hook(key: str):
        def fn(_module, args, kwargs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[key] = _to_numpy(value)
            for index, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    captured[f"{key}_arg{index}"] = _to_numpy(arg)
            for name, arg in kwargs.items():
                if isinstance(arg, torch.Tensor):
                    captured[f"{key}_{name}"] = _to_numpy(arg)

        return fn

    modules = {
        "timesteps_embedding": unet.time_proj,
        "timestep_embedding": unet.time_embedding,
        "resnet": unet.down_blocks[1].resnets[0],
        "attention": unet.down_blocks[1].attentions[0],
        "down_block": unet.down_blocks[1],
    }
    handles = [module.register_forward_hook(hook(key), with_kwargs=True) for key, module in modules.items()]
    noise = {}
    with torch.no_grad():
        for timestep in TIMESTEPS:
            pred = unet(latents, timestep, encoder_hidden_states=embeds, added_cond_kwargs=added).sample
            noise[f"noise_pred_{timestep}"] = _to_numpy(pred)
    for handle in handles:
        handle.remove()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_id = np.asarray(MODEL_ID)
    np.savez(
        OUTPUT_DIR / "sdxl_text_embeddings.npz",
        prompt_embeds=prompt_embeds.astype(np.float32),
        pooled=pooled.astype(np.float32),
        prompts=np.array(PROMPTS),
        model_id=model_id,
    )
    np.savez(
        OUTPUT_DIR / "sdxl_unet.npz",
        latents=_to_numpy(latents),
        timesteps=np.asarray(TIMESTEPS, dtype=np.int64),
        time_ids=_to_numpy(time_ids),
        model_id=model_id,
        **noise,
    )
    record = {
        **captured,
        "timesteps_module": np.asarray("time_proj"),
        "resnet_module": np.asarray("down_blocks.1.resnets.0"),
        "attention_module": np.asarray("down_blocks.1.attentions.0"),
        "down_block_module": np.asarray("down_blocks.1"),
        "hook_timestep": np.int64(TIMESTEPS[0]),
        "model_id": model_id,
    }
    committed = {key: value for key, value in record.items() if _commit(key, value)}
    sidecar = {key: value for key, value in record.items() if key not in committed}
    committed_path = OUTPUT_DIR / "sdxl_modules.npz"
    np.savez(committed_path, **committed)
    size = committed_path.stat().st_size
    if size >= 50_000_000:
        raise SystemExit(f"{committed_path} is {size} bytes")
    input_dir = OUTPUT_DIR / "module_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    np.savez(input_dir / "sdxl_module_inputs.npz", **sidecar)
    print(f"committed {size} bytes, sidecar keys {len(sidecar)}", flush=True)


def write_image(snapshot: Path) -> None:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        snapshot,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    result = pipe(
        CAPTION,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=25,
        guidance_scale=5.0,
        generator=generator,
        output_type="pt",
    )
    image = result.images.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    np.savez(OUTPUT_DIR / "sdxl_euler.npz", image=image, model_id=np.asarray(MODEL_ID))
    from PIL import Image

    pixels = (np.transpose(image[0], (1, 2, 0)) * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(OUTPUT_DIR / "sdxl_euler_torch.png")
    print("wrote reference image", flush=True)


def main() -> None:
    snapshot = _snapshot()
    print("snapshot", snapshot, flush=True)
    if not (OUTPUT_DIR / "sdxl_unet.npz").is_file():
        write_numeric(snapshot)
    if not (OUTPUT_DIR / "sdxl_euler_torch.png").is_file():
        write_image(snapshot)


if __name__ == "__main__":
    main()
