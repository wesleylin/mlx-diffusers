"""Write fp32 CPU golden tensors for a diffusers checkpoint.

Phase 1 runs the SD1.5 family only. SDXL and Animagine ids are accepted on
the CLI and refused before any download.

Prompts, in order:
- index 0: ``a photograph of an astronaut riding a horse``
- index 1: the empty string

Latents are ``torch.randn`` on CPU in fp32 immediately after ``torch.manual_seed(0)``.
Shape is ``(2, unet.config.in_channels, unet.config.sample_size, unet.config.sample_size)``.

UNet noise predictions use that latent batch and both CLIP embeddings at
timesteps 999, 500, and 1.

Module hooks are the outputs of one forward at timestep 999:
- ``time_proj`` (diffusers ``Timesteps``)
- ``down_blocks.0``
- ``down_blocks.0.attentions.0``
- ``down_blocks.0.resnets.0``

VAE goldens are ``decode(latents).sample`` and ``encode(decoded).latent_dist.mean``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel
from huggingface_hub.errors import (
    DisabledRepoError,
    EntryNotFoundError,
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from transformers import CLIPTextModel, CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "tests" / "golden"

SD15_MIRROR = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_FALLBACK = "CompVis/stable-diffusion-v1-4"
SDXL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
ANIMAGINE_ID = "cagliostrolab/animagine-xl-3.1"
LATER_PHASE_IDS = (SDXL_ID, ANIMAGINE_ID)

CAPTION = "a photograph of an astronaut riding a horse"
EMPTY_PROMPT = ""
PROMPTS = (CAPTION, EMPTY_PROMPT)
LATENT_SEED = 0
UNET_TIMESTEPS = (999, 500, 1)
HOOK_TIMESTEP = 999

_MISSING_OR_GATED = (
    DisabledRepoError,
    EntryNotFoundError,
    GatedRepoError,
    LocalEntryNotFoundError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)


def _is_missing_or_gated(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, _MISSING_OR_GATED):
            return True
        if isinstance(current, HfHubHTTPError) and current.response is not None:
            if current.response.status_code in {401, 403, 404}:
                return True
        message = str(current).lower()
        if any(
            marker in message
            for marker in (
                "gated",
                "401 client error",
                "403 client error",
                "404 client error",
                "repository not found",
                "cannot access gated",
                "is not a valid model identifier",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _load_components(model_id: str):
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", dtype=torch.float32)
    text_encoder = text_encoder.to(device="cpu").eval()
    unet = unet.to(device="cpu").eval()
    vae = vae.to(device="cpu").eval()
    return tokenizer, text_encoder, unet, vae


def load_resolved(model_id: str):
    """Load ``model_id``, falling back only when the SD1.5 mirror is missing or gated."""
    if model_id in LATER_PHASE_IDS:
        raise SystemExit(
            f"{model_id} is CLI-capable but is not captured in Phase 1. "
            "This phase runs the SD1.5 family only."
        )
    try:
        return model_id, _load_components(model_id)
    except Exception as exc:
        if model_id != SD15_MIRROR or not _is_missing_or_gated(exc):
            raise
        print(
            f"model {model_id} is missing or gated ({type(exc).__name__}: {exc}). "
            f"falling back to {SD15_FALLBACK}",
            file=sys.stderr,
        )
        return SD15_FALLBACK, _load_components(SD15_FALLBACK)


def _encode_prompts(tokenizer, text_encoder) -> tuple[np.ndarray, np.ndarray]:
    text_inputs = tokenizer(
        list(PROMPTS),
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    attention_mask = None
    if getattr(text_encoder.config, "use_attention_mask", False):
        attention_mask = text_inputs.attention_mask
    with torch.no_grad():
        first = text_encoder(text_inputs.input_ids, attention_mask=attention_mask)[0]
        second = text_encoder(text_inputs.input_ids, attention_mask=attention_mask)[0]
    if not torch.equal(first, second):
        raise RuntimeError("CLIP text encoder forward is not deterministic")
    input_ids = text_inputs.input_ids.detach().cpu().to(dtype=torch.int64).numpy()
    embeddings = first.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return input_ids, embeddings


def _initial_latents(unet) -> torch.Tensor:
    torch.manual_seed(LATENT_SEED)
    return torch.randn(
        (
            len(PROMPTS),
            int(unet.config.in_channels),
            int(unet.config.sample_size),
            int(unet.config.sample_size),
        ),
        device="cpu",
        dtype=torch.float32,
    )


def _tensor_from_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    sample = getattr(output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    if isinstance(output, (tuple, list)) and output:
        return _tensor_from_output(output[0])
    raise TypeError(f"cannot read a tensor from {type(output).__name__}")


def _capture_modules(unet, latents: torch.Tensor, embeddings: torch.Tensor) -> dict[str, torch.Tensor]:
    modules = {
        "timesteps_embedding": unet.time_proj,
        "down_block": unet.down_blocks[0],
        "attention": unet.down_blocks[0].attentions[0],
        "resnet": unet.down_blocks[0].resnets[0],
    }
    if type(modules["timesteps_embedding"]).__name__ != "Timesteps":
        raise RuntimeError(
            "time_proj is "
            f"{type(modules['timesteps_embedding']).__name__}, expected Timesteps"
        )
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def _hook(key: str):
        def fn(_module, _inputs, output):
            captured[key] = (
                _tensor_from_output(output).detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
            )

        return fn

    for key, module in modules.items():
        handles.append(module.register_forward_hook(_hook(key)))
    try:
        with torch.no_grad():
            noise_pred = unet(latents, HOOK_TIMESTEP, encoder_hidden_states=embeddings).sample
    finally:
        for handle in handles:
            handle.remove()
    missing = [key for key in modules if key not in captured]
    if missing:
        raise RuntimeError(f"module hooks did not fire: {missing}")
    noise_pred = noise_pred.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return captured, noise_pred


def _unet_noise_pred(
    unet,
    latents: torch.Tensor,
    embeddings: torch.Tensor,
    timestep: int,
    *,
    check_deterministic: bool,
) -> np.ndarray:
    with torch.no_grad():
        first = unet(latents, timestep, encoder_hidden_states=embeddings).sample
        if check_deterministic:
            second = unet(latents, timestep, encoder_hidden_states=embeddings).sample
            if not torch.equal(first, second):
                raise RuntimeError(f"UNet forward at timestep {timestep} is not deterministic")
    return first.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()


def _vae_goldens(vae, latents: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        decoded = vae.decode(latents).sample
        posterior_mean = vae.encode(decoded).latent_dist.mean
    return (
        decoded.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy(),
        posterior_mean.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy(),
    )


def _savez(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    print(f"wrote {path}")


def write_goldens(model_id: str, output_dir: Path) -> str:
    resolved, (tokenizer, text_encoder, unet, vae) = load_resolved(model_id)
    input_ids, embeddings = _encode_prompts(tokenizer, text_encoder)
    latents = _initial_latents(unet)
    embedding_tensor = torch.from_numpy(embeddings)
    module_outputs, noise_pred_hook = _capture_modules(unet, latents, embedding_tensor)
    noise_preds = {
        f"noise_pred_{HOOK_TIMESTEP}": noise_pred_hook.numpy(),
    }
    with torch.no_grad():
        repeated = unet(latents, HOOK_TIMESTEP, encoder_hidden_states=embedding_tensor).sample
    if not torch.equal(noise_pred_hook, repeated):
        raise RuntimeError(f"UNet forward at timestep {HOOK_TIMESTEP} is not deterministic")
    for timestep in UNET_TIMESTEPS:
        if timestep == HOOK_TIMESTEP:
            continue
        noise_preds[f"noise_pred_{timestep}"] = _unet_noise_pred(
            unet,
            latents,
            embedding_tensor,
            timestep,
            check_deterministic=False,
        )
    decoded, posterior_mean = _vae_goldens(vae, latents)
    model_id_array = np.asarray(resolved)

    _savez(
        output_dir / "sd15_tokens.npz",
        input_ids=input_ids,
        prompts=np.array(PROMPTS),
        model_id=model_id_array,
    )
    _savez(
        output_dir / "sd15_text_embeddings.npz",
        last_hidden_state=embeddings,
        model_id=model_id_array,
    )
    _savez(
        output_dir / "sd15_latents.npz",
        latents=latents.numpy(),
        seed=np.int64(LATENT_SEED),
        model_id=model_id_array,
    )
    _savez(
        output_dir / "sd15_unet.npz",
        timesteps=np.asarray(UNET_TIMESTEPS, dtype=np.int64),
        model_id=model_id_array,
        **noise_preds,
    )
    _savez(
        output_dir / "sd15_vae.npz",
        decoded=decoded,
        posterior_mean=posterior_mean,
        model_id=model_id_array,
    )
    _savez(
        output_dir / "sd15_modules.npz",
        timesteps_embedding=module_outputs["timesteps_embedding"].numpy(),
        down_block=module_outputs["down_block"].numpy(),
        attention=module_outputs["attention"].numpy(),
        resnet=module_outputs["resnet"].numpy(),
        timesteps_module=np.asarray("time_proj"),
        down_block_module=np.asarray("down_blocks.0"),
        attention_module=np.asarray("down_blocks.0.attentions.0"),
        resnet_module=np.asarray("down_blocks.0.resnets.0"),
        hook_timestep=np.int64(HOOK_TIMESTEP),
        model_id=model_id_array,
    )
    print(f"captured_model_id={resolved}")
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default=SD15_MIRROR,
        help=(
            "Diffusers model id. Defaults to the SD1.5 mirror, which falls back to "
            f"{SD15_FALLBACK} when that repo is missing or gated. "
            f"{SDXL_ID} and {ANIMAGINE_ID} are accepted and not run in this phase."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    write_goldens(args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
