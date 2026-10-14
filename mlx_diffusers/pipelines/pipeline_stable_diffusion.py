"""SD1.5 text-to-image pipeline.

One CLIP text encoder, final hidden state, classifier-free guidance, and the
VAE scale from the component (``0.18215`` when the config omits it). The scale
is applied outside ``encode`` and ``decode``. The sampler is the scheduler
object: DDIM with eta 0, or Euler. The tokenizer is ``transformers.CLIPTokenizer``.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_diffusers.configuration import load_model
from mlx_diffusers.pipelines.pipeline_utils import PIPELINE_REGISTRY
from mlx_diffusers.schedulers.scheduling_ddim import DDIMScheduler
from mlx_diffusers.schedulers.scheduling_euler import EulerDiscreteScheduler
from mlx_diffusers.utils.cpu_generator import randn

_SCHEDULERS = {
    "DDIMScheduler": DDIMScheduler,
    "EulerDiscreteScheduler": EulerDiscreteScheduler,
}


class StableDiffusionPipeline:
    def __init__(self, tokenizer, text_encoder, unet, vae, scheduler, scheduler_config: dict, in_channels: int, sample_size: int, vae_scale_factor: int):
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config
        self.in_channels = in_channels
        self.sample_size = sample_size
        self.vae_scale_factor = vae_scale_factor

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, index: dict | None = None) -> "StableDiffusionPipeline":
        from transformers import CLIPTokenizer

        model_dir = Path(model_dir)
        if index is None:
            with (model_dir / "model_index.json").open() as handle:
                index = json.load(handle)
        if index.get("text_encoder", [None, None])[1] != "CLIPTextModel":
            raise ValueError("SD1.5 pipeline expects one CLIPTextModel")
        tokenizer = CLIPTokenizer.from_pretrained(model_dir / "tokenizer")
        text_encoder = load_model(model_dir / "text_encoder").module
        unet_result = load_model(model_dir / "unet")
        vae_result = load_model(model_dir / "vae")
        with (model_dir / "scheduler" / "scheduler_config.json").open() as handle:
            scheduler_config = json.load(handle)
        scheduler_name = index.get("scheduler", [None, "DDIMScheduler"])[1]
        # SD1.5 publishes PNDMScheduler. DDIM reads the same beta fields.
        scheduler_cls = _SCHEDULERS.get(scheduler_name, DDIMScheduler)
        vae_blocks = vae_result.config["block_out_channels"]
        return cls(
            tokenizer,
            text_encoder,
            unet_result.module,
            vae_result.module,
            scheduler_cls.from_config(scheduler_config),
            scheduler_config,
            int(unet_result.config["in_channels"]),
            int(unet_result.config["sample_size"]),
            2 ** (len(vae_blocks) - 1),
        )

    def encode_prompt(self, prompt: str, negative_prompt: str = "") -> mx.array:
        tokens = self.tokenizer(
            [negative_prompt, prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        input_ids = mx.array(np.asarray(tokens["input_ids"], dtype=np.int32))
        hidden = self.text_encoder(input_ids)
        mx.eval(hidden)
        return hidden

    def __call__(
        self,
        prompt: str,
        *,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = 0,
        eta: float = 0.0,
        latents: mx.array | None = None,
    ) -> mx.array:
        self.scheduler.set_timesteps(num_inference_steps)
        prompt_embeds = self.encode_prompt(prompt)
        do_cfg = guidance_scale > 1
        if not do_cfg:
            prompt_embeds = prompt_embeds[1:]
        if latents is None:
            latents = mx.array(randn((1, self.in_channels, self.sample_size, self.sample_size), seed))
        latents = latents * mx.array(np.float32(self.scheduler.init_noise_sigma))
        for timestep in self.scheduler.timesteps:
            latent_input = mx.concatenate([latents, latents], axis=0) if do_cfg else latents
            latent_input = self.scheduler.scale_model_input(latent_input, timestep)
            noise_pred = self.unet(latent_input, _as_timestep(timestep), prompt_embeds)
            mx.eval(noise_pred)
            if do_cfg:
                noise_uncond = noise_pred[:1]
                noise_text = noise_pred[1:]
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            if isinstance(self.scheduler, DDIMScheduler):
                latents = self.scheduler.step(noise_pred, timestep, latents, eta=eta).prev_sample
            else:
                latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
            mx.eval(latents)
        scale = float(self.vae.scaling_factor)
        image = self.vae.decode(latents / scale)
        image = mx.clip(image / 2 + 0.5, 0, 1)
        mx.eval(image)
        return image


def _as_timestep(timestep):
    if isinstance(timestep, np.ndarray):
        return timestep.item()
    return timestep


PIPELINE_REGISTRY["StableDiffusionPipeline"] = StableDiffusionPipeline
