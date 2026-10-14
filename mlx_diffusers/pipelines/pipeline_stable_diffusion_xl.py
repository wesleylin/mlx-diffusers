"""SDXL text-to-image pipeline.

Two encoders. Both contribute ``hidden_states[-2]``, concatenated on the
channel axis. The pooled vector is the bigG projection. Six ``time_ids`` and
that pooled vector go through the UNet ``add_embedding``. The VAE scale is the
component ``scaling_factor`` (``0.13025``), applied outside encode and decode.
Empty negative prompts are zeros when the model index asks for that.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_diffusers.configuration import load_model
from mlx_diffusers.pipelines.pipeline_utils import PIPELINE_REGISTRY
from mlx_diffusers.schedulers.scheduling_euler import EulerDiscreteScheduler
from mlx_diffusers.utils.cpu_generator import randn


class StableDiffusionXLPipeline:
    def __init__(
        self,
        tokenizer,
        tokenizer_2,
        text_encoder,
        text_encoder_2,
        unet,
        vae,
        scheduler,
        scheduler_config: dict,
        in_channels: int,
        sample_size: int,
        vae_scale_factor: int,
        force_zeros_for_empty_prompt: bool,
    ):
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.scheduler_config = scheduler_config
        self.in_channels = in_channels
        self.sample_size = sample_size
        self.vae_scale_factor = vae_scale_factor
        self.force_zeros_for_empty_prompt = force_zeros_for_empty_prompt

    @classmethod
    def from_pretrained(cls, model_dir: str | Path, index: dict | None = None) -> "StableDiffusionXLPipeline":
        from transformers import CLIPTokenizer

        model_dir = Path(model_dir)
        if index is None:
            with (model_dir / "model_index.json").open() as handle:
                index = json.load(handle)
        tokenizer = CLIPTokenizer.from_pretrained(model_dir / "tokenizer")
        tokenizer_2 = CLIPTokenizer.from_pretrained(model_dir / "tokenizer_2")
        text_encoder = load_model(model_dir / "text_encoder").module
        text_encoder_2 = load_model(model_dir / "text_encoder_2").module
        unet_result = load_model(model_dir / "unet")
        vae_result = load_model(model_dir / "vae")
        with (model_dir / "scheduler" / "scheduler_config.json").open() as handle:
            scheduler_config = json.load(handle)
        vae_blocks = vae_result.config["block_out_channels"]
        return cls(
            tokenizer,
            tokenizer_2,
            text_encoder,
            text_encoder_2,
            unet_result.module,
            vae_result.module,
            EulerDiscreteScheduler.from_config(scheduler_config),
            scheduler_config,
            int(unet_result.config["in_channels"]),
            int(unet_result.config["sample_size"]),
            2 ** (len(vae_blocks) - 1),
            bool(index.get("force_zeros_for_empty_prompt", True)),
        )

    def _tokenize(self, tokenizer, prompt: str) -> mx.array:
        tokens = tokenizer(
            [prompt],
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        return mx.array(np.asarray(tokens["input_ids"], dtype=np.int32))

    def encode_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        hidden_1 = self.text_encoder.penultimate(self._tokenize(self.tokenizer, prompt))
        hidden_2, pooled = self.text_encoder_2.penultimate_and_pooled(self._tokenize(self.tokenizer_2, prompt))
        hidden = mx.concatenate([hidden_1, hidden_2], axis=-1)
        mx.eval(hidden, pooled)
        return hidden, pooled

    def __call__(
        self,
        prompt: str,
        *,
        num_inference_steps: int = 25,
        guidance_scale: float = 5.0,
        seed: int = 0,
        eta: float = 0.0,
        height: int | None = None,
        width: int | None = None,
        latents: mx.array | None = None,
        negative_prompt: str = "",
    ) -> mx.array:
        del eta
        height = int(self.sample_size * self.vae_scale_factor if height is None else height)
        width = int(self.sample_size * self.vae_scale_factor if width is None else width)
        self.scheduler.set_timesteps(num_inference_steps)
        prompt_embeds, pooled = self.encode_prompt(prompt)
        do_cfg = guidance_scale > 1
        if do_cfg and negative_prompt == "" and self.force_zeros_for_empty_prompt:
            negative_embeds = mx.zeros_like(prompt_embeds)
            negative_pooled = mx.zeros_like(pooled)
        elif do_cfg:
            negative_embeds, negative_pooled = self.encode_prompt(negative_prompt)
        else:
            negative_embeds = None
            negative_pooled = None
        if do_cfg:
            prompt_embeds = mx.concatenate([negative_embeds, prompt_embeds], axis=0)
            pooled = mx.concatenate([negative_pooled, pooled], axis=0)
        time_ids = mx.array(np.array([[height, width, 0, 0, height, width]], dtype=np.float32))
        if do_cfg:
            time_ids = mx.concatenate([time_ids, time_ids], axis=0)
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        if latents is None:
            latents = mx.array(randn((1, self.in_channels, latent_h, latent_w), seed))
        latents = latents * mx.array(np.float32(self.scheduler.init_noise_sigma))
        # The fp32 UNet is about 10GB. Drop the encoders before denoising so the
        # forward fits on a 16GB machine, and run the two CFG rows separately.
        self.text_encoder = None
        self.text_encoder_2 = None
        gc.collect()
        mx.eval(latents, prompt_embeds, pooled, time_ids)
        previous_cache = mx.set_cache_limit(1024 * 1024 * 1024)
        mx.clear_cache()
        try:
            latents = self._denoise(
                latents,
                prompt_embeds,
                pooled,
                time_ids,
                do_cfg=do_cfg,
                guidance_scale=guidance_scale,
            )
            scale = float(self.vae.scaling_factor)
            image = self.vae.decode(latents / scale)
            image = mx.clip(image / 2 + 0.5, 0, 1)
            mx.eval(image)
        finally:
            mx.set_cache_limit(previous_cache)
        return image

    def _denoise(self, latents, prompt_embeds, pooled, time_ids, *, do_cfg: bool, guidance_scale: float) -> mx.array:
        for timestep in self.scheduler.timesteps:
            latent_input = self.scheduler.scale_model_input(latents, timestep)
            step = _as_timestep(timestep)
            if do_cfg:
                noise_uncond = self.unet(
                    latent_input,
                    step,
                    prompt_embeds[:1],
                    {"text_embeds": pooled[:1], "time_ids": time_ids[:1]},
                )
                mx.eval(noise_uncond)
                noise_text = self.unet(
                    latent_input,
                    step,
                    prompt_embeds[1:],
                    {"text_embeds": pooled[1:], "time_ids": time_ids[1:]},
                )
                mx.eval(noise_text)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            else:
                noise_pred = self.unet(
                    latent_input,
                    step,
                    prompt_embeds,
                    {"text_embeds": pooled, "time_ids": time_ids},
                )
                mx.eval(noise_pred)
            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample
            mx.eval(latents)
            mx.clear_cache()
        return latents


def _as_timestep(timestep):
    if isinstance(timestep, np.ndarray):
        return timestep.item()
    return timestep


PIPELINE_REGISTRY["StableDiffusionXLPipeline"] = StableDiffusionXLPipeline
