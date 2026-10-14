"""Scheduler contract.

Schedulers are duck-typed. A sampler needs:

- ``set_timesteps(num_inference_steps)``
- ``scale_model_input(sample, timestep)``
- ``step(model_output, timestep, sample)`` returning an object with ``prev_sample``
- ``init_noise_sigma``
- ``timesteps``
- ``alphas_cumprod`` for the training forward process

This module is not an ABC hierarchy. It only builds the shared beta schedule
read from a diffusers ``scheduler_config.json``.
"""

from __future__ import annotations

import numpy as np


def betas_from_config(config: dict) -> np.ndarray:
    """Return ``num_train_timesteps`` betas in float32."""
    if config.get("trained_betas") is not None:
        return np.asarray(config["trained_betas"], dtype=np.float32)
    count = int(config.get("num_train_timesteps", 1000))
    beta_start = float(config.get("beta_start", 0.0001))
    beta_end = float(config.get("beta_end", 0.02))
    schedule = config.get("beta_schedule", "linear")
    if schedule == "linear":
        return np.linspace(beta_start, beta_end, count, dtype=np.float32)
    if schedule == "scaled_linear":
        return np.linspace(np.sqrt(beta_start), np.sqrt(beta_end), count, dtype=np.float32) ** 2
    raise ValueError(f"beta_schedule {schedule!r} is not implemented")


def alphas_cumprod_from_betas(betas: np.ndarray) -> np.ndarray:
    alphas = np.float32(1.0) - betas
    return np.cumprod(alphas, axis=0).astype(np.float32)


def add_noise(original: np.ndarray, noise: np.ndarray, timesteps: np.ndarray, alphas_cumprod: np.ndarray) -> np.ndarray:
    """DDPM forward process from ``alphas_cumprod``: ``sqrt(a) x0 + sqrt(1-a) eps``."""
    alpha = alphas_cumprod[np.asarray(timesteps, dtype=np.int64)].astype(np.float32)
    while alpha.ndim < original.ndim:
        alpha = alpha[..., None]
    return np.sqrt(alpha) * original + np.sqrt(np.float32(1.0) - alpha) * noise
