"""Sinusoidal timestep embeddings.

Mirror of diffusers/models/embeddings.py ``get_timestep_embedding``,
``Timesteps``, and ``TimestepEmbedding``, including ``flip_sin_to_cos`` and
``downscale_freq_shift``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_diffusers.layers.activations import SiLU


def get_timestep_embedding(
    timesteps: mx.array,
    embedding_dim: int,
    *,
    flip_sin_to_cos: bool,
    downscale_freq_shift: float,
    scale: float = 1,
    max_period: int = 10000,
) -> mx.array:
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mx.arange(half_dim, dtype=mx.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)
    emb = mx.exp(exponent)
    emb = scale * timesteps.astype(mx.float32)[:, None] * emb[None, :]
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = mx.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = mx.pad(emb, [(0, 0), (0, 1)])
    return emb


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def __call__(self, timesteps: mx.array) -> mx.array:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        sample_proj_bias: bool = True,
    ):
        super().__init__()
        if act_fn not in {"silu", "swish"}:
            raise ValueError(f"TimestepEmbedding act_fn {act_fn!r} is not silu/swish")
        self.linear_1 = nn.Linear(in_channels, time_embed_dim, bias=sample_proj_bias)
        self.act = SiLU()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, bias=sample_proj_bias)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        return self.linear_2(sample)
