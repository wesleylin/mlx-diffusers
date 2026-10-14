"""Shared tensor helpers for the UNet layers."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def nchw_to_nhwc(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 2, 3, 1))


def nhwc_to_nchw(x: mx.array) -> mx.array:
    return mx.transpose(x, (0, 3, 1, 2))


def phase2_to_parameter(name: str, value: mx.array) -> mx.array:
    """Map a Phase 2 weight onto an ``mlx.nn`` parameter.

    Phase 2 stores Linear as ``[in, out]`` and Conv2d as ``[kh, kw, in, out]``.
    ``mlx.nn.Linear`` stores ``[out, in]`` and ``mlx.nn.Conv2d`` stores
    ``[out, kh, kw, in]``.
    """
    if name.endswith(".weight") and value.ndim == 2:
        return mx.contiguous(value.T)
    if name.endswith(".weight") and value.ndim == 4:
        return mx.contiguous(mx.transpose(value, (3, 0, 1, 2)))
    return value


class Conv2d(nn.Conv2d):
    """NCHW conv. Mirror of ``torch.nn.Conv2d`` as used by diffusers.

    MLX convolves channels-last, so this wrapper converts layout around
    ``mlx.nn.Conv2d``. Parameter names stay ``weight`` and ``bias``.
    """

    def __call__(self, x: mx.array) -> mx.array:
        return nhwc_to_nchw(super().__call__(nchw_to_nhwc(x)))
