"""Spatial up and down sampling.

Mirror of ``diffusers/models/downsampling.py`` ``Downsample2D`` and
``diffusers/models/upsampling.py`` ``Upsample2D``. The SD1.5 UNet uses a
strided conv (downsample, attribute ``conv``) and nearest scale-2 followed
by a conv (upsample, attribute ``conv``).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_diffusers.layers import Conv2d, nchw_to_nhwc, nhwc_to_nchw


class Downsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: int | None = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.padding = padding
        out_channels = channels if out_channels is None else out_channels
        if not use_conv:
            raise ValueError("Downsample2D without conv is not the SD1.5 UNet path")
        conv = Conv2d(
            channels,
            out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=bias,
        )
        # name="op" (UNet down blocks) and name="conv" both store the conv as ``conv``.
        if name not in {"conv", "op", "Conv2d_0"}:
            raise ValueError(f"Downsample2D name {name!r} is not a diffusers conv alias")
        self.conv = conv

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if hidden_states.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {hidden_states.shape[1]}")
        if self.padding == 0:
            hidden_states = mx.pad(hidden_states, [(0, 0), (0, 0), (0, 1), (0, 1)])
        return self.conv(hidden_states)


class Upsample2D(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: int | None = None,
        name: str = "conv",
        kernel_size: int = 3,
        padding: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = channels
        out_channels = channels if out_channels is None else out_channels
        self.upsample = nn.Upsample(scale_factor=2.0, mode="nearest")
        if not use_conv:
            raise ValueError("Upsample2D without conv is not the SD1.5 UNet path")
        if name != "conv":
            raise ValueError(f"Upsample2D name {name!r} is not the SD1.5 UNet path")
        self.conv = Conv2d(channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        if hidden_states.shape[1] != self.channels:
            raise ValueError(f"expected {self.channels} channels, got {hidden_states.shape[1]}")
        hidden_states = nhwc_to_nchw(self.upsample(nchw_to_nhwc(hidden_states)))
        return self.conv(hidden_states)
