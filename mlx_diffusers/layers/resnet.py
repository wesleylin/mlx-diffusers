"""2D residual block.

Mirror of ``diffusers/models/resnet.py`` ``ResnetBlock2D`` for the SD1.5
path: GroupNorm, SiLU, conv, timestep add, GroupNorm, SiLU, conv, residual.
``eps``, groups, and channels come from the caller. UNet GroupNorm ``eps`` is
``1e-5`` via ``norm_eps`` in the component config.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_diffusers.layers import Conv2d
from mlx_diffusers.layers.activations import SiLU
from mlx_diffusers.layers.norm import GroupNorm


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        time_embedding_norm: str = "default",
        output_scale_factor: float = 1.0,
        use_in_shortcut: bool | None = None,
    ):
        super().__init__()
        if time_embedding_norm != "default":
            raise ValueError(f"ResnetBlock2D time_embedding_norm {time_embedding_norm!r} is not the SD1.5 path")
        if non_linearity not in {"swish", "silu"}:
            raise ValueError(f"ResnetBlock2D non_linearity {non_linearity!r} is not silu/swish")
        out_channels = in_channels if out_channels is None else out_channels
        self.output_scale_factor = output_scale_factor
        self.norm1 = GroupNorm(groups, in_channels, eps)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        self.norm2 = GroupNorm(groups, out_channels, eps)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.nonlinearity = SiLU()
        if use_in_shortcut is None:
            use_in_shortcut = in_channels != out_channels
        self.conv_shortcut = None
        if use_in_shortcut:
            self.conv_shortcut = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, input_tensor: mx.array, temb: mx.array) -> mx.array:
        hidden_states = self.nonlinearity(self.norm1(input_tensor))
        hidden_states = self.conv1(hidden_states)
        temb = self.time_emb_proj(self.nonlinearity(temb))
        temb = temb[:, :, None, None]
        hidden_states = hidden_states + temb
        hidden_states = self.nonlinearity(self.norm2(hidden_states))
        hidden_states = self.conv2(self.dropout(hidden_states))
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        return (input_tensor + hidden_states) / self.output_scale_factor
