"""SD1.5 conditional UNet.

Mirror of ``diffusers/models/unets/unet_2d_condition.py``
``UNet2DConditionModel`` for the SD1.5 config: positional time embedding,
cross-attention down blocks, a cross-attention mid block, and up blocks with
skip concatenations. Dims, heads, and ``norm_eps`` come from ``config.json``.
``attention_head_dim`` is the number of heads, matching the installed diffusers
naming correction.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_diffusers.layers import Conv2d, phase2_to_parameter
from mlx_diffusers.layers.embeddings import TimestepEmbedding, Timesteps
from mlx_diffusers.layers.norm import GroupNorm
from mlx_diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
)


def _as_list(value, count: int):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * count


class UNet2DConditionModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        block_out = [int(value) for value in config["block_out_channels"]]
        down_types = list(config["down_block_types"])
        up_types = list(config["up_block_types"])
        layers = _as_list(config["layers_per_block"], len(down_types))
        heads = config.get("num_attention_heads") or config["attention_head_dim"]
        heads = _as_list(heads, len(down_types))
        cross = _as_list(config["cross_attention_dim"], len(down_types))
        groups = int(config["norm_num_groups"])
        eps = float(config["norm_eps"])
        act_fn = config["act_fn"]
        padding = int(config["downsample_padding"])
        time_dim = block_out[0] * 4
        self.conv_in = Conv2d(int(config["in_channels"]), block_out[0], kernel_size=3, padding=1)
        self.time_proj = Timesteps(block_out[0], bool(config["flip_sin_to_cos"]), float(config["freq_shift"]))
        self.time_embedding = TimestepEmbedding(block_out[0], time_dim, act_fn=act_fn)

        down_blocks = []
        in_channels = block_out[0]
        for index, kind in enumerate(down_types):
            out_channels = block_out[index]
            add_downsample = index != len(block_out) - 1
            if kind == "CrossAttnDownBlock2D":
                block = CrossAttnDownBlock2D(
                    in_channels,
                    out_channels,
                    time_dim,
                    layers[index],
                    int(heads[index]),
                    int(cross[index]),
                    groups,
                    eps,
                    act_fn,
                    add_downsample,
                    padding,
                )
            elif kind == "DownBlock2D":
                block = DownBlock2D(
                    in_channels,
                    out_channels,
                    time_dim,
                    layers[index],
                    groups,
                    eps,
                    act_fn,
                    add_downsample,
                    padding,
                )
            else:
                raise ValueError(f"unsupported down block {kind}")
            down_blocks.append(block)
            in_channels = out_channels
        self.down_blocks = down_blocks

        self.mid_block = UNetMidBlock2DCrossAttn(
            block_out[-1],
            time_dim,
            int(heads[-1]),
            int(cross[-1]),
            groups,
            eps,
            act_fn,
        )

        reversed_channels = list(reversed(block_out))
        reversed_layers = list(reversed(layers))
        reversed_heads = list(reversed(heads))
        reversed_cross = list(reversed(cross))
        up_blocks = []
        output_channel = reversed_channels[0]
        for index, kind in enumerate(up_types):
            prev_output = output_channel
            output_channel = reversed_channels[index]
            skip_channels = reversed_channels[min(index + 1, len(reversed_channels) - 1)]
            num_layers = reversed_layers[index] + 1
            add_upsample = index != len(reversed_channels) - 1
            if kind == "UpBlock2D":
                block = UpBlock2D(
                    skip_channels,
                    prev_output,
                    output_channel,
                    time_dim,
                    num_layers,
                    groups,
                    eps,
                    act_fn,
                    add_upsample,
                )
            elif kind == "CrossAttnUpBlock2D":
                block = CrossAttnUpBlock2D(
                    skip_channels,
                    prev_output,
                    output_channel,
                    time_dim,
                    num_layers,
                    int(reversed_heads[index]),
                    int(reversed_cross[index]),
                    groups,
                    eps,
                    act_fn,
                    add_upsample,
                )
            else:
                raise ValueError(f"unsupported up block {kind}")
            up_blocks.append(block)
        self.up_blocks = up_blocks
        self.conv_norm_out = GroupNorm(groups, block_out[0], eps)
        self.conv_out = Conv2d(block_out[0], int(config["out_channels"]), kernel_size=3, padding=1)

    def load_converted(self, weights: dict[str, mx.array]) -> None:
        pairs = [(key, phase2_to_parameter(key, value)) for key, value in weights.items()]
        self.load_weights(pairs, strict=True)
        self.eval()

    def __call__(self, sample: mx.array, timestep, encoder_hidden_states: mx.array) -> mx.array:
        if not isinstance(timestep, mx.array):
            timestep = mx.array(timestep)
        if timestep.ndim == 0:
            timestep = mx.full((sample.shape[0],), timestep)
        elif timestep.shape[0] != sample.shape[0]:
            timestep = mx.broadcast_to(timestep, (sample.shape[0],))
        emb = self.time_embedding(self.time_proj(timestep))
        sample = self.conv_in(sample)
        residuals = [sample]
        for block in self.down_blocks:
            if isinstance(block, CrossAttnDownBlock2D):
                sample, states = block(sample, emb, encoder_hidden_states)
            else:
                sample, states = block(sample, emb)
            residuals.extend(states)
        sample = self.mid_block(sample, emb, encoder_hidden_states)
        for block in self.up_blocks:
            count = len(block.resnets)
            skips = residuals[-count:]
            del residuals[-count:]
            if isinstance(block, CrossAttnUpBlock2D):
                sample = block(sample, skips, emb, encoder_hidden_states)
            else:
                sample = block(sample, skips, emb)
        sample = nn.silu(self.conv_norm_out(sample))
        return self.conv_out(sample)
