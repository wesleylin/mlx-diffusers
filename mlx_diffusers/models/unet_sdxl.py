"""SDXL conditional UNet.

Separate from the SD1.5 UNet because the config sets ``use_linear_projection``
and ``addition_embed_type=text_time``. The SD1.5 transformer rejects linear
projection, so this file owns that path. ``transformer_layers_per_block`` and
head counts come from the config. ``layers/`` is not modified.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_diffusers.layers import Conv2d, phase2_to_parameter
from mlx_diffusers.layers.embeddings import TimestepEmbedding, Timesteps
from mlx_diffusers.layers.norm import GroupNorm
from mlx_diffusers.layers.resnet import ResnetBlock2D
from mlx_diffusers.layers.updownsampling import Downsample2D, Upsample2D
from mlx_diffusers.models.transformer_2d import CONTINUOUS_NORM_EPS, BasicTransformerBlock
from mlx_diffusers.models.unet_2d_blocks import DownBlock2D, UpBlock2D


def _as_list(value, count: int):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value] * count


class LinearTransformer2D(nn.Module):
    """Continuous transformer whose ``proj_in`` / ``proj_out`` are linears."""

    def __init__(
        self,
        num_attention_heads: int,
        in_channels: int,
        num_layers: int,
        cross_attention_dim: int,
        norm_num_groups: int,
        norm_eps: float,
    ):
        super().__init__()
        head_dim = in_channels // num_attention_heads
        inner = num_attention_heads * head_dim
        self.norm = GroupNorm(norm_num_groups, in_channels, eps=CONTINUOUS_NORM_EPS)
        self.proj_in = nn.Linear(in_channels, inner)
        self.transformer_blocks = [
            BasicTransformerBlock(
                inner,
                num_attention_heads,
                head_dim,
                cross_attention_dim=cross_attention_dim,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ]
        self.proj_out = nn.Linear(inner, in_channels)

    def __call__(self, hidden_states: mx.array, encoder_hidden_states: mx.array | None = None) -> mx.array:
        residual = hidden_states
        batch, _, height, width = hidden_states.shape
        hidden_states = self.norm(hidden_states)
        inner = hidden_states.shape[1]
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1)).reshape(batch, height * width, inner)
        hidden_states = self.proj_in(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
            # Materialize each block so a 10-layer SDXL transformer does not
            # keep the whole graph live on a 16GB machine.
            mx.eval(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = mx.transpose(hidden_states.reshape(batch, height, width, inner), (0, 3, 1, 2))
        return hidden_states + residual


def _resnet(in_channels: int, out_channels: int, temb_channels: int, groups: int, eps: float, act_fn: str) -> ResnetBlock2D:
    return ResnetBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        eps=eps,
        groups=groups,
        non_linearity=act_fn,
    )


class CrossAttnDownBlockXL(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int,
        transformer_layers: int,
        num_attention_heads: int,
        cross_attention_dim: int,
        groups: int,
        eps: float,
        act_fn: str,
        add_downsample: bool,
        downsample_padding: int,
    ):
        super().__init__()
        resnets = []
        attentions = []
        for index in range(num_layers):
            block_in = in_channels if index == 0 else out_channels
            resnets.append(_resnet(block_in, out_channels, temb_channels, groups, eps, act_fn))
            attentions.append(
                LinearTransformer2D(
                    num_attention_heads,
                    out_channels,
                    transformer_layers,
                    cross_attention_dim,
                    groups,
                    eps,
                )
            )
        self.resnets = resnets
        self.attentions = attentions
        self.downsamplers = None
        if add_downsample:
            self.downsamplers = [
                Downsample2D(out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op")
            ]

    def __call__(self, hidden_states, temb, encoder_hidden_states):
        states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            states.append(hidden_states)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            states.append(hidden_states)
        return hidden_states, states


class CrossAttnUpBlockXL(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int,
        transformer_layers: int,
        num_attention_heads: int,
        cross_attention_dim: int,
        groups: int,
        eps: float,
        act_fn: str,
        add_upsample: bool,
    ):
        super().__init__()
        resnets = []
        attentions = []
        for index in range(num_layers):
            skip = in_channels if index == num_layers - 1 else out_channels
            block_in = prev_output_channel if index == 0 else out_channels
            resnets.append(_resnet(block_in + skip, out_channels, temb_channels, groups, eps, act_fn))
            attentions.append(
                LinearTransformer2D(
                    num_attention_heads,
                    out_channels,
                    transformer_layers,
                    cross_attention_dim,
                    groups,
                    eps,
                )
            )
        self.resnets = resnets
        self.attentions = attentions
        self.upsamplers = [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)] if add_upsample else None

    def __call__(self, hidden_states, res_hidden_states, temb, encoder_hidden_states):
        for resnet, attn in zip(self.resnets, self.attentions):
            skip = res_hidden_states.pop()
            hidden_states = mx.concatenate([hidden_states, skip], axis=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class UNetMidBlockXL(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        transformer_layers: int,
        num_attention_heads: int,
        cross_attention_dim: int,
        groups: int,
        eps: float,
        act_fn: str,
    ):
        super().__init__()
        self.resnets = [
            _resnet(in_channels, in_channels, temb_channels, groups, eps, act_fn),
            _resnet(in_channels, in_channels, temb_channels, groups, eps, act_fn),
        ]
        self.attentions = [
            LinearTransformer2D(
                num_attention_heads,
                in_channels,
                transformer_layers,
                cross_attention_dim,
                groups,
                eps,
            )
        ]

    def __call__(self, hidden_states, temb, encoder_hidden_states):
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states


class UNet2DConditionXL(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if config.get("addition_embed_type") != "text_time":
            raise ValueError("UNet2DConditionXL requires addition_embed_type text_time")
        block_out = [int(value) for value in config["block_out_channels"]]
        down_types = list(config["down_block_types"])
        up_types = list(config["up_block_types"])
        count = len(down_types)
        layers = _as_list(config["layers_per_block"], count)
        head_source = config.get("num_attention_heads") or config["attention_head_dim"]
        heads = [int(value) for value in _as_list(head_source, count)]
        cross = [int(value) for value in _as_list(config["cross_attention_dim"], count)]
        transformers = [int(value) for value in _as_list(config.get("transformer_layers_per_block", 1), count)]
        groups = int(config["norm_num_groups"])
        eps = float(config["norm_eps"])
        act_fn = config["act_fn"]
        padding = int(config["downsample_padding"])
        time_dim = block_out[0] * 4
        self.conv_in = Conv2d(int(config["in_channels"]), block_out[0], kernel_size=3, padding=1)
        self.time_proj = Timesteps(block_out[0], bool(config["flip_sin_to_cos"]), float(config["freq_shift"]))
        self.time_embedding = TimestepEmbedding(block_out[0], time_dim, act_fn=act_fn)
        self.add_time_proj = Timesteps(
            int(config["addition_time_embed_dim"]),
            bool(config["flip_sin_to_cos"]),
            float(config["freq_shift"]),
        )
        self.add_embedding = TimestepEmbedding(int(config["projection_class_embeddings_input_dim"]), time_dim, act_fn=act_fn)

        down_blocks = []
        in_channels = block_out[0]
        for index, kind in enumerate(down_types):
            out_channels = block_out[index]
            add_downsample = index != len(block_out) - 1
            if kind == "CrossAttnDownBlock2D":
                block = CrossAttnDownBlockXL(
                    in_channels,
                    out_channels,
                    time_dim,
                    layers[index],
                    transformers[index],
                    heads[index],
                    cross[index],
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
        self.mid_block = UNetMidBlockXL(
            block_out[-1],
            time_dim,
            transformers[-1],
            heads[-1],
            cross[-1],
            groups,
            eps,
            act_fn,
        )

        reversed_channels = list(reversed(block_out))
        reversed_layers = list(reversed(layers))
        reversed_heads = list(reversed(heads))
        reversed_cross = list(reversed(cross))
        reversed_transformers = list(reversed(transformers))
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
                block = CrossAttnUpBlockXL(
                    skip_channels,
                    prev_output,
                    output_channel,
                    time_dim,
                    num_layers,
                    reversed_transformers[index],
                    reversed_heads[index],
                    reversed_cross[index],
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

    def __call__(
        self,
        sample: mx.array,
        timestep,
        encoder_hidden_states: mx.array,
        added_cond_kwargs: dict[str, mx.array],
    ) -> mx.array:
        if not isinstance(timestep, mx.array):
            timestep = mx.array(timestep)
        if timestep.ndim == 0:
            timestep = mx.full((sample.shape[0],), timestep)
        elif timestep.shape[0] != sample.shape[0]:
            timestep = mx.broadcast_to(timestep, (sample.shape[0],))
        emb = self.time_embedding(self.time_proj(timestep))
        text_embeds = added_cond_kwargs["text_embeds"]
        time_ids = added_cond_kwargs["time_ids"]
        time_embeds = self.add_time_proj(time_ids.reshape((-1,))).reshape((text_embeds.shape[0], -1))
        aug = self.add_embedding(mx.concatenate([text_embeds, time_embeds], axis=-1))
        emb = emb + aug
        sample = self.conv_in(sample)
        residuals = [sample]
        for block in self.down_blocks:
            if isinstance(block, CrossAttnDownBlockXL):
                sample, states = block(sample, emb, encoder_hidden_states)
            else:
                sample, states = block(sample, emb)
            residuals.extend(states)
        sample = self.mid_block(sample, emb, encoder_hidden_states)
        for block in self.up_blocks:
            count = len(block.resnets)
            skips = residuals[-count:]
            del residuals[-count:]
            if isinstance(block, CrossAttnUpBlockXL):
                sample = block(sample, skips, emb, encoder_hidden_states)
            else:
                sample = block(sample, skips, emb)
        sample = nn.silu(self.conv_norm_out(sample))
        return self.conv_out(sample)
