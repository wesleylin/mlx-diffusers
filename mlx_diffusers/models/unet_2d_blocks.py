"""SD1.5 UNet blocks.

Mirror of ``diffusers/models/unets/unet_2d_blocks.py`` for the block types in
the SD1.5 config: ``CrossAttnDownBlock2D``, ``DownBlock2D``,
``UNetMidBlock2DCrossAttn``, ``UpBlock2D``, and ``CrossAttnUpBlock2D``.
Channels, heads, and eps come from the caller.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_diffusers.layers.resnet import ResnetBlock2D
from mlx_diffusers.layers.updownsampling import Downsample2D, Upsample2D
from mlx_diffusers.models.transformer_2d import Transformer2DModel


def _transformer(channels: int, heads: int, cross_attention_dim: int, groups: int, eps: float) -> Transformer2DModel:
    return Transformer2DModel(
        heads,
        channels // heads,
        in_channels=channels,
        norm_num_groups=groups,
        cross_attention_dim=cross_attention_dim,
        norm_eps=eps,
    )


def _resnet(in_channels: int, out_channels: int, temb_channels: int, groups: int, eps: float, act_fn: str) -> ResnetBlock2D:
    return ResnetBlock2D(
        in_channels=in_channels,
        out_channels=out_channels,
        temb_channels=temb_channels,
        eps=eps,
        groups=groups,
        non_linearity=act_fn,
    )


class CrossAttnDownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int,
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
            attentions.append(_transformer(out_channels, num_attention_heads, cross_attention_dim, groups, eps))
        self.resnets = resnets
        self.attentions = attentions
        self.downsamplers = None
        if add_downsample:
            self.downsamplers = [
                Downsample2D(
                    out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                    name="op",
                )
            ]

    def __call__(self, hidden_states: mx.array, temb: mx.array, encoder_hidden_states: mx.array):
        states = []
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
            states.append(hidden_states)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            states.append(hidden_states)
        return hidden_states, states


class DownBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int,
        groups: int,
        eps: float,
        act_fn: str,
        add_downsample: bool,
        downsample_padding: int,
    ):
        super().__init__()
        resnets = []
        for index in range(num_layers):
            block_in = in_channels if index == 0 else out_channels
            resnets.append(_resnet(block_in, out_channels, temb_channels, groups, eps, act_fn))
        self.resnets = resnets
        self.downsamplers = None
        if add_downsample:
            self.downsamplers = [
                Downsample2D(
                    out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    padding=downsample_padding,
                    name="op",
                )
            ]

    def __call__(self, hidden_states: mx.array, temb: mx.array):
        states = []
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
            states.append(hidden_states)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
            states.append(hidden_states)
        return hidden_states, states


class UNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
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
        self.attentions = [_transformer(in_channels, num_attention_heads, cross_attention_dim, groups, eps)]

    def __call__(self, hidden_states: mx.array, temb: mx.array, encoder_hidden_states: mx.array) -> mx.array:
        hidden_states = self.resnets[0](hidden_states, temb)
        hidden_states = self.attentions[0](hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = self.resnets[1](hidden_states, temb)
        return hidden_states


class UpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int,
        groups: int,
        eps: float,
        act_fn: str,
        add_upsample: bool,
    ):
        super().__init__()
        resnets = []
        for index in range(num_layers):
            skip = in_channels if index == num_layers - 1 else out_channels
            block_in = prev_output_channel if index == 0 else out_channels
            resnets.append(_resnet(block_in + skip, out_channels, temb_channels, groups, eps, act_fn))
        self.resnets = resnets
        self.upsamplers = [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)] if add_upsample else None

    def __call__(self, hidden_states: mx.array, res_hidden_states: list[mx.array], temb: mx.array) -> mx.array:
        for resnet in self.resnets:
            skip = res_hidden_states.pop()
            hidden_states = mx.concatenate([hidden_states, skip], axis=1)
            hidden_states = resnet(hidden_states, temb)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class CrossAttnUpBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        num_layers: int,
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
            attentions.append(_transformer(out_channels, num_attention_heads, cross_attention_dim, groups, eps))
        self.resnets = resnets
        self.attentions = attentions
        self.upsamplers = [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)] if add_upsample else None

    def __call__(
        self,
        hidden_states: mx.array,
        res_hidden_states: list[mx.array],
        temb: mx.array,
        encoder_hidden_states: mx.array,
    ) -> mx.array:
        for resnet, attn in zip(self.resnets, self.attentions):
            skip = res_hidden_states.pop()
            hidden_states = mx.concatenate([hidden_states, skip], axis=1)
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states
