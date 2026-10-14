"""SD1.5 variational autoencoder.

Mirror of ``diffusers/models/autoencoders/autoencoder_kl.py`` and
``diffusers/models/autoencoders/vae.py`` ``Encoder`` / ``Decoder``. GroupNorm
``eps`` is ``1e-6``, matching the installed encoder and decoder. Mid-block
attention keeps the checkpoint names ``query``, ``key``, ``value``, and
``proj_attn``. ``decode`` and ``encode`` do not apply ``scaling_factor``;
the pipeline scales latents.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_diffusers.layers import Conv2d, phase2_to_parameter
from mlx_diffusers.layers.norm import GroupNorm
from mlx_diffusers.layers.resnet import ResnetBlock2D
from mlx_diffusers.layers.updownsampling import Downsample2D, Upsample2D

# diffusers Encoder/Decoder hardcode this GroupNorm eps.
VAE_NORM_EPS = 1e-6

# Current diffusers ``Attention`` names. Older VAE checkpoints still use
# ``query`` / ``key`` / ``value`` / ``proj_attn``. The forward is the same.
_VAE_ATTENTION_RENAMES = (
    (".to_q.", ".query."),
    (".to_k.", ".key."),
    (".to_v.", ".value."),
    (".to_out.0.", ".proj_attn."),
)


def _vae_attention_key(key: str) -> str:
    for source, dest in _VAE_ATTENTION_RENAMES:
        if source in key:
            return key.replace(source, dest, 1)
    return key


def _split_heads(x: mx.array, heads: int) -> mx.array:
    batch, seq, channels = x.shape
    head_dim = channels // heads
    x = x.reshape(batch, seq, heads, head_dim)
    return mx.transpose(x, (0, 2, 1, 3))


def _merge_heads(x: mx.array) -> mx.array:
    batch, heads, seq, head_dim = x.shape
    x = mx.transpose(x, (0, 2, 1, 3))
    return x.reshape(batch, seq, heads * head_dim)


class AttentionBlock(nn.Module):
    """Self-attention with the deprecated VAE checkpoint names."""

    def __init__(self, channels: int, num_groups: int, eps: float, head_dim: int):
        super().__init__()
        self.heads = channels // head_dim
        self.scale = head_dim**-0.5
        self.group_norm = GroupNorm(num_groups, channels, eps)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj_attn = nn.Linear(channels, channels)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        residual = hidden_states
        batch, channels, height, width = hidden_states.shape
        hidden_states = self.group_norm(hidden_states)
        hidden_states = mx.transpose(hidden_states.reshape(batch, channels, height * width), (0, 2, 1))
        query = _split_heads(self.query(hidden_states), self.heads)
        key = _split_heads(self.key(hidden_states), self.heads)
        value = _split_heads(self.value(hidden_states), self.heads)
        hidden_states = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale)
        hidden_states = _merge_heads(hidden_states)
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = mx.transpose(hidden_states, (0, 2, 1)).reshape(batch, channels, height, width)
        return hidden_states + residual


class DownEncoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
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
            resnets.append(
                ResnetBlock2D(
                    in_channels=block_in,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=eps,
                    groups=groups,
                    non_linearity=act_fn,
                )
            )
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

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, None)
        if self.downsamplers is not None:
            hidden_states = self.downsamplers[0](hidden_states)
        return hidden_states


class UpDecoderBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        groups: int,
        eps: float,
        act_fn: str,
        add_upsample: bool,
    ):
        super().__init__()
        resnets = []
        for index in range(num_layers):
            block_in = in_channels if index == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=block_in,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=eps,
                    groups=groups,
                    non_linearity=act_fn,
                )
            )
        self.resnets = resnets
        self.upsamplers = None
        if add_upsample:
            self.upsamplers = [Upsample2D(out_channels, use_conv=True, out_channels=out_channels)]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, None)
        if self.upsamplers is not None:
            hidden_states = self.upsamplers[0](hidden_states)
        return hidden_states


class UNetMidBlock2D(nn.Module):
    def __init__(self, channels: int, groups: int, eps: float, act_fn: str, head_dim: int):
        super().__init__()
        self.resnets = [
            ResnetBlock2D(
                in_channels=channels,
                out_channels=channels,
                temb_channels=None,
                eps=eps,
                groups=groups,
                non_linearity=act_fn,
            ),
            ResnetBlock2D(
                in_channels=channels,
                out_channels=channels,
                temb_channels=None,
                eps=eps,
                groups=groups,
                non_linearity=act_fn,
            ),
        ]
        self.attentions = [AttentionBlock(channels, groups, eps, head_dim)]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.resnets[0](hidden_states, None)
        hidden_states = self.attentions[0](hidden_states)
        hidden_states = self.resnets[1](hidden_states, None)
        return hidden_states


class Encoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        channels = [int(value) for value in config["block_out_channels"]]
        groups = int(config["norm_num_groups"])
        layers = int(config["layers_per_block"])
        act_fn = config["act_fn"]
        self.conv_in = Conv2d(int(config["in_channels"]), channels[0], kernel_size=3, padding=1)
        blocks = []
        in_channels = channels[0]
        for index, out_channels in enumerate(channels):
            blocks.append(
                DownEncoderBlock2D(
                    in_channels,
                    out_channels,
                    num_layers=layers,
                    groups=groups,
                    eps=VAE_NORM_EPS,
                    act_fn=act_fn,
                    add_downsample=index != len(channels) - 1,
                    downsample_padding=0,
                )
            )
            in_channels = out_channels
        self.down_blocks = blocks
        self.mid_block = UNetMidBlock2D(channels[-1], groups, VAE_NORM_EPS, act_fn, channels[-1])
        self.conv_norm_out = GroupNorm(groups, channels[-1], VAE_NORM_EPS)
        self.conv_out = Conv2d(channels[-1], 2 * int(config["latent_channels"]), kernel_size=3, padding=1)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.conv_in(sample)
        for block in self.down_blocks:
            sample = block(sample)
        sample = self.mid_block(sample)
        sample = nn.silu(self.conv_norm_out(sample))
        return self.conv_out(sample)


class Decoder(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        channels = [int(value) for value in config["block_out_channels"]]
        groups = int(config["norm_num_groups"])
        layers = int(config["layers_per_block"]) + 1
        act_fn = config["act_fn"]
        reversed_channels = list(reversed(channels))
        self.conv_in = Conv2d(int(config["latent_channels"]), reversed_channels[0], kernel_size=3, padding=1)
        self.mid_block = UNetMidBlock2D(reversed_channels[0], groups, VAE_NORM_EPS, act_fn, reversed_channels[0])
        blocks = []
        in_channels = reversed_channels[0]
        for index, out_channels in enumerate(reversed_channels):
            blocks.append(
                UpDecoderBlock2D(
                    in_channels,
                    out_channels,
                    num_layers=layers,
                    groups=groups,
                    eps=VAE_NORM_EPS,
                    act_fn=act_fn,
                    add_upsample=index != len(reversed_channels) - 1,
                )
            )
            in_channels = out_channels
        self.up_blocks = blocks
        self.conv_norm_out = GroupNorm(groups, channels[0], VAE_NORM_EPS)
        self.conv_out = Conv2d(channels[0], int(config["out_channels"]), kernel_size=3, padding=1)

    def __call__(self, sample: mx.array) -> mx.array:
        sample = self.conv_in(sample)
        sample = self.mid_block(sample)
        for block in self.up_blocks:
            sample = block(sample)
        sample = nn.silu(self.conv_norm_out(sample))
        return self.conv_out(sample)


class AutoencoderKL(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        latent = int(config["latent_channels"])
        self.scaling_factor = float(config.get("scaling_factor", 0.18215))
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = Conv2d(2 * latent, 2 * latent, kernel_size=1)
        self.post_quant_conv = Conv2d(latent, latent, kernel_size=1)

    def load_converted(self, weights: dict[str, mx.array]) -> None:
        pairs = [( _vae_attention_key(key), phase2_to_parameter(key, value)) for key, value in weights.items()]
        self.load_weights(pairs, strict=True)
        self.eval()

    def encode(self, sample: mx.array) -> mx.array:
        """Posterior mean. Matches ``encode(x).latent_dist.mean`` with no latent scaling."""
        moments = self.quant_conv(self.encoder(sample))
        mean, _logvar = mx.split(moments, 2, axis=1)
        return mean

    def decode(self, latents: mx.array) -> mx.array:
        """Decoded image. Matches ``decode(z).sample`` with no latent scaling."""
        return self.decoder(self.post_quant_conv(latents))
