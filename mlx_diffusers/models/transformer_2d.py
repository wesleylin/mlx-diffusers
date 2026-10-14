"""Continuous 2D transformer block.

Mirror of ``diffusers/models/transformers/transformer_2d.py``
``Transformer2DModel`` (continuous input) and
``diffusers/models/attention.py`` ``BasicTransformerBlock`` (layer norm).
The input GroupNorm uses the diffusers hard-coded ``eps=1e-6``. Block
LayerNorm uses ``norm_eps`` (default ``1e-5``). Heads, head dim, and
``cross_attention_dim`` come from the caller.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_diffusers.layers import Conv2d
from mlx_diffusers.layers.attention import Attention
from mlx_diffusers.layers.feedforward import FeedForward
from mlx_diffusers.layers.norm import GroupNorm

# diffusers Transformer2DModel._init_continuous_input hardcodes this eps.
CONTINUOUS_NORM_EPS = 1e-6


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: int | None = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
        )
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
        )
        self.norm3 = nn.LayerNorm(dim, eps=norm_eps)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states), encoder_hidden_states=encoder_hidden_states) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states
        return hidden_states


class Transformer2DModel(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        in_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: int | None = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        norm_eps: float = 1e-5,
        use_linear_projection: bool = False,
    ):
        super().__init__()
        if use_linear_projection:
            raise ValueError("linear projection is not the SD1.5 UNet path")
        inner_dim = num_attention_heads * attention_head_dim
        self.norm = GroupNorm(norm_num_groups, in_channels, eps=CONTINUOUS_NORM_EPS)
        self.proj_in = Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = [
            BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                activation_fn=activation_fn,
                attention_bias=attention_bias,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ]
        self.proj_out = Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        residual = hidden_states
        batch, _, height, width = hidden_states.shape
        hidden_states = self.proj_in(self.norm(hidden_states))
        inner_dim = hidden_states.shape[1]
        hidden_states = mx.transpose(hidden_states, (0, 2, 3, 1)).reshape(batch, height * width, inner_dim)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = mx.transpose(hidden_states.reshape(batch, height, width, inner_dim), (0, 3, 1, 2))
        return self.proj_out(hidden_states) + residual
