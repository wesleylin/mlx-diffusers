"""Self and cross attention.

Mirror of ``diffusers/models/attention_processor.py`` ``Attention`` and
``AttnProcessor2_0``. The score scale is ``1 / sqrt(head_dim)`` and the
scores go through ``mx.fast.scaled_dot_product_attention``. ``to_q`` / ``to_k``
/ ``to_v`` and ``to_out.0`` keep the diffusers names. Dropout is ``to_out.1``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _split_heads(x: mx.array, heads: int) -> mx.array:
    batch, seq, channels = x.shape
    head_dim = channels // heads
    x = x.reshape(batch, seq, heads, head_dim)
    return mx.transpose(x, (0, 2, 1, 3))


def _merge_heads(x: mx.array) -> mx.array:
    batch, heads, seq, head_dim = x.shape
    x = mx.transpose(x, (0, 2, 1, 3))
    return x.reshape(batch, seq, heads * head_dim)


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = False,
        cross_attention_dim: int | None = None,
        out_bias: bool = True,
        rescale_output_factor: float = 1.0,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.rescale_output_factor = rescale_output_factor
        inner_dim = dim_head * heads
        kv_dim = query_dim if cross_attention_dim is None else cross_attention_dim
        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(kv_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, inner_dim, bias=bias)
        self.to_out = [
            nn.Linear(inner_dim, query_dim, bias=out_bias),
            nn.Dropout(dropout),
        ]

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array | None = None,
    ) -> mx.array:
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch, channel, height, width = hidden_states.shape
            hidden_states = mx.transpose(hidden_states.reshape(batch, channel, height * width), (0, 2, 1))

        query = _split_heads(self.to_q(hidden_states), self.heads)
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        key = _split_heads(self.to_k(context), self.heads)
        value = _split_heads(self.to_v(context), self.heads)
        hidden_states = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.scale)
        hidden_states = self.to_out[0](_merge_heads(hidden_states))
        hidden_states = self.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = mx.transpose(hidden_states, (0, 2, 1)).reshape(batch, channel, height, width)
        if self.rescale_output_factor != 1.0:
            hidden_states = hidden_states / self.rescale_output_factor
        return hidden_states
