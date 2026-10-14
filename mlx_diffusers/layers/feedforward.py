"""Feed-forward block.

Mirror of ``diffusers/models/attention.py`` ``FeedForward``. The SD1.5 UNet
uses ``activation_fn="geglu"``, so the parameter path is ``net.0`` (GEGLU),
``net.1`` (dropout), ``net.2`` (linear).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_diffusers.layers.activations import GEGLU, GELU


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        inner_dim: int | None = None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim if dim_out is None else dim_out
        if activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, approximate="none", bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        else:
            raise ValueError(f"FeedForward activation_fn {activation_fn!r} is not used by the SD1.5 UNet")
        self.net = [
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias=bias),
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
