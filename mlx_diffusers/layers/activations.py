"""Activations used by the SD1.5 UNet.

Mirror of diffusers/models/activations.py. ``swish`` and ``silu`` are SiLU.
``GEGLU`` is the feed-forward gate: project to ``2 * dim_out``, then
``hidden * gelu(gate)``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SiLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(x)


class GELU(nn.Module):
    """Project, then exact (erf) GELU. ``approximate='tanh'`` uses the tanh form."""

    def __init__(self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.proj(hidden_states)
        if self.approximate == "tanh":
            return nn.gelu_approx(hidden_states)
        if self.approximate == "none":
            return nn.gelu(hidden_states)
        raise ValueError(f"unknown GELU approximate={self.approximate!r}")


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.proj(hidden_states)
        hidden, gate = mx.split(hidden_states, 2, axis=-1)
        return hidden * nn.gelu(gate)
