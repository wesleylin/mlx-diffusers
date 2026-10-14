"""GroupNorm for NCHW feature maps.

Mirror of ``torch.nn.GroupNorm`` as used by diffusers UNet resnets and the
continuous ``Transformer2DModel`` input norm. ``eps`` is explicit. MLX
normalizes channels-last, so the call converts layout. ``pytorch_compatible``
keeps PyTorch group order. Parameter names stay ``weight`` and ``bias``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_diffusers.layers import nchw_to_nhwc, nhwc_to_nchw


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_groups: int, num_channels: int, eps: float):
        super().__init__(
            num_groups,
            num_channels,
            eps=eps,
            affine=True,
            pytorch_compatible=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return nhwc_to_nchw(super().__call__(nchw_to_nhwc(x)))
