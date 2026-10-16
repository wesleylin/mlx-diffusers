"""Quantize frozen linear weights.

Convolutions stay in their original dtype. LoRA adapters are not linears, so
they stay float32. This is a spike: the training script only calls it behind
``--quantize-base``.
"""

from __future__ import annotations

import mlx.nn as nn


def quantize_frozen_linears(model: nn.Module, group_size: int = 64, bits: int = 4) -> int:
    """Replace frozen ``nn.Linear`` layers with affine quantized weights.

    A linear whose last dimension is not divisible by ``group_size`` is left
    alone. ``nn.quantize`` updates the module in place.
    """

    def predicate(_path, module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        return module.weight.shape[-1] % group_size == 0

    before = sum(1 for _, module in model.named_modules() if isinstance(module, nn.Linear) and predicate("", module))
    if before == 0:
        return 0
    nn.quantize(model, group_size=group_size, bits=bits, mode="affine", class_predicate=predicate)
    return sum(1 for _, module in model.named_modules() if isinstance(module, nn.QuantizedLinear))
