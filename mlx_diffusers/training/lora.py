"""LoRA for attention projections.

``LoRALinear`` follows the MLX-LM tuner (Apple, MIT): frozen base linear, ``A``
initialized uniform in ``[-1/sqrt(in), 1/sqrt(in)]``, ``B`` initialized at
zero, and ``y + scale * dropout(x) @ A @ B``. ``scale`` is ``alpha / rank``.
Injection targets UNet attention ``to_q``, ``to_k``, ``to_v``, and ``to_out.0``.
"""

from __future__ import annotations

import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from safetensors.numpy import save_file

LORA_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        output_dims, input_dims = linear.weight.shape
        self.linear = linear
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = float(alpha) / float(rank)
        bound = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(low=-bound, high=bound, shape=(input_dims, rank))
        self.lora_b = mx.zeros((rank, output_dims))

    def __call__(self, x: mx.array) -> mx.array:
        base = self.linear(x)
        update = (x @ self.lora_a) @ self.lora_b
        return base + (self.scale * update).astype(x.dtype)


def _is_target(name: str) -> bool:
    return any(name == target or name.endswith("." + target) for target in LORA_TARGETS)


def _set_module(root: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


def inject(model: nn.Module, rank: int, alpha: float) -> list[str]:
    """Replace attention projections with LoRA and freeze everything else."""
    names = [name for name, module in model.named_modules() if name and _is_target(name) and isinstance(module, nn.Linear)]
    for name in names:
        current = model
        for part in name.split("."):
            current = current[int(part)] if part.isdigit() else getattr(current, part)
        _set_module(model, name, LoRALinear(current, rank, alpha))
    model.freeze()
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unfreeze()
            module.linear.freeze()
    return names


def lora_arrays(model: nn.Module) -> dict[str, np.ndarray]:
    arrays = {}
    for key, value in tree_flatten(model.parameters()):
        if key.endswith(".lora_a") or key.endswith(".lora_b"):
            mx.eval(value)
            arrays[key] = np.array(value)
    return arrays


def save_lora(model: nn.Module, path: str | Path, rank: int, alpha: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = lora_arrays(model)
    arrays["rank"] = np.array(rank, dtype=np.int64)
    arrays["alpha"] = np.array(alpha, dtype=np.float32)
    save_file(arrays, path)


def zero_lora_b(model: nn.Module) -> None:
    pairs = [(key, mx.zeros_like(value)) for key, value in tree_flatten(model.parameters()) if key.endswith(".lora_b")]
    model.load_weights(pairs, strict=False)
    mx.eval(model.parameters())


def load_lora(model: nn.Module, path: str | Path) -> tuple[int, float]:
    from safetensors import safe_open

    pairs = []
    with safe_open(path, framework="numpy") as handle:
        rank = int(handle.get_tensor("rank"))
        alpha = float(handle.get_tensor("alpha"))
        for key in handle.keys():
            if key in {"rank", "alpha"}:
                continue
            pairs.append((key, mx.array(handle.get_tensor(key))))
    model.load_weights(pairs, strict=False)
    mx.eval(model.parameters())
    return rank, alpha
