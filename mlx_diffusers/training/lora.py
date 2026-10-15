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
        self.lora_a = mx.random.uniform(low=-bound, high=bound, shape=(input_dims, rank)).astype(mx.float32)
        self.lora_b = mx.zeros((rank, output_dims), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        base = self.linear(x)
        # LoRA stays float32 so an fp16 base checkpoint does not overflow the update.
        residual = x.astype(self.lora_a.dtype)
        update = (residual @ self.lora_a) @ self.lora_b
        return base + (self.scale * update).astype(x.dtype)


def _is_target(name: str, targets: tuple[str, ...]) -> bool:
    return any(name == target or name.endswith("." + target) for target in targets)


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


def inject(model: nn.Module, rank: int, alpha: float, targets: tuple[str, ...] = LORA_TARGETS) -> list[str]:
    """Replace attention projections with LoRA and freeze everything else."""
    names = [
        name
        for name, module in model.named_modules()
        if name and _is_target(name, targets) and isinstance(module, nn.Linear)
    ]
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


_KOHYA_PREFIX = {"unet": "lora_unet_", "te": "lora_te_", "te2": "lora_te2_"}


def _kohya_stem(component: str, module_name: str) -> str:
    return _KOHYA_PREFIX[component] + module_name.replace(".", "_")


def save_kohya(models: dict[str, nn.Module], path: str | Path, alpha: float) -> None:
    """Kohya keys: ``lora_unet_`` / ``lora_te_`` / ``lora_te2_`` plus ``.alpha``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for component, model in models.items():
        if component not in _KOHYA_PREFIX:
            raise KeyError(component)
        for key, value in lora_arrays(model).items():
            if key.endswith(".lora_a"):
                module_name = key[: -len(".lora_a")]
                stem = _kohya_stem(component, module_name)
                arrays[stem + ".lora_down.weight"] = np.ascontiguousarray(np.transpose(value))
                arrays[stem + ".alpha"] = np.array(alpha, dtype=np.float32)
            elif key.endswith(".lora_b"):
                module_name = key[: -len(".lora_b")]
                stem = _kohya_stem(component, module_name)
                arrays[stem + ".lora_up.weight"] = np.ascontiguousarray(np.transpose(value))
    save_file(arrays, path)


def load_kohya(models: dict[str, nn.Module], path: str | Path) -> None:
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as handle:
        keys = set(handle.keys())
        for component, model in models.items():
            pairs = []
            for name, module in model.named_modules():
                if not isinstance(module, LoRALinear):
                    continue
                stem = _kohya_stem(component, name)
                down_key = stem + ".lora_down.weight"
                up_key = stem + ".lora_up.weight"
                alpha_key = stem + ".alpha"
                if down_key not in keys or up_key not in keys or alpha_key not in keys:
                    raise KeyError(f"missing kohya entries for {stem}")
                down = handle.get_tensor(down_key)
                up = handle.get_tensor(up_key)
                pairs.append((name + ".lora_a", mx.array(np.ascontiguousarray(np.transpose(down)))))
                pairs.append((name + ".lora_b", mx.array(np.ascontiguousarray(np.transpose(up)))))
            if pairs:
                model.load_weights(pairs, strict=False)
                mx.eval(model.parameters())


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
