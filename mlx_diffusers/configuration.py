"""Architecture registry and weight loading.

One registry, keyed by ``config.json`` ``_class_name``. Converter lookup happens
inside ``load_model``, not at import time. ``CLASS_NAME_ALIASES`` only renames a
transformers class onto that same registry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

ARCH_REGISTRY_KEYS = (
    "UNet2DConditionModel",
    "AutoencoderKL",
    "CLIPTextModel",
)

# Not a second registry. Values are keys of ARCH_REGISTRY.
CLASS_NAME_ALIASES = {
    "CLIPTextModel": "CLIPTextModel",
}


class UnknownClassName(Exception):
    """Raised when ``_class_name`` is not in the registry. Never guessed."""

    def __init__(self, class_name: str):
        self.class_name = class_name
        super().__init__(f"unknown _class_name {class_name!r}")


@dataclass
class ArchEntry:
    """A registered architecture. ``cls`` stays unset until the phase that builds the module."""

    class_name: str
    cls: type | None = None


ARCH_REGISTRY: dict[str, ArchEntry] = {name: ArchEntry(name) for name in ARCH_REGISTRY_KEYS}


@dataclass
class LoadResult:
    class_name: str
    config: dict[str, Any]
    weights: dict[str, mx.array]
    unmapped: tuple[str, ...]
    whitelisted: tuple[str, ...]
    module_registered: bool
    message: str


def class_name_from_config(config: dict[str, Any]) -> str:
    """Read ``_class_name``, or a transformers ``architectures`` entry when it is absent."""
    if "_class_name" in config and config["_class_name"]:
        raw = str(config["_class_name"])
    elif config.get("architectures"):
        raw = str(config["architectures"][0])
    else:
        raise UnknownClassName("<missing>")
    return CLASS_NAME_ALIASES.get(raw, raw)


def resolve_class_name(class_name: str) -> str:
    canonical = CLASS_NAME_ALIASES.get(class_name, class_name)
    if canonical not in ARCH_REGISTRY:
        raise UnknownClassName(class_name)
    return canonical


def load_model(component_dir: str | Path) -> LoadResult:
    """Map a local diffusers component directory to MLX weights.

    The module class is not invented here. When ``ARCH_REGISTRY`` has no ``cls``,
    the converted weights are returned and the module is reported as not registered.
    """
    from mlx_diffusers.utils.convert import CONVERTERS
    from mlx_diffusers.utils.loading import read_safetensors

    config = read_config(component_dir)
    class_name = resolve_class_name(class_name_from_config(config))
    mapper = CONVERTERS.get(class_name)
    if mapper is None:
        raise UnknownClassName(class_name)
    tensors = read_safetensors(component_dir)
    converted = mapper(tensors, config)
    if converted.unmapped:
        joined = ", ".join(converted.unmapped)
        raise RuntimeError(f"unmapped keys in {class_name}: {joined}")

    weights: dict[str, mx.array] = {}
    for key, value in converted.weights.items():
        array = mx.array(value)
        mx.eval(array)
        weights[key] = array

    entry = ARCH_REGISTRY[class_name]
    registered = entry.cls is not None
    return LoadResult(
        class_name=class_name,
        config=config,
        weights=weights,
        unmapped=tuple(converted.unmapped),
        whitelisted=tuple(converted.whitelisted),
        module_registered=registered,
        message="module is registered" if registered else "module is not registered",
    )


def read_config(component_dir: str | Path) -> dict[str, Any]:
    path = Path(component_dir) / "config.json"
    with path.open() as handle:
        return json.load(handle)
