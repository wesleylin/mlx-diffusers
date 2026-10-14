"""Read a local diffusers component directory.

Single-file safetensors and sharded ``*.safetensors.index.json`` checkpoints are
both accepted. No torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open


def read_component(component_dir: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    directory = Path(component_dir)
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing config.json in {directory}")
    with config_path.open() as handle:
        config = json.load(handle)
    return config, read_safetensors(directory)


def read_safetensors(directory: str | Path) -> dict[str, np.ndarray]:
    directory = Path(directory)
    indexes = sorted(directory.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise RuntimeError(f"multiple safetensors indexes in {directory}")
    if len(indexes) == 1:
        return _read_sharded(directory, indexes[0])
    files = sorted(directory.glob("*.safetensors"))
    if len(files) != 1:
        raise FileNotFoundError(f"expected one safetensors file in {directory}, found {len(files)}")
    return _read_file(files[0])


def _read_sharded(directory: Path, index_path: Path) -> dict[str, np.ndarray]:
    with index_path.open() as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"{index_path} has no weight_map")
    by_file: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        by_file.setdefault(str(filename), []).append(str(key))
    tensors: dict[str, np.ndarray] = {}
    for filename, keys in by_file.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"shard {path} listed in {index_path.name} is missing")
        loaded = _read_file(path)
        for key in keys:
            if key not in loaded:
                raise KeyError(f"{key} missing from shard {filename}")
            if key in tensors:
                raise RuntimeError(f"duplicate key {key}")
            tensors[key] = loaded[key]
        extra = set(loaded) - set(keys)
        if extra:
            raise RuntimeError(f"shard {filename} has keys absent from the index: {sorted(extra)}")
    if set(tensors) != set(weight_map):
        raise RuntimeError("sharded checkpoint keys do not match weight_map")
    return tensors


def _read_file(path: Path) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    with safe_open(path, framework="numpy") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    return tensors
