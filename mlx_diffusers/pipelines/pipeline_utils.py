"""Pipeline loader.

``PIPELINE_REGISTRY`` is keyed by ``model_index.json`` ``_class_name``.
``from_pretrained`` accepts a local diffusers directory or a Hub repo id.
Hub downloads go through ``huggingface_hub`` and do not import torch.
"""

from __future__ import annotations

import json
from pathlib import Path

PIPELINE_REGISTRY: dict[str, type] = {}


class UnknownPipeline(Exception):
    def __init__(self, class_name: str):
        self.class_name = class_name
        super().__init__(f"unknown pipeline {class_name!r}")


def resolve_pretrained(model_id_or_path: str | Path) -> Path:
    path = Path(model_id_or_path)
    if path.is_dir() and (path / "model_index.json").is_file():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(str(model_id_or_path)))


def read_model_index(model_dir: str | Path) -> dict:
    with (Path(model_dir) / "model_index.json").open() as handle:
        return json.load(handle)


def from_pretrained(model_id_or_path: str | Path):
    """Build the pipeline named in ``model_index.json``."""
    import mlx_diffusers.pipelines.pipeline_stable_diffusion  # noqa: F401

    model_dir = resolve_pretrained(model_id_or_path)
    index = read_model_index(model_dir)
    class_name = str(index["_class_name"])
    cls = PIPELINE_REGISTRY.get(class_name)
    if cls is None:
        raise UnknownPipeline(class_name)
    return cls.from_pretrained(model_dir, index)
