"""Map diffusers safetensors onto MLX weight layouts.

One mapper per architecture that Phase 2 converts. Keys stay diffusers names.
Transpose only Linear and Conv2d. Layouts follow PyTorch parameter storage, which
the installed diffusers and transformers modules use (Apache-2.0): ``nn.Linear``
is ``[out, in]`` and ``nn.Conv2d`` is ``[out, in, kh, kw]``. Rank comes from the
checkpoint. Embeddings are recognized only when their shape matches the component
``config.json``. Norms, biases, and embeddings are not transposed.

``text_model.embeddings.position_ids`` is the CLIP position-id buffer and is
whitelisted by that name.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

# CLIP text encoder buffer. Named here so it is never a silent drop.
CLIP_POSITION_IDS = "text_model.embeddings.position_ids"

Mapper = Callable[[dict[str, np.ndarray], dict], "Conversion"]


@dataclass
class Conversion:
    weights: dict[str, np.ndarray]
    unmapped: list[str] = field(default_factory=list)
    whitelisted: list[str] = field(default_factory=list)


def transpose_linear(weight: np.ndarray) -> np.ndarray:
    """``[out, in]`` to ``[in, out]``."""
    if weight.ndim != 2:
        raise ValueError(f"Linear weight must be rank 2, got {weight.shape}")
    return np.transpose(weight, (1, 0))


def transpose_conv2d(weight: np.ndarray) -> np.ndarray:
    """``[out, in, kh, kw]`` to ``[kh, kw, in, out]``."""
    if weight.ndim != 4:
        raise ValueError(f"Conv2d weight must be rank 4, got {weight.shape}")
    return np.transpose(weight, (2, 3, 1, 0))


def _convert(
    tensors: dict[str, np.ndarray],
    *,
    whitelist: frozenset[str],
    embedding,
) -> Conversion:
    converted: dict[str, np.ndarray] = {}
    unmapped: list[str] = []
    whitelisted: list[str] = []
    for key in sorted(tensors):
        value = tensors[key]
        if key in whitelist:
            converted[key] = np.array(value, copy=True)
            whitelisted.append(key)
            continue
        kind = _kind(key, value, embedding)
        if kind == "conv2d":
            converted[key] = transpose_conv2d(value)
        elif kind == "linear":
            converted[key] = transpose_linear(value)
        elif kind == "copy":
            converted[key] = np.array(value, copy=True)
        else:
            unmapped.append(key)
    return Conversion(weights=converted, unmapped=unmapped, whitelisted=whitelisted)


def _kind(key: str, value: np.ndarray, embedding) -> str:
    if embedding(key, value):
        return "copy"
    if key.endswith(".bias"):
        return "copy" if value.ndim == 1 else "unmapped"
    if not key.endswith(".weight"):
        return "unmapped"
    if value.ndim == 1:
        return "copy"
    if value.ndim == 2:
        return "linear"
    if value.ndim == 4:
        return "conv2d"
    return "unmapped"


def _no_embedding(key: str, value: np.ndarray) -> bool:
    del key, value
    return False


def _clip_embedding(config: dict):
    vocab = int(config["vocab_size"])
    hidden = int(config["hidden_size"])
    positions = int(config["max_position_embeddings"])

    def embedding(key: str, value: np.ndarray) -> bool:
        if key.endswith("token_embedding.weight"):
            if tuple(value.shape) != (vocab, hidden):
                raise ValueError(
                    f"{key} shape {tuple(value.shape)} does not match config vocab/hidden {(vocab, hidden)}"
                )
            return True
        if key.endswith("position_embedding.weight"):
            if tuple(value.shape) != (positions, hidden):
                raise ValueError(
                    f"{key} shape {tuple(value.shape)} does not match config positions/hidden {(positions, hidden)}"
                )
            return True
        return False

    return embedding


def map_unet(tensors: dict[str, np.ndarray], config: dict) -> Conversion:
    del config
    return _convert(tensors, whitelist=frozenset(), embedding=_no_embedding)


def map_autoencoder_kl(tensors: dict[str, np.ndarray], config: dict) -> Conversion:
    del config
    return _convert(tensors, whitelist=frozenset(), embedding=_no_embedding)


def map_clip_text(tensors: dict[str, np.ndarray], config: dict) -> Conversion:
    return _convert(
        tensors,
        whitelist=frozenset({CLIP_POSITION_IDS}),
        embedding=_clip_embedding(config),
    )


CONVERTERS: dict[str, Mapper] = {
    "UNet2DConditionModel": map_unet,
    "AutoencoderKL": map_autoencoder_kl,
    "CLIPTextModel": map_clip_text,
}
