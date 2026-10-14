"""OpenCLIP text encoder with a projection head.

Mirror of ``transformers`` ``CLIPTextModelWithProjection`` for the SDXL bigG
config (``hidden_act=gelu``, not CLIP ``quick_gelu``). ``penultimate`` is the
encoder output before ``final_layer_norm`` (``hidden_states[-2]``). The pooled
vector is the EOS token of the final layer norm, then ``text_projection``.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from mlx_diffusers.layers import phase2_to_parameter
from mlx_diffusers.models.clip_text_model import CLIPAttention, CLIPTextEmbeddings

_EMBEDDING_WEIGHTS = (
    "text_model.embeddings.token_embedding.weight",
    "text_model.embeddings.position_embedding.weight",
)


def gelu(x: mx.array) -> mx.array:
    """Exact erf GELU, matching ``transformers.activations.gelu``."""
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


class CLIPGeluMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.fc2(gelu(self.fc1(hidden_states)))


class CLIPGeluEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, eps: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = CLIPGeluMLP(hidden_size, intermediate_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class CLIPTextModelWithProjection(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        if config.get("hidden_act") != "gelu":
            raise ValueError(f"CLIPTextModelWithProjection expected gelu, got {config.get('hidden_act')!r}")
        hidden_size = int(config["hidden_size"])
        eps = float(config["layer_norm_eps"])
        self.embeddings = CLIPTextEmbeddings(
            int(config["vocab_size"]),
            hidden_size,
            int(config["max_position_embeddings"]),
        )
        self.layers = [
            CLIPGeluEncoderLayer(
                hidden_size,
                int(config["num_attention_heads"]),
                int(config["intermediate_size"]),
                eps,
            )
            for _ in range(int(config["num_hidden_layers"]))
        ]
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=eps)
        self.text_projection = nn.Linear(hidden_size, int(config["projection_dim"]), bias=False)

    def load_converted(self, weights: dict[str, mx.array]) -> None:
        pairs = []
        renamed = {}
        for key, value in weights.items():
            mapped = value if key in _EMBEDDING_WEIGHTS or key.endswith("position_ids") else phase2_to_parameter(key, value)
            if key.startswith("text_model.encoder.layers."):
                key = "layers." + key[len("text_model.encoder.layers.") :]
            elif key.startswith("text_model.embeddings."):
                key = "embeddings." + key[len("text_model.embeddings.") :]
            elif key.startswith("text_model.final_layer_norm."):
                key = "final_layer_norm." + key[len("text_model.final_layer_norm.") :]
            renamed[key] = mapped
        if "embeddings.position_ids" not in renamed:
            renamed["embeddings.position_ids"] = self.embeddings.position_ids
        pairs = list(renamed.items())
        self.load_weights(pairs, strict=True)
        self.eval()

    def penultimate_and_pooled(self, input_ids: mx.array) -> tuple[mx.array, mx.array]:
        hidden = self.embeddings(input_ids)
        # Same hidden_states[-2] index as CLIPTextModel.penultimate. Pooled output
        # still uses the last encoder layer, final norm, and text projection.
        for layer in self.layers[:-1]:
            hidden = layer(hidden)
        penultimate = hidden
        hidden = self.layers[-1](hidden)
        final = self.final_layer_norm(hidden)
        index = mx.argmax(input_ids, axis=-1)
        pooled = final[mx.arange(final.shape[0]), index]
        return penultimate, self.text_projection(pooled)
