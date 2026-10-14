"""SD1.5 CLIP text encoder.

Mirror of ``transformers/models/clip/modeling_clip.py`` ``CLIPTextModel``
(ViT-L/14). Checkpoint keys keep the ``text_model.`` prefix from the
safetensors file. ``hidden_act`` is ``quick_gelu`` (``x * sigmoid(1.702 * x)``).
Self-attention is causal via ``mx.fast.scaled_dot_product_attention`` with
``mask="causal"`` and scale ``1/sqrt(head_dim)``. ``final_layer_norm`` runs
after the encoder. Dims, heads, and ``layer_norm_eps`` come from the
component config.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_diffusers.layers import phase2_to_parameter

_EMBEDDING_WEIGHTS = (
    "text_model.embeddings.token_embedding.weight",
    "text_model.embeddings.position_embedding.weight",
)


def quick_gelu(x: mx.array) -> mx.array:
    """Mirror of ``transformers.activations.QuickGELUActivation``."""
    return x * mx.sigmoid(1.702 * x)


def _split_heads(x: mx.array, heads: int) -> mx.array:
    batch, seq, channels = x.shape
    head_dim = channels // heads
    x = x.reshape(batch, seq, heads, head_dim)
    return mx.transpose(x, (0, 2, 1, 3))


def _merge_heads(x: mx.array) -> mx.array:
    batch, heads, seq, head_dim = x.shape
    x = mx.transpose(x, (0, 2, 1, 3))
    return x.reshape(batch, seq, heads * head_dim)


class CLIPAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.scale = (hidden_size // num_heads) ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        query = _split_heads(self.q_proj(hidden_states), self.num_heads)
        key = _split_heads(self.k_proj(hidden_states), self.num_heads)
        value = _split_heads(self.v_proj(hidden_states), self.num_heads)
        hidden_states = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            mask="causal",
        )
        return self.out_proj(_merge_heads(hidden_states))


class CLIPMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.fc2(quick_gelu(self.fc1(hidden_states)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, eps: float):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = CLIPMLP(hidden_size, intermediate_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = hidden_states + self.self_attn(self.layer_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.position_ids = mx.arange(max_position_embeddings)[None, :]

    def __call__(self, input_ids: mx.array) -> mx.array:
        seq = input_ids.shape[-1]
        if seq > self.position_ids.shape[-1]:
            raise ValueError(
                f"sequence length {seq} exceeds max_position_embeddings {self.position_ids.shape[-1]}"
            )
        position_ids = self.position_ids[:, :seq]
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class CLIPTextTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        hidden_size = int(config["hidden_size"])
        eps = float(config["layer_norm_eps"])
        self.embeddings = CLIPTextEmbeddings(
            int(config["vocab_size"]),
            hidden_size,
            int(config["max_position_embeddings"]),
        )
        self.encoder = CLIPEncoder(
            hidden_size=hidden_size,
            num_heads=int(config["num_attention_heads"]),
            intermediate_size=int(config["intermediate_size"]),
            eps=eps,
            num_layers=int(config["num_hidden_layers"]),
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=eps)

    def __call__(self, input_ids: mx.array) -> mx.array:
        hidden_states = self.encoder(self.embeddings(input_ids))
        return self.final_layer_norm(hidden_states)


class CLIPEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        eps: float,
        num_layers: int,
    ):
        super().__init__()
        self.layers = [
            CLIPEncoderLayer(hidden_size, num_heads, intermediate_size, eps) for _ in range(num_layers)
        ]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class CLIPTextModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.text_model = CLIPTextTransformer(config)

    def load_converted(self, weights: dict[str, mx.array]) -> None:
        """Assign Phase 2 weights. Embeddings stay ``[vocab, dim]``; linears are transposed."""
        pairs = []
        for key, value in weights.items():
            if key in _EMBEDDING_WEIGHTS or key.endswith("position_ids"):
                pairs.append((key, value))
            else:
                pairs.append((key, phase2_to_parameter(key, value)))
        self.load_weights(pairs, strict=True)
        self.eval()

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self.text_model(input_ids)
