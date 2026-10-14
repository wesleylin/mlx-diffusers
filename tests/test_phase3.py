"""Phase 3: SD1.5 UNet primitives against hook goldens."""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors import safe_open

from mlx_diffusers.layers import phase2_to_parameter
from mlx_diffusers.layers.attention import Attention
from mlx_diffusers.layers.embeddings import TimestepEmbedding, Timesteps
from mlx_diffusers.layers.feedforward import FeedForward
from mlx_diffusers.layers.resnet import ResnetBlock2D
from mlx_diffusers.layers.updownsampling import Downsample2D, Upsample2D
from mlx_diffusers.models.transformer_2d import Transformer2DModel
from mlx_diffusers.utils.golden import compare, load_golden

GOLDEN = Path(__file__).parent / "goldens" / "sd15" / "sd15_modules.npz"
WEIGHTS = Path(__file__).resolve().parents[1] / "converted" / "sd15" / "unet.safetensors"
SD15_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MEAN_LIMIT = 1e-5
MAX_LIMIT = 5e-4


def _snapshot() -> Path:
    repo = "models--" + SD15_ID.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / repo / "snapshots"
    snaps = [path for path in sorted(root.iterdir()) if (path / "unet" / "config.json").is_file()]
    assert snaps, f"Phase 1 cache for {SD15_ID} is not on disk"
    return snaps[-1]


def _config() -> dict:
    return json.loads((_snapshot() / "unet" / "config.json").read_text())


def _mx(array) -> mx.array:
    return mx.array(array)


def _load(module: nn.Module, prefix: str) -> None:
    assert WEIGHTS.is_file(), f"Phase 2 converted UNet is missing at {WEIGHTS}"
    weights = []
    with safe_open(WEIGHTS, framework="numpy") as handle:
        for key in handle.keys():
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :]
            value = mx.array(handle.get_tensor(key))
            weights.append((name, phase2_to_parameter(name, value)))
    module.load_weights(weights, strict=True)
    module.eval()


def _heads(config: dict) -> int:
    heads = config.get("num_attention_heads") or config["attention_head_dim"]
    if isinstance(heads, list):
        return int(heads[0])
    return int(heads)


def _cross_dim(config: dict) -> int:
    dim = config["cross_attention_dim"]
    if isinstance(dim, list):
        return int(dim[0])
    return int(dim)


class _CrossAttnDownBlock(nn.Module):
    """First SD1.5 cross-attention down block, assembled for the hook golden."""

    def __init__(self, config: dict):
        super().__init__()
        channels = int(config["block_out_channels"][0])
        heads = _heads(config)
        head_dim = channels // heads
        layers = int(config["layers_per_block"])
        temb = channels * 4
        groups = int(config["norm_num_groups"])
        eps = float(config["norm_eps"])
        act = config["act_fn"]
        cross = _cross_dim(config)
        self.resnets = [
            ResnetBlock2D(
                in_channels=channels,
                out_channels=channels,
                temb_channels=temb,
                eps=eps,
                groups=groups,
                non_linearity=act,
            )
            for _ in range(layers)
        ]
        self.attentions = [
            Transformer2DModel(
                heads,
                head_dim,
                in_channels=channels,
                norm_num_groups=groups,
                cross_attention_dim=cross,
                norm_eps=eps,
            )
            for _ in range(layers)
        ]
        self.downsamplers = [
            Downsample2D(
                channels,
                use_conv=True,
                out_channels=channels,
                padding=int(config["downsample_padding"]),
                name="op",
            )
        ]

    def __call__(self, hidden_states, temb, encoder_hidden_states):
        for resnet, attn in zip(self.resnets, self.attentions):
            hidden_states = resnet(hidden_states, temb)
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)
        return self.downsamplers[0](hidden_states)


def _gate(name: str, actual: mx.array, expected) -> tuple[str, float, float, bool]:
    mx.eval(actual)
    mean_abs, max_abs = compare(actual, expected)
    ok = mean_abs < MEAN_LIMIT and max_abs < MAX_LIMIT
    return name, mean_abs, max_abs, ok


def test_layers_do_not_hardcode_cross_attention_width():
    root = Path(__file__).resolve().parents[1] / "mlx_diffusers" / "layers"
    for path in root.glob("*.py"):
        text = path.read_text()
        assert "import torch" not in text
        assert "768" not in text


def test_phase3_module_gates():
    config = _config()
    golden = load_golden(GOLDEN)
    channels = int(config["block_out_channels"][0])
    temb_channels = channels * 4
    heads = _heads(config)
    head_dim = channels // heads
    groups = int(config["norm_num_groups"])
    eps = float(config["norm_eps"])
    cross = _cross_dim(config)
    act = config["act_fn"]

    timesteps = Timesteps(channels, bool(config["flip_sin_to_cos"]), float(config["freq_shift"]))
    time_embedding = TimestepEmbedding(channels, temb_channels, act_fn=act)
    _load(time_embedding, "time_embedding.")

    resnet = ResnetBlock2D(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        eps=eps,
        groups=groups,
        non_linearity=act,
    )
    _load(resnet, "down_blocks.0.resnets.0.")

    self_attn = Attention(query_dim=channels, heads=heads, dim_head=head_dim)
    _load(self_attn, "down_blocks.0.attentions.0.transformer_blocks.0.attn1.")
    cross_attn = Attention(
        query_dim=channels,
        heads=heads,
        dim_head=head_dim,
        cross_attention_dim=cross,
    )
    _load(cross_attn, "down_blocks.0.attentions.0.transformer_blocks.0.attn2.")

    feedforward = FeedForward(channels, activation_fn="geglu")
    _load(feedforward, "down_blocks.0.attentions.0.transformer_blocks.0.ff.")

    transformer = Transformer2DModel(
        heads,
        head_dim,
        in_channels=channels,
        norm_num_groups=groups,
        cross_attention_dim=cross,
        norm_eps=eps,
    )
    _load(transformer, "down_blocks.0.attentions.0.")

    down = _CrossAttnDownBlock(config)
    _load(down, "down_blocks.0.")

    up_channels = int(list(reversed(config["block_out_channels"]))[0])
    upsample = Upsample2D(up_channels, use_conv=True, out_channels=up_channels)
    _load(upsample, "up_blocks.0.upsamplers.0.")

    downsample = Downsample2D(
        channels,
        use_conv=True,
        out_channels=channels,
        padding=int(config["downsample_padding"]),
        name="op",
    )
    _load(downsample, "down_blocks.0.downsamplers.0.")

    rows = [
        _gate("timesteps", timesteps(_mx(golden["timesteps_embedding_arg0"])), golden["timesteps_embedding"]),
        _gate(
            "timestep_embedding",
            time_embedding(_mx(golden["timestep_embedding_arg0"])),
            golden["timestep_embedding"],
        ),
        _gate(
            "resnet",
            resnet(_mx(golden["resnet_arg0"]), _mx(golden["resnet_arg1"])),
            golden["resnet"],
        ),
        _gate("self_attn", self_attn(_mx(golden["self_attn_arg0"])), golden["self_attn"]),
        _gate(
            "cross_attn",
            cross_attn(
                _mx(golden["cross_attn_arg0"]),
                encoder_hidden_states=_mx(golden["cross_attn_encoder_hidden_states"]),
            ),
            golden["cross_attn"],
        ),
        _gate("feedforward", feedforward(_mx(golden["feedforward_arg0"])), golden["feedforward"]),
        _gate(
            "attention",
            transformer(
                _mx(golden["attention_arg0"]),
                encoder_hidden_states=_mx(golden["attention_encoder_hidden_states"]),
            ),
            golden["attention"],
        ),
        _gate("downsample", downsample(_mx(golden["downsample_arg0"])), golden["downsample"]),
        _gate("upsample", upsample(_mx(golden["upsample_arg0"])), golden["upsample"]),
        _gate(
            "down_block",
            down(
                _mx(golden["down_block_hidden_states"]),
                _mx(golden["down_block_temb"]),
                _mx(golden["down_block_encoder_hidden_states"]),
            ),
            golden["down_block"],
        ),
    ]
    lines = ["module mean_abs max_abs"]
    for name, mean_abs, max_abs, ok in rows:
        lines.append(f"{name} {mean_abs:.3e} {max_abs:.3e} {'ok' if ok else 'FAIL'}")
    print("\n".join(lines))
    failed = [row for row in rows if not row[3]]
    assert not failed, lines
