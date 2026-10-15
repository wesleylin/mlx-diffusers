"""Phase 10: aspect buckets, SDXL conditioning in the trainer, kohya keys."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_diffusers.training.dataset import area_buckets, choose_bucket
from mlx_diffusers.training.lora import inject, load_kohya, lora_arrays, save_kohya
from mlx_diffusers.training.trainer import fixed_batch_loss, train


def test_area_buckets_are_aligned_and_include_the_square():
    for target in (768, 1024):
        buckets = area_buckets(target)
        assert (target, target) in buckets
        areas = [height * width for height, width in buckets]
        assert min(areas) > 0
        assert all(height % 64 == 0 and width % 64 == 0 for height, width in buckets)
        square = target * target
        assert min(abs(area - square) for area in areas) == 0
    image = choose_bucket(800, 500, area_buckets(768))
    assert image[0] % 64 == 0 and image[1] % 64 == 0
    assert image[0] / image[1] > 1


class TinyText(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)


class TinyXL(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4, bias=False)

    def __call__(self, sample, timestep, embedding, added):
        del timestep
        pooled = added["text_embeds"]
        times = added["time_ids"]
        return self.to_q(sample) + mx.mean(embedding) + mx.mean(pooled) + mx.mean(times)


def test_kohya_export_roundtrip_includes_alpha(tmp_path: Path):
    mx.random.seed(0)
    unet = TinyXL()
    text = TinyText()
    inject(unet, rank=4, alpha=8)
    inject(text, rank=4, alpha=8, targets=("q_proj",))
    path = tmp_path / "kohya.safetensors"
    save_kohya({"unet": unet, "te": text}, path, alpha=8)
    from safetensors import safe_open

    with safe_open(path, framework="numpy") as handle:
        keys = set(handle.keys())
    assert any(key.startswith("lora_unet_") and key.endswith(".lora_down.weight") for key in keys)
    assert any(key.startswith("lora_te_") and key.endswith(".lora_up.weight") for key in keys)
    assert any(key.startswith("lora_unet_") and key.endswith(".alpha") for key in keys)
    original = lora_arrays(unet)
    unet.to_q.lora_b = mx.zeros_like(unet.to_q.lora_b)
    load_kohya({"unet": unet, "te": text}, path)
    copied = lora_arrays(unet)
    for key in original:
        assert np.max(np.abs(original[key] - copied[key])) == 0


def test_trainer_uses_pooled_and_time_ids():
    mx.random.seed(0)
    model = TinyXL()
    inject(model, rank=2, alpha=2)
    generator = np.random.Generator(np.random.PCG64(0))
    examples = []
    for _ in range(2):
        examples.append(
            {
                "latent": generator.standard_normal((1, 4), dtype=np.float32),
                "embedding": generator.standard_normal((1, 2, 4), dtype=np.float32),
                "text_embeds": generator.standard_normal((1, 4), dtype=np.float32),
                "time_ids": np.array([[768, 768, 0, 0, 768, 768]], dtype=np.float32),
                "caption": "toy",
            }
        )
    alphas = mx.array(np.linspace(0.99, 0.1, 8, dtype=np.float32))
    before = fixed_batch_loss(model, examples[0], alphas, step=3)
    result = train(
        model,
        examples,
        alphas,
        steps=2,
        learning_rate=1e-2,
        rank=2,
        alpha=2,
    )
    after = fixed_batch_loss(model, examples[0], alphas, step=3)
    assert np.isfinite(before) and np.isfinite(after)
    assert result["losses"]
