"""Phase 10: aspect buckets, SDXL conditioning in the trainer, kohya keys."""

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from mlx_diffusers.pipelines.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from mlx_diffusers.training.dataset import (
    area_buckets,
    cache_examples,
    choose_bucket,
    parse_resolutions,
    stage_step_counts,
)
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


def test_lora_and_adam_stay_fp32_on_an_fp16_base():
    mx.random.seed(0)
    model = TinyXL()
    model.to_q.weight = model.to_q.weight.astype(mx.float16)
    inject(model, rank=2, alpha=2)
    assert model.to_q.linear.weight.dtype == mx.float16
    assert model.to_q.lora_a.dtype == mx.float32
    assert model.to_q.lora_b.dtype == mx.float32
    generator = np.random.Generator(np.random.PCG64(0))
    example = {
        "latent": generator.standard_normal((1, 4), dtype=np.float32),
        "embedding": generator.standard_normal((1, 2, 4), dtype=np.float32),
        "text_embeds": generator.standard_normal((1, 4), dtype=np.float32),
        "time_ids": np.array([[64, 64, 0, 0, 64, 64]], dtype=np.float32),
        "caption": "toy",
    }
    alphas = mx.array(np.linspace(0.99, 0.1, 8, dtype=np.float32))
    result = train(model, [example], alphas, steps=1, learning_rate=1e-2, rank=2, alpha=2)
    from mlx.utils import tree_flatten

    floating = [
        value.dtype
        for _, value in tree_flatten(result["optimizer"].state)
        if isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating)
    ]
    assert floating
    assert set(floating) == {mx.float32}


def test_release_for_training_drops_vae_and_encoders():
    pipeline = StableDiffusionXLPipeline(None, None, object(), object(), object(), object(), None, {}, 4, 128, 8, True)
    pipeline.release_for_training()
    assert pipeline.conditioning_resident() == []


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


def test_resolution_choices_keep_768_as_the_default():
    assert parse_resolutions(None) == [768]
    assert parse_resolutions(["512"]) == [512]
    assert parse_resolutions(["512,768"]) == [512, 768]
    assert parse_resolutions(["768", "512"]) == [768, 512]
    assert stage_step_counts(10, 2) == [5, 5]
    assert stage_step_counts(10, 1) == [10]
    try:
        parse_resolutions(["1024"])
    except ValueError as exc:
        assert "1024" in str(exc)
    else:
        raise AssertionError("1024 was accepted")


def test_second_resolution_cache_is_not_reencoded(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), "white").save(image_dir / "a.png")
    (image_dir / "a.txt").write_text("caption")
    cache = tmp_path / "cache"
    cache.mkdir()
    np.savez(cache / "a.npz", latent=np.zeros((1, 4, 1, 1), dtype=np.float32))
    cache_examples(object(), image_dir, cache, target_size=512)


def test_attention_sublayer_tape_matches_full_backward():
    from mlx.utils import tree_flatten

    from mlx_diffusers.models.unet_sdxl import UNet2DConditionXL

    mx.random.seed(0)
    model = UNet2DConditionXL(
        {
            "addition_embed_type": "text_time",
            "in_channels": 4,
            "out_channels": 4,
            "block_out_channels": [32],
            "down_block_types": ["CrossAttnDownBlock2D"],
            "up_block_types": ["CrossAttnUpBlock2D"],
            "layers_per_block": 1,
            "transformer_layers_per_block": 2,
            "num_attention_heads": 4,
            "cross_attention_dim": 16,
            "norm_num_groups": 4,
            "norm_eps": 1e-5,
            "act_fn": "silu",
            "downsample_padding": 1,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "addition_time_embed_dim": 4,
            "projection_class_embeddings_input_dim": 32,
        }
    )
    inject(model, rank=2, alpha=2)
    for _, module in model.named_modules():
        if hasattr(module, "lora_b"):
            module.lora_b = mx.random.normal(module.lora_b.shape) * 0.02
    latent = mx.random.normal((1, 4, 8, 8))
    embedding = mx.random.normal((1, 4, 16))
    noise = mx.random.normal((1, 4, 8, 8))
    text = mx.random.normal((1, 8))
    time_ids = mx.zeros((1, 6))
    alpha = mx.array(0.5)
    timestep = mx.array([3])
    loss_s, grads_s = model.segmented_loss(latent, embedding, noise, alpha, timestep, text, time_ids)

    def full(latent, embedding, noise, alpha, timestep, text, time_ids):
        noisy = mx.sqrt(alpha) * latent + mx.sqrt(1.0 - alpha) * noise
        pred = model(noisy, timestep, embedding, {"text_embeds": text, "time_ids": time_ids})
        diff = pred.astype(mx.float32) - noise.astype(mx.float32)
        return mx.mean(diff * diff)

    loss_f, grads_f = nn.value_and_grad(model, full)(latent, embedding, noise, alpha, timestep, text, time_ids)
    mx.eval(loss_s, loss_f, grads_s, grads_f)
    assert abs(float(loss_s) - float(loss_f)) < 1e-5
    worst = 0.0
    nonzero = 0
    for (key_s, grad_s), (key_f, grad_f) in zip(tree_flatten(grads_s), tree_flatten(grads_f)):
        assert key_s == key_f
        gap = float(mx.max(mx.abs(grad_s - grad_f)))
        worst = max(worst, gap)
        if key_s.endswith(".lora_b") and float(mx.max(mx.abs(grad_s))) > 0:
            nonzero += 1
    assert nonzero > 0
    assert worst < 1e-4
