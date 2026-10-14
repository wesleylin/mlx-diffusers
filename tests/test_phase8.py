"""Phase 8: LoRA injection, export roundtrip, accumulation, and resume."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import mlx.optimizers as optim

from mlx_diffusers.training.lora import inject, load_lora, lora_arrays, save_lora
from mlx_diffusers.training.trainer import fixed_batch_loss, load_checkpoint, train

GOLDEN_DIR = Path(__file__).parent / "goldens" / "sd15" / "overfit"


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(4, 4, bias=False)
        self.to_k = nn.Linear(4, 4, bias=False)
        self.query = nn.Linear(4, 4, bias=False)
        self.to_out = [nn.Linear(4, 4, bias=False), nn.Dropout(0.0)]

    def __call__(self, sample, timestep, embedding):
        del timestep, embedding
        hidden = self.to_q(sample) + self.to_k(sample) + self.query(sample)
        return self.to_out[0](hidden)


def _examples():
    generator = np.random.Generator(np.random.PCG64(0))
    rows = []
    for _ in range(4):
        rows.append(
            {
                "latent": generator.standard_normal((1, 4), dtype=np.float32),
                "embedding": generator.standard_normal((1, 2, 4), dtype=np.float32),
                "caption": "toy",
            }
        )
    return rows


def test_inject_targets_attention_projections_only():
    mx.random.seed(0)
    model = TinyDenoiser()
    names = inject(model, rank=4, alpha=4)
    assert set(names) == {"to_q", "to_k", "to_out.0"}
    assert "query" not in names
    trainable = [key for key, _value in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.trainable_parameters())]
    assert trainable
    assert all(key.endswith(".lora_a") or key.endswith(".lora_b") for key in trainable)
    assert not any("query" in key for key in trainable)


def test_export_roundtrip_and_resume_loss(tmp_path: Path):
    alphas = mx.array(np.linspace(0.99, 0.1, 16, dtype=np.float32))
    examples = _examples()
    mx.random.seed(1)
    first = TinyDenoiser()
    inject(first, rank=4, alpha=8)
    checkpoint = tmp_path / "checkpoint.npz"
    export = tmp_path / "lora.safetensors"
    half = train(
        first,
        examples,
        alphas,
        steps=4,
        learning_rate=1e-2,
        grad_accumulation=2,
        max_grad_norm=1.0,
        rank=4,
        alpha=8,
        checkpoint_path=checkpoint,
        checkpoint_every=4,
    )
    before = fixed_batch_loss(first, examples[0], alphas, step=123)
    mx.random.seed(1)
    second = TinyDenoiser()
    inject(second, rank=4, alpha=8)
    optimizer = optim.AdamW(learning_rate=1e-2)
    start = load_checkpoint(second, optimizer, checkpoint)
    assert start == 4
    resumed = fixed_batch_loss(second, examples[0], alphas, step=123)
    assert abs(before - resumed) < 1e-5
    continued = train(
        second,
        examples,
        alphas,
        steps=2,
        learning_rate=1e-2,
        grad_accumulation=2,
        max_grad_norm=1.0,
        rank=4,
        alpha=8,
        optimizer=optimizer,
        start_step=start,
    )
    assert continued["losses"][0] < half["losses"][0] * 5
    save_lora(first, export, 4, 8)
    fresh = TinyDenoiser()
    inject(fresh, rank=4, alpha=8)
    rank, alpha = load_lora(fresh, export)
    assert rank == 4 and alpha == 8.0
    left = lora_arrays(first)
    right = lora_arrays(fresh)
    assert set(left) == set(right)
    for key in left:
        assert np.allclose(left[key], right[key])


def test_overfit_loss_log_falls_and_resume_matches():
    record = json.loads((GOLDEN_DIR / "loss.json").read_text())
    probes = record["fixed_probes"]
    assert probes[-1] < probes[0]
    assert all(later <= earlier for earlier, later in zip(probes, probes[1:]))
    assert abs(record["fixed_after_resume"] - record["fixed_before"]) < 1e-4
    assert record["fixed_after_zero"] != record["fixed_before"]
    assert record["export_max_abs"] == 0.0
    assert (GOLDEN_DIR / "early.png").is_file()
    assert (GOLDEN_DIR / "final.png").is_file()
