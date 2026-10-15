"""Epsilon-prediction LoRA trainer.

Noise comes from ``alphas_cumprod``. The optimizer is ``mlx.optimizers.AdamW``.
Gradients are evaluated before the optimizer step. Updated parameters are
evaluated before a checkpoint is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from mlx_diffusers.training.lora import lora_arrays, save_lora


def prediction_loss(
    model,
    latent,
    embedding,
    noise,
    alpha,
    timestep,
    text_embeds=None,
    time_ids=None,
) -> mx.array:
    noisy = mx.sqrt(alpha) * latent + mx.sqrt(1.0 - alpha) * noise
    if text_embeds is None:
        prediction = model(noisy, timestep, embedding)
    else:
        prediction = model(
            noisy,
            timestep,
            embedding,
            {"text_embeds": text_embeds, "time_ids": time_ids},
        )
    # Compare in fp32. An fp16 UNet output stays fp16 so the backward graph does not upcast.
    return mx.mean((prediction.astype(mx.float32) - noise.astype(mx.float32)) ** 2)


def _batch_noise(example: dict[str, np.ndarray], step: int, num_train_timesteps: int):
    generator = np.random.Generator(np.random.PCG64(step))
    timestep = int(generator.integers(0, num_train_timesteps))
    noise = generator.standard_normal(size=example["latent"].shape, dtype=np.float32)
    return timestep, noise


def train(
    model: nn.Module,
    examples: list[dict[str, np.ndarray]],
    alphas_cumprod: mx.array,
    *,
    steps: int,
    learning_rate: float,
    grad_accumulation: int = 1,
    max_grad_norm: float = 1.0,
    weight_decay: float = 0.01,
    rank: int,
    alpha: float,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 0,
    resume_path: Path | None = None,
    export_path: Path | None = None,
    on_step=None,
    optimizer: optim.AdamW | None = None,
    start_step: int = 0,
) -> dict:
    if optimizer is None:
        optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    if resume_path is not None:
        start_step = load_checkpoint(model, optimizer, resume_path)
    loss_and_grad = nn.value_and_grad(model, prediction_loss)
    losses = []
    num_train = int(alphas_cumprod.shape[0])
    for step in range(start_step, start_step + steps):
        accumulated = None
        micro_losses = []
        for micro in range(grad_accumulation):
            example = examples[(step * grad_accumulation + micro) % len(examples)]
            timestep, noise_np = _batch_noise(example, step * grad_accumulation + micro, num_train)
            alpha = alphas_cumprod[timestep]
            arguments = [
                model,
                mx.array(example["latent"]),
                mx.array(example["embedding"]),
                mx.array(noise_np),
                alpha,
                mx.array(timestep),
            ]
            if "text_embeds" in example:
                arguments.append(mx.array(example["text_embeds"]))
                arguments.append(mx.array(example["time_ids"]))
            if getattr(model, "gradient_checkpointing", False):
                loss, grads = model.segmented_loss(*arguments[1:])
            else:
                loss, grads = loss_and_grad(*arguments)
            mx.eval(loss, grads)
            micro_losses.append(loss)
            accumulated = grads if accumulated is None else tree_map(lambda left, right: left + right, accumulated, grads)
        accumulated = tree_map(lambda value: value / grad_accumulation, accumulated)
        clipped, _norm = optim.clip_grad_norm(accumulated, max_grad_norm)
        mx.eval(clipped)
        optimizer.update(model, clipped)
        mx.eval(model.parameters())
        mean_loss = sum(float(value.item()) for value in micro_losses) / len(micro_losses)
        losses.append(mean_loss)
        if on_step is not None:
            on_step(step, mean_loss)
        if checkpoint_path is not None and checkpoint_every and (step + 1) % checkpoint_every == 0:
            save_checkpoint(checkpoint_path, step + 1, model, optimizer, rank, alpha)
    if export_path is not None:
        save_lora(model, export_path, rank, alpha)
    return {"losses": losses, "optimizer": optimizer, "start_step": start_step, "steps": steps}


def save_checkpoint(path: Path, step: int, model: nn.Module, optimizer: optim.Optimizer, rank: int, alpha: float) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mx.eval(model.parameters())
    payload = {
        "step": np.int64(step),
        "rank": np.int64(rank),
        "alpha": np.float32(alpha),
    }
    for key, value in lora_arrays(model).items():
        payload["lora." + key] = value
    for key, value in tree_flatten(optimizer.state):
        payload["opt." + key] = np.array(value)
    np.savez(path, **payload)


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, path: Path) -> int:
    with np.load(path, allow_pickle=False) as data:
        step = int(data["step"])
        pairs = []
        opt_pairs = []
        for key in data.files:
            if key.startswith("lora."):
                pairs.append((key.removeprefix("lora."), mx.array(data[key])))
            elif key.startswith("opt."):
                opt_pairs.append((key.removeprefix("opt."), mx.array(data[key])))
        model.load_weights(pairs, strict=False)
        if opt_pairs:
            optimizer.state = tree_unflatten(opt_pairs)
        mx.eval(model.parameters())
    return step


def fixed_batch_loss(model: nn.Module, example: dict[str, np.ndarray], alphas_cumprod: mx.array, step: int) -> float:
    timestep, noise = _batch_noise(example, step, int(alphas_cumprod.shape[0]))
    arguments = [
        model,
        mx.array(example["latent"]),
        mx.array(example["embedding"]),
        mx.array(noise),
        alphas_cumprod[timestep],
        mx.array(timestep),
    ]
    if "text_embeds" in example:
        arguments.append(mx.array(example["text_embeds"]))
        arguments.append(mx.array(example["time_ids"]))
    loss = prediction_loss(*arguments)
    mx.eval(loss)
    return float(loss.item())


def write_loss_log(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
