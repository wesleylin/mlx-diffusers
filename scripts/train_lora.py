"""Overfit an SD1.5 attention LoRA on the generated concept set.

Rank 8, learning rate 1e-4. The default step count is a short overfit, not
the multi-hour 300-step run: enough steps to see the loss fall, with a
checkpoint reload before the second half. Validation uses DDIM eta 0.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_diffusers.pipelines.pipeline_utils import from_pretrained
from mlx_diffusers.training.dataset import cache_examples, list_examples, load_cache
from mlx_diffusers.training.lora import inject, lora_arrays, load_lora, save_lora, zero_lora_b
from mlx_diffusers.training.trainer import fixed_batch_loss, load_checkpoint, train, write_loss_log

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERFIT = REPO_ROOT / "tests" / "goldens" / "sd15" / "overfit"
CAPTION = "a photograph of a zxqblorptax, a bright cyan teapot"


def _generate():
    path = Path(__file__).resolve().parent / "generate.py"
    spec = importlib.util.spec_from_file_location("generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sheet(sample: mx.array, dataset: Path, output: Path) -> None:
    thumbs = []
    for image_path, _caption in list_examples(dataset):
        thumbs.append(Image.open(image_path).convert("RGB").resize((128, 128)))
    sample_img = np.array(sample[0])
    sample_img = (np.transpose(sample_img, (1, 2, 0)) * 255.0).round().clip(0, 255).astype(np.uint8)
    sample_thumb = Image.fromarray(sample_img).resize((128, 128))
    width = 128 * (len(thumbs) + 1)
    canvas = Image.new("RGB", (width, 128))
    for index, thumb in enumerate(thumbs):
        canvas.paste(thumb, (index * 128, 0))
    canvas.paste(sample_thumb, (len(thumbs) * 128, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _validate(pipeline, dataset: Path, output: Path) -> None:
    pipeline.unet.eval()
    image = pipeline(CAPTION, num_inference_steps=8, seed=0, eta=0.0)
    mx.eval(image)
    _sheet(image, dataset, output)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=_generate().default_model())
    parser.add_argument("--dataset", type=Path, default=OVERFIT / "dataset")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args()
    mx.random.seed(0)
    pipeline = from_pretrained(args.model)
    names = inject(pipeline.unet, args.rank, args.alpha)
    print(f"injected {len(names)} projections")
    cache_dir = cache_examples(pipeline, args.dataset, OVERFIT / "cache")
    examples = load_cache(cache_dir)
    alphas = pipeline.scheduler.alphas_cumprod
    _validate(pipeline, args.dataset, OVERFIT / "early.png")
    half = args.steps // 2
    checkpoint = OVERFIT / "checkpoint.npz"
    first = train(
        pipeline.unet,
        examples,
        alphas,
        steps=half,
        learning_rate=args.lr,
        grad_accumulation=args.grad_accumulation,
        max_grad_norm=args.max_grad_norm,
        rank=args.rank,
        alpha=args.alpha,
        checkpoint_path=checkpoint,
        checkpoint_every=half,
        start_step=0,
    )
    before = fixed_batch_loss(pipeline.unet, examples[0], alphas, step=10_000)
    zero_lora_b(pipeline.unet)
    jumped = fixed_batch_loss(pipeline.unet, examples[0], alphas, step=10_000)
    import mlx.optimizers as optim

    optimizer = optim.AdamW(learning_rate=args.lr)
    load_checkpoint(pipeline.unet, optimizer, checkpoint)
    restored = fixed_batch_loss(pipeline.unet, examples[0], alphas, step=10_000)
    second = train(
        pipeline.unet,
        examples,
        alphas,
        steps=args.steps - half,
        learning_rate=args.lr,
        grad_accumulation=args.grad_accumulation,
        max_grad_norm=args.max_grad_norm,
        rank=args.rank,
        alpha=args.alpha,
        optimizer=optimizer,
        start_step=half,
    )
    _validate(pipeline, args.dataset, OVERFIT / "final.png")
    export_path = OVERFIT / "zxqblorptax.safetensors"
    save_lora(pipeline.unet, export_path, args.rank, args.alpha)
    original = lora_arrays(pipeline.unet)
    zero_lora_b(pipeline.unet)
    load_lora(pipeline.unet, export_path)
    copied = lora_arrays(pipeline.unet)
    max_diff = max(float(np.max(np.abs(original[key] - copied[key]))) for key in original)
    losses = first["losses"] + second["losses"]
    record = {
        "losses": losses,
        "first": losses[0],
        "mid": losses[len(losses) // 2],
        "last": losses[-1],
        "fixed_before": before,
        "fixed_after_zero": jumped,
        "fixed_after_resume": restored,
        "export_max_abs": max_diff,
        "injected": len(names),
    }
    write_loss_log(OVERFIT / "loss.json", record)
    media = Path.home() / "Library" / "Application Support" / "Cursor" / "AgentStores" / "cursor_agent_stores" / "bc-c831f663-f1f0-4ce6-85d7-6549b6e82a9f" / "files" / "media" / "phase8"
    media.mkdir(parents=True, exist_ok=True)
    for path in (OVERFIT / "early.png", OVERFIT / "final.png"):
        shutil.copy(path, media / path.name)
    print(
        f"loss first {record['first']:.6f} mid {record['mid']:.6f} last {record['last']:.6f} "
        f"resume {before:.6f} -> {restored:.6f} export {max_diff:.3e}"
    )


if __name__ == "__main__":
    main()
