"""Short SDXL LoRA overfit on two Phase 9 images.

The base pipeline draws the images. Training loads
``cagliostrolab/animagine-xl-3.1`` with the SDXL classes. Rank 8, lr 1e-4,
about 40 steps, 768px, batch 1, cached conditioning. ``--stability``
continues that cache toward 2000 steps and does not draw new images.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from PIL import Image

from mlx_diffusers.configuration import load_model
from mlx_diffusers.pipelines.pipeline_utils import from_pretrained
from mlx_diffusers.schedulers.scheduling_euler import EulerDiscreteScheduler
from mlx_diffusers.training.dataset import cache_examples, list_examples, load_cache
from mlx_diffusers.training.lora import inject, load_kohya, lora_arrays, save_kohya, zero_lora_b
from mlx_diffusers.training.trainer import fixed_batch_loss, load_checkpoint, train, write_loss_log

REPO_ROOT = Path(__file__).resolve().parents[1]
OVERFIT = REPO_ROOT / "tests" / "goldens" / "sdxl" / "overfit"
DATASET = OVERFIT / "dataset"
CAPTION = "a photograph of a brilvexonq, a bright magenta lantern"
ANIMAGINE = "cagliostrolab/animagine-xl-3.1"
MEMORY_LIMIT = 12 * 1024 ** 3
STABILITY_LOG = OVERFIT / "stability.json"
MEDIA = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Cursor"
    / "AgentStores"
    / "cursor_agent_stores"
    / "bc-c831f663-f1f0-4ce6-85d7-6549b6e82a9f"
    / "files"
    / "media"
    / "phase10"
)


def _generate():
    path = Path(__file__).resolve().parent / "generate.py"
    spec = importlib.util.spec_from_file_location("generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sdxl_snapshot() -> str:
    repo = Path.home() / ".cache" / "huggingface" / "hub" / "models--stabilityai--stable-diffusion-xl-base-1.0" / "snapshots"
    snaps = [path for path in sorted(repo.iterdir()) if (path / "model_index.json").is_file()]
    if not snaps:
        raise SystemExit("SDXL base snapshot is not on disk")
    return str(snaps[-1])


def _make_dataset(steps: int) -> None:
    existing = list(DATASET.glob("*.png")) if DATASET.is_dir() else []
    if len(existing) >= 2:
        print(f"dataset already has {len(existing)} images", flush=True)
        return
    generate = _generate()
    DATASET.mkdir(parents=True, exist_ok=True)
    print("generating overfit images with the phase 9 pipeline", flush=True)
    pipeline = from_pretrained(_sdxl_snapshot())
    for seed in (0, 1):
        image = pipeline(
            CAPTION,
            num_inference_steps=steps,
            guidance_scale=5.0,
            seed=seed,
            height=768,
            width=768,
        )
        mx.eval(image)
        stem = f"brilvexonq_{seed:02d}"
        generate.save_image(image, DATASET / f"{stem}.png")
        (DATASET / f"{stem}.txt").write_text(CAPTION + "\n")
        print(f"wrote {stem}", flush=True)
    del pipeline
    gc.collect()
    mx.clear_cache()


def _sheet(sample: mx.array, output: Path) -> None:
    thumbs = [Image.open(path).convert("RGB").resize((128, 128)) for path, _caption in list_examples(DATASET)]
    array = np.array(sample[0])
    pixels = (np.transpose(array, (1, 2, 0)) * 255.0).round().clip(0, 255).astype(np.uint8)
    sample_thumb = Image.fromarray(pixels).resize((128, 128))
    canvas = Image.new("RGB", (128 * (len(thumbs) + 1), 128))
    for index, thumb in enumerate(thumbs):
        canvas.paste(thumb, (index * 128, 0))
    canvas.paste(sample_thumb, (len(thumbs) * 128, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _validate(pipeline, output: Path) -> None:
    pipeline.unet.eval()
    image = pipeline(CAPTION, num_inference_steps=8, guidance_scale=5.0, seed=0, height=768, width=768)
    mx.eval(image)
    _sheet(image, output)
    print(f"wrote {output}", flush=True)


def _stability_record(losses: list[float], peaks: list[float], *, stopped: str | None) -> dict:
    finite = bool(losses) and all(math.isfinite(value) for value in losses)
    return {
        "steps_completed": len(losses),
        "first": losses[0] if losses else None,
        "last": losses[-1] if losses else None,
        "max_peak_gb": max(peaks) if peaks else None,
        "finite": finite and stopped is None,
        "stopped": stopped,
        "losses": losses,
        "peak_gb": peaks,
    }


def _run_stability(args) -> None:
    """2000-step finite check on the cached 2-image set. No new images."""
    cache_dir = OVERFIT / "cache"
    if not any(cache_dir.glob("*.npz")):
        raise SystemExit(f"missing cached latents in {cache_dir}")
    mx.set_memory_limit(MEMORY_LIMIT)
    mx.set_cache_limit(512 * 1024 ** 2)
    from huggingface_hub import snapshot_download

    model_dir = Path(snapshot_download(args.model))
    print(f"loading unet from {model_dir.name}", flush=True)
    unet = load_model(model_dir / "unet").module
    with (model_dir / "scheduler" / "scheduler_config.json").open() as handle:
        scheduler_config = json.load(handle)
    alphas = EulerDiscreteScheduler.from_config(scheduler_config).alphas_cumprod
    mx.random.seed(0)
    names = inject(unet, args.rank, args.alpha)
    unet.enable_gradient_checkpointing()
    print(f"injected {len(names)} projections", flush=True)
    print("train holds the unet only", flush=True)
    examples = load_cache(cache_dir)
    gc.collect()
    mx.clear_cache()
    fixed_before = fixed_batch_loss(unet, examples[0], alphas, step=10_000)
    print(f"fixed_batch_before {fixed_before:.6f}", flush=True)
    mx.clear_cache()
    mx.reset_peak_memory()
    losses: list[float] = []
    peaks: list[float] = []
    seconds: list[float] = []
    stopped: str | None = None
    clock = time.perf_counter()

    def _log(step, loss):
        nonlocal stopped, clock
        now = time.perf_counter()
        elapsed = now - clock
        clock = now
        peak = mx.get_peak_memory() / (1024 ** 3)
        active = mx.get_active_memory() / (1024 ** 3)
        print(
            f"step {step} loss {loss:.6f} peak_gb {peak:.2f} active_gb {active:.2f} seconds {elapsed:.2f}",
            flush=True,
        )
        seconds.append(elapsed)
        losses.append(loss)
        peaks.append(peak)
        if not math.isfinite(loss):
            stopped = f"nan at step {step}"
        elif peak > 12.0:
            stopped = f"memory {peak:.2f} GB at step {step}"
        if stopped is not None or step % 25 == 0:
            write_loss_log(STABILITY_LOG, _stability_record(losses, peaks, stopped=stopped))
        mx.clear_cache()
        mx.reset_peak_memory()
        if stopped is not None:
            raise SystemExit(stopped)

    try:
        train(
            unet,
            examples,
            alphas,
            steps=args.steps,
            learning_rate=args.lr,
            rank=args.rank,
            alpha=args.alpha,
            on_step=_log,
        )
    except SystemExit:
        write_loss_log(STABILITY_LOG, _stability_record(losses, peaks, stopped=stopped))
        raise
    fixed_after = fixed_batch_loss(unet, examples[0], alphas, step=10_000)
    record = _stability_record(losses, peaks, stopped=None)
    record["fixed_before"] = fixed_before
    record["fixed_after"] = fixed_after
    record["seconds_per_step"] = sum(seconds) / len(seconds) if seconds else None
    write_loss_log(STABILITY_LOG, record)
    print(
        json.dumps(
            {
                "steps_completed": record["steps_completed"],
                "first": record["first"],
                "last": record["last"],
                "max_peak_gb": record["max_peak_gb"],
                "finite": record["finite"],
                "fixed_before": fixed_before,
                "fixed_after": fixed_after,
                "seconds_per_step": record["seconds_per_step"],
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=ANIMAGINE)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-steps", type=int, default=8)
    parser.add_argument("--target-size", type=int, default=768)
    parser.add_argument("--stability", action="store_true", help="train the cached 2-image set and skip image generation")
    args = parser.parse_args()
    if args.stability:
        _run_stability(args)
        return
    # Leave headroom on a 16GB machine. The limit is a ceiling, not a target allocation.
    mx.set_memory_limit(12 * 1024 ** 3)
    mx.set_cache_limit(512 * 1024 ** 2)
    _make_dataset(args.image_steps)
    print(f"loading {args.model}", flush=True)
    pipeline = from_pretrained(args.model)
    print(f"loaded {type(pipeline).__name__} unet {type(pipeline.unet).__name__}", flush=True)
    mx.random.seed(0)
    names = inject(pipeline.unet, args.rank, args.alpha)
    pipeline.unet.enable_gradient_checkpointing()
    print(f"injected {len(names)} projections", flush=True)
    cache_dir = cache_examples(pipeline, DATASET, OVERFIT / "cache", target_size=args.target_size)
    examples = load_cache(cache_dir)
    alphas = pipeline.scheduler.alphas_cumprod
    if not (OVERFIT / "early.png").is_file():
        _validate(pipeline, OVERFIT / "early.png")
    pipeline.release_for_training()
    gc.collect()
    mx.clear_cache()
    held = pipeline.conditioning_resident()
    if held:
        raise RuntimeError(f"still resident during train: {held}")
    print("train holds the unet only", flush=True)
    mx.reset_peak_memory()
    start_fixed = fixed_batch_loss(pipeline.unet, examples[0], alphas, step=10_000)
    half = args.steps // 2
    checkpoint = OVERFIT / "checkpoint.npz"

    def _log(step, loss):
        peak = mx.get_peak_memory() / (1024 ** 3)
        active = mx.get_active_memory() / (1024 ** 3)
        print(f"step {step} loss {loss:.6f} peak_gb {peak:.2f} active_gb {active:.2f}", flush=True)
        mx.clear_cache()
        mx.reset_peak_memory()

    first = train(
        pipeline.unet,
        examples,
        alphas,
        steps=half,
        learning_rate=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        checkpoint_path=checkpoint,
        checkpoint_every=half,
        start_step=0,
        on_step=_log,
    )
    before = fixed_batch_loss(pipeline.unet, examples[0], alphas, step=10_000)
    zero_lora_b(pipeline.unet)
    jumped = fixed_batch_loss(pipeline.unet, examples[0], alphas, step=10_000)
    optimizer = optim.AdamW(learning_rate=args.lr)
    load_checkpoint(pipeline.unet, optimizer, checkpoint)
    restored = fixed_batch_loss(pipeline.unet, examples[0], alphas, step=10_000)
    second = train(
        pipeline.unet,
        examples,
        alphas,
        steps=args.steps - half,
        learning_rate=args.lr,
        rank=args.rank,
        alpha=args.alpha,
        optimizer=optimizer,
        start_step=half,
        on_step=_log,
    )
    pipeline.vae = load_model(pipeline.model_dir / "vae").module
    gc.collect()
    mx.clear_cache()
    _validate(pipeline, OVERFIT / "final.png")
    export_path = OVERFIT / "brilvexonq.safetensors"
    save_kohya({"unet": pipeline.unet}, export_path, args.alpha)
    original = lora_arrays(pipeline.unet)
    zero_lora_b(pipeline.unet)
    load_kohya({"unet": pipeline.unet}, export_path)
    copied = lora_arrays(pipeline.unet)
    max_diff = max(float(np.max(np.abs(original[key] - copied[key]))) for key in original)
    losses = first["losses"] + second["losses"]
    end_fixed = fixed_batch_loss(pipeline.unet, examples[0], alphas, step=10_000)
    record = {
        "losses": losses,
        "first": losses[0],
        "mid": losses[len(losses) // 2],
        "last": losses[-1],
        "fixed_start": start_fixed,
        "fixed_before": before,
        "fixed_after_zero": jumped,
        "fixed_after_resume": restored,
        "fixed_end": end_fixed,
        "export_max_abs": max_diff,
        "injected": len(names),
        "model": args.model,
    }
    write_loss_log(OVERFIT / "loss.json", record)
    MEDIA.mkdir(parents=True, exist_ok=True)
    for path, _caption in list_examples(DATASET):
        shutil.copy(path, MEDIA / path.name)
    for name in ("early.png", "final.png"):
        shutil.copy(OVERFIT / name, MEDIA / name)
    print(json.dumps({key: record[key] for key in record if key != "losses"}), flush=True)


if __name__ == "__main__":
    main()
