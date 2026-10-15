"""Image folder with sidecar captions, cached to VAE latents and CLIP states."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image


def list_examples(folder: str | Path) -> list[tuple[Path, str]]:
    root = Path(folder)
    images = sorted(path for path in root.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    examples = []
    for image in images:
        caption_path = image.with_suffix(".txt")
        if not caption_path.is_file():
            raise FileNotFoundError(f"missing caption for {image.name}")
        examples.append((image, caption_path.read_text().strip()))
    if not examples:
        raise FileNotFoundError(f"no images in {root}")
    return examples


def _image_tensor(path: Path) -> mx.array:
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    array = array * 2.0 - 1.0
    return mx.array(np.transpose(array, (2, 0, 1))[None])


def area_buckets(target_size: int = 1024, alignment: int = 64, max_aspect: float = 2.0) -> list[tuple[int, int]]:
    """Height, width pairs near ``target_size ** 2``, multiples of ``alignment``."""
    if alignment <= 0 or target_size < alignment:
        raise ValueError(f"target_size {target_size} alignment {alignment} is not usable")
    target_area = target_size * target_size
    limit = int(target_size * max_aspect)
    found: dict[tuple[int, int], int] = {}
    side = alignment
    while side <= limit:
        other = max(alignment, int(round((target_area / side) / alignment)) * alignment)
        if other <= limit and max(side, other) / min(side, other) <= max_aspect + 1e-6:
            for pair in ((other, side), (side, other)):
                found[pair] = abs(pair[0] * pair[1] - target_area)
        side += alignment
    return sorted(found, key=lambda pair: (found[pair], abs(pair[0] - pair[1]), pair[0], pair[1]))


def choose_bucket(height: int, width: int, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    """Pick the bucket whose aspect ratio is closest to the image."""
    aspect = height / width
    return min(buckets, key=lambda pair: (abs(pair[0] / pair[1] - aspect), abs(pair[0] * pair[1] - height * width)))


def cache_examples(
    pipeline,
    folder: str | Path,
    cache_dir: str | Path,
    target_size: int | None = None,
) -> Path:
    """Write one ``.npz`` per image: scaled latent and CLIP hidden state.

    SDXL also stores the second encoder's pooled vector and six ``time_ids``
    for the aspect bucket. ``target_size`` is the square side of the area
    (768 on a 16GB machine, 1024 otherwise).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sdxl = getattr(pipeline, "text_encoder_2", None) is not None
    buckets = area_buckets(target_size or 1024) if sdxl else None
    for image_path, caption in list_examples(folder):
        target = cache_dir / f"{image_path.stem}.npz"
        if target.is_file():
            continue
        if sdxl:
            image = Image.open(image_path).convert("RGB")
            bucket_h, bucket_w = choose_bucket(image.height, image.width, buckets)
            image = image.resize((bucket_w, bucket_h), Image.Resampling.LANCZOS)
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = array * 2.0 - 1.0
            pixels = mx.array(np.transpose(array, (2, 0, 1))[None])
            latent = pipeline.vae.encode(pixels) * pipeline.vae.scaling_factor
            embedding, pooled = pipeline.encode_prompt(caption)
            time_ids = np.array([[bucket_h, bucket_w, 0, 0, bucket_h, bucket_w]], dtype=np.float32)
            mx.eval(latent, embedding, pooled)
            np.savez(
                target,
                latent=np.array(latent),
                embedding=np.array(embedding),
                text_embeds=np.array(pooled),
                time_ids=time_ids,
                caption=np.asarray(caption),
            )
        else:
            pixels = _image_tensor(image_path)
            latent = pipeline.vae.encode(pixels) * pipeline.vae.scaling_factor
            tokens = pipeline.tokenizer(
                [caption],
                padding="max_length",
                max_length=pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            embedding = pipeline.text_encoder(mx.array(np.asarray(tokens["input_ids"], dtype=np.int32)))
            mx.eval(latent, embedding)
            np.savez(
                target,
                latent=np.array(latent),
                embedding=np.array(embedding),
                caption=np.asarray(caption),
            )
        print(f"cached {image_path.stem}")
    return cache_dir


def load_cache(cache_dir: str | Path) -> list[dict[str, np.ndarray]]:
    paths = sorted(Path(cache_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no cached examples in {cache_dir}")
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            row = {
                "latent": data["latent"],
                "embedding": data["embedding"],
                "caption": str(data["caption"].item()),
            }
            if "text_embeds" in data.files:
                row["text_embeds"] = data["text_embeds"]
                row["time_ids"] = data["time_ids"]
            rows.append(row)
    return rows
