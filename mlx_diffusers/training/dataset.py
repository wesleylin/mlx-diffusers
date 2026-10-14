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


def cache_examples(pipeline, folder: str | Path, cache_dir: str | Path) -> Path:
    """Write one ``.npz`` per image: scaled latent and CLIP hidden state."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for image_path, caption in list_examples(folder):
        target = cache_dir / f"{image_path.stem}.npz"
        if target.is_file():
            continue
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
            rows.append(
                {
                    "latent": data["latent"],
                    "embedding": data["embedding"],
                    "caption": str(data["caption"].item()),
                }
            )
    return rows
