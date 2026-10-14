"""Build the SD1.5 LoRA overfit set with the Phase 7 pipeline.

Four images, fixed seeds, one caption. The concept name is ``zxqblorptax`` and
every caption names it. Nothing is downloaded. Sampling is 8 DDIM steps so the
set stays small.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import mlx.core as mx

from mlx_diffusers.pipelines.pipeline_utils import from_pretrained


def _generate():
    path = Path(__file__).resolve().parent / "generate.py"
    spec = importlib.util.spec_from_file_location("generate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

CAPTION = "a photograph of a zxqblorptax, a bright cyan teapot"
SEEDS = tuple(range(4))
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "goldens" / "sd15" / "overfit" / "dataset"


def main() -> None:
    generate = _generate()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=generate.default_model())
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    print(f"loading {args.model}", flush=True)
    pipeline = from_pretrained(args.model)
    print("loaded", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        image = pipeline(CAPTION, num_inference_steps=args.steps, seed=seed, eta=0.0)
        mx.eval(image)
        stem = f"zxqblorptax_{seed:02d}"
        generate.save_image(image, args.output / f"{stem}.png")
        (args.output / f"{stem}.txt").write_text(CAPTION + "\n")
        print(f"wrote {stem}", flush=True)


if __name__ == "__main__":
    main()
