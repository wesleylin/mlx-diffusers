"""Generate an image with the SD1.5 pipeline.

``--prompt`` and ``--seed`` select the conditioning and the initial noise.
The default sampler is 20-step DDIM with eta 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image

from mlx_diffusers.pipelines.pipeline_utils import from_pretrained
from mlx_diffusers.schedulers.scheduling_euler import EulerDiscreteScheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "tests" / "goldens" / "sd15" / "sd15_ddim_mlx.png"
DEFAULT_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def default_model() -> str:
    repo = Path.home() / ".cache" / "huggingface" / "hub" / "models--stable-diffusion-v1-5--stable-diffusion-v1-5" / "snapshots"
    if repo.is_dir():
        snaps = [path for path in sorted(repo.iterdir()) if (path / "model_index.json").is_file()]
        if snaps:
            return str(snaps[-1])
    return DEFAULT_MODEL


def save_image(image: mx.array, path: Path) -> None:
    array = np.array(image)
    if array.ndim == 4:
        array = array[0]
    pixels = (np.transpose(array, (1, 2, 0)) * 255.0).round().clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--scheduler", choices=("ddim", "euler"), default="ddim")
    parser.add_argument(
        "--model",
        default=default_model(),
        help="Local diffusers directory or Hub repo id.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    pipeline = from_pretrained(args.model)
    if args.scheduler == "euler":
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler_config)
    image = pipeline(
        args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        eta=0.0,
    )
    mx.eval(image)
    save_image(image, args.output)
    float_path = args.output.with_suffix(".npy")
    np.save(float_path, np.array(image))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
