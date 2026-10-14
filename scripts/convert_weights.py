"""Write mapper output for a local diffusers component or a cached snapshot.

Does not download weights. Does not construct modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from mlx_diffusers.configuration import class_name_from_config, resolve_class_name
from mlx_diffusers.utils.convert import CONVERTERS
from mlx_diffusers.utils.loading import read_component

SD15_MIRROR = "stable-diffusion-v1-5/stable-diffusion-v1-5"
COMPONENTS = ("unet", "vae", "text_encoder")


def cached_snapshot(model_id: str) -> Path:
    repo = "models--" + model_id.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / repo / "snapshots"
    if not root.is_dir():
        raise FileNotFoundError(f"no Hugging Face cache for {model_id} at {root}")
    snapshots = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not snapshots:
        raise FileNotFoundError(f"no snapshots for {model_id} at {root}")
    return snapshots[-1]


def convert_directory(component_dir: Path, output_path: Path) -> None:
    config, tensors = read_component(component_dir)
    class_name = resolve_class_name(class_name_from_config(config))
    converted = CONVERTERS[class_name](tensors, config)
    if converted.unmapped:
        joined = ", ".join(converted.unmapped)
        raise SystemExit(f"unmapped keys in {class_name}: {joined}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # safetensors writes the base buffer of a non-contiguous view, which turns
    # a transpose into a reshape. Copy first so the file matches the mapper.
    contiguous = {key: np.ascontiguousarray(value) for key, value in converted.weights.items()}
    save_file(contiguous, output_path)
    print(
        f"wrote {output_path} class={class_name} "
        f"keys={len(converted.weights)} unmapped={len(converted.unmapped)} "
        f"whitelisted={converted.whitelisted}"
    )


def convert_model(model_id: str, output_dir: Path) -> None:
    snapshot = cached_snapshot(model_id)
    found = [name for name in COMPONENTS if (snapshot / name / "config.json").is_file()]
    if not found:
        raise SystemExit(f"{snapshot} has no unet, vae, or text_encoder component")
    for name in found:
        convert_directory(snapshot / name, output_dir / f"{name}.safetensors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=SD15_MIRROR)
    parser.add_argument(
        "--component",
        type=Path,
        default=None,
        help="Local component directory, or a snapshot root that contains component folders.",
    )
    parser.add_argument("--output", type=Path, default=Path("converted"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.component is None:
        convert_model(args.model_id, args.output)
        return
    component = args.component
    if (component / "config.json").is_file():
        destination = args.output
        if destination.suffix != ".safetensors":
            destination = destination / f"{component.name}.safetensors"
        convert_directory(component, destination)
        return
    found = [name for name in COMPONENTS if (component / name / "config.json").is_file()]
    if not found:
        raise SystemExit(f"{component} is not a component directory or a snapshot root")
    for name in found:
        convert_directory(component / name, args.output / f"{name}.safetensors")


if __name__ == "__main__":
    main()
