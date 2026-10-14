"""Phase 7: DDIM and Euler match torch, and 20-step DDIM matches the reference image."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler as TorchDDIM
from diffusers import EulerDiscreteScheduler as TorchEuler

from mlx_diffusers.schedulers.scheduling_ddim import DDIMScheduler
from mlx_diffusers.schedulers.scheduling_euler import EulerDiscreteScheduler
from mlx_diffusers.utils.cpu_generator import randn
from mlx_diffusers.utils.golden import compare

GOLDEN_DIR = Path(__file__).parent / "goldens" / "sd15"
SD15_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
PIXEL_LIMIT = 2.5 / 255


def _scheduler_config() -> dict:
    repo = "models--" + SD15_ID.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / repo / "snapshots"
    snaps = [path for path in sorted(root.iterdir()) if (path / "scheduler" / "scheduler_config.json").is_file()]
    assert snaps, "SD1.5 scheduler config is not on disk"
    return json.loads((snaps[-1] / "scheduler" / "scheduler_config.json").read_text())


def test_cpu_randn_matches_torch_seed_0():
    torch.manual_seed(0)
    generator = torch.Generator(device="cpu").manual_seed(0)
    expected = torch.randn((1, 4, 64, 64), generator=generator, dtype=torch.float32).numpy()
    actual = randn((1, 4, 64, 64), 0)
    mean_abs, max_abs = compare(actual, expected)
    print(f"randn mean_abs {mean_abs:.3e} max_abs {max_abs:.3e}")
    assert mean_abs < 1e-6 and max_abs < 1e-5


def test_ddim_schedule_and_step_match_torch():
    config = _scheduler_config()
    ours = DDIMScheduler.from_config(config)
    reference = TorchDDIM.from_config(config)
    ours.set_timesteps(20)
    reference.set_timesteps(20)
    assert ours.timesteps.tolist() == reference.timesteps.tolist()
    generator = torch.Generator(device="cpu").manual_seed(1)
    sample = torch.randn(1, 4, 8, 8, generator=generator)
    model_output = torch.randn(1, 4, 8, 8, generator=generator)
    timestep = int(reference.timesteps[0])
    expected = reference.step(model_output, timestep, sample, eta=0.0).prev_sample
    actual = ours.step(mx.array(model_output.numpy()), timestep, mx.array(sample.numpy()), eta=0.0).prev_sample
    mx.eval(actual)
    mean_abs, max_abs = compare(actual, expected.numpy())
    print(f"ddim step mean_abs {mean_abs:.3e} max_abs {max_abs:.3e}")
    assert mean_abs < 1e-6 and max_abs < 1e-5


def test_euler_schedule_and_step_match_torch():
    config = _scheduler_config()
    ours = EulerDiscreteScheduler.from_config(config)
    reference = TorchEuler.from_config(config)
    ours.set_timesteps(20)
    reference.set_timesteps(20)
    assert np.allclose(ours.timesteps, reference.timesteps.numpy())
    generator = torch.Generator(device="cpu").manual_seed(2)
    sample = torch.randn(1, 4, 8, 8, generator=generator)
    model_output = torch.randn(1, 4, 8, 8, generator=generator)
    timestep = reference.timesteps[0]
    reference.scale_model_input(sample, timestep)
    scaled = ours.scale_model_input(mx.array(sample.numpy()), timestep)
    mx.eval(scaled)
    scale_mean, scale_max = compare(scaled, reference.scale_model_input(sample, timestep).numpy())
    expected = reference.step(model_output, timestep, sample).prev_sample
    actual = ours.step(mx.array(model_output.numpy()), timestep, mx.array(sample.numpy())).prev_sample
    mx.eval(actual)
    mean_abs, max_abs = compare(actual, expected.numpy())
    print(
        f"euler scale mean_abs {scale_mean:.3e} max_abs {scale_max:.3e} "
        f"step mean_abs {mean_abs:.3e} max_abs {max_abs:.3e}"
    )
    assert scale_mean < 1e-6 and scale_max < 1e-5
    # Beta linspace differs from torch by 1 ulp, which shows up in the Euler step.
    assert mean_abs < 1e-5 and max_abs < 1e-5


def test_ddim_20_step_image_matches_reference():
    prompt = "a photograph of an astronaut riding a horse"
    script = Path(__file__).resolve().parents[1] / "scripts" / "generate.py"
    output = GOLDEN_DIR / "sd15_ddim_mlx.png"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--prompt",
            prompt,
            "--seed",
            "0",
            "--steps",
            "20",
            "--output",
            str(output),
        ],
        check=True,
    )
    actual = np.load(output.with_suffix(".npy"))
    expected = np.load(GOLDEN_DIR / "sd15_ddim.npz")["image"]
    mean_abs, max_abs = compare(actual, expected)
    print(f"ddim image mean_abs {mean_abs:.6f} max_abs {max_abs:.6f} limit {PIXEL_LIMIT:.6f}")
    torch_png = Image.open(GOLDEN_DIR / "sd15_ddim_torch.png").convert("RGB")
    mlx_png = Image.open(output).convert("RGB")
    side = Image.new("RGB", (torch_png.width + mlx_png.width, max(torch_png.height, mlx_png.height)))
    side.paste(torch_png, (0, 0))
    side.paste(mlx_png, (torch_png.width, 0))
    side_path = GOLDEN_DIR / "sd15_ddim_side_by_side.png"
    side.save(side_path)
    media = Path.home() / "Library" / "Application Support" / "Cursor" / "AgentStores" / "cursor_agent_stores" / "bc-c831f663-f1f0-4ce6-85d7-6549b6e82a9f" / "files" / "media" / "phase7"
    media.mkdir(parents=True, exist_ok=True)
    for path in (GOLDEN_DIR / "sd15_ddim_torch.png", output, side_path):
        shutil.copy(path, media / path.name)
    output.with_suffix(".npy").unlink()
    assert mean_abs < PIXEL_LIMIT, (mean_abs, max_abs)
