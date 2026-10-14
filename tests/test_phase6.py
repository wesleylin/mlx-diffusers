"""Phase 6: SD1.5 UNet single-step noise prediction."""

from pathlib import Path

import mlx.core as mx

from mlx_diffusers.configuration import load_model
from mlx_diffusers.utils.golden import compare, load_golden

GOLDEN_DIR = Path(__file__).parent / "goldens" / "sd15"
SD15_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MEAN_LIMIT = 1e-5
MAX_LIMIT = 1e-3
TIMESTEPS = (999, 500, 1)


def _snapshot() -> Path:
    repo = "models--" + SD15_ID.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / repo / "snapshots"
    snaps = [path for path in sorted(root.iterdir()) if (path / "unet" / "config.json").is_file()]
    assert snaps, f"Phase 1 cache for {SD15_ID} is not on disk"
    return snaps[-1]


def test_unet_noise_pred_at_three_timesteps():
    latents = mx.array(load_golden(GOLDEN_DIR / "sd15_latents.npz")["latents"])
    embeddings = mx.array(load_golden(GOLDEN_DIR / "sd15_text_embeddings.npz")["last_hidden_state"])
    golden = load_golden(GOLDEN_DIR / "sd15_unet.npz")
    unet = load_model(_snapshot() / "unet").module
    rows = []
    failed = []
    for timestep in TIMESTEPS:
        pred = unet(latents, timestep, embeddings)
        mx.eval(pred)
        mean_abs, max_abs = compare(pred, golden[f"noise_pred_{timestep}"])
        ok = mean_abs < MEAN_LIMIT and max_abs < MAX_LIMIT
        rows.append((timestep, mean_abs, max_abs, ok))
        if not ok:
            failed.append((timestep, mean_abs, max_abs))
    lines = ["timestep mean_abs max_abs"]
    for timestep, mean_abs, max_abs, ok in rows:
        lines.append(f"{timestep} {mean_abs:.3e} {max_abs:.3e} {'ok' if ok else 'FAIL'}")
    print("\n".join(lines))
    assert not failed, lines
