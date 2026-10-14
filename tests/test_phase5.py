"""Phase 5: SD1.5 AutoencoderKL against the VAE golden."""

from pathlib import Path

import mlx.core as mx

from mlx_diffusers.configuration import load_model
from mlx_diffusers.utils.golden import compare, load_golden

GOLDEN_DIR = Path(__file__).parent / "goldens" / "sd15"
SD15_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MEAN_LIMIT = 1e-5
MAX_LIMIT = 1e-3


def _snapshot() -> Path:
    repo = "models--" + SD15_ID.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / repo / "snapshots"
    snaps = [path for path in sorted(root.iterdir()) if (path / "vae" / "config.json").is_file()]
    assert snaps, f"Phase 1 cache for {SD15_ID} is not on disk"
    return snaps[-1]


def test_vae_decode_and_posterior_mean():
    golden = load_golden(GOLDEN_DIR / "sd15_vae.npz")
    latents = load_golden(GOLDEN_DIR / "sd15_latents.npz")["latents"]
    vae = load_model(_snapshot() / "vae").module
    decoded = vae.decode(mx.array(latents))
    mx.eval(decoded)
    # The golden posterior is encode(reference decode), so the encoder sees that image.
    posterior = vae.encode(mx.array(golden["decoded"]))
    mx.eval(posterior)
    rows = [
        ("decoded", *compare(decoded, golden["decoded"])),
        ("posterior_mean", *compare(posterior, golden["posterior_mean"])),
    ]
    lines = ["tensor mean_abs max_abs"]
    failed = []
    for name, mean_abs, max_abs in rows:
        ok = mean_abs < MEAN_LIMIT and max_abs < MAX_LIMIT
        lines.append(f"{name} {mean_abs:.3e} {max_abs:.3e} {'ok' if ok else 'FAIL'}")
        if not ok:
            failed.append((name, mean_abs, max_abs))
    print("\n".join(lines))
    assert not failed, lines
