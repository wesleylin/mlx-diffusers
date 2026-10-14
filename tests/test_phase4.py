"""Phase 4: SD1.5 CLIP text encoder against the embedding golden."""

from pathlib import Path

import mlx.core as mx

from mlx_diffusers.configuration import ARCH_REGISTRY, load_model
from mlx_diffusers.models.clip_text_model import CLIPTextModel
from mlx_diffusers.utils.golden import compare, load_golden

GOLDEN_DIR = Path(__file__).parent / "goldens" / "sd15"
SD15_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
MEAN_LIMIT = 1e-5
MAX_LIMIT = 1e-3


def _snapshot() -> Path:
    repo = "models--" + SD15_ID.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / repo / "snapshots"
    snaps = [path for path in sorted(root.iterdir()) if (path / "text_encoder" / "config.json").is_file()]
    assert snaps, f"Phase 1 cache for {SD15_ID} is not on disk"
    return snaps[-1]


def test_clip_text_model_is_registered():
    assert ARCH_REGISTRY["CLIPTextModel"].cls is CLIPTextModel
    result = load_model(_snapshot() / "text_encoder")
    assert result.module_registered is True
    assert result.message == "module is registered"
    assert isinstance(result.module, CLIPTextModel)
    assert result.unmapped == ()


def test_clip_last_hidden_state_matches_golden():
    tokens = load_golden(GOLDEN_DIR / "sd15_tokens.npz")
    embeddings = load_golden(GOLDEN_DIR / "sd15_text_embeddings.npz")
    prompts = [str(item) for item in tokens["prompts"].tolist()]
    result = load_model(_snapshot() / "text_encoder")
    output = result.module(mx.array(tokens["input_ids"]))
    mx.eval(output)
    expected = embeddings["last_hidden_state"]
    rows = [("batch", *compare(output, expected))]
    for index, prompt in enumerate(prompts):
        rows.append((prompt, *compare(output[index], expected[index])))
    lines = ["prompt mean_abs max_abs"]
    failed = []
    for name, mean_abs, max_abs in rows:
        ok = mean_abs < MEAN_LIMIT and max_abs < MAX_LIMIT
        lines.append(f"{name!r} {mean_abs:.3e} {max_abs:.3e} {'ok' if ok else 'FAIL'}")
        if not ok:
            failed.append((name, mean_abs, max_abs))
    print("\n".join(lines))
    assert not failed, lines
