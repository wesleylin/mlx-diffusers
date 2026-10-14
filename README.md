# mlx-diffusers

Pure-MLX diffusion models for Apple Silicon. Ports are validated numerically against PyTorch reference tensors. See `SPEC.md` for the rules and `PROGRESS.md` for phase status.

## Setup

Requires macOS on Apple Silicon, Python 3.11, and [uv](https://docs.astral.sh/uv/).

```bash
cd ~/code/mlx-diffusers
uv sync
uv run pytest
```

`uv sync` creates `.venv` from `.python-version`, installs this project, and installs the runtime dependencies plus pytest (the default `dev` group). It does not install the `reference` group.

Sync the locked reference packages (`torch`, `diffusers`, `transformers`, `huggingface_hub`) when a phase needs the PyTorch reference:

```bash
uv sync --group reference
```

Confirm MLX evaluates:

```bash
uv run python -c "import mlx.core as mx; mx.eval(mx.array([1.0]))"
```
