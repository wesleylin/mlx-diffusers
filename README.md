# mlx-diffusers

Pure-MLX diffusion models for Apple Silicon. Ports are validated numerically against PyTorch reference tensors. See `SPEC.md` for the rules and `PROGRESS.md` for phase status.

## Setup

Requires macOS on Apple Silicon and Python 3.11.

```bash
cd ~/code/mlx-diffusers
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e .
pytest
```

`pip install -e .` installs runtime dependencies and pytest. It does not install the optional `reference` extra (`torch`, `diffusers`, `transformers`, `huggingface_hub`).

Confirm MLX evaluates:

```bash
python -c "import mlx.core as mx; mx.eval(mx.array([1.0]))"
```
