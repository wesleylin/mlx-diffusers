# mlx-diffusers

Pure-MLX library for diffusion models on Apple Silicon. End state: train an Animagine XL (SDXL-architecture) LoRA entirely with this library and load the result into ComfyUI.

## Prime directives

- Weights are the spec. Architecture comes from each model's `config.json` and the state dict, never from memory or papers. Module names mirror diffusers checkpoint key names exactly.
- Correctness is numeric, never visual. Every port is validated against golden tensors from a PyTorch reference. "It generated an image" is not a pass.
- fp32 first. All ports and all tests run fp32. dtype flags are added later, never during a port.
- One phase at a time. A phase is DONE only when every acceptance criterion passes. Commit after each phase. Never start Phase N+1 with Phase N red.
- License hygiene. May translate from and reference diffusers and transformers (Apache-2.0) and mlx-lm (MIT), with attribution in the docstring (e.g. "mirror of diffusers/models/resnet.py"). May NOT read or copy code from mflux (GPL) or any unlicensed/abandoned repo. This library is MIT. Do not shallow-clone mflux into a gitignored `references/` folder; GPL source stays unread. A later phase may inspect https://github.com/AmirHossein-razlighi/mlx_diffuser only after confirming its license allows it.
- Library is pure MLX. torch/diffusers/transformers may only be imported under `scripts/` and `tests/`. Zero torch imports in `mlx_diffusers/`. Exception: `CLIPTokenizer` from transformers may be used at pipeline runtime (pure Python, no torch).
- Never loosen a tolerance to make a test pass. If a threshold seems wrong, stop and report. Numerics failures use the bisecting playbook, not hyperparameter tweaking.
- MLX eval discipline. MLX is lazy. Every test must `mx.eval(...)` outputs before asserting. Every training step must eval loss/grads before the optimizer update, and eval updated params before checkpointing.

## Environment

- macOS, Apple Silicon, Python 3.11 (`.python-version`).
- uv is the only install path. `uv sync` installs the project, runtime dependencies, and the default `dev` group (`pytest`). Run tests with `uv run pytest`. `torch`, `diffusers`, `transformers`, and `huggingface_hub` are the `reference` dependency group, locked in `uv.lock`, and installed with `uv sync --group reference`.
- Runtime: `mlx`, `numpy`, `safetensors`, `pillow`, `tqdm`.
- Maintain `PROGRESS.md` as a phase checklist. Update it at every phase completion.

## Golden-tensor protocol

Phase 1 writes the SD1.5 files. Later phases extend this protocol per model family. Do not loosen the gates below.

`scripts/make_goldens.py` runs the PyTorch reference on CPU, fp32, seeded, and saves float32 `.npz` under `tests/goldens/<family>/`. Phase 1 writes `tests/goldens/sd15/`.

- token ids for 2 prompts (one real caption, one empty string)
- CLIP text embeddings for both prompts
- fixed initial latents `torch.randn(seed=0)`
- UNet noise pred at timesteps `{999, 500, 1}` for (latents, both embeds, batch=2)
- VAE: `decode(fixed latents)` → image; `encode(fixed image)` → posterior mean
- Module hooks: Timesteps embedding output, one down-block output, one attention block output, one ResNet output

Model IDs are CLI args, never hardcoded.

- SD1.5 mirror: `stable-diffusion-v1-5/stable-diffusion-v1-5` (original runwayml repo was removed; fall back to `CompVis/stable-diffusion-v1-4`).
- SDXL: `stabilityai/stable-diffusion-xl-base-1.0`.
- Animagine: `cagliostrolab/animagine-xl-3.1`.

Gates after `mx.eval`, fp32:

- mean abs diff `< 1e-5` AND max abs `< 1e-3` for full-UNet outputs and embeddings
- mean `< 1e-5`, max `< 5e-4` for individual modules
- End-to-end: DDIM 20-step final image mean abs pixel diff `< 2.5/255` vs the torch reference image

## Debugging playbook

1. Weight layouts: PyTorch `Linear` is `[out, in]`, `Conv2d` is `[out, in, kh, kw]`; MLX `nn.Linear` is `[in, out]`, `nn.Conv2d` is `[kh, kw, in, out]` — transposes required at conversion.
2. Diff the module input first.
3. Compare against the captured torch hook for that exact module.
4. GroupNorm eps: UNet `1e-5`, VAE `1e-6`; do not unify.
5. Attention scale `1/sqrt(head_dim)`. SD1.5 Timesteps uses diffusers' exact `flip_sin_to_cos` / `downscale_freq_shift` — read installed diffusers source, mirror it, unit-test it.
6. Only then suspect accumulation; re-verify in fp32.

## Layout

The Phase 0 package directories are not the target tree. The target layout is [Architecture — modularity](#architecture--modularity). Create a file only in the phase that implements it. Do not add empty directories or placeholder classes for SD3, Flux, or any other future model.

## Architecture — modularity

One registry, keyed by `config.json` `_class_name`. No YAML, entry points, or ABC hierarchies. Schedulers are duck-typed: `set_timesteps`, `scale_model_input`, and `step`. `schedulers/scheduling_base.py` documents that contract. It is not an ABC tree.

Configs supply dims, channels, heads, and eps. The string `768` appears in configs and tests, never in `layers/`.

Components are folders. The loader reads `model_index.json` and wires components by name. `PIPELINE_REGISTRY` maps `StableDiffusionPipeline` and `StableDiffusionXLPipeline`. Pipelines own only conditioning, the CFG loop, and the trainer cache builder.

- SD1.5: one CLIP, final hidden state, VAE scale `0.18215`.
- SDXL: two CLIPs, `hidden_states[-2]` concatenated, pooled text embeds, 6 `time_ids`, VAE scale `0.13025`.

Rule of two. SD1.5 is concrete. SDXL is the second concrete implementation. Extract a shared abstraction only when that second implementation shows the seam. Do not pre-build T5, Flux, SD3, or a conditioning framework.

Converter lookup happens inside `load_model`, not at decorator import time, so import order cannot matter. An unknown `_class_name` raises. Never guess. A tiny alias map may rename transformers class names (`CLIPTextModel`) onto the same `ARCH_REGISTRY`. That alias is not a second registry. If a second registry shape appears, delete it.

Weight load, without freezing an MLX `load_weights` call signature: construct the arch from its config, run that arch's mapper (transpose and rename only), assign the arrays under diffusers key names.

`from_pretrained` accepts a local diffusers directory. `huggingface_hub` is allowed inside the package for downloads only, with no torch import. Add that dependency when pipelines need it (Phase 7), not before.

Target file map. Create these in the phase that implements them. This is not a Phase 1 scaffold.

- `configuration.py` — registry, `load_model`, pipeline loader
- `layers/` — `embeddings.py`, `norm.py`, `activations.py`, `feedforward.py`, `attention.py`, `resnet.py`, `updownsampling.py`
- `models/` — `unet_2d_condition.py`, `unet_2d_blocks.py`, `transformer_2d.py`, `autoencoder_kl.py`, `clip_text_model.py`
- `schedulers/` — `scheduling_base.py`, `scheduling_ddim.py`, `scheduling_euler.py`
- `pipelines/` — `pipeline_utils.py`, `pipeline_stable_diffusion.py`, `pipeline_sdxl.py`
- `training/` — the trainer consumes cached latents, cached conditioning, and trainable params, and does not see arch internals. LoRA `inject()` targets attention modules by name pattern.
- `utils/` — `golden.py` stays. `convert.py` holds one mapper per arch.
- `tests/goldens/<family>/`

Register `UNet2DConditionModel`, `AutoencoderKL`, and `CLIPTextModel`. One mapper per arch in `utils/convert.py`, the same `CONVERTERS` dict the loader uses. `scripts/convert_weights.py` only pre-caches mapper output.

Anti-stubs: no empty dirs or placeholder classes for future models. `layers/` may hold a module used by only one model.

Smoke test for the registry: a known `_class_name` builds from a config, and an unknown `_class_name` raises. Do not add a pytest that calls `from_pretrained` and generates with every registered pipeline. That downloads weights on every run. Full generate stays the phase acceptance.

Future plugs, not to implement now:

- Animagine 4.x is zero new code only if its `model_index` class names stay the SDXL ones.
- SD 2.x may be a v-prediction flag.
- SD3.5 and Flux would be new model files plus a pipeline, leaving `layers/`, VAE, LoRA, the trainer, export, and schedulers alone.
- A new scheduler is one registered file.

## Phases

Implement one phase at a time. A phase is DONE only when every acceptance criterion passes.

### 0 — Scaffold + environment

DONE when `uv sync` and `uv run pytest` pass on Apple Silicon, MLX eval works, and `PROGRESS.md` exists.

### 1 — Golden harness for SD1.5

`scripts/make_goldens.py` writes `tests/goldens/sd15/*.npz`. `utils/golden.py` loader + compare helper (mean/max abs diff).

Acceptance: script runs top-to-bottom on the user's Mac; smoke test loads each npz and asserts shapes/dtypes.

### 2 — Weight conversion

Delivers the registry, mappers, and `load_model`. `utils/convert.py`: HF diffusers safetensors (single or sharded) → MLX safetensors, diffusers key names, Linear/Conv transposes. Architecture params from `config.json`.

Acceptance: converted unet/vae/text_encoder; every state-dict key consumed or explicitly whitelisted; test asserts zero unmapped keys and correct post-transpose shapes. Unknown `_class_name` raises.

### 3 — Core modules fp32

Create the `layers/` and `models/` files this phase needs, for real. Dims, channels, heads, and eps come from config. Unit-tested vs torch hooks: Timesteps/TimestepEmbedding, ResnetBlock2D, Attention (self+cross, `mx.fast.scaled_dot_product_attention`), FeedForward, Transformer2DModel, up/downsample. Mirror diffusers class structure and key names. Module gates.

### 4 — CLIP text encoder

Create `models/clip_text_model.py` for real. Dims come from config. SD1.5 ViT-L/14, 77 tokens, causal mask. `last_hidden_state` matches golden for both prompts.

### 5 — AutoencoderKL

Create `models/autoencoder_kl.py` for real. Dims come from config. Decode image and encode posterior mean match gates.

### 6 — SD1.5 UNet2DConditionModel

Create `models/unet_2d_condition.py` and `models/unet_2d_blocks.py` for real. Dims come from config. Single-step tests at timesteps `{999, 500, 1}` within full-UNet gates. No sampling in this phase.

### 7 — Schedulers + SD1.5 pipeline

DDIM (`eta=0`) and Euler. `StableDiffusionPipeline`-shaped class. Tokenizer via transformers. Add `huggingface_hub` to the package here, for downloads only, with no torch import. 20-step DDIM within the end-to-end gate. `scripts/generate.py --prompt ... --seed 0`. Commit reference + MLX PNGs side by side in `tests/goldens/sd15/`.

### 8 — SD1.5 LoRA training

LoRA rank/alpha (may reference mlx-lm MIT). Inject attention q/k/v (`+to_out.0` optional). Offline VAE latent + CLIP caches. Folder of images + `.txt` captions. Noise from `alphas_cumprod`. eps-prediction MSE. AdamW, grad clip, grad accumulation, periodic sampler validation, resume, save LoRA safetensors.

Overfit: ~10 images, rank 8, lr `1e-4`, ~300 steps, loss falls, samples reproduce the concept. Checkpoint roundtrip without loss jump. Clean export.

Eyeballing overfit samples is a human check, recorded in `PROGRESS.md`.

### 9 — SDXL inference

Dual text encoders (CLIP ViT-L 768 + OpenCLIP bigG 1280), penultimate hidden states concatenated to 2048, pooled 1280 from bigG. `time_ids` (6) + pooled embeds via `add_embedding`. VAE scale `0.13025` (not `0.18215`). 25-step end-to-end matches reference. Module + single-step goldens fp32 CPU; end-to-end reference may be torch fp16 MPS with the standard end-to-end gate.

SDXL lands as new arch and pipeline code. It must not special-case SDXL inside an existing `layers/` module. A numeric bugfix that is true for the module in general is allowed. A new config-driven layer file is allowed only if reported as a design note in `PROGRESS.md`.

### 10 — Animagine / SDXL LoRA

Reuse the phase 8 trainer. `from_pretrained` on `cagliostrolab/animagine-xl-3.1` works with zero new model code. Two encoder caches + `time_ids` per aspect bucket. Area bucketing ~1024px area, 64px alignment. Batch 1–4 + grad accum. Overfit reproduces 10 images. Real small dataset, no NaN over 2k steps. Export kohya-style keys `lora_unet_...` / `lora_te...` plus alpha.

ComfyUI load is a manual human check recorded in `PROGRESS.md`.

Memory: 16GB uses 768px batch 1 cached; 32GB+ uses 1024px.

## Non-goals

Flux, SD3, T5, RoPE, video, ControlNet, quantization, distributed training, EMA (optional stretch after phase 10), textual inversion.

## Session rules

Read `SPEC.md` fully, then `PROGRESS.md`. Work ONLY on Phase N. Never loosen tolerances. Run the full test suite; all green required. On completion update `PROGRESS.md`, commit `phase(N): <summary>`, stop. If blocked more than 3 attempts on a numeric gate, stop and report the per-module diff table.
