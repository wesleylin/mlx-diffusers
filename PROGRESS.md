# Progress

One phase at a time. A phase is done only when every acceptance criterion passes. Do not start the next phase while the current one is red.

Human checks (not automated): eyeballing overfit samples (phase 8) and ComfyUI load (phase 10). Record those in this file when a person confirms them.

- [x] **0 — Scaffold + environment.** `uv sync` and `uv run pytest` pass on Apple Silicon, `mx.eval(mx.array([1.0]))` works, this file exists.
- [x] **1 — Golden harness for SD1.5.** `scripts/make_goldens.py` writes `tests/goldens/sd15/*.npz`; smoke test loads each npz and asserts shapes/dtypes. Captured `stable-diffusion-v1-5/stable-diffusion-v1-5` on CPU in fp32. Caption: "a photograph of an astronaut riding a horse"; the second prompt is empty.
- [x] **2 — Weight conversion.** Registry, mappers, and `load_model`. Zero unmapped keys, post-transpose shapes match, unknown `_class_name` raises. SD1.5 unet, vae, and text_encoder from the local cache each have 0 unmapped keys. Module classes are not registered yet.
- [x] **3 — Core modules fp32.** Timesteps, TimestepEmbedding, Resnet, self/cross Attention, FeedForward, Transformer2D, up/downsample, and the first down block pass the module gate (mean abs < 1e-5 and max abs < 5e-4) on the timestep-999 hooks.
- [x] **4 — CLIP text encoder.** SD1.5 ViT-L/14 `last_hidden_state` matches the embedding gate (mean abs < 1e-5 and max abs < 1e-3) for both prompts and the batch.
- [x] **5 — AutoencoderKL.** Decode image and encode posterior mean match the full-model gate (mean abs < 1e-5 and max abs < 1e-3). `scaling_factor` is not applied inside encode or decode.
- [x] **6 — SD1.5 UNet.** Single-step noise pred at timesteps 999, 500, and 1 within full-UNet gates. No sampling.
- [x] **7 — Schedulers + SD1.5 pipeline.** 20-step DDIM within the end-to-end gate; reference and MLX PNGs committed side by side.
- [x] **8 — SD1.5 LoRA training.** Short overfit: 2 images, rank 8, lr 1e-4, 40 steps. Fixed-batch loss fell from 0.2589 to 0.2556. Resume matched and the export roundtrip was exact. Human check: overfit samples visually accepted on 2026-10-14.
- [x] **9 — SDXL inference.** `stabilityai/stable-diffusion-xl-base-1.0`. 25-step Euler at 768, guidance 5, seed 0. Pixel mean abs 0.00423, under 2.5/255. Module, text, and UNet gates passed on CPU fp32. No file under `layers/` changed. Linear projection lives in `models/unet_sdxl.py` because the SD1.5 transformer rejects that config. The image is 768, not 1024, so the fp32 UNet fits in 16GB. Installed transformers records encoder-layer outputs with `tie_last_hidden_states` false, so `hidden_states[-2]` is the second-to-last encoder layer. Newer CLIP checkpoints omit `position_ids`; the module keeps its arange buffer. SDXL VAE attention uses `to_q` / `to_k` / `to_v` / `to_out.0`; `AutoencoderKL.load_converted` renames those onto the existing `query` / `key` / `value` / `proj_attn` modules.
- [ ] **10 — Animagine / SDXL LoRA.** `from_pretrained` on `cagliostrolab/animagine-xl-3.1` with zero new model code. Overfit and a real small run stay finite; kohya-style export. Human check: ComfyUI loads the LoRA.
