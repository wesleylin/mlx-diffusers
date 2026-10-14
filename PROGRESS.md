# Progress

One phase at a time. A phase is done only when every acceptance criterion passes. Do not start the next phase while the current one is red.

Human checks (not automated): eyeballing overfit samples (phase 8) and ComfyUI load (phase 10). Record those in this file when a person confirms them.

- [x] **0 — Scaffold + environment.** `uv sync` and `uv run pytest` pass on Apple Silicon, `mx.eval(mx.array([1.0]))` works, this file exists.
- [x] **1 — Golden harness for SD1.5.** `scripts/make_goldens.py` writes `tests/goldens/sd15/*.npz`; smoke test loads each npz and asserts shapes/dtypes. Captured `stable-diffusion-v1-5/stable-diffusion-v1-5` on CPU in fp32. Caption: "a photograph of an astronaut riding a horse"; the second prompt is empty.
- [ ] **2 — Weight conversion.** Registry, mappers, and `load_model`. Zero unmapped keys, post-transpose shapes match, unknown `_class_name` raises.
- [ ] **3 — Core modules fp32.** Timesteps, Resnet, Attention, FeedForward, Transformer2D, up/downsample pass module gates vs torch hooks.
- [ ] **4 — CLIP text encoder.** SD1.5 ViT-L/14 `last_hidden_state` matches golden for both prompts.
- [ ] **5 — AutoencoderKL.** Decode image and encode posterior mean match gates.
- [ ] **6 — SD1.5 UNet.** Single-step noise pred at timesteps 999, 500, and 1 within full-UNet gates. No sampling.
- [ ] **7 — Schedulers + SD1.5 pipeline.** 20-step DDIM within the end-to-end gate; reference and MLX PNGs committed side by side.
- [ ] **8 — SD1.5 LoRA training.** Overfit loss falls and checkpoint roundtrip is clean. Human check: eyeball overfit samples.
- [ ] **9 — SDXL inference.** 25-step end-to-end matches the reference gate. New arch and pipeline code; no SDXL special case inside an existing `layers/` module. A general numeric bugfix is allowed. A new config-driven layer file requires a design note here.
- [ ] **10 — Animagine / SDXL LoRA.** `from_pretrained` on `cagliostrolab/animagine-xl-3.1` with zero new model code. Overfit and a real small run stay finite; kohya-style export. Human check: ComfyUI loads the LoRA.
