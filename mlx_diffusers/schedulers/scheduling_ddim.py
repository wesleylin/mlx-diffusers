"""DDIM scheduler, eta=0 path used by SD1.5.

Mirror of ``diffusers.schedulers.scheduling_ddim.DDIMScheduler`` for the
fields stored in the SD1.5 scheduler config: scaled-linear betas, ``steps_offset``,
``set_alpha_to_one``, and ``clip_sample``. ``timestep_spacing`` defaults to
``leading``, matching the installed diffusers default. ``step`` follows formulas
(12) and (16) of the DDIM paper. Eta 0 drops the stochastic term.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_diffusers.schedulers.scheduling_base import alphas_cumprod_from_betas, betas_from_config


@dataclass
class DDIMSchedulerOutput:
    prev_sample: mx.array
    pred_original_sample: mx.array


class DDIMScheduler:
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas=None,
        clip_sample: bool = True,
        set_alpha_to_one: bool = True,
        steps_offset: int = 0,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "leading",
        clip_sample_range: float = 1.0,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_schedule = beta_schedule
        self.clip_sample = bool(clip_sample)
        self.set_alpha_to_one = bool(set_alpha_to_one)
        self.steps_offset = int(steps_offset)
        self.prediction_type = prediction_type
        self.timestep_spacing = timestep_spacing
        self.clip_sample_range = float(clip_sample_range)
        betas = betas_from_config(
            {
                "num_train_timesteps": self.num_train_timesteps,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "beta_schedule": self.beta_schedule,
                "trained_betas": trained_betas,
            }
        )
        self.betas = betas
        self._alphas_cumprod = alphas_cumprod_from_betas(betas)
        self.alphas_cumprod = mx.array(self._alphas_cumprod)
        if self.set_alpha_to_one:
            final = np.float32(1.0)
        else:
            final = self._alphas_cumprod[0]
        self.final_alpha_cumprod = mx.array(final)
        self.init_noise_sigma = 1.0
        self.num_inference_steps: int | None = None
        self.timesteps = np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)

    @classmethod
    def from_config(cls, config: dict) -> "DDIMScheduler":
        return cls(
            num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
            beta_start=float(config.get("beta_start", 0.0001)),
            beta_end=float(config.get("beta_end", 0.02)),
            beta_schedule=config.get("beta_schedule", "linear"),
            trained_betas=config.get("trained_betas"),
            clip_sample=bool(config.get("clip_sample", True)),
            set_alpha_to_one=bool(config.get("set_alpha_to_one", True)),
            steps_offset=int(config.get("steps_offset", 0)),
            prediction_type=config.get("prediction_type", "epsilon"),
            timestep_spacing=config.get("timestep_spacing", "leading"),
            clip_sample_range=float(config.get("clip_sample_range", 1.0)),
        )

    def scale_model_input(self, sample: mx.array, timestep=None) -> mx.array:
        del timestep
        return sample

    def set_timesteps(self, num_inference_steps: int) -> None:
        if num_inference_steps > self.num_train_timesteps:
            raise ValueError(
                f"num_inference_steps {num_inference_steps} exceeds {self.num_train_timesteps}"
            )
        self.num_inference_steps = int(num_inference_steps)
        if self.timestep_spacing == "linspace":
            timesteps = (
                np.linspace(0, self.num_train_timesteps - 1, num_inference_steps).round()[::-1].copy().astype(np.int64)
            )
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.int64)
            timesteps += self.steps_offset
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps
            timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.int64)
            timesteps -= 1
        else:
            raise ValueError(f"timestep_spacing {self.timestep_spacing!r} is not supported")
        self.timesteps = timesteps

    def _alpha(self, timestep: int) -> mx.array:
        if timestep >= 0:
            return self.alphas_cumprod[int(timestep)]
        return self.final_alpha_cumprod

    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
        eta: float = 0.0,
    ) -> DDIMSchedulerOutput:
        if self.num_inference_steps is None:
            raise ValueError("set_timesteps must run before step")
        if self.prediction_type != "epsilon":
            raise ValueError(f"prediction_type {self.prediction_type!r} is not the SD1.5 epsilon path")
        timestep = int(timestep)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self._alpha(timestep)
        alpha_prod_t_prev = self._alpha(prev_timestep)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - mx.sqrt(beta_prod_t) * model_output) / mx.sqrt(alpha_prod_t)
        pred_epsilon = model_output
        if self.clip_sample:
            pred_original_sample = mx.clip(
                pred_original_sample, -self.clip_sample_range, self.clip_sample_range
            )
        variance = self._variance(timestep, prev_timestep)
        std_dev_t = eta * mx.sqrt(variance)
        pred_sample_direction = mx.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * pred_epsilon
        prev_sample = mx.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction
        if eta > 0:
            raise ValueError("DDIM eta > 0 is not the SD1.5 deterministic path")
        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def _variance(self, timestep: int, prev_timestep: int) -> mx.array:
        alpha_prod_t = self._alpha(timestep)
        alpha_prod_t_prev = self._alpha(prev_timestep)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        return (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
