"""Euler discrete scheduler.

Mirror of ``diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler``
for the SD1.5 config: scaled-linear betas, ``linspace`` spacing when the config
does not set one, epsilon prediction, and a zero terminal sigma. ``scale_model_input``
divides by ``sqrt(sigma**2 + 1)``. Stochastic churn (``s_churn``) stays at 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from mlx_diffusers.schedulers.scheduling_base import alphas_cumprod_from_betas, betas_from_config


@dataclass
class EulerDiscreteSchedulerOutput:
    prev_sample: mx.array
    pred_original_sample: mx.array


class EulerDiscreteScheduler:
    def __init__(
        self,
        *,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas=None,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
        interpolation_type: str = "linear",
        final_sigmas_type: str = "zero",
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.timestep_spacing = timestep_spacing
        self.steps_offset = int(steps_offset)
        self.interpolation_type = interpolation_type
        self.final_sigmas_type = final_sigmas_type
        betas = betas_from_config(
            {
                "num_train_timesteps": self.num_train_timesteps,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "beta_schedule": self.beta_schedule,
                "trained_betas": trained_betas,
            }
        )
        self._alphas_cumprod = alphas_cumprod_from_betas(betas)
        self.alphas_cumprod = mx.array(self._alphas_cumprod)
        sigmas = np.sqrt((1 - self._alphas_cumprod) / self._alphas_cumprod).astype(np.float32)[::-1]
        self._sigmas = np.concatenate([sigmas, np.array([0], dtype=np.float32)])
        self.num_inference_steps: int | None = None
        self.timesteps = np.linspace(0, self.num_train_timesteps - 1, self.num_train_timesteps, dtype=np.float32)[::-1].copy()
        self._step_index: int | None = None
        self._scale_input_called = False

    @classmethod
    def from_config(cls, config: dict) -> "EulerDiscreteScheduler":
        return cls(
            num_train_timesteps=int(config.get("num_train_timesteps", 1000)),
            beta_start=float(config.get("beta_start", 0.0001)),
            beta_end=float(config.get("beta_end", 0.02)),
            beta_schedule=config.get("beta_schedule", "linear"),
            trained_betas=config.get("trained_betas"),
            prediction_type=config.get("prediction_type", "epsilon"),
            timestep_spacing=config.get("timestep_spacing", "linspace"),
            steps_offset=int(config.get("steps_offset", 0)),
            interpolation_type=config.get("interpolation_type", "linear"),
            final_sigmas_type=config.get("final_sigmas_type", "zero"),
        )

    @property
    def init_noise_sigma(self) -> float:
        maximum = float(np.max(self._sigmas))
        if self.timestep_spacing in {"linspace", "trailing"}:
            return maximum
        return float(np.sqrt(maximum**2 + 1))

    @property
    def sigmas(self) -> mx.array:
        return mx.array(self._sigmas)

    def scale_model_input(self, sample: mx.array, timestep) -> mx.array:
        if self._step_index is None:
            self._step_index = self._index_for_timestep(timestep)
        sigma = np.float32(self._sigmas[self._step_index])
        self._scale_input_called = True
        return sample / mx.sqrt(mx.array(sigma) ** 2 + 1)

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.num_inference_steps = int(num_inference_steps)
        if self.timestep_spacing == "linspace":
            timesteps = np.linspace(
                0, self.num_train_timesteps - 1, num_inference_steps, dtype=np.float32
            )[::-1].copy()
        elif self.timestep_spacing == "leading":
            step_ratio = self.num_train_timesteps // self.num_inference_steps
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
            timesteps += np.float32(self.steps_offset)
        elif self.timestep_spacing == "trailing":
            step_ratio = self.num_train_timesteps / self.num_inference_steps
            timesteps = np.round(np.arange(self.num_train_timesteps, 0, -step_ratio)).astype(np.float32)
            timesteps -= 1
        else:
            raise ValueError(f"timestep_spacing {self.timestep_spacing!r} is not supported")
        train_sigmas = np.sqrt((1 - self._alphas_cumprod) / self._alphas_cumprod).astype(np.float32)
        if self.interpolation_type != "linear":
            raise ValueError(f"interpolation_type {self.interpolation_type!r} is not the SD1.5 path")
        sigmas = np.interp(timesteps, np.arange(0, len(train_sigmas)), train_sigmas).astype(np.float32)
        if self.final_sigmas_type == "zero":
            sigma_last = np.float32(0)
        elif self.final_sigmas_type == "sigma_min":
            sigma_last = np.sqrt((1 - self._alphas_cumprod[0]) / self._alphas_cumprod[0]).astype(np.float32)
        else:
            raise ValueError(f"final_sigmas_type {self.final_sigmas_type!r} is not supported")
        self.timesteps = timesteps
        self._sigmas = np.concatenate([sigmas, np.array([sigma_last], dtype=np.float32)])
        self._step_index = None
        self._scale_input_called = False

    def step(self, model_output: mx.array, timestep, sample: mx.array) -> EulerDiscreteSchedulerOutput:
        if self.prediction_type != "epsilon":
            raise ValueError(f"prediction_type {self.prediction_type!r} is not the SD1.5 epsilon path")
        if isinstance(timestep, (int, np.integer)):
            raise ValueError("EulerDiscreteScheduler.step expects a scheduler timestep, not an integer index")
        if self._step_index is None:
            self._step_index = self._index_for_timestep(timestep)
        sigma = mx.array(np.float32(self._sigmas[self._step_index]))
        pred_original_sample = sample - sigma * model_output
        derivative = (sample - pred_original_sample) / sigma
        dt = mx.array(np.float32(self._sigmas[self._step_index + 1])) - sigma
        prev_sample = sample + derivative * dt
        self._step_index += 1
        return EulerDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    def _index_for_timestep(self, timestep) -> int:
        schedule = self.timesteps
        value = np.float32(timestep.item() if isinstance(timestep, np.ndarray) else timestep)
        matches = np.nonzero(schedule == value)[0]
        if len(matches) == 0:
            raise ValueError(f"timestep {value} is not in the Euler schedule")
        position = 1 if len(matches) > 1 else 0
        return int(matches[position])
