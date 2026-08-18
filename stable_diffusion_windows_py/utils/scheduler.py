# ---------------------------------------------------------------------
# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Pure-numpy Euler discrete scheduler for Stable Diffusion v2.1.

Matches ``diffusers.EulerDiscreteScheduler`` with ``beta_schedule="scaled_linear"``,
``timestep_spacing="leading"`` and ``prediction_type="v_prediction"`` (SD v2.1's
config), implemented in numpy.
"""

from __future__ import annotations

import numpy as np

# Stable Diffusion v2.1's fixed scheduler config -- there is only ever one
# correct set of values for this model, so they're hardcoded here.
BETA_START = 0.00085
BETA_END = 0.012
NUM_TRAIN_TIMESTEPS = 1000
STEPS_OFFSET = 1


class EulerScheduler:
    """Euler discrete scheduler for a fixed number of SD v2.1 inference steps.

    The timesteps and per-step noise sigmas are precomputed at construction for
    ``num_steps`` steps. Uses ``v_prediction`` (SD v2.1's prediction type).

    Attributes
    ----------
    timesteps : np.ndarray
        The ``num_steps`` timesteps (descending), fed to the UNet's time input.
    sigmas : np.ndarray
        Noise sigma per step, with a trailing ``0.0`` (length ``num_steps + 1``);
        ``sigmas[i]`` and ``sigmas[i + 1]`` are the current/next sigma for step ``i``.
    """

    timesteps: np.ndarray
    sigmas: np.ndarray

    def __init__(self, num_steps: int) -> None:
        """Build the timesteps and sigmas for ``num_steps`` of inference.

        Parameters
        ----------
        num_steps
            Number of diffusion (inference) steps.
        """
        betas = np.linspace(BETA_START**0.5, BETA_END**0.5, NUM_TRAIN_TIMESTEPS) ** 2
        alphas_cumprod = np.cumprod(1.0 - betas)
        all_sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5

        step_ratio = NUM_TRAIN_TIMESTEPS / num_steps
        ts = (np.arange(0, num_steps) * step_ratio).round().astype(np.int64)
        ts += STEPS_OFFSET
        self.timesteps = ts[::-1].copy()
        self.sigmas = np.append(all_sigmas[self.timesteps], 0.0)

    @property
    def init_noise_sigma(self) -> float:
        """Sigma the initial latent noise should be scaled by (the max sigma)."""
        return float(self.sigmas.max())

    def scale_model_input(self, sample: np.ndarray, step_index: int) -> np.ndarray:
        """Scale the latent by ``1 / sqrt(sigma^2 + 1)`` before the UNet (Euler).

        Parameters
        ----------
        sample
            Current latent sample.
        step_index
            Index into :attr:`sigmas`/:attr:`timesteps` for the current step.

        Returns
        -------
        np.ndarray
            The scaled latent to feed to the UNet.
        """
        sigma = float(self.sigmas[step_index])
        return sample / float((sigma**2 + 1) ** 0.5)

    def step(
        self, model_output: np.ndarray, step_index: int, sample: np.ndarray
    ) -> np.ndarray:
        """Take one Euler step, returning the previous (denoised) latent sample.

        Parameters
        ----------
        model_output
            The UNet's noise prediction for this step.
        step_index
            Index into :attr:`sigmas` for the current step; ``sigmas[step_index]``
            and ``sigmas[step_index + 1]`` are the current/next sigma.
        sample
            Current latent sample.

        Returns
        -------
        np.ndarray
            The latent sample for the next step.
        """
        sigma = float(self.sigmas[step_index])
        sigma_next = float(self.sigmas[step_index + 1])
        pred_orig = model_output * float(
            -sigma / (sigma**2 + 1) ** 0.5
        ) + sample * float(1.0 / (sigma**2 + 1))
        derivative = (sample - pred_orig) / sigma
        return sample + derivative * float(sigma_next - sigma)
