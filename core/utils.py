# core/utils.py

"""
General utilities: normalisation, smoothing, and field initialisation.
"""

from __future__ import annotations

import numpy as np

from .operators import laplacian


def normalize_field(field: np.ndarray, target_rms: float = 1.0) -> np.ndarray:
    """Rescale field so that RMS(field) = target_rms."""

    rms = np.sqrt(np.mean(field * field))
    if rms == 0:
        return field
    return field * (target_rms / rms)


def smooth_field(field: np.ndarray, dx: float, n_steps: int = 1, kappa: float = 0.1) -> np.ndarray:
    """
    Simple diffusion-like smoothing using the Laplacian:

        f_{t+dt} = f_t + kappa * Δf * dt

    We set dt = dx here so only kappa and n_steps matter.
    """

    f = field.copy()
    for _ in range(n_steps):
        f += kappa * laplacian(f, dx)
    return f


def random_initial_field(
    shape: tuple[int, int],
    amplitude: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Gaussian random initial condition of given amplitude."""

    rng = np.random.default_rng(seed)
    f = rng.normal(loc=0.0, scale=1.0, size=shape)
    return normalize_field(f, target_rms=amplitude)
