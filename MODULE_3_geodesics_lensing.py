# MODULE_3_geodesics_lensing.py

"""
MODULE 3 — GEODESICS + LENSING ENGINE

Implements weak-field MTS lensing using curvature persistence Γ_eff
from MODULE_2_gamma_kappa_engine.

Core equations:

    Φ_Γ(x, y) = ∫ W(z) Γ_eff(x, y, z) dz  (2D approximation uses Γ_eff directly)
    α_x = ∂Φ_Γ/∂x
    α_y = ∂Φ_Γ/∂y

Ray propagation:
    x_{n+1} = x_n + α_x ds
    y_{n+1} = y_n + α_y ds
"""

from __future__ import annotations

import numpy as np

from core.operators import gradient


class LensingEngine:
    """Computes lensing potential and deflection angles from Γ_eff."""

    def __init__(self, dx: float):
        self.dx = dx

    def lensing_potential(self, gamma_eff: np.ndarray) -> np.ndarray:
        """
        For a 2D simulation slice we take Φ_Γ = Γ_eff.
        Later this can integrate along z for full 3D.
        """

        return gamma_eff.copy()

    def deflection_angles(self, gamma_eff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute lensing deflection field:

            α_x = ∂Φ_Γ/∂x
            α_y = ∂Φ_Γ/∂y

        Uses central-difference gradients.
        """

        Phi_Gamma = self.lensing_potential(gamma_eff)
        ax, ay = gradient(Phi_Gamma, self.dx)
        return ax, ay


class RayTracer:
    """Propagates rays across the deflection field."""

    def __init__(self, ax: np.ndarray, ay: np.ndarray, dx: float):
        self.ax = ax
        self.ay = ay
        self.dx = dx
        self.nx, self.ny = ax.shape

    def _sample(self, field: np.ndarray, x: float, y: float) -> float:
        """Bilinear sampling of field at fractional coordinates with periodic wrap."""

        x = x % self.nx
        y = y % self.ny

        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = (x0 + 1) % self.nx
        y1 = (y0 + 1) % self.ny

        fx = x - x0
        fy = y - y0

        return (
            field[x0, y0] * (1 - fx) * (1 - fy)
            + field[x1, y0] * fx * (1 - fy)
            + field[x0, y1] * (1 - fx) * fy
            + field[x1, y1] * fx * fy
        )

    def trace_rays(
        self,
        x0: np.ndarray,
        y0: np.ndarray,
        ds: float,
        n_steps: int,
    ) -> np.ndarray:
        """
        Trace multiple rays.

        Args:
            x0, y0: arrays of initial positions
            ds: step size
            n_steps: number of propagation steps

        Returns:
            trajectories: shape (N_rays, n_steps, 2)
        """

        n_rays = len(x0)
        traj = np.zeros((n_rays, n_steps, 2))

        x = x0.astype(float).copy()
        y = y0.astype(float).copy()

        for t in range(n_steps):
            traj[:, t, 0] = x
            traj[:, t, 1] = y

            # sample deflection
            ax_vals = np.array([self._sample(self.ax, x[i], y[i]) for i in range(n_rays)])
            ay_vals = np.array([self._sample(self.ay, x[i], y[i]) for i in range(n_rays)])

            # update positions
            x += ay_vals * ds
            y += ax_vals * ds

        return traj
