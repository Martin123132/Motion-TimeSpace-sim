# MODULE_8_structure_formation_mts.py

"""
MODULE 8 — STRUCTURE FORMATION WITH MTS

Implements the coupled evolution system:

    δ̇ = - ∇·[(1 + δ) v]
    v̇ = - ∇Φ + Γ_eff ∇δ
    ∇² Φ = δ

Produces δ(x,y,t), v(x,y,t), Φ(x,y,t) along with power spectrum utilities.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fftfreq, irfft2, rfft2

from core.operators import divergence, gradient, laplacian


# ----------------------------------------------------------------------
# 1. Poisson solver: ∇² Φ = δ  (FFT-based)
# ----------------------------------------------------------------------
def poisson_potential(delta: np.ndarray, dx: float) -> np.ndarray:
    """
    Solve Poisson equation ∇² Φ = δ using FFT with periodic boundaries.

    Φ_k = - δ_k / k^2 with k^2 = kx^2 + ky^2.
    """

    nx, ny = delta.shape
    kx = fftfreq(nx, d=dx)
    ky = fftfreq(ny, d=dx)

    kx2 = kx * kx
    ky2 = ky * ky

    delta_k = rfft2(delta)

    k2 = np.zeros_like(delta_k)
    for i in range(nx):
        for j in range(delta_k.shape[1]):
            k2[i, j] = kx2[i] + ky2[j]

    k2[0, 0] = 1.0

    phi_k = -delta_k / k2
    phi = irfft2(phi_k, s=delta.shape)
    return phi


# ----------------------------------------------------------------------
# 2. Structure Formation Engine
# ----------------------------------------------------------------------
class StructureFormationMTS:
    """Evolves density contrast δ and velocity field v under MTS curvature."""

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dt: float,
        delta_init: np.ndarray,
        vx_init: np.ndarray | None = None,
        vy_init: np.ndarray | None = None,
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt

        self.delta = delta_init.copy()
        self.vx = np.zeros((nx, ny)) if vx_init is None else vx_init.copy()
        self.vy = np.zeros((nx, ny)) if vy_init is None else vy_init.copy()

    def compute_potential(self) -> np.ndarray:
        """Compute Φ from δ using FFT Poisson solver."""

        return poisson_potential(self.delta, self.dx)

    def step(self, gamma_eff: np.ndarray, n_steps: int = 1) -> None:
        """Advance (δ, v) forward in time using explicit Euler."""

        for _ in range(n_steps):
            dt = self.dt

            Phi = self.compute_potential()
            dPhidx, dPhidy = gradient(Phi, self.dx)
            dDdx, dDdy = gradient(self.delta, self.dx)

            vx_dot = -dPhidx + gamma_eff * dDdx
            vy_dot = -dPhidy + gamma_eff * dDdy

            flux_x = (1 + self.delta) * self.vx
            flux_y = (1 + self.delta) * self.vy
            div_flux = divergence(flux_x, flux_y, self.dx)
            delta_dot = -div_flux

            self.vx += dt * vx_dot
            self.vy += dt * vy_dot
            self.delta += dt * delta_dot

    def power_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute isotropic matter power spectrum P(k) from δ(x,y)."""

        delta_k = rfft2(self.delta)
        Pk2d = np.abs(delta_k) ** 2

        nx, ny_half = delta_k.shape
        kx = fftfreq(nx, d=self.dx)
        ky = fftfreq(2 * (ny_half - 1), d=self.dx)[:ny_half]

        k_vals = []
        P_vals = []
        for i in range(nx):
            for j in range(ny_half):
                k = np.sqrt(kx[i] ** 2 + ky[j] ** 2)
                k_vals.append(k)
                P_vals.append(Pk2d[i, j])

        k_vals = np.array(k_vals)
        P_vals = np.array(P_vals)
        idx = np.argsort(k_vals)
        return k_vals[idx], P_vals[idx]
