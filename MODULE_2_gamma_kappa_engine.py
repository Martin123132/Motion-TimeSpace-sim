# MODULE_2_gamma_kappa_engine.py

"""
MODULE 2 — Γ–κ CURVATURE MEMORY ENGINE

Implements the curvature-memory dynamics:

    κ̇ = η Φ − ζ κ
    Γ̇ = D_Γ ∇²Γ + α Φ − β κ Γ

with the effective curvature persistence defined by MBT-5:

    Γ_eff = (1 − κ) Φ c_g²

where Φ = |∇ψ| from MODULE_1_psi_field.
"""

from __future__ import annotations

import numpy as np

from core.constants import C_G
from core.operators import laplacian


class CurvatureMemoryEngine:
    """
    Evolves the κ and Γ fields on the same grid as ψ.
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dt: float,
        eta: float,
        zeta: float,
        D_gamma: float,
        alpha: float,
        beta: float,
        c_g: float = C_G,
        kappa_init: np.ndarray | None = None,
        gamma_init: np.ndarray | None = None,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt

        self.eta = eta
        self.zeta = zeta
        self.D_gamma = D_gamma
        self.alpha = alpha
        self.beta = beta
        self.c_g = c_g

        self.kappa = np.zeros((nx, ny), dtype=np.float64) if kappa_init is None else kappa_init.astype(np.float64)
        self.gamma = np.zeros((nx, ny), dtype=np.float64) if gamma_init is None else gamma_init.astype(np.float64)

    def step(self, phi: np.ndarray, n_steps: int = 1) -> None:
        """
        Advance κ and Γ forward in time given the current Φ field.

        Args:
            phi: 2D array Φ(x,y) = |∇ψ| from the ψ-field simulation.
        """

        dt = self.dt
        dx = self.dx

        for _ in range(n_steps):
            # κ̇ = η Φ − ζ κ
            kappa_dot = self.eta * phi - self.zeta * self.kappa

            # Γ̇ = D_Γ ∇²Γ + α Φ − β κ Γ
            lap_gamma = laplacian(self.gamma, dx)
            gamma_dot = self.D_gamma * lap_gamma + self.alpha * phi - self.beta * self.kappa * self.gamma

            self.kappa += dt * kappa_dot
            self.gamma += dt * gamma_dot

    def gamma_effective(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute the MBT-5 curvature persistence:

            Γ_eff = (1 − κ) Φ c_g²
        """

        return (1.0 - self.kappa) * phi * (self.c_g ** 2)

    def energy_like_density(self, phi: np.ndarray) -> np.ndarray:
        """
        Simple diagnostic "stored curvature energy" density:

            ρ_Γ ~ Γ_eff
        """

        return self.gamma_effective(phi)
