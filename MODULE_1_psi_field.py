# MODULE_1_psi_field.py

"""
MODULE 1 — ψ-FIELD NONLINEAR EVOLUTION

Implements the microscopic MTS field equation

    ∂²_t ψ − c² ∇²ψ + γ ∂_t ψ + λ |ψ|^{n−1} = 0

as derived from the fundamental action. We discretise it on a 2D grid with
simple finite differences, evolving ψ(t, x, y).
"""

from __future__ import annotations

import numpy as np

from core.constants import N_NONLINEAR
from core.operators import laplacian, grad_magnitude


class PsiFieldSimulator:
    """
    Time integrator for the nonlinear ψ-field in 2D.

    State variables:
        psi    : scalar field ψ(t, x, y)
        psi_t  : time derivative ∂ψ/∂t

    Evolution equation (rearranged):

        ∂²_t ψ = c² ∇²ψ − γ ∂_t ψ − λ |ψ|^{n−1}

    Updates use an explicit Euler scheme:

        psi_t  ← psi_t + dt * psi_tt
        psi    ← psi    + dt * psi_t
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        dx: float,
        dt: float,
        c: float,
        gamma: float,
        lam: float,
        n: float = N_NONLINEAR,
        psi_init: np.ndarray | None = None,
        psi_t_init: np.ndarray | None = None,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dt = dt

        self.c = c
        self.gamma = gamma
        self.lam = lam
        self.n = n

        self.psi = np.zeros((nx, ny), dtype=np.float64) if psi_init is None else psi_init.astype(np.float64)
        self.psi_t = np.zeros_like(self.psi) if psi_t_init is None else psi_t_init.astype(np.float64)

    def _nonlinear_term(self) -> np.ndarray:
        """
        Compute λ |ψ|^{n−1} with sign(ψ) symmetry.

        Written form λ |ψ|^{n−1} sign(ψ) preserves symmetry under ψ ↦ -ψ.
        """

        abs_psi = np.abs(self.psi)
        return self.lam * np.sign(self.psi) * (abs_psi ** (self.n - 1.0))

    def compute_psi_tt(self) -> np.ndarray:
        """Compute ∂²_t ψ at the current state."""

        lap = laplacian(self.psi, self.dx)
        nonlinear = self._nonlinear_term()
        psi_tt = (self.c * self.c) * lap - self.gamma * self.psi_t - nonlinear
        return psi_tt

    def step(self, n_steps: int = 1) -> None:
        """Advance the ψ-field forward in time by n_steps * dt."""

        dt = self.dt
        for _ in range(n_steps):
            psi_tt = self.compute_psi_tt()
            self.psi_t += dt * psi_tt
            self.psi += dt * self.psi_t

    def phi_field(self) -> np.ndarray:
        """
        Compute Φ = |∇ψ| from the current field configuration.

        This "motion layer" feeds the curvature-memory engine (Γ–κ).
        """

        return grad_magnitude(self.psi, self.dx)

    def energy_density(self) -> np.ndarray:
        """
        Approximate MTS energy density:

            ρ ≈ 1/2 (ψ_t² + |∇ψ|²) + (λ/n) |ψ|^n
        """

        grad_mag = self.phi_field()
        kinetic = 0.5 * (self.psi_t ** 2)
        gradient = 0.5 * (grad_mag ** 2)
        potential = (self.lam / self.n) * (np.abs(self.psi) ** self.n)
        return kinetic + gradient + potential
