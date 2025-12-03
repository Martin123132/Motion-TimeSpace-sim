# MODULE_6_stress_energy.py

"""
MODULE 6 — STRESS–ENERGY FOR MTS

Contains:
    6A. Stress–energy from ψ-field: ρ, pressures, shear, tensor components
    6B. Curvature-coupled stress tensor: Γ_eff T_{μν}, κ T_{μν}, total tensor

Depends on ψ, ψ_t, gradients, Γ_eff, and κ fields.
"""

from __future__ import annotations

import numpy as np

from core.operators import gradient, grad_magnitude


# ----------------------------------------------------------------------
# MODULE 6A — Stress–Energy from the ψ-field
# ----------------------------------------------------------------------
def compute_energy_density(
    psi: np.ndarray,
    psi_t: np.ndarray,
    lam: float,
    n: float,
    dx: float,
) -> np.ndarray:
    """
    ρ = 1/2 (ψ_t^2 + |∇ψ|^2) + (λ/n) |ψ|^n
    """

    grad_mag = grad_magnitude(psi, dx)
    kinetic = 0.5 * psi_t * psi_t
    gradient = 0.5 * grad_mag * grad_mag
    potential = (lam / n) * np.abs(psi) ** n
    return kinetic + gradient + potential


def compute_pressures(
    psi: np.ndarray,
    lam: float,
    n: float,
    dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    p_x = 1/2 (∂ψ/∂x)^2 - V(ψ)
    p_y = 1/2 (∂ψ/∂y)^2 - V(ψ)
    """

    dpsidx, dpsidy = gradient(psi, dx)
    V = (lam / n) * np.abs(psi) ** n
    px = 0.5 * dpsidx * dpsidx - V
    py = 0.5 * dpsidy * dpsidy - V
    return px, py


def compute_shear(psi: np.ndarray, dx: float) -> np.ndarray:
    """τ_xy = (∂ψ/∂x)(∂ψ/∂y)."""

    dpsidx, dpsidy = gradient(psi, dx)
    return dpsidx * dpsidy


def stress_energy_tensor(
    psi: np.ndarray,
    psi_t: np.ndarray,
    lam: float,
    n: float,
    dx: float,
) -> dict[str, np.ndarray]:
    """
    Returns dictionary containing components:

        ρ, px, py, τ_xy
        Tμν in 2+1D approximation:
            T00 = ρ
            T11 = px
            T22 = py
            T12 = T21 = τ_xy
    """

    rho = compute_energy_density(psi, psi_t, lam, n, dx)
    px, py = compute_pressures(psi, lam, n, dx)
    tau_xy = compute_shear(psi, dx)

    return {
        "rho": rho,
        "px": px,
        "py": py,
        "tau_xy": tau_xy,
        "T00": rho,
        "T11": px,
        "T22": py,
        "T12": tau_xy,
        "T21": tau_xy,
    }


# ----------------------------------------------------------------------
# MODULE 6B — Curvature-coupled stress tensor
# ----------------------------------------------------------------------
def gamma_coupled_tensor(T: dict[str, np.ndarray], gamma_eff: np.ndarray) -> dict[str, np.ndarray]:
    """T_{μν}^{(Γ)} = Γ_eff * T_{μν}."""

    return {k: gamma_eff * v for k, v in T.items()}


def kappa_coupled_tensor(T: dict[str, np.ndarray], kappa: np.ndarray) -> dict[str, np.ndarray]:
    """κ T_{μν} coupling."""

    return {k: kappa * v for k, v in T.items()}


def total_stress_tensor(
    T: dict[str, np.ndarray],
    gamma_eff: np.ndarray,
    kappa: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Full combined tensor:

        T_total = (1 + Γ_eff + κ) T
    """

    factor = 1.0 + gamma_eff + kappa
    return {k: factor * v for k, v in T.items()}
