# MODULE_9_global_curvature_gradient.py

"""
MODULE 9 — GLOBAL CURVATURE GRADIENT Γ_G

Computes the large-scale curvature invariant:

    Γ_G = < |∇Γ| >

and the preferred cosmic direction:

    n_G = < ∇Γ > / | < ∇Γ > |

Inputs:
    gamma_field   : Γ_eff field from MODULE 2
    dx            : spatial grid spacing
"""

from __future__ import annotations

import numpy as np

from core.operators import gradient, grad_magnitude


def global_curvature_gradient(gamma_eff: np.ndarray, dx: float) -> float:
    """Compute Γ_G = < |∇Γ| >."""

    grad_mag = grad_magnitude(gamma_eff, dx)
    return float(np.mean(grad_mag))


def curvature_direction(gamma_eff: np.ndarray, dx: float) -> np.ndarray:
    """Compute the unit vector n_G = <∇Γ> / |<∇Γ>|."""

    dGdx, dGdy = gradient(gamma_eff, dx)
    mean_dx = float(np.mean(dGdx))
    mean_dy = float(np.mean(dGdy))

    vec = np.array([mean_dx, mean_dy], dtype=float)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return np.array([0.0, 0.0])
    return vec / norm


def compute_global_curvature_stats(gamma_eff: np.ndarray, dx: float) -> dict:
    """
    Returns a dictionary:
        {
            "Gamma_G": scalar curvature invariant,
            "direction_G": unit direction vector,
        }
    """

    GG = global_curvature_gradient(gamma_eff, dx)
    nG = curvature_direction(gamma_eff, dx)
    return {"Gamma_G": GG, "direction_G": nG}
