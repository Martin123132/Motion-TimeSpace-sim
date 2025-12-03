# core/constants.py

"""
Core physical constants and Motion–TimeSpace (MTS) parameters.

These are symbolic defaults. The simulation accepts parameters so you can
fit or explore values without hard-coding guesses.
"""

from __future__ import annotations

import numpy as np

# Fundamental constants (SI units)
C_LIGHT = 2.99792458e8  # m/s
HBAR = 1.054571817e-34  # J·s
G_NEWTON = 6.67430e-11  # m^3 kg^-1 s^-2

# MTS-specific theoretical constants
# Φ_G is a geometric constant; placeholder of order unity by default.
PHI_G = 1.0

# Nonlinear exponent n = 4/3 from the MTS field theory
#   ∂²_t ψ – c² ∇²ψ + γ ∂_t ψ + λ |ψ|^{n−1} = 0,  with  n = 4/3
N_NONLINEAR = 4.0 / 3.0

# Curvature propagation coefficient c_g – default to c_light
C_G = C_LIGHT


def gamma_from_phiG(
    phi_g: float = PHI_G,
    c: float = C_LIGHT,
    G: float = G_NEWTON,
    hbar: float = HBAR,
) -> float:
    """Compute γ = Φ_G √(c⁵/(G ħ))."""

    return phi_g * np.sqrt(c ** 5 / (G * hbar))


def lambda_from_phiG(
    phi_g: float = PHI_G,
    c: float = C_LIGHT,
    G: float = G_NEWTON,
    gamma: float | None = None,
    hbar: float = HBAR,
) -> float:
    """Compute λ = Φ_G³ (c³/G) γ using the supplied or derived γ."""

    gamma_val = gamma_from_phiG(phi_g=phi_g, c=c, G=G, hbar=hbar) if gamma is None else gamma
    return (phi_g ** 3) * (c ** 3 / G) * gamma_val
