# MODULE_4_cosmology_distance_engine.py

"""
MODULE 4 — COSMOLOGY DISTANCE ENGINE (MTS)

Implements the MTS cosmology:

    H_eff(z) = H0 * (1 + α ln(1+z) + β z) / (1 + τ z)
    d_M(z) = ∫[0→z] c / H_eff(z') dz'
    d_L(z) = (1+z) d_M(z) [1 + h tanh(z/τ_h) + q z^2]
    μ(z) = 5 log10(d_L / 10pc)
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from core.constants import C_LIGHT


class MTSCosmologyEngine:
    """
    Fully specified MTS cosmology engine.

    Parameters:
        H0      : present-day Hubble constant in km/s/Mpc
        alpha   : curvature-log coefficient
        beta    : linear curvature term
        tau     : denominator term for low–mid-z correction
        h       : tanh amplitude in luminosity distance
        tau_h   : tanh scaling parameter
        q       : quadratic correction
    """

    def __init__(
        self,
        H0: float = 72.41,  # km/s/Mpc
        alpha: float = 0.2016,
        beta: float = 0.0034,
        tau: float = 0.47,
        h: float = 0.032,
        tau_h: float = 0.72,
        q: float = 0.004,
    ):
        self.H0 = H0
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.h = h
        self.tau_h = tau_h
        self.q = q

        # convert H0 to SI for distance integrals:
        self.H0_si = H0 * 1000 / (3.085677581e22)  # km/s/Mpc → 1/s

    # ------------------------------------------------------------------
    # Effective Hubble parameter
    # ------------------------------------------------------------------
    def H_eff(self, z: float | np.ndarray) -> np.ndarray:
        """Effective Hubble expansion rate H_eff(z)."""

        z_arr = np.asarray(z)
        top = 1 + self.alpha * np.log1p(z_arr) + self.beta * z_arr
        bottom = 1 + self.tau * z_arr
        return self.H0 * (top / bottom)

    # ------------------------------------------------------------------
    # Comoving distance
    # ------------------------------------------------------------------
    def comoving_distance(self, z: float | np.ndarray) -> np.ndarray:
        """Numerically computes comoving distance d_M(z)."""

        def integrand(zp: float) -> float:
            return C_LIGHT / (self.H_eff(zp) * 1000)  # convert km/s/Mpc → m/Mpc

        if np.isscalar(z):
            result, _ = quad(integrand, 0, z)
            return result

        out = np.zeros_like(z, dtype=float)
        for i, zi in enumerate(z):
            out[i], _ = quad(integrand, 0, zi)
        return out

    # ------------------------------------------------------------------
    # Luminosity distance
    # ------------------------------------------------------------------
    def luminosity_distance(self, z: float | np.ndarray) -> np.ndarray:
        """d_L(z) = (1+z) d_M(z) [1 + h tanh(z/τ_h) + q z^2]."""

        z_arr = np.asarray(z)
        dM = self.comoving_distance(z_arr)
        correction = 1 + self.h * np.tanh(z_arr / self.tau_h) + self.q * z_arr * z_arr
        return (1 + z_arr) * dM * correction

    # ------------------------------------------------------------------
    # Distance modulus
    # ------------------------------------------------------------------
    def distance_modulus(self, z: float | np.ndarray) -> np.ndarray:
        """μ(z) = 5 log10(d_L / 10 pc)."""

        dL = self.luminosity_distance(z)
        dL_parsec = dL * (1 / 3.085677581e16)  # m → pc
        return 5 * np.log10(dL_parsec / 10.0)

    # ------------------------------------------------------------------
    # Helper: chi-square for distance fits
    # ------------------------------------------------------------------
    def chi_square(self, z: np.ndarray, mu_obs: np.ndarray, sigma: np.ndarray) -> float:
        """Chi-square statistic: χ² = Σ[(μ_obs − μ_model)² / σ²]."""

        mu_model = self.distance_modulus(z)
        return np.sum(((mu_obs - mu_model) ** 2) / (sigma ** 2))
