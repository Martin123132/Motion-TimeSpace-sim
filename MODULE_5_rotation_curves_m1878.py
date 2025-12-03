# MODULE_5_rotation_curves_m1878.py

"""
MODULE 5 — GALAXY ROTATION CURVES (m ≈ 1.878)

Implements the universal MTS rotation-curve scaling law:

    μ_MTS(y) = sqrt( (1 + y^m) / (1 + y^2) )

where y = a_N / a_Γ with
    a_N     = GM(r)/r^2
    a_Γ(r)  = sqrt(<Γ_eff>)

Rotation velocity:

    V_MTS(r) = sqrt( r * a_N * μ_MTS(y) )

Utilities are provided to load SPARC-like data and compute χ² fits.
"""

from __future__ import annotations

import numpy as np

from core.constants import G_NEWTON


# ----------------------------------------------------------------------
# 1. Universal MTS renormalisation function μ_MTS(y)
# ----------------------------------------------------------------------
def mu_MTS(y: np.ndarray, m: float = 1.878) -> np.ndarray:
    """
    μ_MTS(y) = sqrt( (1 + y^m) / (1 + y^2) )

    Vectorised, stable for small y.
    """

    y_arr = np.asarray(y)
    num = 1.0 + np.power(y_arr, m)
    den = 1.0 + y_arr * y_arr
    return np.sqrt(num / den)


# ----------------------------------------------------------------------
# 2. Newtonian acceleration from baryonic mass model
# ----------------------------------------------------------------------
def a_newtonian(r: np.ndarray, M_r: np.ndarray) -> np.ndarray:
    """
    a_N(r) = G M(r) / r^2

    Inputs:
        r    : radii (kpc)
        M_r  : enclosed mass (Msun)

    Output:
        a_N in m/s^2
    """

    r_m = r * 3.085677581e19  # kpc → m
    M_kg = M_r * 1.98847e30   # Msun → kg
    return G_NEWTON * M_kg / (r_m * r_m)


# ----------------------------------------------------------------------
# 3. Curvature-acceleration from Γ_eff
# ----------------------------------------------------------------------
def a_gamma_from_field(gamma_eff: np.ndarray) -> float:
    """
    a_Γ = sqrt( mean(Γ_eff) )

    Effective curvature-memory acceleration scale for 2D slice.
    """

    return np.sqrt(np.mean(gamma_eff))


# ----------------------------------------------------------------------
# 4. Compute MTS rotation curve
# ----------------------------------------------------------------------
def rotation_curve_MTS(
    r: np.ndarray,
    M_r: np.ndarray,
    gamma_eff: np.ndarray,
    m: float = 1.878,
) -> np.ndarray:
    """
    Compute predicted rotation velocity V_MTS(r).

    Inputs:
        r         : radius array (kpc)
        M_r       : baryonic enclosed mass (Msun)
        gamma_eff : Γ_eff field from MODULE 2
        m         : MTS exponent (default 1.878)

    Returns:
        V_MTS in km/s.
    """

    aN = a_newtonian(r, M_r)
    aG = a_gamma_from_field(gamma_eff)

    y = aN / aG
    mu = mu_MTS(y, m=m)

    r_m = r * 3.085677581e19  # kpc → m
    V = np.sqrt(r_m * aN * mu)
    return V / 1000.0  # m/s → km/s


# ----------------------------------------------------------------------
# 5. SPARC loader utility
# ----------------------------------------------------------------------
def load_sparc_file(path: str):
    """
    Load a SPARC galaxy file (CSV with R, Mass, V_obs).

    Expected columns:
        R_kpc, M_enclosed_Msun, V_obs_kms, eV_obs

    Returns:
        r, M_r, V_obs, eV_obs
    """

    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    r = data[:, 0]
    M_r = data[:, 1]
    V_obs = data[:, 2]
    eV_obs = data[:, 3]
    return r, M_r, V_obs, eV_obs


# ----------------------------------------------------------------------
# 6. Fit statistics
# ----------------------------------------------------------------------
def chi_square_rotation(
    r: np.ndarray,
    M_r: np.ndarray,
    V_obs: np.ndarray,
    eV_obs: np.ndarray,
    gamma_eff: np.ndarray,
    m: float = 1.878,
) -> float:
    """
    χ² = Σ[(V_obs - V_MTS)² / σ²]
    """

    V_model = rotation_curve_MTS(r, M_r, gamma_eff, m=m)
    return np.sum(((V_obs - V_model) ** 2) / (eV_obs ** 2))
