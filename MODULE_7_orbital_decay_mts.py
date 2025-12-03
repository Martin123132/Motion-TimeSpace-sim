# MODULE_7_orbital_decay_mts.py

"""
MODULE 7 — ORBITAL DECAY WITH MTS CORRECTION

Implements the curvature-driven orbital decay term:

    da/dt = da_GR/dt + C_mts * Γ_eff * Ω^(m - 1)

where:
    a       : semi-major axis
    Ω       : orbital frequency = sqrt(G M / a^3)
    Γ_eff   : curvature persistence from MODULE 2
    m       : universal MTS scaling exponent (~1.878)
"""

from __future__ import annotations

import numpy as np

from core.constants import G_NEWTON


# ----------------------------------------------------------------------
# 1. Orbital frequency
# ----------------------------------------------------------------------
def orbital_frequency(a: float, M: float) -> float:
    """Ω = sqrt(G M / a^3)."""

    return np.sqrt(G_NEWTON * M / (a ** 3))


# ----------------------------------------------------------------------
# 2. MTS decay term
# ----------------------------------------------------------------------
def mts_decay_term(
    a: float,
    M: float,
    gamma_eff: np.ndarray,
    C_mts: float = 1.0,
    m: float = 1.878,
) -> float:
    """
    Compute the MTS curvature-driven decay component:

        da_MTS/dt = C_mts * Γ_eff_orbit * Ω^(m - 1)

    where Γ_eff_orbit = mean(gamma_eff).
    """

    Gamma_orb = float(np.mean(gamma_eff))
    Omega = orbital_frequency(a, M)
    return C_mts * Gamma_orb * (Omega ** (m - 1))


# ----------------------------------------------------------------------
# 3. Single-step orbital evolution
# ----------------------------------------------------------------------
def evolve_orbit_step(
    a: float,
    M: float,
    dt: float,
    da_dt_GR: float,
    gamma_eff: np.ndarray,
    C_mts: float = 1.0,
    m: float = 1.878,
) -> float:
    """
    Evolve semi-major axis a by one timestep:

        a_new = a + ( da_GR/dt + da_MTS/dt ) * dt
    """

    da_dt_MTS = mts_decay_term(a, M, gamma_eff, C_mts=C_mts, m=m)
    da_total = da_dt_GR + da_dt_MTS
    return a + da_total * dt


# ----------------------------------------------------------------------
# 4. Full orbital evolution over time
# ----------------------------------------------------------------------
def evolve_orbit(
    a0: float,
    M: float,
    t_array: np.ndarray,
    da_dt_GR_func,
    gamma_eff: np.ndarray,
    C_mts: float = 1.0,
    m: float = 1.878,
) -> np.ndarray:
    """
    Evolves orbit over an array of times t_array.

    Inputs:
        a0          : initial semi-major axis
        M           : total mass
        t_array     : array of times (seconds)
        da_dt_GR_func : function giving GR decay rate da_GR/dt(t)
        gamma_eff   : curvature field
        C_mts       : coupling constant
        m           : MTS exponent

    Output:
        a(t) array of same length as t_array
    """

    a = a0
    a_vals = np.zeros_like(t_array)

    for i in range(len(t_array)):
        if i == 0:
            a_vals[i] = a
            continue

        dt = t_array[i] - t_array[i - 1]
        da_dt_GR = da_dt_GR_func(t_array[i])

        a = evolve_orbit_step(
            a,
            M,
            dt,
            da_dt_GR=da_dt_GR,
            gamma_eff=gamma_eff,
            C_mts=C_mts,
            m=m,
        )

        a_vals[i] = a

    return a_vals
