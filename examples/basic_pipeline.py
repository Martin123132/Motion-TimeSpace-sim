"""
Example: run ψ + Γ–κ + lensing on a small grid.

This script is lightweight and intended to be run locally to verify that the
pipeline executes without errors. It evolves the ψ-field, feeds Φ into the
curvature engine, computes Γ_eff, and then derives a lensing deflection map.
"""

from __future__ import annotations

import numpy as np

from core.constants import C_LIGHT, N_NONLINEAR
from core.utils import random_initial_field
from MODULE_1_psi_field import PsiFieldSimulator
from MODULE_2_gamma_kappa_engine import CurvatureMemoryEngine
from MODULE_3_geodesics_lensing import LensingEngine, RayTracer


def main():
    nx = ny = 64
    dx = 1.0
    dt = 0.01

    c = C_LIGHT
    gamma = 0.1
    lam = 0.01
    n = N_NONLINEAR

    eta = 0.5
    zeta = 0.1
    D_gamma = 0.2
    alpha = 0.3
    beta = 0.1

    psi_init = random_initial_field((nx, ny), amplitude=0.1, seed=42)
    psi_t_init = np.zeros((nx, ny))

    psi = PsiFieldSimulator(nx, ny, dx, dt, c, gamma, lam, n, psi_init, psi_t_init)
    curv = CurvatureMemoryEngine(nx, ny, dx, dt, eta, zeta, D_gamma, alpha, beta)
    lens = LensingEngine(dx)

    for step in range(30):
        psi.step(1)
        phi = psi.phi_field()
        curv.step(phi, n_steps=5)

    gamma_eff = curv.gamma_effective(phi)
    ax, ay = lens.deflection_angles(gamma_eff)

    x0 = np.linspace(0, nx - 1, 10)
    y0 = np.zeros_like(x0)
    tracer = RayTracer(ax, ay, dx)
    traj = tracer.trace_rays(x0, y0, ds=0.5, n_steps=50)

    print("Trajectories shape:", traj.shape)
    print("Mean deflection ax, ay:", float(np.mean(ax)), float(np.mean(ay)))


if __name__ == "__main__":
    main()
