"""
Main entry point for the Motion–TimeSpace simulation engine.

Runs a minimal end-to-end simulation using:
    - MODULE 1: ψ-field evolution
    - MODULE 2: Γ–κ curvature memory
    - MODULE 8: structure formation
    - MODULE 9: global curvature gradient
    - MODULE 10: full cosmic evolution orchestrator

Use this file as a quick sanity check; for focused studies see scripts in examples/.
"""

from __future__ import annotations

import numpy as np

from core.constants import C_LIGHT, N_NONLINEAR
from core.utils import random_initial_field
from MODULE_1_psi_field import PsiFieldSimulator
from MODULE_2_gamma_kappa_engine import CurvatureMemoryEngine
from MODULE_8_structure_formation_mts import StructureFormationMTS
from MODULE_9_global_curvature_gradient import compute_global_curvature_stats
from MODULE_10_full_cosmic_evolution import MTSCosmicSimulator


def build_simulators(nx: int, ny: int, dx: float, dt: float):
    """Construct base simulators with simple initial conditions."""

    # ψ-field parameters
    c = C_LIGHT
    gamma = 0.1
    lam = 0.01
    n = N_NONLINEAR

    psi_init = random_initial_field((nx, ny), amplitude=0.1, seed=1)
    psi_t_init = np.zeros((nx, ny), dtype=np.float64)

    psi_sim = PsiFieldSimulator(
        nx=nx,
        ny=ny,
        dx=dx,
        dt=dt,
        c=c,
        gamma=gamma,
        lam=lam,
        n=n,
        psi_init=psi_init,
        psi_t_init=psi_t_init,
    )

    # curvature memory parameters
    eta = 0.5
    zeta = 0.1
    D_gamma = 0.2
    alpha = 0.3
    beta = 0.1

    curv_engine = CurvatureMemoryEngine(
        nx=nx,
        ny=ny,
        dx=dx,
        dt=dt,
        eta=eta,
        zeta=zeta,
        D_gamma=D_gamma,
        alpha=alpha,
        beta=beta,
    )

    # structure formation initial conditions
    delta_init = random_initial_field((nx, ny), amplitude=1e-3, seed=2)
    struct_engine = StructureFormationMTS(
        nx=nx,
        ny=ny,
        dx=dx,
        dt=dt,
        delta_init=delta_init,
    )

    return psi_sim, curv_engine, struct_engine


def main():
    nx = ny = 64
    dx = 1.0
    dt = 0.01

    psi_sim, curv_engine, struct_engine = build_simulators(nx, ny, dx, dt)

    cosmic = MTSCosmicSimulator(
        psi_sim=psi_sim,
        curv_engine=curv_engine,
        struct_engine=struct_engine,
        dx=dx,
        dt_cosmic=dt,
        store_history=True,
    )

    def log_callback(step: int, GG: float, nG: np.ndarray, Gamma_eff: np.ndarray):
        if step % 10 == 0:
            print(
                f"step={step:03d}  "
                f"<Γ_G>={GG:.4e}  "
                f"|n_G|={np.linalg.norm(nG):.3f}  "
                f"<δ>={np.mean(cosmic.struct.delta):.4e}  "
                f"<Γ_eff>={np.mean(Gamma_eff):.4e}"
            )

    cosmic.run(N_steps=50, psi_steps=1, curv_steps=5, struct_steps=1, callback=log_callback)

    # Final global curvature stats
    Phi_final = psi_sim.phi_field()
    Gamma_eff_final = curv_engine.gamma_effective(Phi_final)
    stats = compute_global_curvature_stats(Gamma_eff_final, dx)
    print("Final Γ_G:", stats["Gamma_G"])
    print("Final direction n_G:", stats["direction_G"])


if __name__ == "__main__":
    main()
