# MODULE_10_full_cosmic_evolution.py

"""
MODULE 10 — FULL COSMIC EVOLUTION ENGINE

Links all subsystems:
    • ψ-field evolution (Module 1)
    • Γ–κ curvature memory (Module 2)
    • Structure formation (Module 8)
    • Global curvature gradient Γ_G (Module 9)
Optional hooks: lensing, rotation curves, orbital decay.
"""

from __future__ import annotations

import numpy as np

from MODULE_1_psi_field import PsiFieldSimulator
from MODULE_2_gamma_kappa_engine import CurvatureMemoryEngine
from MODULE_8_structure_formation_mts import StructureFormationMTS
from MODULE_9_global_curvature_gradient import compute_global_curvature_stats


class MTSCosmicSimulator:
    """High-level unified simulator for the Motion–TimeSpace framework."""

    def __init__(
        self,
        psi_sim: PsiFieldSimulator,
        curv_engine: CurvatureMemoryEngine,
        struct_engine: StructureFormationMTS,
        dx: float,
        dt_cosmic: float,
        store_history: bool = True,
    ):
        self.psi = psi_sim
        self.curv = curv_engine
        self.struct = struct_engine
        self.dx = dx
        self.dt_cosmic = dt_cosmic
        self.store_history = store_history

        self.history = {
            "Gamma_G": [],
            "direction_G": [],
            "mean_delta": [],
            "mean_Gamma_eff": [],
            "psi_energy": [],
        }

    def step(self, psi_steps: int = 1, curv_steps: int = 5, struct_steps: int = 1):
        """
        Perform one cosmic evolution block:
            1. evolve ψ
            2. compute Φ
            3. evolve Γ, κ
            4. compute Γ_eff
            5. evolve δ, v (structure)
            6. compute Γ_G, direction
            7. save diagnostics
        """

        self.psi.step(n_steps=psi_steps)
        Phi = self.psi.phi_field()

        self.curv.step(phi=Phi, n_steps=curv_steps)
        Gamma_eff = self.curv.gamma_effective(Phi)

        self.struct.step(Gamma_eff, n_steps=struct_steps)

        stats = compute_global_curvature_stats(Gamma_eff, self.dx)
        GG = stats["Gamma_G"]
        nG = stats["direction_G"]

        if self.store_history:
            self.history["Gamma_G"].append(GG)
            self.history["direction_G"].append(nG)
            self.history["mean_delta"].append(float(np.mean(self.struct.delta)))
            self.history["mean_Gamma_eff"].append(float(np.mean(Gamma_eff)))
            self.history["psi_energy"].append(float(np.mean(self.psi.energy_density())))

        return GG, nG, Gamma_eff

    def run(
        self,
        N_steps: int,
        psi_steps: int = 1,
        curv_steps: int = 5,
        struct_steps: int = 1,
        callback=None,
    ):
        """Run full simulation for N_steps cosmic blocks."""

        for i in range(N_steps):
            GG, nG, Gamma_eff = self.step(
                psi_steps=psi_steps,
                curv_steps=curv_steps,
                struct_steps=struct_steps,
            )

            if callback is not None:
                callback(i, GG, nG, Gamma_eff)

        return self.history
