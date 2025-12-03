# Motion–TimeSpace Simulation Engine

This repository implements the Motion–TimeSpace (MTS) simulation architecture as laid out in the provided blueprints. All modules are fully specified and wired to run as a coherent pipeline.

## Modules

- `MODULE_1_psi_field.py` — Nonlinear ψ-field evolution with n = 4/3.
- `MODULE_2_gamma_kappa_engine.py` — Curvature memory engine evolving Γ and κ and exposing Γ_eff.
- `MODULE_3_geodesics_lensing.py` — Weak-field lensing potential and ray tracing from Γ_eff.
- `MODULE_4_cosmology_distance_engine.py` — MTS cosmology distances (H_eff, d_M, d_L, μ).
- `MODULE_5_rotation_curves_m1878.py` — Universal rotation-curve scaling with m ≈ 1.878.
- `MODULE_6_stress_energy.py` — ψ stress–energy tensor and curvature-coupled extensions.
- `MODULE_7_orbital_decay_mts.py` — Orbital decay with Γ_eff correction.
- `MODULE_8_structure_formation_mts.py` — Structure formation via δ̇ and v̇ with curvature forcing.
- `MODULE_9_global_curvature_gradient.py` — Global curvature invariant Γ_G and direction n_G.
- `MODULE_10_full_cosmic_evolution.py` — High-level orchestrator linking all subsystems.

Core utilities live in `core/` (constants, operators, utils). Example scripts are in `examples/`.

## Quick start

Run the minimal end-to-end simulation:

```bash
python main.py
```

A lighter pipeline demo with lensing is available:

```bash
python examples/basic_pipeline.py
```

Both scripts use small grid sizes for quick execution; adjust parameters inside to explore different regimes.
