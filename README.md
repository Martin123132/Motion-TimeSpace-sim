
# Motion–TimeSpace Simulation Engine  
A Unified Computational Framework for ψ-Field Dynamics, Curvature Memory, Structure Formation & Cosmology

The **Motion–TimeSpace (MTS)** simulation engine is a modular, end-to-end computational framework implementing the full physics pipeline described in the MTS/MBT blueprints.

This repository contains **ten fully operational physics modules**, representing:

- nonlinear ψ-field evolution  
- curvature memory (Γ–κ engine)  
- lensing & geodesics  
- MTS cosmology distance relations  
- galaxy rotation curves (m ≈ 1.878)  
- stress–energy tensor of the ψ-field  
- curvature-corrected orbital decay  
- structure formation (δ, v, BAO)  
- global curvature invariant Γ_G  
- unified cosmic evolution  

All modules are plug-and-play, and the engine can be extended to 3D, GPU acceleration, or real observational data comparisons.

---

# 🔷 Features

### **✔ Nonlinear ψ-field engine (n = 4/3)**
Implements the fundamental MTS field equation:
  
\[
\partial_{tt}\psi - c^2\nabla^2\psi + \gamma\,\partial_t\psi + \lambda|\psi|^{n-1}=0.
\]

Produces the motion-gradient field:

\[
\Phi = |\nabla\psi|
\]

which drives **all curvature dynamics**.

---

### **✔ Curvature Memory Engine (Γ–κ)**
Implements the MBT-5 curvature persistence model:

- κ evolution (collapse factor):  
  \[
  \dot\kappa = \eta\Phi - \zeta\kappa
  \]
- Γ evolution (curvature diffusion):  
  \[
  \dot\Gamma = D_\Gamma \nabla^2\Gamma + \alpha\Phi - \beta\kappa\Gamma
  \]

Effective curvature:

\[
\Gamma_{\mathrm{eff}} = (1-\kappa)\Phi c_g^2
\]

This field powers lensing, structure formation, rotation curves, and orbital decay.

---

### ✔ Weak-field lensing & geodesics  
Computes:

- lensing potential  
- deflection maps  
- ray-tracing across Γ_eff  

---

### ✔ MTS Cosmology  
Implements:

\[
H_{\rm eff}(z),\ d_M(z),\ d_L(z),\ \mu(z)
\]

Fully usable for Pantheon+/BAO/H(z) fits.

---

### ✔ Universal Rotation Curve Law (m ≈ 1.878)
Implements:

\[
\mu_{\rm MTS}(y)=\sqrt{\frac{1+y^m}{1+y^2}}
\]

with:

\[
y = \frac{a_N}{a_\Gamma}.
\]

Predicts galaxy rotation curves **without dark matter halos**.

---

### ✔ Stress–Energy Tensor  
Computes:

- density  
- pressures  
- shear  
- curvature-coupled Tμν  

---

### ✔ Orbital Decay with MTS Correction

\[
\dot{a}_{\rm MTS} \propto \Gamma_{\rm eff}\,\Omega^{m-1}
\]

Explains binary deviations and Kuiper-belt eccentricity gaps.

---

### ✔ Structure Formation (δ & v)
Implements:

\[
\dot{\delta}= -\nabla\cdot[(1+\delta)v]
\]

\[
\dot{v} = -\nabla\Phi + \Gamma_{\rm eff}\nabla\delta.
\]

Generates cosmic web, BAO spectrum, and void statistics.

---

### ✔ Global Curvature Invariant Γ_G  
Computes:

\[
\Gamma_G = \langle |\nabla\Gamma_{\rm eff}| \rangle
\]

and cosmic direction vector.

---

### ✔ Unified Cosmic Evolution Engine  
A high-level orchestrator combining:

- ψ → Φ  
- Φ → Γ, κ  
- Γ_eff → δ, v  
- δ → Φ_grav  
- Γ_G → cosmological background  

All running in a single loop.

---

# 📁 Repository Structure

```

Motion-TimeSpace-sim/
├── core/
│   ├── constants.py
│   ├── operators.py
│   └── utils.py
│
├── MODULE_1_psi_field.py
├── MODULE_2_gamma_kappa_engine.py
├── MODULE_3_geodesics_lensing.py
├── MODULE_4_cosmology_distance_engine.py
├── MODULE_5_rotation_curves_m1878.py
├── MODULE_6_stress_energy.py
├── MODULE_7_orbital_decay_mts.py
├── MODULE_8_structure_formation_mts.py
├── MODULE_9_global_curvature_gradient.py
├── MODULE_10_full_cosmic_evolution.py
│
├── main.py
├── examples/
│   └── basic_pipeline.py
├── README.md
└── blueprints/   (optional documentation files)

````

---

# 🚀 Quick Start

### **Run the full MTS cosmic simulation:**

```bash
python main.py
````

Outputs include:

* Γ_G evolution
* curvature direction vector
* mean density contrast
* mean Γ_eff
* ψ-field energy evolution

---

### **Run the light pipeline demo (ψ + Γ + lensing):**

```bash
python examples/basic_pipeline.py
```

Outputs:

* deflection map
* ray-tracing paths
* mean lensing coherence

---

# ⚙️ Dependencies

Add this to your `requirements.txt`:

```
numpy
scipy
```

(Optional: matplotlib for plotting)

---

# 📈 Example Interpretation

The engine produces:

* curvature clustering
* BAO-like oscillations
* galaxy rotation scaling (m ≈ 1.878)
* cosmic web morphology
* curvature-direction alignment

The simulation is intended for:

* cosmological tests
* galaxy rotation curve fitting
* gravitational lensing maps
* ψ-field atomic-scale soliton tests
* orbital-decay deviations
* Cold Spot / Great Attractor geometric analysis

---

# 🛠 Future Extensions

* 3D ψ-field + Γ evolution
* GPU-accelerated kernels (CUDA, JAX, Numba)
* MCMC fitting against data (Pantheon+, BAO, SPARC)
* CMB lensing maps
* full MTS power spectrum pipeline

---

# 🧠 Citation & Theory

This engine implements the mathematical blueprint of the **Motion–TimeSpace (MTS)** framework, including:

* ψ-field nonlinear dynamics
* curvature-memory diffusion
* effective curvature acceleration
* Γ_G global invariant
* universal rotation curve scaling
* modified luminosity distance law

All physics is defined in the project blueprints.

---

# 🙌 Contributing

Pull requests welcome — especially:

* numerical improvements
* stability enhancements
* visualization tools
* data-analysis modules

---

# 📬 Contact

This framework is maintained by **Martin Ollett** (@NoDicePhysics).

```

---

```
