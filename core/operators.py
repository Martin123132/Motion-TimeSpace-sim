# core/operators.py

"""
Discrete differential operators on a 2D grid for the MTS simulation.

We use second-order finite differences with periodic boundary conditions
(np.roll) to avoid introducing boundary artefacts.
"""

from __future__ import annotations

import numpy as np


def laplacian(field: np.ndarray, dx: float) -> np.ndarray:
    """
    2D scalar Laplacian with periodic boundary conditions.

    Δf ≈ (f_{i+1,j} + f_{i-1,j} + f_{i,j+1} + f_{i,j-1} - 4 f_{i,j}) / dx²
    """

    f = field
    lap = (
        np.roll(f, +1, axis=0)
        + np.roll(f, -1, axis=0)
        + np.roll(f, +1, axis=1)
        + np.roll(f, -1, axis=1)
        - 4.0 * f
    ) / (dx * dx)
    return lap


def gradient(field: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """
    2D central-difference gradient with periodic boundary conditions.

    ∂f/∂x ≈ (f_{i+1,j} - f_{i-1,j}) / (2 dx)
    ∂f/∂y ≈ (f_{i,j+1} - f_{i,j-1}) / (2 dx)
    """

    f = field
    dfdx = (np.roll(f, -1, axis=0) - np.roll(f, +1, axis=0)) / (2.0 * dx)
    dfdy = (np.roll(f, -1, axis=1) - np.roll(f, +1, axis=1)) / (2.0 * dx)
    return dfdx, dfdy


def grad_magnitude(field: np.ndarray, dx: float) -> np.ndarray:
    """
    Φ = |∇ψ| = sqrt( (∂ψ/∂x)² + (∂ψ/∂y)² )
    """

    dfdx, dfdy = gradient(field, dx)
    return np.sqrt(dfdx * dfdx + dfdy * dfdy)


def divergence(fx: np.ndarray, fy: np.ndarray, dx: float) -> np.ndarray:
    """
    Compute divergence of vector field (fx, fy):

        ∇·F ≈ (f_x(x+dx) - f_x(x-dx)) / (2 dx)
             + (f_y(y+dx) - f_y(y-dx)) / (2 dx)

    Uses periodic boundaries consistent with other operators.
    """

    dfxdx = (np.roll(fx, -1, axis=0) - np.roll(fx, +1, axis=0)) / (2.0 * dx)
    dfydy = (np.roll(fy, -1, axis=1) - np.roll(fy, +1, axis=1)) / (2.0 * dx)
    return dfxdx + dfydy
