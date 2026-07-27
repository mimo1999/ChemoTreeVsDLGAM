"""
Basis-function and roughness-penalty primitives for the Shared-Shape Temporal
GAM.

Every shape function in this model is a linear combination of fixed radial
(Gaussian) basis functions placed on a regular grid in a *normalised*
coordinate. A second-difference penalty on *adjacent* coefficients yields
the "penalise wiggliness" smoothing a penalised spline gives, so the shape
functions stay interpretable and stable when observations are sparse.

This module is pure numerics and can be unit-tested in isolation.
"""

import numpy as np


def make_rbf_centers(lo: float, hi: float, n: int) -> tuple[np.ndarray, float]:
    """Return ``n`` equally-spaced RBF centers on ``[lo, hi]`` and a bandwidth.

    The bandwidth is set to the center spacing, which gives ~50% neighbour
    overlap — smooth reconstruction without leaving gaps between bumps.
    """
    n = max(int(n), 1)
    centers = np.linspace(lo, hi, n).astype(np.float32)
    spacing = (hi - lo) / (n - 1) if n > 1 else (hi - lo)
    bandwidth = float(spacing) if spacing > 1e-8 else 1.0
    return centers, bandwidth


def rbf_np(z: np.ndarray, centers: np.ndarray, bandwidth: float) -> np.ndarray:
    """NumPy Gaussian design matrix. ``z`` shape (...) -> output (..., P)."""
    d = (np.asarray(z, np.float32)[..., None] - np.asarray(centers, np.float32)) / bandwidth
    return np.exp(-0.5 * d * d).astype(np.float32)


def second_diff_matrix_np(n: int) -> np.ndarray:
    """NumPy second-order difference operator, shape ``(n-2, n)``.

    ``||D c||^2 = sum_i (c_{i-1} - 2 c_i + c_{i+1})^2`` penalises curvature of
    the coefficient sequence — the discrete analogue of a spline's integrated
    squared second derivative. Returns an empty ``(0, n)`` matrix when there
    are too few centers to define curvature.
    """
    n = int(n)
    if n < 3:
        return np.zeros((0, n), dtype=np.float64)
    D = np.zeros((n - 2, n), dtype=np.float64)
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D
