"""The geometry core: centroid, reflections, D, description length and alignment, in numpy.

States are rows. Every reflection is taken about the landmark centroid `c`, so structures found early and late share
a frame. Description length is probabilistic PCA's two-part code (PLAN.md, step 2), provisional until the toy bed
has tested it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def centroid(X: np.ndarray) -> np.ndarray:
    return np.asarray(X).mean(axis=0)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def reflect(X: np.ndarray, c: np.ndarray, u: np.ndarray) -> np.ndarray:
    """X mirrored in the hyperplane through c normal to u: c + H(x - c), H = I - 2uu^T."""
    u = unit(u)
    return X - 2.0 * ((X - c) @ u)[:, None] * u


def spread(sets: list[np.ndarray]) -> float:
    """D: the root-mean-square distance of a state from its set's mean, averaged over sets."""
    return float(np.mean([np.sqrt(np.mean(np.sum((S - S.mean(0)) ** 2, axis=1))) for S in sets]))


def between(sets: list[np.ndarray]) -> float:
    """The root-mean-square distance of the set means from their overall mean."""
    M = np.stack([S.mean(0) for S in sets])
    return float(np.sqrt(np.mean(np.sum((M - M.mean(0)) ** 2, axis=1))))


@dataclass
class Code:
    """Axes fitted to a window of states: their affine subspace (mean + span of the rows of `axes`)."""
    mean: np.ndarray
    axes: np.ndarray          # k x d, orthonormal rows
    bits: float               # the description length that chose k


def description_length(eigs: np.ndarray, n: int, d: int, k: int) -> float:
    """Bits for n states in d dimensions under k axes: Gaussian bits for the data (constants dropped, so only
    differences between k mean anything) plus half log2 n bits per parameter. Fewer states than dimensions span only
    n - 1 of them, so the code is priced within that span; priced over all d it keeps every axis it can."""
    d = min(d, n - 1)
    total = float(eigs.sum())
    kept = eigs[:k]
    sigma2 = max(total - float(kept.sum()), 1e-300) / (d - k)
    data = 0.5 * n * (float(np.log2(kept).sum()) + (d - k) * np.log2(sigma2))
    params = d * k - k * (k + 1) / 2 + k + 1
    return data + 0.5 * params * np.log2(n)


def fit_code(X: np.ndarray, max_k: int | None = None) -> Code:
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    mean = X.mean(0)
    _, s, vt = np.linalg.svd(X - mean, full_matrices=False)
    eigs = s ** 2 / n
    eigs = eigs[eigs > 1e-12 * max(eigs.max(), 1e-300)] if eigs.size else eigs
    top = min(len(eigs), d - 1, n - 2, max_k if max_k is not None else d)
    bits = [description_length(eigs, n, d, k) for k in range(top + 1)]
    k = int(np.argmin(bits))
    return Code(mean=mean, axes=vt[:k], bits=float(bits[k]))


def residual(code: Code, X: np.ndarray) -> float:
    """The mean squared distance of the states from the code's affine subspace."""
    Z = np.asarray(X, dtype=float) - code.mean
    if len(code.axes):
        Z = Z - (Z @ code.axes.T) @ code.axes
    return float(np.mean(np.sum(Z ** 2, axis=1)))


def alignment(A: np.ndarray, B: np.ndarray) -> float:
    """How far two sets of orthonormal axes span the same space: the mean squared cosine of their principal angles,
    1 for the same space, 0 for orthogonal ones."""
    if len(A) == 0 or len(B) == 0:
        return 0.0
    return float(np.sum((A @ B.T) ** 2) / min(len(A), len(B)))


def fit_mirror(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """The normal of the mirror taking X to Y: the main direction of their differences."""
    _, _, vt = np.linalg.svd(X - Y, full_matrices=False)
    return vt[0]


def fit_additive(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return (Y - X).mean(0)


def rmse(Y: np.ndarray, Yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((Y - Yhat) ** 2, axis=1))))


def mirror_or_additive(X: np.ndarray, Y: np.ndarray, c: np.ndarray, D: float, n_fit: int) -> tuple[str, float, float]:
    """Fit a mirror about c and an additive direction on the first n_fit pairs; on the rest, whichever predicts Y
    better by more than D wins, else "undecided". Returns the verdict and both held-out errors."""
    u = fit_mirror(X[:n_fit], Y[:n_fit])
    v = fit_additive(X[:n_fit], Y[:n_fit])
    e_mirror = rmse(Y[n_fit:], reflect(X[n_fit:], c, u))
    e_add = rmse(Y[n_fit:], X[n_fit:] + v)
    if e_add - e_mirror > D:
        return "mirror", e_mirror, e_add
    if e_mirror - e_add > D:
        return "additive", e_mirror, e_add
    return "undecided", e_mirror, e_add
