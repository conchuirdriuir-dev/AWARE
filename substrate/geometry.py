"""The geometry core, in numpy. States are rows; every quantity is taken about the origin's centroid `c`.

Shared by both tracks. Fitting and admission of axes are in `axes.py`; the hypotheses about negation are the research
track's (`research/gate1.py`).
"""
from __future__ import annotations

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


def off(X: np.ndarray, c: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """X about c, with the part in the span of `axes` (orthonormal rows) removed."""
    Z = np.asarray(X, dtype=float) - c
    if len(axes):
        Z = Z - (Z @ axes.T) @ axes
    return Z


def rms(Z: np.ndarray) -> float:
    """Root-mean-square length of the rows: the unit of D."""
    return float(np.sqrt(np.mean(np.sum(Z ** 2, axis=1))))


def top_direction(Z: np.ndarray) -> np.ndarray:
    """The direction of largest second moment of the rows (not centred: a shift counts)."""
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return vt[0]


def alignment(A: np.ndarray, B: np.ndarray) -> float:
    """How far two sets of orthonormal axes span the same space: the mean squared cosine of their principal angles,
    1 for the same space, 0 for orthogonal ones."""
    if len(A) == 0 or len(B) == 0:
        return 0.0
    return float(np.sum((A @ B.T) ** 2) / min(len(A), len(B)))


def rmse(Y: np.ndarray, Yhat: np.ndarray) -> float:
    return rms(Y - Yhat)
