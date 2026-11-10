"""Area, a sense of order (shape's new form; owner, 2026-11-09, proposals/area-sense.md section 2).

Over a window, the world's replies as states on one read layer, projected into the episode's own R principal
directions fitted on its world chunks up to the window's end (causal, as rate's X_c). A is the R x R antisymmetric
Levy-area matrix of that path, the level-2 part of its log-signature that order alone sets. The sense is

    rho = sigma_max(A) / sum_t |dx_t|^2

scale-free; the Frobenius form ||A||_F in place of sigma_max(A) is reported beside it. Mean over the read layers, as
the other senses. Stage A's toy check (proposals/toy_area.py) computes the same quantities from the signature; the
closed form here is checked against it in tests/test_area.py.
"""
from __future__ import annotations

import numpy as np

R = 8


def levy_area(path: np.ndarray) -> np.ndarray:
    """A_ij = 1/2 sum_t (X_t,i dx_t,j - dx_t,i X_t,j), X_t the displacement before step t: the antisymmetric part of
    the path's level-2 signature."""
    dx = np.diff(path, axis=0)
    X = np.cumsum(np.vstack([np.zeros((1, path.shape[1])), dx[:-1]]), axis=0)
    M = X.T @ dx
    return (M - M.T) / 2.0


def rho(path: np.ndarray) -> tuple[float, float]:
    """(sigma_max(A), ||A||_F), each over the sum of squared steps. NaN for a path that does not move."""
    steps = float((np.diff(path, axis=0) ** 2).sum())
    if steps <= 0:
        return float("nan"), float("nan")
    A = levy_area(path)
    return float(np.linalg.svd(A, compute_uv=False)[0]) / steps, float(np.linalg.norm(A)) / steps


def basis(X: np.ndarray, r: int = R) -> tuple[np.ndarray, np.ndarray]:
    """The mean and the top r principal directions (as columns) of the rows of X."""
    mu = X.mean(0)
    Vt = np.linalg.svd(X - mu, full_matrices=False)[2]
    return mu, Vt[: min(r, Vt.shape[0])].T


def window_rho(states: np.ndarray, fit: np.ndarray, r: int = R) -> tuple[float, float]:
    """rho of the path `states` (points x width) in the basis of `fit` (the episode's chunks up to the window's
    end, points x width)."""
    mu, V = basis(fit, r)
    return rho((states - mu) @ V)


def spread(S: np.ndarray, groups: np.ndarray) -> dict:
    """Within-content spread over between-content distance (stage A's measure, proposals/area_noise.py): S is
    texts x width at one layer, groups the content of each text."""
    ids = np.unique(groups)
    means = np.stack([S[groups == g].mean(0) for g in ids])
    k = np.array([(groups == g).sum() for g in ids])
    res = np.concatenate([S[groups == g] - means[i] for i, g in enumerate(ids)])
    within = float(np.sqrt((res ** 2).sum(1).mean() * k.mean() / (k.mean() - 1)))
    iu = np.triu_indices(len(ids), 1)
    d2 = ((means[:, None] - means[None]) ** 2).sum(-1)[iu] - 2 * within ** 2 / k.mean()
    between = float(np.sqrt(max(np.median(d2), 1e-12)))
    return {"within": within, "between": between, "ratio": within / between}
