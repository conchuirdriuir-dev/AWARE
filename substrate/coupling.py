"""The window's code and coupling: shared observer primitives (owner, 2026-11-03, DECISIONS.md).

Moved unchanged from `product/coupling.py`, which re-exports them, so the product rounds run and read as before. They
moved because they describe the entity's dynamics, not drift: the body readout (`body/`) reads them too.

The observer is a joint fit in the code, s(t+1) ~ A s(t) + B E(t). Coupling is the unique contribution of E: the fall
in held-out one-step error when E(t) is added to a fit already using the last p states, so an echo of an earlier
state reads zero whatever its provenance. Dynamic residual is that fit's held-out error.
"""
from __future__ import annotations

import numpy as np

from substrate.geometry import rms

RIDGE = 1.0
COUPLING_WINDOW, COUPLING_AXES, LAGS = 160, 3, 12


def basis(S: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    """The code for a window: its top k principal axes about c."""
    return np.linalg.svd(S - c, full_matrices=False)[2][:k]


def ridge(X: np.ndarray, Y: np.ndarray, alpha: float = RIDGE) -> np.ndarray:
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)


def coupling(S: np.ndarray, X: np.ndarray, c: np.ndarray, k: int = COUPLING_AXES, p: int = LAGS,
             train: float = 0.7) -> dict:
    """Coupling and dynamic residual on one window of states S with externals X."""
    U = basis(S, c, k)
    z, e = (S - c) @ U.T, (X - c) @ U.T
    rows = range(p - 1, len(S) - 1)
    past = np.array([np.concatenate([z[t - j] for j in range(p)] + [[1.0]]) for t in rows])
    full = np.hstack([past, e[p - 1:len(S) - 1]])
    y = z[p:]
    n = int(train * len(y))
    err = {}
    for name, M in (("past", past), ("full", full)):
        beta = ridge(M[:n], y[:n])
        err[name] = rms(y[n:] - M[n:] @ beta)
    return {"coupling": err["past"] - err["full"], "residual": err["full"]}
