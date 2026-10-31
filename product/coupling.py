"""v10's detectors (the Drift detection (v10) tab, "Operators"; PLAN.md, product step 3, round 2).

The observer is a joint fit in the code, s(t+1) ~ A s(t) + B E(t). Coupling is the unique contribution of E: the fall
in held-out one-step error when E(t) is added to a fit already using the last p states, so an echo of an earlier
state reads zero whatever its provenance. Dynamic residual is that fit's held-out error. Point contrast is taken on
the residual, its candidate tested by admission; the spectral descriptor reads A with B E removed.
"""
from __future__ import annotations

import numpy as np

from substrate.axes import Code
from substrate.geometry import rms

RIDGE = 1.0
COUPLING_WINDOW, COUPLING_AXES, LAGS = 160, 3, 12
CONTRAST_AXES, BEFORE, AFTER = 4, 60, 30


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


class Joint:
    """A joint fit with p = 1 in a window's own code; it predicts the full state, the part off the code as its mean."""

    def __init__(self, S: np.ndarray, X: np.ndarray, c: np.ndarray, k: int = CONTRAST_AXES):
        self.c, self.U = c, basis(S, c, k)
        z, e = (S - c) @ self.U.T, (X - c) @ self.U.T
        self.beta = ridge(self._design(z[:-1], e[:-1]), z[1:])
        Z = S - c
        self.off_mean = (Z - (Z @ self.U.T) @ self.U).mean(0)
        self.k = k

    @staticmethod
    def _design(z, e):
        return np.hstack([z, e, np.ones((len(z), 1))])

    def residuals(self, S: np.ndarray, X: np.ndarray) -> np.ndarray:
        """s(t+1) minus its prediction from s(t) and E(t), for each step in S after the first."""
        z, e = (S - self.c) @ self.U.T, (X - self.c) @ self.U.T
        pred = self.c + (self._design(z[:-1], e[:-1]) @ self.beta) @ self.U + self.off_mean
        return S[1:] - pred

    def modes(self) -> np.ndarray:
        """Sorted eigenvalue magnitudes of A: the dynamics with B E taken out."""
        return np.sort(np.abs(np.linalg.eigvals(self.beta[:self.k].T)))[::-1]


def contrast(run: dict, floor: float | None = None) -> dict:
    """Point contrast on the residual at onset, and, given the floor, whether admission keeps its candidate."""
    s, E, c, t0 = run["s"], run["E"], run["c"], run["t0"]
    model = Joint(s[t0 - BEFORE:t0], E[t0 - BEFORE:t0], c)
    r_before = model.residuals(s[t0 - BEFORE:t0], E[t0 - BEFORE:t0])
    r_after = model.residuals(s[t0 - 1:t0 + AFTER], E[t0 - 1:t0 + AFTER])
    candidate = r_after.mean(0) - r_before.mean(0)
    half = AFTER // 2
    fit, held = r_after[:half], r_after[half:]

    def residual(axes):
        if not len(axes):
            return rms(held)
        level = fit.mean(0) @ axes.T
        return rms(held - level @ axes)

    out = {"statistic": float(np.linalg.norm(candidate)), "empty residual": residual(np.zeros((0, len(c))))}
    if floor is not None:
        axis = Code(c, 1.0, "contrast", unit=floor).consider(candidate, residual=residual, source="point contrast",
                                                             at=t0)
        out["admitted"] = axis.kind == "predictive"
    return out


def descriptor(run: dict, start: int = 180, stop: int = 240) -> float:
    """How far the dynamics moved: modes before onset against a later stretch."""
    s, E, c, t0 = run["s"], run["E"], run["c"], run["t0"]
    before = Joint(s[t0 - BEFORE:t0], E[t0 - BEFORE:t0], c).modes()
    later = Joint(s[start:stop], E[start:stop], c).modes()
    return float(np.linalg.norm(before - later))
