"""Gate 1: what elicited negation does to a state, about the origin (v9, "Tests, in order"). Research track.

Three hypotheses, each fitted on pairs (x, y) of a statement and its elicited opposite:
- mirror:      y = c + H(x - c), H a reflection;
- additive:    y = x + v;
- contraction: y = c + lam (x - c), lam a scalar.
Whichever predicts held-out pairs better than both others by more than D wins; otherwise undecided. Mirror structure
passing would reinstate self-negation as a mechanism; the gate cannot end the project. v9 runs it as an analysis on
the accumulated axes, after the main demo's arms exist.
"""
from __future__ import annotations

import numpy as np

from substrate.geometry import reflect, rmse, top_direction

HYPOTHESES = ("mirror", "additive", "contraction")


def fit(X: np.ndarray, Y: np.ndarray, c: np.ndarray) -> dict:
    Xc, Yc = X - c, Y - c
    return {"mirror": top_direction(X - Y),
            "additive": (Y - X).mean(0),
            "contraction": float(np.sum(Xc * Yc) / np.sum(Xc * Xc))}


def predict(name: str, param, X: np.ndarray, c: np.ndarray) -> np.ndarray:
    if name == "mirror":
        return reflect(X, c, param)
    if name == "additive":
        return X + param
    return c + param * (X - c)


def verdict(X: np.ndarray, Y: np.ndarray, c: np.ndarray, D: float, n_fit: int) -> tuple[str, dict]:
    """Fit on the first n_fit pairs; the winner must beat both others on the rest by more than D."""
    params = fit(X[:n_fit], Y[:n_fit], c)
    errors = {h: rmse(Y[n_fit:], predict(h, params[h], X[n_fit:], c)) for h in HYPOTHESES}
    best = min(errors, key=errors.get)
    if all(errors[h] - errors[best] > D for h in HYPOTHESES if h != best):
        return best, errors
    return "undecided", errors
