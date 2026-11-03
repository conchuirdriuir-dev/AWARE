"""Progress by source: the body readout's learning signal (PLAN.md, "Body readout", Step 0).

One formula over the whole record, T2's (`goalspace/formula.py`): PPCA, noise floored at D^2/d, k chosen by the two-part
code within the span. For a scored self chunk t, the window W is every chunk since the 20th self chunk before t. The
chunk is coded four times: by the formula fitted on everything before t, and by the same fit with W's chunks of one
source left out. The differences are the bits each source's recent chunks save.

`GramFit` is the same fit computed from the session's Gram matrix (inner products of its rows), so its cost depends on
the number of rows fitted, not on the width of the space (4,096 on the 8B; thousands of words in the text baseline),
and leaving rows out is indexing. It agrees with `goalspace.formula.Fit` (tests/test_body.py). Plain numpy.
"""
from __future__ import annotations

import math

import numpy as np

from goalspace.formula import KMAX, LN2, Fit

SOURCES = ("partner", "self", "tool")
WINDOW_SELF = 20


def _top_eigh(A: np.ndarray, top: int):
    """The `top` largest eigenpairs of a symmetric matrix, largest first (scipy's subset solver when present)."""
    n = len(A)
    top = min(top, n)
    try:
        from scipy.linalg import eigh
        w, v = eigh(A, subset_by_index=[n - top, n - 1])
    except ImportError:
        w, v = np.linalg.eigh(A)
        w, v = w[n - top:], v[:, n - top:]
    return w[::-1], v[:, ::-1]


class GramFit(Fit):
    """`Fit` on the rows `idx` of a session, from its Gram matrix G. The sample covariance's nonzero eigenvalues are
    those of the doubly-centred Gram over n; projections of a new row come from its inner products with the fitted
    rows. Only the top KMAX + 1 eigenpairs and the trace are kept: all `Fit` reads, since k never exceeds KMAX."""

    def __init__(self, G: np.ndarray, idx: np.ndarray, d: int, floor_var: float, kmax: int = KMAX):
        self.G, self.idx = G, np.asarray(idx)
        self.n, self.d = len(self.idx), d
        Gs = G[np.ix_(self.idx, self.idx)]
        self.row_mu = Gs.mean(1)                              # X_j . mu for each fitted row j
        self.mu_mu = float(self.row_mu.mean())                # mu . mu
        Gc = Gs - self.row_mu[:, None] - self.row_mu[None, :] + self.mu_mu
        Gc = (Gc + Gc.T) / 2
        w, u = _top_eigh(Gc, kmax + 1)
        w = np.maximum(w, 0.0)
        self.lam = w / self.n
        self.U, self.s = u, np.sqrt(w)
        self.total = float(max(np.trace(Gc), 0.0)) / self.n
        self.floor = max(float(floor_var), 1e-12)
        self.mu = None                                         # not held: the width is never materialised

    def bits(self, t: int, k: int) -> float:
        """Code length of session row t (not among the fitted rows) under the k-axis model."""
        k, lam_k, sigma2 = self.params(k)
        g = self.G[self.idx, t]                               # X_j . x
        mu_x = float(g.mean())
        a = g - mu_x - self.row_mu + self.mu_mu               # (X_j - mu) . (x - mu)
        safe = np.where(self.s[:k] > 0, self.s[:k], 1.0)
        p = np.where(self.s[:k] > 0, (self.U[:, :k].T @ a) / safe, 0.0)
        rr = float(self.G[t, t]) - 2 * mu_x + self.mu_mu
        maha = float((p ** 2 / lam_k).sum()) + (rr - float(p @ p)) / sigma2
        logdet = float(np.log(lam_k).sum()) + (self.d - k) * math.log(sigma2)
        return 0.5 * (self.d * math.log(2 * math.pi) + logdet + maha) / LN2


def code(G: np.ndarray, idx: np.ndarray, t: int, d: int, floor_var: float, kmax: int = KMAX) -> tuple[float, int]:
    """Bits of row t under the formula fitted on rows idx, and the k chosen."""
    fit = GramFit(G, idx, d, floor_var, kmax)
    k = fit.choose_k(kmax)
    return fit.bits(t, k), k


def by_source(X: np.ndarray, sources: list[str], floor_var: float, burn_self: int = 40,
              window_self: int = WINDOW_SELF, kmax: int = KMAX, every: int = 1) -> dict:
    """Per chunk of one session (rows of X, in time order, each with its source): for each scored self chunk, the four
    code lengths. A self chunk is scored from the `burn_self`-th on (counting from 0), every `every`-th. Unscored rows
    are NaN.

    Returns L_now, L_nopartner, L_notool, L_noself, k, and the cross-, tool- and self-progress they give."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    G = X @ X.T
    src = np.asarray(sources)
    selves = np.flatnonzero(src == "self")
    keys = ("L_now", "L_nopartner", "L_notool", "L_noself")
    out = {k: np.full(n, np.nan) for k in keys}
    out["k"] = np.full(n, -1)
    for j, t in enumerate(selves):
        if j < burn_self or (j - burn_self) % every:
            continue
        before = np.arange(t)
        w0 = selves[j - window_self] + 1 if j >= window_self else 0   # chunks since the 20th self chunk before t
        out["L_now"][t], out["k"][t] = code(G, before, t, d, floor_var, kmax)
        for key, s in (("L_nopartner", "partner"), ("L_notool", "tool"), ("L_noself", "self")):
            drop = (before >= w0) & (src[:t] == s)
            out[key][t] = out["L_now"][t] if not drop.any() else code(G, before[~drop], t, d, floor_var, kmax)[0]
    out["X"] = out["L_nopartner"] - out["L_now"]
    out["T"] = out["L_notool"] - out["L_now"]
    out["S"] = out["L_noself"] - out["L_now"]
    return out


def cost(L_now: np.ndarray) -> np.ndarray:
    """C_t: a scored chunk's code length minus the median over the session's earlier scored chunks (NaN for the
    first)."""
    c = np.full(len(L_now), np.nan)
    seen = []
    for t in np.flatnonzero(np.isfinite(L_now)):
        if seen:
            c[t] = L_now[t] - float(np.median(seen))
        seen.append(L_now[t])
    return c
