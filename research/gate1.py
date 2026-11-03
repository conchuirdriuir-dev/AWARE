"""Gate 1: what elicited negation does to a state, and whether its centre is the origin (v9, "Tests, in order"; PLAN.md).
Research track.

Three forms, each fitted on pairs (x, y) of a statement and its elicited opposite. With the centre fitted (`fit_free`):
- mirror:      y = x - 2((x.u) - b) u, u the top direction of x - y, b the mean of (x + y)/2 along u;
- additive:    y = x + v;
- contraction: y = lam x + v, its centre v / (1 - lam).
With the centre fixed at c (`fit`), reported beside: mirror through c; additive; y = c + lam (x - c). A form must predict
held-out pairs better than both others and identity by more than D. Mirror structure passing would reinstate
self-negation as a mechanism; the gate cannot end the project. It runs standalone on elicited pairs, lifted out of v9's
build order (owner, DECISIONS.md, 2026-11-02).
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


# The centre fitted, and the scopes the verdict is read in (PLAN.md).

IDENTITY = "identity"
NO_CENTRE = 0.1         # |1 - lam| below this: no finite centre
MIN_GROUP = 3           # a topic with fewer pairs is left out of the per-topic scope


def fit_free(X: np.ndarray, Y: np.ndarray) -> dict:
    u = top_direction(X - Y)
    xm, ym = X.mean(0), Y.mean(0)
    lam = float(np.sum((X - xm) * (Y - ym)) / np.sum((X - xm) ** 2))
    return {"mirror": (u, float(((X + Y) / 2 @ u).mean())),
            "additive": ym - xm,
            "contraction": (lam, ym - lam * xm)}


def predict_free(name: str, param, X: np.ndarray) -> np.ndarray:
    if name == "mirror":
        u, b = param
        return X - 2.0 * (X @ u - b)[:, None] * u
    if name == "additive":
        return X + param
    lam, v = param
    return lam * X + v


def fixed_centre(c: np.ndarray) -> tuple:
    return (lambda X, Y: fit(X, Y, c)), (lambda name, param, X: predict(name, param, X, c))


FREE = (fit_free, predict_free)


def topic_folds(groups: np.ndarray, k: int = 5, seed: int = 0) -> list[np.ndarray]:
    return np.array_split(np.random.default_rng(seed).permutation(np.unique(groups)), k)


def _errors(Y: np.ndarray, X: np.ndarray, out: dict, lams: list) -> dict:
    errors = {h: rmse(Y, out[h]) for h in HYPOTHESES}
    errors[IDENTITY] = rmse(Y, X)
    errors["lam"] = float(np.median(lams))      # the contraction's lambda, the median over the fits
    return errors


def _lam(param) -> float:
    return param[0] if isinstance(param, tuple) else float(param)


def held_out_by_topic(X, Y, groups, model=FREE, k: int = 5) -> dict:
    """The global scope: one fit across topics, its error on the topics held out."""
    fitter, predictor = model
    out, order, lams = {h: [] for h in HYPOTHESES}, [], []
    for fold in topic_folds(groups, k):
        test = np.isin(groups, fold)
        params = fitter(X[~test], Y[~test])
        lams.append(_lam(params["contraction"]))
        for h in HYPOTHESES:
            out[h].append(predictor(h, params[h], X[test]))
        order.append(np.flatnonzero(test))
    order = np.concatenate(order)
    return _errors(Y[order], X[order], {h: np.concatenate(out[h]) for h in HYPOTHESES}, lams)


def leave_one_out_per_topic(X, Y, groups, model=FREE) -> dict:
    """The per-topic scope: one fit per topic, each pair left out in turn."""
    fitter, predictor = model
    out, order, lams = {h: [] for h in HYPOTHESES}, [], []
    for g in np.unique(groups):
        where = np.flatnonzero(groups == g)
        if len(where) < MIN_GROUP:
            continue
        for i in where:
            rest = where[where != i]
            params = fitter(X[rest], Y[rest])
            lams.append(_lam(params["contraction"]))
            for h in HYPOTHESES:
                out[h].append(predictor(h, params[h], X[[i]]))
            order.append(i)
    return _errors(Y[order], X[order], {h: np.concatenate(out[h]) for h in HYPOTHESES}, lams)


def decide(errors: dict, D: float, nesting: bool = True) -> str:
    """The form that beats its rivals and identity by more than D, or undecided. A free contraction with lambda within
    NO_CENTRE of 1 is the additive form, so it is no rival and cannot win (owner, 2026-11-02); `nesting=False` is the
    rule as first written, under which additive can never win."""
    forms = [h for h in HYPOTHESES if not (nesting and h == "contraction" and abs(1 - errors["lam"]) < NO_CENTRE)]
    best = min(forms, key=errors.get)
    return best if all(errors[h] - errors[best] > D for h in (*forms, IDENTITY) if h != best) else "undecided"


def scope_verdict(by_layer: list[str], need: int = 3) -> str:
    """The form winning at `need` or more of the layers, or undecided."""
    wins = [v for v in by_layer if v != "undecided"]
    best = max(set(wins), key=wins.count, default="undecided")
    return best if wins.count(best) >= need else "undecided"


def shuffle_opposites(groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """A permutation giving each statement another topic's opposite."""
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(groups))
    while (clash := np.flatnonzero(groups == groups[p])).size:
        for i in clash:
            j = rng.integers(len(groups))
            p[[i, j]] = p[[j, i]]
    return p


def paraphrase_pairs(S: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each set's pairs, each once: S is sets x texts x width; 20 sets of 5 give 20 groups of 10, the shape of the pairs."""
    n = S.shape[1]
    i, j = np.triu_indices(n, 1)
    X, Y = S[:, i].reshape(-1, S.shape[-1]), S[:, j].reshape(-1, S.shape[-1])
    return X, Y, np.repeat(np.arange(len(S)), len(i))


def contraction_centre(lam: float, v: np.ndarray) -> np.ndarray | None:
    return None if abs(1 - lam) < NO_CENTRE else v / (1 - lam)


def distance_from_centre(name: str, param, c: np.ndarray) -> float | None:
    """How far c sits from a form's centre: along u only for a mirror; None if the form has no finite centre."""
    if name == "mirror":
        u, b = param
        return float(abs(c @ u - b))
    centre = contraction_centre(*param)
    return None if centre is None else float(np.linalg.norm(c - centre))
