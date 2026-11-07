"""Rate, conditional (a repair drafted by the agent, 2026-11-07, for the owner to fix; PLAN.md, "Rate's repair").

Rate's world part as first built (`body/progress.py`, X) codes the agent's next chunk under one Gaussian fitted to every
earlier chunk, with and without the window's world chunks in the fit. That is a marginal measure: nothing in it
predicts the action from the reply before it. The calibration's two arms play the same actions (the blind arm replays
the contingent arm's moves), so a marginal measure can only differ between them through which replies enter the fit;
it failed the calibration in the 8B's states and in the text space alike (PLAN.md, "Rate's calibration: the result").

Here each action is coded *given* what came before it, prequentially:

- **Space.** At each scored action t, every chunk before t is projected on the top R principal directions of the
  chunks before t (centred on their mean): a basis fitted on the past only. The action's own state y_t, the agent's
  previous action s_t and the world's chunks since that action w_t (their mean: the reply, and a planted event if any)
  are taken in that basis.
- **Two predictors**, ridge regressions fitted on the earlier scored pairs of the same episode: self-only, y from
  [1, s]; with the world, y from [1, s, w]. Ridge LAMBDA, on predictors scaled by the past's spread.
- **Code length** of y_t under each: the residual coded as an isotropic Gaussian whose variance is the mean squared
  *prequential* error of that predictor on the earlier turns (each made before its own turn was seen), over all R
  components, with the past's mean variance of y as one prior observation, floored at the floor variance. One
  variance, not one per component: the basis is refitted every turn, so its components do not line up across turns.
- **X_c(t)** = bits(self-only) − bits(with the world): the bits the world's last reply saves in predicting the action.
  **S_c(t)** = bits(mean only) − bits(self-only), reported: what the agent's own last action saves.

Domain-free: chunk states and their sources only; nothing reads a direction word, a reply's text or the arm.
Plain numpy.
"""
from __future__ import annotations

import math

import numpy as np

# Fixed by the owner, 2026-11-07, before 7100-7139 are played (PLAN.md, "Rate, conditional", "The defaults"): an
# override of probes/x_c_defaults.py's rule (its pick, 4, 0.1, 10, misses a spread-out dependence), chosen on
# synthetic episodes and the development seeds 907000-907039 only. First drafted as 16, 1 and 20.
R = 8             # principal directions of the past
LAMBDA = 10.0     # ridge, on scaled predictors
MIN_PAIRS = 20    # earlier scored pairs before an action is scored
LN2 = math.log(2)


def _bits(e: np.ndarray, var: np.ndarray) -> float:
    return float(0.5 * np.sum(np.log2(2 * math.pi * var) + e ** 2 / (var * LN2)))


def _ridge(Z: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Weights mapping [1, Z] to Y; the intercept unpenalised, Z scaled by its spread."""
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    Zs = (Z - mu) / sd
    A = np.hstack([np.ones((len(Z), 1)), Zs])
    P = lam * np.eye(A.shape[1])
    P[0, 0] = 0.0
    W = np.linalg.solve(A.T @ A + P, A.T @ Y)
    return W, mu, sd


def _predict(fit, z: np.ndarray) -> np.ndarray:
    W, mu, sd = fit
    return np.concatenate([[1.0], (z - mu) / sd]) @ W


def pairs(slots: list[str], turns: list[int]) -> list[tuple[int, int, list[int]]]:
    """(t, s, ws) for every self chunk t that has an earlier self chunk s: ws are the world chunks between them."""
    out, last_self = [], None
    for i, slot in enumerate(slots):
        if slot == "self":
            if last_self is not None:
                out.append((i, last_self, [j for j in range(last_self + 1, i) if slots[j] != "self"]))
            last_self = i
    return out


def conditional(X: np.ndarray, slots: list[str], turns: list[int], floor_var: float, burn_turns: int = 10,
                r: int = R, lam: float = LAMBDA, min_pairs: int = MIN_PAIRS) -> dict:
    """X_c and S_c per chunk (NaN where not scored): a self chunk is scored from turn `burn_turns` on, once
    `min_pairs` earlier pairs with a world chunk are there to fit on."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    Xc, Sc = np.full(n, np.nan), np.full(n, np.nan)
    P = [p for p in pairs(slots, turns) if p[2]]
    err = {"mean": [], "self": [], "world": []}        # prequential mean squared errors, one per turn
    for k, (t, s, ws) in enumerate(P):
        past = [p for p in P[:k]]
        if len(past) < min_pairs:
            continue
        before = X[:t]
        mu = before.mean(0)
        _, _, Vt = np.linalg.svd(before - mu, full_matrices=False)
        B = Vt[: min(r, Vt.shape[0])].T                 # d x r
        proj = lambda rows: (np.atleast_2d(rows) - mu) @ B
        Y = proj(X[[p[0] for p in past]])
        S = proj(X[[p[1] for p in past]])
        W = np.vstack([proj(X[p[2]]).mean(0) for p in past])
        y, s_t, w_t = proj(X[t])[0], proj(X[s])[0], proj(X[ws]).mean(0)
        prior = max(float(Y.var(0).mean()), floor_var)
        preds = {"mean": Y.mean(0),
                 "self": _predict(_ridge(S, Y, lam), s_t),
                 "world": _predict(_ridge(np.hstack([S, W]), Y, lam), np.concatenate([s_t, w_t]))}
        bits = {}
        for name, yhat in preds.items():
            e = y - yhat
            past = err[name]
            var = max((sum(past) + prior) / (len(past) + 1), floor_var)
            bits[name] = _bits(e, np.full(len(e), var))
            err[name].append(float(np.mean(e ** 2)))
        if turns[t] >= burn_turns:
            Xc[t] = bits["self"] - bits["world"]
            Sc[t] = bits["mean"] - bits["self"]
    return {"X_c": Xc, "S_c": Sc}
