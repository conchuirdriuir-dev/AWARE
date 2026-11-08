"""The body vector for babbling (PLAN.md, "Babbling (v0.2 Step 2): the gate"): the two senses that passed their
instrument tests, computed chunk by chunk as an episode arrives, exactly as they were tested.

- **Boundary (v0)**, Step -1's news (`body.stepm1.news`): for a reply, its nearest distance to every earlier chunk of
  the episode, per layer, in echo-floor units, then the mean over the read layers. Read on the full text.
- **Rate's world part, X_c** (`body.conditional.conditional`): for an action, the bits the world's last reply saves
  in predicting it, per layer, then the mean over the read layers (as rate's calibration scores it). Read on the
  text cut at 256 tokens (`body.step0_habitat.CAP`), as the calibration read it.

The body vector after turn t is b_t = (boundary of reply t, X_c of action t, 1 if X_c is defined else 0); X_c is
undefined until MIN_PAIRS earlier pairs are there to fit on, and before turn 10. Before turn 0 the vector is zeros.
`Episode` computes X_c incrementally, one new action at a time; tests check it equals `conditional` on the whole
episode, so the live senses are the tested ones. States are read with any graft off. Plain numpy.
"""
from __future__ import annotations

import numpy as np

from body.conditional import LAMBDA, MIN_PAIRS, R, _bits, _predict, _ridge

BURN_TURNS = 10                     # as `conditional`: an action is scored from turn 10 on


class _XcLayer:
    """`conditional`'s loop for one layer, run one pair at a time as chunks arrive."""

    def __init__(self, floor_var: float, r: int = R, lam: float = LAMBDA, min_pairs: int = MIN_PAIRS):
        self.fv, self.r, self.lam, self.min_pairs = floor_var, r, lam, min_pairs
        self.X: list[np.ndarray] = []
        self.P: list[tuple[int, int, list[int]]] = []
        self.err = {"mean": [], "self": [], "world": []}

    def add(self, x: np.ndarray, slot: str, turn: int, slots: list[str]) -> float:
        """Add one chunk's state (slots: every chunk's slot so far, this one included); X_c if this chunk is a
        scored action, else NaN."""
        self.X.append(np.asarray(x, dtype=np.float64))
        t = len(self.X) - 1
        if slot != "self":
            return np.nan
        prev = [i for i in range(t) if slots[i] == "self"]
        if not prev:
            return np.nan
        s = prev[-1]
        ws = [j for j in range(s + 1, t) if slots[j] != "self"]
        if not ws:
            return np.nan
        past = list(self.P)
        self.P.append((t, s, ws))
        if len(past) < self.min_pairs:
            return np.nan
        X = np.vstack(self.X)
        before = X[:t]
        mu = before.mean(0)
        _, _, Vt = np.linalg.svd(before - mu, full_matrices=False)
        B = Vt[: min(self.r, Vt.shape[0])].T
        proj = lambda rows: (np.atleast_2d(rows) - mu) @ B
        Y = proj(X[[p[0] for p in past]])
        S = proj(X[[p[1] for p in past]])
        W = np.vstack([proj(X[p[2]]).mean(0) for p in past])
        y, s_t, w_t = proj(X[t])[0], proj(X[s])[0], proj(X[ws]).mean(0)
        prior = max(float(Y.var(0).mean()), self.fv)
        preds = {"mean": Y.mean(0),
                 "self": _predict(_ridge(S, Y, self.lam), s_t),
                 "world": _predict(_ridge(np.hstack([S, W]), Y, self.lam), np.concatenate([s_t, w_t]))}
        bits = {}
        for name, yhat in preds.items():
            e = y - yhat
            hist = self.err[name]
            var = max((sum(hist) + prior) / (len(hist) + 1), self.fv)
            bits[name] = _bits(e, np.full(len(e), var))
            self.err[name].append(float(np.mean(e ** 2)))
        return bits["self"] - bits["world"] if turn >= BURN_TURNS else np.nan


class Episode:
    """One episode's senses, chunk by chunk. `unit`: the echo floor per layer in raw distance (floor x D);
    `floor_var`: X_c's floor variance per layer (D^2 / width)."""

    def __init__(self, unit, floor_var):
        self.unit = np.asarray(unit, dtype=np.float64)
        self.layers = [_XcLayer(float(v)) for v in floor_var]
        self.full: list[np.ndarray] = []
        self.slots: list[str] = []

    def add(self, full: np.ndarray, capped: np.ndarray, slot: str, turn: int) -> dict:
        """One chunk: its state per layer on the full text and on the capped text (layers x width each). Returns
        {"boundary": float or NaN, "X_c": float or NaN}."""
        full = np.asarray(full, dtype=np.float64)
        i = len(self.full)
        boundary = np.nan
        if slot == "world" and i > 0:
            S = np.stack(self.full)
            boundary = float(np.mean(np.min(np.linalg.norm(S - full[None], axis=2), axis=0) / self.unit))
        self.full.append(full)
        self.slots.append(slot)
        xs = [layer.add(capped[j], slot, turn, self.slots) for j, layer in enumerate(self.layers)]
        x_c = float(np.mean(xs)) if slot == "self" and np.isfinite(xs).all() else np.nan
        return {"boundary": boundary, "X_c": x_c}


def body_vectors(chunks: list[dict], senses: list[dict]) -> dict:
    """Turn -> (boundary of reply t, X_c of action t) from each chunk's senses, NaN where undefined. The opening
    (turn -1) and any spontaneous chunk carry none."""
    out: dict = {}
    for c, s in zip(chunks, senses):
        t = c["turn"]
        if t < 0:
            continue
        b = out.setdefault(t, [np.nan, np.nan])
        if c["slot"] == "self":
            b[1] = s["X_c"]
        elif c.get("cause", "elicited") == "elicited":
            b[0] = s["boundary"]
    return {t: tuple(v) for t, v in out.items()}


def zscore(vectors: dict, z: dict, clip: float | None = None) -> np.ndarray:
    """turns x 3: (boundary z, X_c z or 0, 1 if X_c is defined); `z` holds each sense's mean and sd. With `clip`,
    the two z-scores are cut to [-clip, clip]: the channel's choice, never the sense's (a looping episode can give
    X_c of 1e14 bits, the instrument unchanged)."""
    rows = []
    for t in sorted(vectors):
        b, x = vectors[t]
        bz = (b - z["boundary"][0]) / z["boundary"][1] if np.isfinite(b) else 0.0
        ok = bool(np.isfinite(x))
        rows.append([bz, (x - z["X_c"][0]) / z["X_c"][1] if ok else 0.0, 1.0 if ok else 0.0])
    out = np.array(rows, dtype=np.float64).reshape(-1, 3)
    if clip is not None:
        out[:, :2] = np.clip(out[:, :2], -clip, clip)
    return out


def fit_z(all_vectors: list[dict]) -> dict:
    b = np.array([v[0] for vs in all_vectors for v in vs.values() if np.isfinite(v[0])])
    x = np.array([v[1] for vs in all_vectors for v in vs.values() if np.isfinite(v[1])])
    return {"boundary": [float(b.mean()), float(b.std() + 1e-12)], "X_c": [float(x.mean()), float(x.std() + 1e-12)]}
