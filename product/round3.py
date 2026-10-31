"""Toy bed round 3: the step, coupling at zero against the post-intake reference, and the probe (PLAN.md, product
step 3, round 3; the v10 tab's draft section "The step, and coupling at zero").

    python -m product.round3                         # 200 trials a structure, seed 0
    python -m product.round3 --trials 50 --out runs/round3.json

Its own generator: each step is one self chunk, externals arrive sparsely after an opening intake, and the entity is
driven by the latest external, which stays in context. Rounds 1 and 2 keep theirs, so their records still reproduce.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from product import coupling as cp
from product.toybed import auc, orthonormal, rotation
from substrate.geometry import rms

D_DIM, T, T0, RAMP, INTAKE, ARRIVAL = 32, 500, 200, 60, 40, 0.4
PROBE, PROBE_LEN, PAUSE = 460, 20, 40
REFERENCE, LATE, AFTER_PROBE = (40, 200), (300, 460), (340, 500)
AFTER_RESUME = (240, 260, 280, 300)
WINDOW, AXES, LAGS, FOLDS, GAP, BOOT, RIDGE, LAMBDA = 160, 3, 12, 5, 12, 200, 1.0, 20.0
STRUCTURES = {                                  # name: (kind, echo lag)
    "topic shift": ("topic shift", 0), "dynamics shift": ("dynamics shift", 0),
    "closed loop (dullness)": ("dullness", 0),
    "proliferation, lag 1": ("proliferation", 1), "proliferation, lag 3": ("proliferation", 3),
    "proliferation, lag 10": ("proliferation", 10),
    "mastery": ("mastery", 0), "noise spike": ("noise spike", 0), "progress": ("progress", 0),
    "valid inference": ("valid inference", 0), "healthy, declining": ("declining", 0)}
LOOPS = ("closed loop (dullness)", "proliferation, lag 1", "proliferation, lag 3", "proliferation, lag 10")
ECHOES = LOOPS[1:]
HEALTHY_PROBE = ("progress", "mastery", "declining")


def generate(kind: str, rng, lag: int = 0, D: float = 1.0) -> dict:
    """One run with sparse externals and a probe at step 460. Returns the entity's states s, the externals as the
    observer sees them (E, zero where none arrived), the arrival marker m, the external read in context (E_in), the
    latest external in context at each step (H), and the origin's centroid c."""
    d = D_DIM
    c0 = rng.normal(scale=10.0, size=d)
    t = np.arange(T)
    novelty = np.ones(T)
    if kind == "declining":
        novelty = 1 - 0.85 * np.clip((t - T0) / RAMP, 0, 1)
    novelty[PROBE:PROBE + PROBE_LEN] = 1.0                       # the partner's response is fresh
    walk = np.cumsum(rng.normal(scale=0.3 * D / np.sqrt(d), size=(T, d)) * novelty[:, None], axis=0)
    if kind == "topic shift":
        walk[T0:] += 6 * D * orthonormal(rng, 1, d)[0]
    trend = np.zeros(d)
    if kind == "valid inference":
        trend = 0.3 * D * orthonormal(rng, 1, d)[0]
        walk += t[:, None] * trend
    noise_scale = np.where(t < INTAKE, 2 * D, D * novelty)
    fresh = c0 + walk + rng.normal(size=(T, d)) * (noise_scale / np.sqrt(d))[:, None]
    m = (rng.random(T) < ARRIVAL) | (t < INTAKE)
    if kind == "valid inference":
        m[T0:T0 + PAUSE] = False
    m[PROBE:PROBE + PROBE_LEN] = True
    I = np.eye(d)
    rot = 0.5 * I
    if kind == "dynamics shift":
        Q = orthonormal(rng, 4, d)
        R = np.zeros((4, 4))
        R[:2, :2], R[2:, 2:] = rotation(0.6, 0.9), rotation(0.6, 0.9)
        rot = 0.5 * I + Q.T @ (R - 0.5 * np.eye(4)) @ Q
    u = orthonormal(rng, 1, d)[0]
    s, E, H = np.zeros((T, d)), np.zeros((T, d)), np.zeros((T, d))
    s[0] = held = fresh[0]
    for k in range(T):
        if m[k]:
            if kind == "proliferation" and k >= T0:
                E[k] = s[k - lag] + rng.normal(scale=0.3 * D / np.sqrt(d), size=d)
            else:
                E[k] = fresh[k]
            held = E[k]
        H[k] = held
        if k == T - 1:
            break
        f = min(max((k - T0) / RAMP, 0.0), 1.0)
        A, beta, sigma, drift = 0.5 * I, 1.0, D, np.zeros(d)
        if kind == "dynamics shift" and k >= T0:
            A = rot
        elif kind == "dullness":
            A, beta, drift = (0.5 + 0.48 * f) * I, 1.0 - f, f * 0.3 * D * u
        elif kind == "mastery":
            A, sigma = (0.5 - 0.3 * f) * I, D * (1 - 0.7 * f)
        elif kind == "noise spike" and T0 <= k < T0 + 3:
            sigma = 20 * D
        drive = held - c0
        if kind == "valid inference" and T0 <= k < T0 + PAUSE:
            drive = walk[T0 - 1] + trend * (k - T0 + 1)           # the entity's own on-trend extrapolation
        s[k + 1] = c0 + A @ (s[k] - c0) + beta * (I - A) @ drive + drift + rng.normal(scale=sigma / np.sqrt(d), size=d)
    E_in = np.where(m[:, None], 0.5 * E + 0.5 * s, 0.0)
    return {"s": s, "E": E, "m": m, "E_in": E_in, "H": H, "c": fresh[:10].mean(0), "t0": T0, "kind": kind}


def make(name: str, rng, D: float = 1.0) -> dict:
    kind, lag = STRUCTURES[name]
    return generate(kind, rng, lag, D)


def _fit(X, Y, prior=None, lam=RIDGE):
    target = X.T @ Y + (lam * prior if prior is not None else 0.0)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), target)


def coupling(run: dict, start: int, external: str = "E", a_prior: float | None = None, rng=None) -> dict:
    """Coupling on the window [start, start + 160): held-out errors by contiguous 5-fold with a 12-step gap, the
    arrival marker in both fits, block-bootstrap 90% bounds. With `a_prior`, the past fit is shrunk toward a scalar
    persistence instead of toward zero."""
    sl = slice(start, start + WINDOW)
    S, X, M, c = run["s"][sl], run[external][sl], run["m"][sl].astype(float), run["c"]
    U = cp.basis(S, c, AXES)
    z, e = (S - c) @ U.T, ((X - c) @ U.T) * M[:, None]
    rows = np.arange(LAGS - 1, WINDOW - 1)
    past = np.array([np.concatenate([z[t - j] for j in range(LAGS)] + [[1.0, M[t]]]) for t in rows])
    full = np.hstack([past, e[rows]])
    y = z[rows + 1]
    n = len(y)
    fold = np.arange(n) * FOLDS // n
    sq = {"past": np.zeros(n), "full": np.zeros(n)}
    for f in range(FOLDS):
        test = fold == f
        lo, hi = np.flatnonzero(test)[[0, -1]]
        train = ~test & ((np.arange(n) < lo - GAP) | (np.arange(n) > hi + GAP))
        for name, Xd in (("past", past), ("full", full)):
            prior, lam = None, RIDGE
            if a_prior is not None:
                prior = np.zeros((Xd.shape[1], AXES))
                prior[:AXES] = a_prior * np.eye(AXES)
                lam = LAMBDA
            b = _fit(Xd[train], y[train], prior, lam)
            sq[name][test] = np.sum((y[test] - Xd[test] @ b) ** 2, axis=1)
    value = float(np.sqrt(sq["past"].mean()) - np.sqrt(sq["full"].mean()))
    rng = rng if rng is not None else np.random.default_rng(start)
    blocks = [np.arange(i, min(i + GAP, n)) for i in range(0, n, GAP)]
    boot = []
    for _ in range(BOOT):
        idx = np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))])
        boot.append(np.sqrt(sq["past"][idx].mean()) - np.sqrt(sq["full"][idx].mean()))
    lo90, hi90 = np.percentile(boot, [10, 90])
    return {"coupling": value, "low": float(lo90), "high": float(hi90), "residual": float(np.sqrt(sq["full"].mean()))}


def reading(w: dict, delta: float | None) -> str:
    if delta is None:
        return "no reference"
    if w["low"] > delta:
        return "open"
    if w["high"] < delta:
        return "at zero"
    return "undecided"


def reference(run: dict, **kw) -> float | None:
    ref = coupling(run, REFERENCE[0], **kw)["coupling"]
    return 0.5 * ref if ref > 0 else None


def _held(run):
    """The point contrast and descriptor read the external in context at each step: the latest one to arrive."""
    return {"s": run["s"], "E": run["H"], "c": run["c"], "t0": run["t0"]}


def contrast(run: dict, floor: float | None = None) -> dict:
    """Round 2's point contrast and admission at the new onset (before 140-200, after 200-230)."""
    return cp.contrast(_held(run), floor)


def descriptor(run: dict) -> float:
    return cp.descriptor(_held(run), 260, 320)


def calibrate(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 10_000)
    runs = [make("progress", rng) for _ in range(n)]
    con = [contrast(r) for r in runs]
    prior_rng = np.random.default_rng(seed + 30_000)
    diag = []
    for _ in range(50):
        r = make("progress", prior_rng)
        S, c = r["s"][LATE[0]:LATE[0] + WINDOW], r["c"]
        U = cp.basis(S, c, AXES)
        z = (S - c) @ U.T
        b = _fit(np.hstack([z[:-1], np.ones((len(z) - 1, 1))]), z[1:])
        diag.append(np.mean(np.diag(b[:AXES])))
    return {"floor": float(np.median([x["empty residual"] for x in con])),
            "point contrast": float(np.percentile([x["statistic"] for x in con], 99)),
            "descriptor": float(np.percentile([descriptor(r) for r in runs], 99)),
            "a_prior": float(np.median(diag))}


def read(run: dict, th: dict, name: str) -> dict:
    delta = reference(run)
    late = coupling(run, LATE[0])
    before = reading(late, delta)
    after = before if before == "open" else reading(coupling(run, AFTER_PROBE[0]), delta)
    con = contrast(run, th["floor"])
    point, desc = con["statistic"] > th["point contrast"], descriptor(run) > th["descriptor"]
    if point:
        label = "topic shift" if con["admitted"] else "noise spike"
    elif desc:
        label = "no reference" if delta is None else ("dynamics shift" if before != "at zero" else "closed loop")
    else:
        label = "progress"
    out = {"label": label, "delta": delta, "late": late["coupling"], "before": before, "after": after}
    if name == "progress" or name in ECHOES:
        sd = reference(run, a_prior=th["a_prior"])
        shrunk = coupling(run, LATE[0], a_prior=th["a_prior"])
        out |= {"late shrunk": shrunk["coupling"], "before shrunk": reading(shrunk, sd),
                "late in context": coupling(run, LATE[0], external="E_in")["coupling"]}
    if name == "valid inference":
        out["after resume"] = [reading(coupling(run, a), delta) for a in AFTER_RESUME]
    return out


def evaluate(trials: int = 200, calibration: int = 500, seed: int = 0) -> dict:
    th = calibrate(calibration, seed)
    rng = np.random.default_rng(seed)
    res = {name: [read(make(name, rng), th, name) for _ in range(trials)] for name in STRUCTURES}

    def share(name, key, value):
        return float(np.mean([r[key] == value for r in res[name]]))

    def values(name, key):
        return [r[key] for r in res[name]]

    separation = {n: share(n, "label", n) for n in ("topic shift", "dynamics shift", "noise spike", "progress")}
    loops_as_dynamics = {n: share(n, "label", "dynamics shift") for n in LOOPS}
    mastery = {n: auc(values("mastery", "late"), values(n, "late")) for n in LOOPS}
    progress = {n: auc(values("progress", "late"), values(n, "late")) for n in ECHOES}
    vi = [x for r in res["valid inference"] for x in r["after resume"]]
    valid = float(np.mean([x not in ("at zero", "no reference") for x in vi]))
    zero_before = {n: float(np.mean([r["before"] in ("at zero", "no reference") for r in res[n]]))
                   for n in ("progress", "mastery", "topic shift", "dynamics shift")}
    loops_after = {n: share(n, "after", "at zero") for n in LOOPS}
    declining_after = float(np.mean([r["after"] not in ("at zero", "no reference") for r in res["healthy, declining"]]))
    sd = {"unshrunk": float(np.std(values("progress", "late"))), "shrunk": float(np.std(values("progress", "late shrunk")))}
    false_open = {n: {"unshrunk": share(n, "before", "open"), "shrunk": share(n, "before shrunk", "open")} for n in ECHOES}
    framing = {n: {"alone": auc(values("progress", "late"), values(n, "late")),
                   "in context": auc(values("progress", "late in context"), values(n, "late in context"))} for n in ECHOES}
    passed = {
        "1 separation": all(v >= 0.9 for v in separation.values()) and all(v <= 0.1 for v in loops_as_dynamics.values()),
        "2 mastery over loops": all(v >= 0.8 for v in mastery.values()),
        "3 progress over echoes": all(v >= 0.8 for v in progress.values()),
        "4 valid inference": valid >= 0.9,
        "5 healthy not at zero": all(v <= 0.1 for v in zero_before.values()),
        "6 after the probe": all(v >= 0.8 for v in loops_after.values()) and declining_after >= 0.9}
    decided = {
        "7 shrinkage kept": sd["shrunk"] < sd["unshrunk"]
        and all(v["shrunk"] - v["unshrunk"] <= 0.02 for v in false_open.values()),
        "8 reading alone shown to help": all(v["alone"] - v["in context"] >= 0.05 for v in framing.values())}
    readings = {n: {"before": {k: share(n, "before", k) for k in ("open", "undecided", "at zero", "no reference")},
                    "after": {k: share(n, "after", k) for k in ("open", "undecided", "at zero", "no reference")},
                    "coupling, median": float(np.median(values(n, "late")))} for n in STRUCTURES}
    return {"trials": trials, "calibration": calibration, "seed": seed, "thresholds": th,
            "separation": separation, "loops called dynamics shift": loops_as_dynamics,
            "coupling AUC, mastery over loop": mastery, "coupling AUC, progress over echo": progress,
            "valid inference not at zero, after resume": valid, "at zero before probe": zero_before,
            "loops at zero after probe": loops_after, "healthy declining not at zero after probe": declining_after,
            "shrinkage: progress spread": sd, "shrinkage: echo false-open": false_open,
            "framing: AUC progress over echo": framing, "passed": passed, "decided": decided, "readings": readings}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--calibration", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the full report as JSON")
    a = p.parse_args(argv)
    rep = evaluate(a.trials, a.calibration, a.seed)

    def show(v):
        if isinstance(v, dict):
            return {k: show(x) for k, x in v.items()}
        return round(v, 3) if isinstance(v, float) else v
    for key, v in rep.items():
        if key not in ("trials", "calibration", "seed"):
            print(f"{key}: {show(v)}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
