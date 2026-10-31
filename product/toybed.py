"""The toy bed (v9, "Toy bed specification"; PLAN.md, product step 3): six planted structures, each changing only A, B
or E in s(t+1) = A s(t) + B E(t) + noise at D, and the detectors asked to separate them.

    python -m product.toybed                       # 200 trials a structure, thresholds from 500 progress runs
    python -m product.toybed --trials 50 --out runs/toybed.json

Synthetic for now; v9 wants it in the real model's activation space so D is real. It tests the detectors, never the
generator: the closed loop planted here is a conjecture that the labeled logs check.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from product import detectors
from product.compass import cell_v9 as cell

KINDS = ("topic shift", "dynamics shift", "closed loop (dullness)", "closed loop (proliferation)", "mastery",
         "noise spike", "progress")
EXTRA = ("valid inference",)            # v10, round 2


def orthonormal(rng, k: int, d: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return q.T


def rotation(angle: float, scale: float) -> np.ndarray:
    return scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def generate(kind: str, rng, d: int = 32, D: float = 1.0, T: int = 240, t0: int = 120, ramp: int = 60,
             n_origin: int = 10, lag: int = 0, pause: int = 40) -> dict:
    """One run: the entity's states s, the external states E (both T x d), the origin's centroid c, the onset t0.
    In proliferation the externals echo the entity's state `lag` steps back (round 1: 0). In valid inference the
    externals carry a trend and pause for `pause` steps from onset while the entity extrapolates on trend."""
    if kind not in KINDS + EXTRA:
        raise ValueError(kind)
    c0 = rng.normal(scale=10.0, size=d)
    walk = np.cumsum(rng.normal(scale=0.3 * D / np.sqrt(d), size=(T, d)), axis=0)
    if kind == "topic shift":
        walk[t0:] += 6 * D * orthonormal(rng, 1, d)[0]
    trend = np.zeros(d)
    if kind == "valid inference":
        trend = 0.3 * D * orthonormal(rng, 1, d)[0]
        walk += np.arange(T)[:, None] * trend
    E = c0 + walk + rng.normal(scale=D / np.sqrt(d), size=(T, d))
    if kind == "valid inference":
        E[t0:t0 + pause] = E[t0 - 1]                    # nothing new arrives: the last external, repeated
    I = np.eye(d)
    rot = I * 0.5
    if kind == "dynamics shift":
        Q = orthonormal(rng, 4, d)
        R = np.zeros((4, 4))
        R[:2, :2], R[2:, 2:] = rotation(0.6, 0.9), rotation(0.6, 0.9)
        rot = 0.5 * I + Q.T @ (R - 0.5 * np.eye(4)) @ Q
    u = orthonormal(rng, 1, d)[0]
    s = np.zeros((T, d))
    s[0] = E[0]
    for t in range(T - 1):
        f = min(max((t - t0) / ramp, 0.0), 1.0)
        A, beta, sigma, drift = 0.5 * I, 1.0, D, np.zeros(d)
        if kind == "dynamics shift" and t >= t0:
            A = rot
        elif kind == "closed loop (dullness)":
            A, beta, drift = (0.5 + 0.48 * f) * I, 1.0 - f, f * 0.3 * D * u
        elif kind == "mastery":
            A, sigma = (0.5 - 0.3 * f) * I, D * (1 - 0.7 * f)
        elif kind == "noise spike" and t0 <= t < t0 + 3:
            sigma = 20 * D
        if kind == "closed loop (proliferation)" and t >= t0:
            E[t] = s[t - lag] + rng.normal(scale=0.3 * D / np.sqrt(d), size=d)
        x, e = s[t] - c0, E[t] - c0
        if kind == "valid inference" and t0 <= t < t0 + pause:
            e = walk[t0 - 1] + trend * (t - t0 + 1)     # the entity's own on-trend extrapolation drives it
        s[t + 1] = c0 + A @ x + beta * (I - A) @ e + drift + rng.normal(scale=sigma / np.sqrt(d), size=d)
    return {"s": s, "E": E, "c": E[:n_origin].mean(0), "t0": t0, "kind": kind}


def auc(pos, neg) -> float:
    """The chance a positive ranks above a negative, ties counting half."""
    pos, neg = np.asarray(pos, dtype=float)[:, None], np.asarray(neg, dtype=float)[None, :]
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def evaluate(trials: int = 200, calibration: int = 500, seed: int = 0, D: float = 1.0) -> dict:
    cal_rng = np.random.default_rng(seed + 10_000)
    null = [detectors.contrasts(generate("progress", cal_rng, D=D), D) for _ in range(calibration)]
    thresholds = {k: float(np.percentile([n[k] for n in null], 99)) for k in null[0]}
    rng = np.random.default_rng(seed)
    results = {}
    for kind in KINDS:
        runs = []
        for _ in range(trials):
            run = generate(kind, rng, D=D)
            stats = detectors.contrasts(run, D)
            fires = {k: stats[k] > thresholds[k] for k in stats}
            blocks = detectors.observe(run["s"], run["E"], run["c"], D)
            runs.append({"fires": fires, "label": detectors.classify(fires), "blocks": blocks})
        results[kind] = runs

    def late(kind, key, start):
        return [b[key] for r in results[kind] for b in r["blocks"] if b["start"] >= start]

    def growth(kind):
        return [sum(b["growth"] for b in r["blocks"] if b["start"] >= 120) for r in results[kind]]

    separation = {k: float(np.mean([r["label"] == k for r in results[k]]))
                  for k in ("topic shift", "dynamics shift", "noise spike", "progress")}
    margin = {v: auc(late(v, "margin", 180), late("mastery", "margin", 180))
              for v in ("closed loop (dullness)", "closed loop (proliferation)")}
    frame = auc(growth("progress"), growth("closed loop (proliferation)"))
    described = {}
    for kind in KINDS:
        rs = results[kind]
        cells = [cell(b["self"], b["frame"]) for r in rs for b in r["blocks"] if b["start"] >= 180]
        described[kind] = {
            **{f"fires {k}": float(np.mean([r["fires"][k] for r in rs])) for k in thresholds},
            "self-residual, 180-240": float(np.median(late(kind, "self", 180))),
            "frame-residual, 180-240": float(np.median(late(kind, "frame", 180))),
            "margin, 180-240": float(np.median(late(kind, "margin", 180))),
            "frame growth, 120-240": float(np.mean(growth(kind))),
            "distance to origin, 180-240": float(np.median(late(kind, "distance", 180))),
            "commonest cell, 180-240": max(set(cells), key=cells.count),
        }
    passed = {"separation": all(v >= 0.9 for v in separation.values()),
              "margin": all(v >= 0.8 for v in margin.values()), "frame growth": frame >= 0.8}
    return {"trials": trials, "calibration": calibration, "seed": seed, "thresholds": thresholds,
            "separation": separation, "margin AUC vs mastery": margin, "frame growth AUC, progress over 3b": frame,
            "passed": passed, "described": described}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--calibration", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the full report as JSON")
    a = p.parse_args(argv)
    rep = evaluate(a.trials, a.calibration, a.seed)
    print("separation (pass >= 0.90 each):", {k: round(v, 2) for k, v in rep["separation"].items()})
    print("margin AUC vs mastery (pass >= 0.80 each):", {k: round(v, 2) for k, v in rep["margin AUC vs mastery"].items()})
    print(f"frame growth AUC, progress over proliferation (pass >= 0.80): {rep['frame growth AUC, progress over 3b']:.2f}")
    print("passed:", rep["passed"])
    for kind, row in rep["described"].items():
        print(f"  {kind}: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
