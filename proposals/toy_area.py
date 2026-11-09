"""Toy worlds for "Meaning as Transport": checks that bag, order, holonomy and flux are all signed area.

    python -m proposals.toy_area            # all four checks, prints a table, exits 1 if any fails
    python -m proposals.toy_area --quick    # smaller sizes (what the unit test runs)

No language and no model: each check is built so its answer is known in closed form. It checks the math of the
doc's "One quantity: signed area" section before any reading on text relies on it. Nothing here is a sense or a
step's test; it decides nothing in PLAN.md.

  1. bag      The mean of a path (minus its start) is the (i, t) term of the depth-2 signature of (t, X).
              Reordering the points keeps the bag and changes the area: order lives in the area.
  2. area     Level 2 of the log-signature is the Levy area; reversing a path flips it, and the symmetric part of
              level 2 is fixed by level 1 (shuffle identity), so it carries no order.
  3. holonomy A fixed connection A_1, A_2 carried round a circle: holonomy = exp(area * [A_1, A_2]) to leading
              order (error per unit area falls with the radius), and the depth-N signature expansion converges to
              the exact holonomy as N grows (transport is a function of the signature).
  4. flux     dX = -(k I + w J) X dt + s dW in 2-D. Mean area rate = -w s^2 / (2k) in closed form: zero for a pure
              landscape (w = 0) and proportional to the flux w otherwise. Reversed paths flip its sign.

Numpy only. Tolerances are set from the closed forms before running and printed beside each result.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


# --------------------------------------------------------------------------------------------------------------
# Signatures of piecewise-linear paths (depth <= 3) via Chen's identity, and the Levy area
# --------------------------------------------------------------------------------------------------------------
def signature(path, depth=3):
    """Levels 1..depth of the signature of a piecewise-linear path, as a list of arrays (d,), (d, d), (d, d, d)."""
    path = np.asarray(path, dtype=np.float64)
    d = path.shape[1]
    s1, s2, s3 = np.zeros(d), np.zeros((d, d)), np.zeros((d, d, d))
    for dx in np.diff(path, axis=0):
        b2 = np.outer(dx, dx) / 2.0
        if depth >= 3:
            b3 = np.einsum("i,j,k->ijk", dx, dx, dx) / 6.0
            s3 = s3 + b3 + np.einsum("ij,k->ijk", s2, dx) + np.einsum("i,jk->ijk", s1, b2)
        s2 = s2 + b2 + np.outer(s1, dx)
        s1 = s1 + dx
    return [s1, s2, s3][:depth]


def levy_area(path):
    """Antisymmetric part of level 2: A_ij = (S_ij - S_ji) / 2. Equals the level-2 log-signature coefficient."""
    s2 = signature(path, 2)[1]
    return (s2 - s2.T) / 2.0


def expm(M):
    """Matrix exponential by scaling and squaring with a Taylor series (numpy only; fine for small matrices)."""
    M = np.asarray(M, dtype=np.float64)
    n = max(0, int(np.ceil(np.log2(max(np.abs(M).sum(axis=1).max(), 1e-300)))) + 1)
    A = M / 2.0 ** n
    out, term = np.eye(len(M)), np.eye(len(M))
    for k in range(1, 20):
        term = term @ A / k
        out = out + term
    for _ in range(n):
        out = out @ out
    return out


# --------------------------------------------------------------------------------------------------------------
# The four checks. Each returns a list of (name, value, bar, passed) rows.
# --------------------------------------------------------------------------------------------------------------
def check_bag(seed=0, n=9, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    t = np.linspace(0.0, 1.0, n)[:, None]
    s2 = signature(np.hstack([X, t]), 2)[1]                 # coordinates 0..d-1 are X, coordinate d is time
    term = s2[:d, d]                                         # S^{(i, t)} = int (X^i_u - X^i_0) du
    mean_pl = (((X[:-1] + X[1:]) / 2.0) * np.diff(t, axis=0)).sum(axis=0)   # mean of the piecewise-linear path
    err = float(np.abs(term - (mean_pl - X[0])).max())

    # Same points, other order: the bag of points is unchanged, the area is not.
    perm = rng.permutation(n)
    bag_moved = float(np.abs(X.mean(0) - X[perm].mean(0)).max())
    area_moved = float(np.abs(levy_area(X) - levy_area(X[perm])).max())
    return [
        ("bag: S^(i,t) = mean - start, max error", err, "< 1e-12", err < 1e-12),
        ("bag: reorder moves the point mean by", bag_moved, "< 1e-12", bag_moved < 1e-12),
        ("bag: reorder moves the Levy area by", area_moved, "> 0.1", area_moved > 0.1),
    ]


def check_area(seed=1, n=12, d=4):
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(size=(n, d)), axis=0)
    s1, s2 = signature(X, 2)
    A, Ar = levy_area(X), levy_area(X[::-1])
    flip = float(np.abs(A + Ar).max())                       # reversal flips the area
    sym = float(np.abs((s2 + s2.T) / 2.0 - np.outer(s1, s1) / 2.0).max())   # symmetric part = level 1 squared
    return [
        ("area: reversed path has area -A, max error", flip, "< 1e-9", flip < 1e-9),
        ("area: sym. part of level 2 = (level 1)^2 / 2", sym, "< 1e-9", sym < 1e-9),
    ]


def _transport(A1, A2, P):
    """Exact transport of a fixed linear connection along a polyline P in the plane."""
    H = np.eye(len(A1))
    for d in np.diff(P, axis=0):
        H = expm(A1 * d[0] + A2 * d[1]) @ H
    return H


def _rho(sig, mats):
    """Holonomy from the signature: sum over words of S^{word} * A_{word} (later letters act on the left)."""
    m = len(mats[0])
    out = np.eye(m)
    for level, S in enumerate(sig, start=1):
        for idx in np.ndindex(*S.shape):
            M = np.eye(m)
            for i in idx:
                M = mats[i] @ M
            out = out + S[idx] * M
    return out


def check_holonomy(seed=2, m=3, radii=(0.3, 0.1, 0.03), steps=2000):
    rng = np.random.default_rng(seed)
    A1, A2 = rng.normal(size=(2, m, m))
    F = A2 @ A1 - A1 @ A2                                    # curvature of a fixed connection, in this ordering
    rows, ratios = [], []
    th = np.linspace(0.0, 2 * np.pi, steps + 1)
    for r in radii:
        P = r * np.c_[np.cos(th), np.sin(th)]
        H = _transport(A1, A2, P)
        area = levy_area(P)[0, 1]                            # = pi r^2, counter-clockwise
        ratio = float(np.abs(H - expm(area * F)).max() / area)
        ratios.append(ratio)
        rows.append((f"holonomy: |H - exp(area F)| / area, r = {r}", ratio, "falls with r", None))
    falls = all(b < a * 0.6 for a, b in zip(ratios, ratios[1:]))
    rows = [(n, v, b, falls) for n, v, b, _ in rows]

    # Transport is a function of the signature: depth-N expansion converges to the exact transport.
    P = np.cumsum(0.15 * rng.normal(size=(30, 2)), axis=0)
    H = _transport(A1, A2, P)
    errs = [float(np.abs(H - _rho(signature(P, N), (A1, A2))).max()) for N in (1, 2, 3)]
    conv = errs[0] > errs[1] > errs[2]
    for N, e in zip((1, 2, 3), errs):
        rows.append((f"holonomy: |H - rho(S_depth{N})| on an open path", e, "falls with depth", conv))
    return rows


def area_rate(k, w, s=1.0, chains=64, T=400.0, dt=0.01, burn=20.0, seed=3, reverse=False):
    """Mean Levy-area rate of dX = -(k I + w J) X dt + s dW, J = [[0, -1], [1, 0]], over parallel chains."""
    rng = np.random.default_rng(seed)
    B = k * np.eye(2) + w * np.array([[0.0, -1.0], [1.0, 0.0]])
    x = rng.normal(scale=s / np.sqrt(2 * k), size=(chains, 2))   # start at the stationary spread
    n_burn, n = int(burn / dt), int(T / dt)
    xs = np.empty((n + 1, chains, 2))
    for i in range(n_burn + n):
        x = x - dt * x @ B.T + s * np.sqrt(dt) * rng.normal(size=(chains, 2))
        if i >= n_burn - 1:
            xs[i - n_burn + 1] = x
    if reverse:
        xs = xs[::-1]
    dx = np.diff(xs, axis=0)
    mid = (xs[:-1] + xs[1:]) / 2.0                           # Stratonovich midpoint, the area of the polyline
    a = 0.5 * (mid[..., 0] * dx[..., 1] - mid[..., 1] * dx[..., 0]).sum(axis=0) / T
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(chains))


def check_flux(k=1.0, s=1.0, ws=(0.0, 0.25, 0.5, 1.0), quick=False):
    kw = dict(chains=32, T=150.0) if quick else {}
    rows = []
    for w in ws:
        est, se = area_rate(k, w, s, **kw)
        theory = -w * s * s / (2 * k)
        ok = abs(est - theory) < 4 * se + 0.02
        rows.append((f"flux: area rate, w = {w} (theory {theory:+.3f})", est, f"theory +/- 4se ({se:.3f})", ok))
    est_r, se_r = area_rate(k, 1.0, s, reverse=True, **kw)
    ok = abs(est_r - s * s / (2 * k)) < 4 * se_r + 0.02
    rows.append(("flux: reversed paths, w = 1 (theory +0.500)", est_r, f"theory +/- 4se ({se_r:.3f})", ok))
    return rows


def run(quick=False):
    rows = check_bag() + check_area() + check_holonomy() + check_flux(quick=quick)
    width = max(len(r[0]) for r in rows)
    for name, value, bar, ok in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {value:>11.4g}   {bar}")
    return all(r[3] for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller flux simulation")
    ok = run(quick=ap.parse_args().quick)
    print("all checks pass" if ok else "SOME CHECKS FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
