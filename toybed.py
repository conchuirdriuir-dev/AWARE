"""The toy bed: synthetic states with noise of size D, structures planted where the answer is known (PLAN.md, step 2).

    python -m toybed                                 # sweep D/R, 200 trials of each test
    python -m toybed --ratios 0.1 --trials 500 --dim 512

It tests the geometry core and the compass against each other; it says nothing about a real model.
"""
from __future__ import annotations

import argparse

import numpy as np

import compass
from geometry import mirror_or_additive, reflect, unit

KINDS = {"on track": "on track", "progress": "progress", "drift": "lost", "rabbit hole": "rabbit hole"}


def noise(rng, n: int, d: int, D: float) -> np.ndarray:
    """n vectors whose length is about D."""
    return rng.normal(scale=D / np.sqrt(d), size=(n, d))


def orthonormal(rng, k: int, d: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return q.T


def landmarks(rng, d: int, D: float, n: int = 6) -> np.ndarray:
    """The task statement and its paraphrases: one point, read n times with noise D."""
    return rng.normal(size=d) + noise(rng, n, d, D)


def pairs(kind: str, rng, c: np.ndarray, R: float, D: float, n: int = 40, k: int = 3):
    """n pairs (x, y) related by a mirror about c or by an additive direction, observed with noise D. Each x varies
    along the planted direction and k other axes, so the two structures differ by about R."""
    d = len(c)
    basis = orthonormal(rng, k + 1, d)
    w, A = basis[0], basis[1:]
    X = c + R * (rng.normal(size=(n, 1)) * w + rng.normal(size=(n, k)) @ A)
    Y = reflect(X, c, w) if kind == "mirror" else X + R * w
    return X + noise(rng, n, d, D), Y + noise(rng, n, d, D)


def trajectory(kind: str, rng, c: np.ndarray, R: float, D: float, w: int = 20, k: int = 3):
    """Three windows of w states (first, prev, recent) for one of the four kinds."""
    d = len(c)
    basis = orthonormal(rng, 2 * k + 2, d)
    A, new, away, A2 = basis[:k], basis[k], basis[k + 1], basis[k + 2:]

    def along(axes, centre):
        return centre + R * rng.normal(size=(w, len(axes))) @ axes + noise(rng, w, d, D)

    first = along(A, c)
    if kind == "on track":
        prev, recent = along(A, c), along(A, c)
    elif kind == "progress":
        prev, recent = along(A, c), along(np.vstack([A, new]), c)
    elif kind == "rabbit hole":
        far = c + 4 * R * away
        prev, recent = along(A2, far), along(A2, far)
    elif kind == "drift":
        step = 4 * R / np.sqrt(2 * w)
        walk = c + np.cumsum(rng.normal(scale=step / np.sqrt(d), size=(2 * w, d)), axis=0)
        walk += noise(rng, 2 * w, d, D)
        prev, recent = walk[:w], walk[w:]
    else:
        raise ValueError(kind)
    return first, prev, recent


def separation_rate(rng, d: int, R: float, D: float, trials: int, n_fit: int = 30) -> float:
    right = 0
    for t in range(trials):
        kind = "mirror" if t % 2 == 0 else "additive"
        c = landmarks(rng, d, D).mean(0)
        X, Y = pairs(kind, rng, c, R, D)
        right += mirror_or_additive(X, Y, c, D, n_fit)[0] == kind
    return right / trials


def placement_rate(rng, d: int, R: float, D: float, trials: int) -> float:
    right = 0
    kinds = list(KINDS)
    for t in range(trials):
        kind = kinds[t % len(kinds)]
        c = landmarks(rng, d, D).mean(0)
        first, prev, recent = trajectory(kind, rng, c, R, D)
        right += compass.read(prev, recent, c, compass.anchor_scale(first, c), D).cell == KINDS[kind]
    return right / trials


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ratios", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0], help="D/R values")
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    R = 1.0
    print(f"{'D/R':>6}  {'separate':>9}  {'place':>6}   (pass: both >= 0.90, at the measured D)")
    for ratio in a.ratios:
        rng = np.random.default_rng(a.seed)
        sep = separation_rate(rng, a.dim, R, ratio * R, a.trials)
        place = placement_rate(rng, a.dim, R, ratio * R, a.trials)
        print(f"{ratio:>6.2f}  {sep:>9.2f}  {place:>6.2f}")


if __name__ == "__main__":
    main()
