"""The product track's detectors (PLAN.md, product step 3): point and trajectory contrast, and the observer that keeps
a self code and a frame and reads self-residual, frame-residual, the margin, frame growth and distance to the origin.
"""
from __future__ import annotations

import numpy as np

from substrate.axes import Code
from substrate.geometry import off, top_direction

WINDOW, BLOCK, MODES = 40, 20, 4


def point_contrast(a: np.ndarray, b: np.ndarray, D: float) -> float:
    """How far the mean state moved between two windows, in D: sees topics."""
    return float(np.linalg.norm(b.mean(0) - a.mean(0)) / D)


def _modes(W: np.ndarray, basis: np.ndarray) -> np.ndarray:
    Z = (W - W.mean(0)) @ basis.T
    A = np.linalg.lstsq(Z[:-1], Z[1:], rcond=None)[0].T
    return np.sort(np.abs(np.linalg.eigvals(A)))[::-1]


def trajectory_contrast(a: np.ndarray, b: np.ndarray, k: int = MODES) -> float:
    """How far the local dynamics changed between two windows: the sorted eigenvalue magnitudes of a linear fit in
    each, in the top k principal axes of the two together, each window's own mean removed. Sees modes of working."""
    union = np.vstack([a - a.mean(0), b - b.mean(0)])
    basis = np.linalg.svd(union, full_matrices=False)[2][:k]
    return float(np.linalg.norm(_modes(a, basis) - _modes(b, basis)))


def contrasts(run: dict, D: float, w: int = WINDOW) -> dict:
    """Both contrasts at onset and for persistence (the window before onset against one a window after it)."""
    s, t0 = run["s"], run["t0"]
    before, after, later = s[t0 - w:t0], s[t0:t0 + w], s[t0 + w:t0 + 2 * w]
    return {"point at onset": point_contrast(before, after, D),
            "trajectory at onset": trajectory_contrast(before, after),
            "point persists": point_contrast(before, later, D),
            "trajectory persists": trajectory_contrast(before, later)}


def classify(fires: dict) -> str:
    """The signatures pre-registered for rows 1, 2, 5 and 6."""
    point, traj = fires["point persists"], fires["trajectory persists"]
    onset = fires["point at onset"] or fires["trajectory at onset"]
    if point and not traj:
        return "topic shift"
    if traj and not point:
        return "dynamics shift"
    if not point and not traj:
        return "noise spike" if onset else "progress"
    return "unclassified"


def observe(s: np.ndarray, E: np.ndarray, c: np.ndarray, D: float, h: int = BLOCK) -> list[dict]:
    """Run the observer over the trajectory in blocks. Each block is read under the codes as they stood before it;
    then each code considers one candidate from the block's first half, judged on its second half of s."""
    self_code, frame = Code(c, D, "self"), Code(c, D, "frame")
    rows, half = [], h // 2
    for b in range(0, len(s) - h + 1, h):
        S, X = s[b:b + h], E[b:b + h]
        r_self, r_frame = self_code.residual(S) / D, frame.residual(S) / D
        mine = top_direction(off(np.vstack([S[:half], X[:half]]), c, self_code.predictive()))
        world = top_direction(off(X[:half], c, frame.predictive()))
        self_code.consider(mine, S[half:], "self", b)
        admitted = frame.consider(world, S[half:], "frame", b).kind == "predictive"
        rows.append({"start": b, "self": r_self, "frame": r_frame, "margin": r_frame - r_self,
                     "growth": int(admitted), "distance": float(np.linalg.norm(S.mean(0) - c) / D)})
    return rows
