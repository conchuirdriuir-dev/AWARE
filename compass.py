"""The compass: two signals read on a recent window of states, against the code fitted to the window before it.

Self-residual (how much of the recent work the code does not explain) says whether the agent is coherent; anchor
distance (how far its work is from the landmark centroid) says whether it is on task. Together they place it in one
of four cells (the Big Picture doc, "The compass"). Thresholds are fixed in PLAN.md before any run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry import fit_code, residual

R_HIGH = 2.0       # self-residual, in units of D^2, above which the recent work is unexplained
A_FAR = 1.0        # anchor distance, in units of the first window's spread about c, above which the work is far

CELLS = {(False, False): "on track", (True, False): "progress", (True, True): "lost", (False, True): "rabbit hole"}


@dataclass
class Reading:
    self_residual: float
    anchor_distance: float
    cell: str


def anchor_scale(first: np.ndarray, c: np.ndarray) -> float:
    """The root-mean-square distance from c of the first window's states: the start of the work, taken as on task."""
    return float(np.sqrt(np.mean(np.sum((first - c) ** 2, axis=1))))


def read(prev: np.ndarray, recent: np.ndarray, c: np.ndarray, scale: float, D: float,
         r_high: float = R_HIGH, a_far: float = A_FAR) -> Reading:
    r = residual(fit_code(prev), recent) / D ** 2
    a = float(np.linalg.norm(recent.mean(0) - c)) / scale
    return Reading(r, a, CELLS[(r > r_high, a > a_far)])
