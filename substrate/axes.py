"""Retained axes and their admission (the Entity tab, substrate decisions 2 and 3).

The code is retained, not consumed: every axis considered is kept as an object after it has served. Admission has two
criteria. An axis that lowers held-out residual by more than D enters the predictive code (the product criterion). One
that does not is retained as structural, tagged untested and excluded from residual, until the research track states a
structural test. Nothing is discarded.

By default residual is static: the root-mean-square distance from the code's affine subspace through the origin, in
the units of D. A caller may supply its own residual (v10: dynamic, the one-step error of a fit) and its own unit (v10:
the measured residual floor, since D as noise scale and D as threshold proved not to be the same unit).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from substrate.geometry import off, rms, unit


@dataclass
class Axis:
    vector: np.ndarray
    kind: str                 # "predictive" or "structural"
    gain: float               # held-out residual removed, in the units of D (negative: it made things worse)
    source: str               # where the candidate came from, e.g. "self" or "frame"
    at: int                   # the step it was considered at
    tags: list[str] = field(default_factory=list)


class Code:
    """Axes through the origin's centroid `c`. Only predictive axes take part in residual."""

    def __init__(self, c: np.ndarray, D: float, name: str = "code", unit: float | None = None):
        self.c, self.D, self.name = np.asarray(c, dtype=float), float(D), name
        self.unit = float(unit) if unit is not None else self.D       # what a gain must exceed to admit
        self.axes: list[Axis] = []

    def predictive(self) -> np.ndarray:
        rows = [a.vector for a in self.axes if a.kind == "predictive"]
        return np.stack(rows) if rows else np.zeros((0, len(self.c)))

    def structural(self) -> list[Axis]:
        return [a for a in self.axes if a.kind == "structural"]

    def residual(self, X: np.ndarray) -> float:
        return rms(off(X, self.c, self.predictive()))

    def consider(self, candidate: np.ndarray, held_out: np.ndarray | None = None, source: str = "", at: int = 0,
                 residual=None) -> Axis:
        """Test a candidate and retain it either way. `residual(axes)` gives held-out residual under a set of axes
        (orthonormal rows); without it, static residual on `held_out`."""
        P = self.predictive()
        v = unit(candidate - (P.T @ (P @ candidate)) if len(P) else candidate)
        if residual is None:
            residual = lambda axes: rms(off(held_out, self.c, axes))  # noqa: E731
        gain = (residual(P) - residual(np.vstack([P, v[None]]))) / self.unit
        if gain > 1.0:
            axis = Axis(v, "predictive", gain, source, at)
        else:
            axis = Axis(v, "structural", gain, source, at, tags=["untested"])
        self.axes.append(axis)
        return axis

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), np.stack([a.vector for a in self.axes]) if self.axes else np.zeros((0, 0)))
        meta = {"name": self.name, "c": self.c.tolist(), "D": self.D, "unit": self.unit,
                "axes": [{k: v for k, v in asdict(a).items() if k != "vector"} for a in self.axes]}
        path.with_suffix(".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str | Path) -> "Code":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        code = cls(np.array(meta["c"]), meta["D"], meta["name"], meta.get("unit"))
        vectors = np.load(path.with_suffix(".npy"))
        code.axes = [Axis(vector=vectors[i], **a) for i, a in enumerate(meta["axes"])]
        return code
