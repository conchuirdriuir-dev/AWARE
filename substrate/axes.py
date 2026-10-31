"""Retained axes and their admission (the Entity tab, substrate decisions 2 and 3).

The code is retained, not consumed: every axis considered is kept as an object after it has served. Admission has two
criteria. An axis that lowers held-out residual by more than D enters the predictive code (the product criterion). One
that does not is retained as structural, tagged untested and excluded from residual, until the research track states a
structural test. Nothing is discarded.

Residual here is static: the root-mean-square distance from the code's affine subspace through the origin, in the
units of D. Which residual the compass should use is open (v9 tension log, "Which residual"); so is D serving as a
threshold on a reduction in it ("D is three units"). Both are provisional.
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

    def __init__(self, c: np.ndarray, D: float, name: str = "code"):
        self.c, self.D, self.name = np.asarray(c, dtype=float), float(D), name
        self.axes: list[Axis] = []

    def predictive(self) -> np.ndarray:
        rows = [a.vector for a in self.axes if a.kind == "predictive"]
        return np.stack(rows) if rows else np.zeros((0, len(self.c)))

    def structural(self) -> list[Axis]:
        return [a for a in self.axes if a.kind == "structural"]

    def residual(self, X: np.ndarray) -> float:
        return rms(off(X, self.c, self.predictive()))

    def consider(self, candidate: np.ndarray, held_out: np.ndarray, source: str, at: int) -> Axis:
        """Test a candidate on held-out states and retain it either way."""
        P = self.predictive()
        v = unit(candidate - (P.T @ (P @ candidate)) if len(P) else candidate)
        before = self.residual(held_out)
        after = rms(off(held_out, self.c, np.vstack([P, v[None]])))
        gain = (before - after) / self.D
        if gain > 1.0:
            axis = Axis(v, "predictive", gain, source, at)
        else:
            axis = Axis(v, "structural", gain, source, at, tags=["untested"])
        self.axes.append(axis)
        return axis

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), np.stack([a.vector for a in self.axes]) if self.axes else np.zeros((0, 0)))
        meta = {"name": self.name, "c": self.c.tolist(), "D": self.D,
                "axes": [{k: v for k, v in asdict(a).items() if k != "vector"} for a in self.axes]}
        path.with_suffix(".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str | Path) -> "Code":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        code = cls(np.array(meta["c"]), meta["D"], meta["name"])
        vectors = np.load(path.with_suffix(".npy"))
        code.axes = [Axis(vector=vectors[i], **a) for i, a in enumerate(meta["axes"])]
        return code
