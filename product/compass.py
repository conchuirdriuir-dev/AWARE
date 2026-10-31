"""The compass (v9, "The compass"): self-residual against the band, frame-residual high or low, on the same window.

The band's edges and the frame threshold are provisional and descriptive (PLAN.md, product step 3): the toy bed reports
cells, it does not pass on them. v9 makes the band's lower edge a warning only while the margin grows.
"""
from __future__ import annotations

BAND = (0.75, 2.0)        # self-residual, in D
FRAME_HIGH = 2.0          # frame-residual, in D

CELLS = {("in band", False): "on track", ("in band", True): "progress",
         ("high", False): "confused about itself", ("high", True): "lost",
         ("below band", False): "mastery", ("below band", True): "closed loop"}


def cell(self_residual: float, frame_residual: float, band=BAND, frame_high: float = FRAME_HIGH) -> str:
    level = "below band" if self_residual < band[0] else "high" if self_residual > band[1] else "in band"
    return CELLS[(level, frame_residual > frame_high)]
