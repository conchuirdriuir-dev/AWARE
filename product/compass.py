"""The compass. v10 (current): does the prediction hold (dynamic residual at the floor or above it), and is the channel
open (coupling above zero or at it). No band. v9's cells are kept for round 1's record (`cell_v9`).
"""
from __future__ import annotations

CELLS = {(False, False): "on track", (False, True): "closed loop", (True, False): "progress", (True, True): "lost"}


def cell(residual_above_floor: bool, coupling_at_zero: bool) -> str:
    return CELLS[(bool(residual_above_floor), bool(coupling_at_zero))]


# v9, round 1 only: self-residual against a band, frame-residual high or low
BAND = (0.75, 2.0)
FRAME_HIGH = 2.0
CELLS_V9 = {("in band", False): "on track", ("in band", True): "progress",
            ("high", False): "confused about itself", ("high", True): "lost",
            ("below band", False): "mastery", ("below band", True): "closed loop"}


def cell_v9(self_residual: float, frame_residual: float, band=BAND, frame_high: float = FRAME_HIGH) -> str:
    level = "below band" if self_residual < band[0] else "high" if self_residual > band[1] else "in band"
    return CELLS_V9[(level, frame_residual > frame_high)]
