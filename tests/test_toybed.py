"""The toy bed's pass criterion (PLAN.md, step 2), here at D/R = 0.2 until D is measured on the real model."""
import numpy as np

import compass
import toybed


def test_structures_are_separated_and_trajectories_placed():
    rng = np.random.default_rng(0)
    assert toybed.separation_rate(rng, 256, 1.0, 0.2, 200) >= 0.9
    assert toybed.placement_rate(rng, 256, 1.0, 0.2, 200) >= 0.9


def test_each_kind_lands_in_its_own_cell():
    rng = np.random.default_rng(1)
    for kind, cell in toybed.KINDS.items():
        c = toybed.landmarks(rng, 256, 0.1).mean(0)
        first, prev, recent = toybed.trajectory(kind, rng, c, 1.0, 0.1)
        assert compass.read(prev, recent, c, compass.anchor_scale(first, c), 0.1).cell == cell
