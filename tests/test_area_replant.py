import numpy as np

from proposals import area_stage_b_replant as P


def fake(rng, pre=5, width=24, layers=2, noise=0.05):
    """An episode in the diagnostic's form: replies before the window, then 24 (command, reply) pairs."""
    cmds, C = rng.normal(size=(4, width)), rng.normal(size=(4, width)) * 3
    lab = np.repeat(np.arange(4), 6)
    rows = [rng.normal(size=width) * 5 for _ in range(pre)]
    for g in lab:
        rows += [cmds[g], C[g] + rng.normal(size=width) * noise]
    S = np.repeat(np.array(rows)[:, None, :], layers, axis=1)
    world = pre + 1 + 2 * np.arange(24)
    return {"S": S, "world": world, "fit_world": np.concatenate([np.arange(pre), world])}


def test_orders_are_loops_and_dithers():
    rng = np.random.default_rng(0)
    l, d = P.order("loop", rng), P.order("dither", rng)
    assert (l[4:] == l[:-4]).all() and (np.bincount(d) == 6).all() and (d[1:] != d[:-1]).all()


def test_replanting_tells_loop_from_dither_on_clean_material():
    rng = np.random.default_rng(1)
    eps = {s: fake(rng) for s in range(12)}
    for variant in ("real", "resid_shuffled", "iso64"):
        win, run = P.replant(eps, sorted(eps), variant, repeats=1)
        assert win > 0.9 and 0 <= run <= 1, (variant, win, run)
