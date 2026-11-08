"""body/senses.py: the live senses equal the tested ones (Step -1's news, rate's X_c), chunk by chunk."""
import numpy as np

from body.conditional import conditional
from body.senses import Episode, body_vectors, fit_z, zscore
from body.stepm1 import news


def _episode(turns=40, layers=2, width=24, seed=0):
    rng = np.random.default_rng(seed)
    slots, tlist = ["world"], [-1]
    for t in range(turns):
        slots += ["self", "world"]
        tlist += [t, t]
    S = rng.normal(size=(len(slots), layers, width))
    S[2::2] += 0.5 * S[1:-1:2]                       # replies depend on the action before them
    return S, slots, tlist


def test_live_senses_equal_the_tested_ones():
    S, slots, turns = _episode()
    C = S + 0.1                                       # the capped reading differs from the full one
    unit, fv = np.array([1.3, 0.7]), np.array([0.01, 0.02])
    ep = Episode(unit, fv)
    got = [ep.add(S[i], C[i], slots[i], turns[i]) for i in range(len(slots))]
    want_b = np.nanmean(news(S, slots, unit), axis=1)
    want_x = np.mean([conditional(C[:, j], slots, turns, fv[j])["X_c"] for j in range(2)], axis=0)
    for i, g in enumerate(got):
        assert np.isnan(g["boundary"]) == np.isnan(want_b[i])
        if np.isfinite(want_b[i]):
            assert abs(g["boundary"] - want_b[i]) < 1e-9
        assert np.isnan(g["X_c"]) == np.isnan(want_x[i])
        if np.isfinite(want_x[i]):
            assert abs(g["X_c"] - want_x[i]) < 1e-9
    assert np.isfinite(want_x).sum() > 5               # some actions are scored, the early ones are not


def test_body_vectors_and_zscores():
    S, slots, turns = _episode(turns=30)
    ep = Episode(np.ones(2), np.full(2, 0.01))
    senses = [ep.add(S[i], S[i], slots[i], turns[i]) for i in range(len(slots))]
    chunks = [{"slot": s, "turn": t} for s, t in zip(slots, turns)]
    v = body_vectors(chunks, senses)
    assert sorted(v) == list(range(30))
    z = fit_z([v])
    B = zscore(v, z)
    assert B.shape == (30, 3)
    assert (B[:21, 2] == 0).all() and B[:21, 1].tolist() == [0.0] * 21   # X_c undefined early: zero, flagged
    assert abs(B[:, 0].mean()) < 1e-9


def test_clip_cuts_the_channel_not_the_flag():
    z = {"boundary": [0.0, 1.0], "X_c": [0.0, 1.0]}
    B = zscore({0: (50.0, 1e14), 1: (1.0, float("nan"))}, z, clip=10)
    assert B.tolist() == [[10.0, 10.0, 1.0], [1.0, 0.0, 0.0]]
