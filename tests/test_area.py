"""Area's sense (body/area.py) and stage B's reading and scoring (body/area_fg.py)."""
import numpy as np
import pytest

from body import area, area_fg
from proposals import toy_area


def test_closed_form_matches_the_signature():
    rng = np.random.default_rng(0)
    for n, d in ((5, 2), (24, 8), (40, 3)):
        p = rng.normal(size=(n, d))
        assert np.allclose(area.levy_area(p), toy_area.levy_area(p), atol=1e-10)
        assert np.allclose(area.rho(p), toy_area.area_rho(p), atol=1e-12)


def test_area_reads_order_not_position():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(4, 8)) * 3
    loop = np.vstack([pts[i % 4] for i in range(24)]) + rng.normal(size=(24, 8)) * 0.3
    order = [0, 2, 1, 3, 0, 1, 3, 2, 0, 3, 1, 2, 3, 0, 2, 1, 2, 0, 3, 1, 0, 2, 3, 1]
    dither = np.vstack([pts[i] for i in order]) + rng.normal(size=(24, 8)) * 0.3
    assert area.rho(loop)[0] > 3 * area.rho(dither)[0]
    assert np.isclose(area.rho(loop[::-1])[0], area.rho(loop)[0])             # reversal flips A, not |sigma|


def fake_episode(rng, kind, width=16, contents=None):
    """Chunks in the plant's structure, and states: intake, 10 burn turns, 24 window turns."""
    order = (list(range(4)) * 6 if kind == "loop" else
             [0, 2, 1, 3, 0, 1, 3, 2, 0, 3, 1, 2, 3, 0, 2, 1, 2, 0, 3, 1, 0, 2, 3, 1])
    pts = contents if contents is not None else rng.normal(size=(4, width)) * 3
    chunks, S = [{"slot": "world", "turn": -1, "text": "intake"}], [rng.normal(size=width)]
    for t in range(area_fg.BURN + area_fg.WINDOW):
        if t < area_fg.BURN:
            cmd, x = f"burn {t}", rng.normal(size=width)
        else:
            c = order[t - area_fg.BURN]
            cmd, x = f"measure V p{c} q{c}", pts[c] + rng.normal(size=width) * 0.5
        chunks += [{"slot": "self", "turn": t, "text": cmd}, {"slot": "world", "turn": t, "text": f"reply {t}"}]
        S += [rng.normal(size=width), x]
    S = np.stack(S)[:, None, :].repeat(2, axis=1)                                 # two "layers"
    keys = {t: {"area_kind": kind, "phase": "window" if t >= area_fg.BURN else "burn"}
            for t in range(area_fg.BURN + area_fg.WINDOW)}
    return chunks, S, keys


def test_window_is_read_from_chunks_alone():
    chunks, S, _ = fake_episode(np.random.default_rng(2), "loop")
    w = area_fg.window_of(chunks)
    assert len(w["world"]) == 24 and len(w["all"]) == 48 and len(set(w["commands"])) == 4
    assert all(chunks[i]["slot"] == "world" and chunks[i]["turn"] >= 10 for i in w["world"])
    with pytest.raises(ValueError):
        area_fg.window_of([c for c in chunks if c["turn"] < 30])


def test_score_passes_a_clear_loop_and_fails_a_blind_sense():
    rng = np.random.default_rng(3)
    per, kind, eps = {}, {}, {}
    for s in range(80):
        k = "loop" if s % 2 == 0 else "dither"
        chunks, S, keys = fake_episode(rng, k)
        per[s] = area_fg.senses(S, chunks, np.ones(2))
        kind[s] = k
        eps[(s, None)] = {"keys": keys}
    assert area_fg.kinds(eps) == kind
    out = area_fg.score(per, kind, draws=200)
    assert out["bar1_area"]["auc"] > 0.95 and out["pass"]
    assert 0.25 < out["bar2_apart_from_boundary"]["boundary_auc"] < 0.75
    blind = {s: dict(v, rho=rng.normal(size=2)) for s, v in per.items()}               # a sense that reads nothing
    assert not area_fg.score(blind, kind, draws=200)["pass"]


def test_kinds_refuses_a_misplaced_window():
    eps = {(1, None): {"keys": {t: {"area_kind": "loop", "phase": "window" if t >= 9 else "burn"} for t in range(34)}}}
    with pytest.raises(SystemExit):
        area_fg.kinds(eps)


def test_form_b2_fits_the_basis_on_the_window_alone():
    chunks, S, _ = fake_episode(np.random.default_rng(4), "loop")
    b, b2 = area_fg.senses(S, chunks, np.ones(2), "b"), area_fg.senses(S, chunks, np.ones(2), "b2")
    w = area_fg.window_of(chunks)
    for j in range(2):
        X = S[:, j]
        assert np.isclose(b2["rho"][j], area.window_rho(X[w["world"]], X[w["world"]])[0])
        assert np.isclose(b2["rho_inter"][j], area.window_rho(X[w["all"]], X[w["all"]])[0])
    assert np.allclose(b2["rho_asrun"], b["rho"]) and "rho_asrun" not in b
    assert np.allclose(b2["boundary"], b["boundary"]) and np.allclose(b2["ratio"], b["ratio"])


def test_form_b2_survives_a_burn_that_crowds_the_basis():
    """Replies before the window, large and off the contents' plane, take every place in the as-run basis; the
    window's own basis keeps the loop."""
    rng = np.random.default_rng(5)
    per = {"b": {}, "b2": {}}
    kind = {}
    for s in range(40):
        k = "loop" if s % 2 == 0 else "dither"
        contents = np.zeros((4, 32))
        contents[:, :3] = rng.normal(size=(4, 3)) * [8, 4, 2]
        chunks, S, _ = fake_episode(rng, k, width=32, contents=contents)
        before = [i for i, c in enumerate(chunks) if c["slot"] == "world" and c["turn"] < area_fg.BURN]
        S[before] = 0
        S[before, :, 3:3 + len(before)] = np.eye(len(before))[:, None, :] * 40 * rng.choice([-1, 1], len(before))[:, None, None]
        kind[s] = k
        for f in per:
            per[f][s] = area_fg.senses(S, chunks, np.ones(2), f)
    a = {f: area_fg.score(per[f], kind, draws=100)["bar1_area"]["auc"] for f in per}
    assert a["b2"] > 0.85 and a["b2"] > a["b"] + 0.25, a
