"""Step 0's habitat form (body/step0_habitat.py) on toy episodes: windows, controls, the readings, the decision, and
the placebo. No model and no TextWorld (the one test on a real world is in test_habitat.py)."""
import numpy as np
import pytest

from body import step0_habitat as Z

ONSET, STALL, ASC = (50, 12), 12, (12, 75)


def _toy(seed, signal=True, n=100, params=None, rng=None):
    """An episode of n turns: a closure at turn 50 (12 flat turns repeating one action), ascents at 12 and 75 (the
    quest's count falls), every other turn new. With `signal`, X is low in the 10 turns before the onset and high in
    the 10 after each ascent; otherwise noise alone. One layer and the text space, the text with no signal."""
    rng = rng or np.random.default_rng(seed)
    left, turns = 5, []
    for t in range(n):
        flat = ONSET[0] <= t < ONSET[0] + ONSET[1]
        if t in ASC:
            left -= 1
        turns.append({"turn": t, "action": "look" if flat else f"take thing {t}", "room": "kitchen",
                      "new_state": not flat, "score_change": 0, "quest_left": left})
    turns[ONSET[0] - 1]["action"] = "look"                 # the stall repeats an earlier (action, room) pair
    scored = list(range(Z.BURN, n))
    X = {t: float(rng.normal(1.0, 0.3)) for t in scored}
    if signal:
        for t in range(ONSET[0] - 10, ONSET[0]):
            X[t] -= 1.0
        for a in ASC:
            for t in range(a + 1, a + 11):
                if t in X:
                    X[t] += 1.0
    layer = {f: dict(X) for f in Z.FIELDS}
    text = {f: {t: float(rng.normal(1.0, 0.3)) for t in scored} for f in Z.FIELDS}
    return {"seed": seed, "params": params or {"rooms": 6}, "scored": scored, "spaces": [layer, text],
            "turns": turns, "start": 5}


def _eps(n, signal=True, turns=160):
    return {s: _toy(s, signal, n=turns) for s in range(3000, 3000 + n)}


def test_windows_end_before_the_onset_and_start_after_the_ascent():
    e = _toy(3000)
    lab = Z.labels(e, 5)
    assert lab["closures"] == [(50, 12)] and lab["ascents"] == list(ASC)
    evs, dropped = Z.windows(e, lab)
    clo = [x for x in evs if x["kind"] == "closure"][0]
    assert clo["rows"] == list(range(40, 50))                           # the onset's own action is not in it
    assert [x["rows"] for x in evs if x["kind"] == "ascent"] == [list(range(13, 23)), list(range(76, 86))]
    short = dict(e, scored=[t for t in e["scored"] if t < 46 or t >= 50])
    assert [x["rows"] for x in Z.windows(short, lab)[0] if x["kind"] == "closure"] == [list(range(36, 46))]
    late = dict(e, scored=list(range(48, 100)))                         # two scored turns before the onset
    assert "3000/closure@50" in Z.windows(late, Z.labels(late, 5))[1]


def test_controls_keep_away_from_events_and_are_never_reused():
    e = _toy(3000)
    lab = Z.labels(e, 5)
    ok = Z.eligible(e, lab, set())
    assert not set(range(50, 62)) & set(ok)                              # outside the closure
    assert all(abs(t - m) >= Z.GAP for t in ok for m in (50,) + ASC)
    eps = {3000: e}
    m = Z.match(eps, {3000: lab}, Z.windows(e, lab)[0], {3000: set()})
    used = [t for x in m["events"] for t in x["control"]]
    assert len(used) == len(set(used)) and all(x["control_seed"] == 3000 for x in m["events"])


def test_a_control_comes_from_the_nearest_world_with_the_same_settings_when_its_own_has_none():
    full = _toy(3001)
    bare = dict(_toy(3000), scored=list(range(40, 60)) + list(range(76, 86)))   # no room for a control of its own
    other = _toy(3005, params={"rooms": 9})
    eps = {3000: bare, 3001: full, 3005: other}
    labs = {s: Z.labels(e, 5) for s, e in eps.items()}
    evs = [x for x in Z.windows(bare, labs[3000])[0] if x["kind"] == "closure"]
    m = Z.match(eps, labs, evs, {s: set() for s in eps})
    assert m["events"][0]["control_seed"] == 3001 and m["cross-episode controls"]


def test_the_readings_pass_on_a_signal_and_not_on_noise():
    eps = _eps(120)
    out = Z.score(eps, 5, 1, placebo_draws=20, draws=200)
    r1, r2 = out["reading 1, closures"], out["reading 2, ascents"]
    assert r1["n"] == 120 and r2["n"] == 240 and r1["episodes"] == 120
    assert r1["AUC"] > 0.95 and r1["90% interval"][0] > 0.5 and r1["text AUC"] < 0.7
    assert r2["90% interval"][0] > 0.5 and out["decision"]["passed"] is True
    assert set(out["reported"]) >= {"at k = 3", "at k = 10", "placebo windows", "same-episode pairs only",
                                    "broad ascents", "X with W = 5", "X with W = 20"}
    assert abs(out["reported"]["placebo windows"]["closures"]["mean"] - 0.5) < 0.15
    null = Z.score(_eps(120, signal=False), 5, 1, placebo_draws=5, draws=200)
    assert null["decision"]["passed"] is False and abs(null["reading 1, closures"]["AUC"] - 0.5) < 0.15


def test_short_episodes_run_out_of_controls_and_say_so():
    out = Z.events_at(_eps(4, turns=100), 5)                             # room for two controls an episode, not three
    assert len(out["events"]) + len(out["no control"]) == 12 and out["no control"]


def test_too_few_events_decide_nothing():
    out = Z.score(_eps(30), 5, 1, placebo_draws=2, draws=50)
    assert out["decision"]["passed"] is None and "too few" in out["decision"]["note"]


def test_codes_score_from_the_tenth_turn_with_the_world_as_partner():
    rng = np.random.default_rng(0)
    chunks = [{"slot": "world", "turn": -1}] + [{"slot": s, "turn": t} for t in range(20) for s in ("self", "world")]
    assert Z.sources(chunks)[:3] == ["partner", "self", "partner"]
    X = rng.normal(size=(len(chunks), 6))
    c = Z.codes(X, Z.sources(chunks), 0.1, variants=False)
    selves = [i for i, ch in enumerate(chunks) if ch["slot"] == "self"]
    assert np.isnan(c["L_now"][selves[Z.BURN - 1]]) and np.isfinite(c["L_now"][selves[Z.BURN]])
    assert np.isfinite(c["X"][selves[Z.BURN:]]).all() and np.isnan(c["X_w5"]).all()
