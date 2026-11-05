"""Step 0's habitat form (body/step0_habitat.py, amendment 2) on toy episodes: windows, pre-progress pairs, the
readings, the decision, and the placebo. No model and no TextWorld (the one test on a real world is in
test_habitat.py)."""
import numpy as np

from body import step0_habitat as Z

CLOSURE, ASC = (60, 12), (25, 30, 100, 130)


def _toy(seed, signal=True, n=160, params=None, ascents=ASC):
    """An episode of n turns: a closure at turn 60 (12 flat turns repeating one action), ascents (the quest's count
    falls) at `ascents`, every other turn new. With `signal`, X is low in the 10 turns before the onset and baseline
    elsewhere; otherwise noise alone. One layer and the text space, the text with no signal."""
    rng = np.random.default_rng(seed)
    left, turns = len(ascents), []
    for t in range(n):
        flat = CLOSURE[0] <= t < CLOSURE[0] + CLOSURE[1]
        if t in ascents:
            left -= 1
        turns.append({"turn": t, "action": "look" if flat else f"take thing {t}", "room": "kitchen",
                      "new_state": not flat, "score_change": 0, "quest_left": left})
    turns[CLOSURE[0] - 1]["action"] = "look"                 # the stall repeats an earlier (action, room) pair
    scored = list(range(Z.BURN, n))
    X = {t: float(rng.normal(1.0, 0.3)) for t in scored}
    if signal:
        for t in range(CLOSURE[0] - 10, CLOSURE[0]):
            X[t] -= 1.0
    layer = {f: dict(X) for f in Z.FIELDS}
    text = {f: {t: float(rng.normal(1.0, 0.3)) for t in scored} for f in Z.FIELDS}
    return {"seed": seed, "params": params or {"rooms": 6}, "scored": scored, "spaces": [layer, text],
            "turns": turns, "start": len(ascents)}


def _eps(n, signal=True):
    return {s: _toy(s, signal) for s in range(3000, 3000 + n)}


def test_windows_end_before_the_onset_and_start_after_the_ascent():
    e = _toy(3000)
    lab = Z.labels(e, 5)
    assert lab["closures"] == [(60, 12)] and lab["ascents"] == list(ASC)
    evs, dropped = Z.windows(e, lab)
    assert [x["rows"] for x in evs if x["kind"] == "closure"] == [list(range(50, 60))]   # the onset's action not in it
    assert [x["rows"] for x in evs if x["kind"] == "ascent"][0] == list(range(26, 36))
    late = dict(e, scored=list(range(58, 160)))                          # two scored turns before the onset
    assert "3000/closure@60" in Z.windows(late, Z.labels(late, 5))[1]


def test_pre_progress_windows_end_before_the_ascent_and_stay_out_of_closures():
    e = _toy(3000, ascents=(25, 66, 100))                                # 66: its window runs into the closure
    pre = dict(Z.pre_progress(e, Z.labels(e, 5)))
    assert pre[25] == list(range(15, 25)) and pre[100] == list(range(90, 100)) and 66 not in pre


def test_each_closure_takes_the_nearest_pre_progress_window_once():
    e = _toy(3000)
    lab = {3000: Z.labels(e, 5)}
    clo = [x for x in Z.windows(e, lab[3000])[0] if x["kind"] == "closure"]
    m = Z.match({3000: e}, lab, clo + [dict(clo[0], at=61)], {3000: set()})
    assert [x["control_ascent"] for x in m["events"]] == [30, 25]        # middle 55: 30's window, then 25's
    assert m["events"][0]["control"] == list(range(20, 30))


def test_a_closure_pairs_with_the_nearest_world_with_the_same_settings_when_its_own_has_no_progress():
    bare = _toy(3000, ascents=())
    eps = {3000: bare, 3002: _toy(3002), 3001: _toy(3001, params={"rooms": 9})}
    labs = {s: Z.labels(x, 5) for s, x in eps.items()}
    clo = [x for x in Z.windows(bare, labs[3000])[0] if x["kind"] == "closure"]
    m = Z.match(eps, labs, clo, {s: set() for s in eps})
    assert m["events"][0]["control_seed"] == 3002 and m["cross-episode controls"]


def test_the_readings_pass_on_a_signal_and_not_on_noise():
    out = Z.score(_eps(120), 5, 1, placebo_draws=10, draws=200)
    r1, r2 = out["reading 1, closures"], out["reading 2, ascents"]
    assert r1["n"] == 120 and r1["episodes"] == 120 and r2["n"] == 480 and r2["against"] == 120
    assert r1["AUC"] > 0.95 and r1["90% interval"][0] > 0.5 and r1["text AUC"] < 0.7
    assert r2["90% interval"][0] > 0.5 and out["decision"]["passed"] is True
    assert set(out["reported"]) >= {"at k = 3", "at k = 10", "placebo windows", "same-episode pairs only (reading 1)",
                                    "broad ascents", "X with W = 5", "X with W = 20"}
    null = Z.score(_eps(120, signal=False), 5, 1, placebo_draws=5, draws=200)
    assert null["decision"]["passed"] is False and abs(null["reading 1, closures"]["AUC"] - 0.5) < 0.15
    assert abs(null["reading 2, ascents"]["AUC"] - 0.5) < 0.15


def test_the_placebo_reads_pretend_windows_by_the_same_rules():
    out = Z.score(_eps(60, signal=False), 5, 1, placebo_draws=20, draws=50)["reported"]["placebo windows"]
    assert out["n closures"] == 60 and out["n ascents"] == 240
    assert abs(out["closures"]["mean"] - 0.5) < 0.15 and abs(out["ascents"]["mean"] - 0.5) < 0.15


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
