"""The body readout's progress by source (body/progress.py): MomentFit agrees with T2's Fit, and the three progress
readings behave as Step 0 registers them on toy streams."""
import numpy as np

from body.progress import GramFit, by_source, code, cost, reported
from goalspace.formula import Fit


def test_gramfit_matches_fit_below_and_above_width():
    rng = np.random.default_rng(0)
    for n, d in ((30, 50), (200, 20)):
        X = rng.normal(size=(n + 1, d)) @ rng.normal(size=(d, d)) * 0.3 + rng.normal(size=d)
        a = Fit(X[:n], 1e-3)
        b = GramFit(X @ X.T, np.arange(n), d, 1e-3)
        assert a.choose_k() == b.choose_k()
        for k in (0, 1, 3, a.choose_k()):
            assert np.isclose(a.bits(X[n], k), b.bits(n, k), rtol=1e-7, atol=1e-6)
            assert np.isclose(a.code_length(k), b.code_length(k), rtol=1e-7, atol=1e-5)


def test_gramfit_on_a_subset_equals_fit_on_those_rows():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(81, 12))
    keep = np.setdiff1d(np.arange(80), [70, 72, 75])
    a = Fit(X[keep], 1e-3)
    assert np.isclose(code(X @ X.T, keep, 80, 12, 1e-3)[0], a.bits(X[80], a.choose_k()))


def _stream(rng, n, d, partner_teaches):
    """Self chunks drift along a direction the partner announces (or does not); tool chunks are noise."""
    u = np.zeros(d)
    u[0] = 1.0
    X, src = [], []
    for t in range(n):
        drift = 0.08 * t * u
        if t % 5 == 0:
            X.append(drift + 0.3 * u + 0.2 * rng.normal(size=d) if partner_teaches else rng.normal(size=d))
            src.append("partner")
        elif t % 5 == 4:
            X.append(rng.normal(size=d))
            src.append("tool")
        else:
            X.append(drift + 0.2 * rng.normal(size=d))
            src.append("self")
    return np.array(X), src


def test_cross_progress_higher_when_the_partner_carries_the_direction():
    rng = np.random.default_rng(2)
    a = by_source(*_stream(rng, 300, 8, True), floor_var=1e-3, burn_self=40)
    b = by_source(*_stream(rng, 300, 8, False), floor_var=1e-3, burn_self=40)
    assert np.nanmean(a["X"]) > np.nanmean(b["X"])


def test_no_partner_in_window_reads_zero_and_unscored_rows_are_nan():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 6))
    src = ["self"] * 120
    out = by_source(X, src, floor_var=1e-3, burn_self=40)
    scored = np.isfinite(out["L_now"])
    assert scored.sum() == 80 and not scored[:40].any()
    assert np.all(out["X"][scored] == 0.0) and np.all(out["T"][scored] == 0.0)
    assert np.all(np.isfinite(out["S"][scored]))


def test_cost_is_relative_to_the_recent_scored_chunks():
    L = np.array([np.nan, 10.0, 12.0, np.nan, 8.0, 20.0])
    c = cost(L, window=2)
    assert np.isnan(c[1]) and np.isnan(c[2]) and c[4] == -3.0 and c[5] == 10.0


def test_reported_readings_run_label_blind():
    rng = np.random.default_rng(4)
    X, src = _stream(rng, 200, 8, True)
    scored = np.zeros(200, bool)
    scored[[t for t in range(150, 200) if src[t] == "self"]] = True
    r = reported(X, src, scored, unit=0.5)
    assert np.isfinite(r["coupling"][scored]).all() and np.isfinite(r["B"][scored]).any()
    assert set(np.unique(r["opens"][scored])) <= {0.0, 1.0}


def test_world_and_echo_split_tool_progress():
    rng = np.random.default_rng(6)
    X, src = _stream(rng, 200, 8, True)
    split = [("echo" if i % 2 else "world") if s == "tool" else s for i, s in enumerate(src)]
    out = by_source(X, split, floor_var=1e-3, burn_self=40)
    scored = np.isfinite(out["L_now"])
    for k in ("X", "T", "T_world", "T_echo", "S"):
        assert np.isfinite(out[k][scored]).all()
    plain = by_source(X, src, floor_var=1e-3, burn_self=40)
    assert np.allclose(out["T"][scored], plain["T"][scored]) and np.allclose(out["X"][scored], plain["X"][scored])


# --- Step 0's matching and its placebo null (amendment 5) -----------------------------------------------------------

def _sessions(spec, rng):
    """spec: {key: (n chunks, X offset)}; one scored chunk a minute, X noise plus the session's offset."""
    import datetime as dt

    out, t0 = {}, dt.datetime(2026, 10, 1)
    for j, (key, (n, offset)) in enumerate(spec.items()):
        start = t0 + dt.timedelta(days=j)
        chunks = [{"at": (start + dt.timedelta(minutes=i)).isoformat(), "ref": f"{key}:{i}"} for i in range(n)]
        out[key] = {"chunks": chunks, "codes": [{"L_now": np.ones(n), "X": rng.normal(offset, 1, n)}]}
    return out


def _closure(key, sessions, at, back, n=10):
    """A confirmed agent closure whose first return is chunk `at`, departed `back` minutes before."""
    import datetime as dt

    ch = sessions[key]["chunks"]
    ret = dt.datetime.fromisoformat(ch[at]["at"])
    return {"id": f"{key}-{at}", "ord": key, "time": (ret - dt.timedelta(minutes=back)).isoformat(),
            "return": ch[at]["at"], "label": {"real": ["yes"], "kind": ["closure"], "who": ["agent"]}}


def test_controls_sit_outside_every_departure_and_fall_back_to_the_nearest_session():
    from body import step0 as s0

    rng = np.random.default_rng(0)
    ses = _sessions({"A": (400, 0.0), "B": (40, 0.0)}, rng)
    deps = [_closure("A", ses, 200, 30), _closure("B", ses, 35, 34)]
    w = s0.windows(ses, deps)
    a, b = sorted(w["events"], key=lambda e: e["ord"])
    for e in (a, b):
        assert len(e["rows"]) == 10 and len(e["control"]) == 10
    assert a["control_ord"] == "A" and b["control_ord"] == "A"            # B has no room: the nearest session
    assert w["cross-session controls"] == ["B-35/closure from A"]
    span = range(200 - 30 - 60, 200 + 60 + 1)
    assert not set(a["control"]) & set(span) and not set(a["control"]) & set(b["control"])


def test_the_placebo_sits_at_half_without_bias_and_moves_when_the_matching_does():
    """The null must be able to fail: controls drawn from a session whose X runs higher push it, as they would the
    reading."""
    from body import step0 as s0

    rng = np.random.default_rng(1)
    fair = _sessions({f"S{j}": (600, 0.0) for j in range(6)}, rng)
    deps = [_closure(f"S{j}", fair, 300, 20) for j in range(6)] + [_closure(f"S{j}", fair, 500, 20) for j in range(6)]
    clo = s0.windows(fair, deps)["events"]
    null = s0.placebo_null(fair, deps, clo, [], [0], 0, draws=40)
    assert abs(null["closures"]["mean"] - 0.5) < 0.1

    spec = {"BIG": (2000, 3.0), **{f"s{j}": (30, 0.0) for j in range(12)}}  # short sessions: controls come from BIG
    biased = _sessions(spec, rng)
    deps = [_closure(f"s{j}", biased, 28, 27) for j in range(12)]
    clo = s0.windows(biased, deps)["events"]
    assert all(e["control_ord"] == "BIG" for e in clo)
    null = s0.placebo_null(biased, deps, clo, [], [0], 0, draws=40)
    assert null["closures"]["mean"] > 0.9                                # the bias shows in the null


def test_same_session_pairs_leave_out_cross_session_controls():
    from body import step0 as s0

    rng = np.random.default_rng(2)
    ses = _sessions({"A": (400, 0.0), "B": (40, 0.0)}, rng)
    deps = [_closure("A", ses, 200, 30), _closure("B", ses, 35, 34)]
    clo = s0.windows(ses, deps)["events"]
    r = s0.reading(ses, [e for e in clo if e.get("control_ord", e["ord"]) == e["ord"]], [0], 0, True,
                   np.random.default_rng(0))
    assert r["n"] == 1


def test_a_session_with_no_chunks_reads_as_empty(tmp_path):
    """The 0.6B rehearsal crashed on one; the text-only dry run never reads states, so it could not catch it."""
    from body import step0 as s0
    from substrate.record import Record

    class Reader:
        info = {"width": 8}

    S = s0.read_states(Reader(), [], [1, 2], Record(tmp_path / "r.jsonl"), "states-empty")
    assert S.shape == (0, 2, 8)


def test_the_floor_file_takes_the_midpoint_of_repeats_95th_and_news_5th():
    from body.floor_file import floors

    kinds = np.array(["exact repeat"] * 20 + ["near repeat"] * 20 + ["new"] * 100 + ["between"] * 10)
    dist = np.r_[np.zeros(20), np.linspace(0.1, 0.2, 20), np.linspace(0.5, 1.5, 100), np.full(10, 9.0)][:, None]
    want = 0.5 * (np.percentile(dist[:40, 0], 95) + np.percentile(dist[40:140, 0], 5))
    assert floors(dist, kinds, [24]) == {"24": want}


def test_shape_by_source_codes_a_chunk_against_its_own_source_only():
    from body.stepm1 import shape, shape_by_source
    rng = np.random.default_rng(0)
    n, d = 60, 30
    slots = ["world"] + ["self", "world"] * ((n - 1) // 2)
    turns = [-1] + [t for t in range(30) for _ in (0, 1)]
    S = np.zeros((len(slots), 1, d))
    for i, s in enumerate(slots):                     # two sources far apart: commands near 0, replies near 10
        S[i, 0] = rng.normal(size=d) + (0 if s == "self" else 10)
    S[-1, 0] = rng.normal(size=d) + 10                # an ordinary reply, last
    joint = shape(S, turns, [1.0])[-1, 0]
    by_src = shape_by_source(S, turns, slots, [1.0])[-1, 0]
    assert np.isfinite(by_src) and by_src < joint     # under replies alone, a reply is no longer far from the mean
    assert np.isnan(shape_by_source(S, turns, slots, [1.0])[: 2 * 10, 0]).all()   # the burn-in never scores
