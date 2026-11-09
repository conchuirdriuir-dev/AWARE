"""Rate's predictive test on Fault Garage (body/step0_fg.py; owner, 2026-11-08) on toy records shaped like Fault
Garage's export: X_c's scoring rule, the labels from the keys, pairs matched on position, the screen, and a score.
No model and no Fault Garage."""
import numpy as np

from body import step0_fg as F
from body import step0_habitat as Z
from body.conditional import conditional

ONSET, LEN, ASC = (60, 12), 12, (40, 45, 90)


def _record(seeds, n=100, correct=True, ascents=ASC, onset=60, stall=LEN):
    """Entries in Fault Garage's export form: per episode, an opening, then per turn the model's reply, the world's
    reply and a key with Step 0's labels; a stall of `stall` turns at `onset` repeating one command with the key on."""
    out, seq = [], 0

    def add(kind, by="code", **f):
        nonlocal seq
        out.append({"seq": seq, "kind": kind, "by": by, **f}); seq += 1
    for s in seeds:
        add("episode", seed=s, environment="fault_garage", level=2)
        add("chunk", text="Customer complaint", provenance="external", cause="spontaneous", seed=s, turn=-1,
            slot="world")
        for t in range(n):
            flat = onset <= t < onset + stall or t == onset - 1
            cmd = "measure V a b" if flat else f"lookup page {t}"
            add("chunk", by="model", text=f"COMMAND: {cmd}", provenance="self", seed=s, turn=t, slot="self")
            add("chunk", text=f"reply {t}", provenance="external", cause="elicited", seed=s, turn=t, slot="world")
            add("key", seed=s, turn=t, accepted=True, command=cmd, room="key_on", score_change=0,
                new_state=not (onset <= t < onset + stall), ascent=t in ascents, quest_left=None)
        add("end", seed=s, turns=n, correct=correct)
    return out


def test_defined_is_the_rule_conditional_scores_by():
    entries = _record([903600], n=40)
    ep = F.episodes_in(entries, "step0")[903600]
    slots, turns = [c["slot"] for c in ep["chunks"]], [c["turn"] for c in ep["chunks"]]
    X = np.random.default_rng(0).normal(size=(len(slots), 6))
    xc = conditional(X, slots, turns, 1e-3)["X_c"]
    assert F.defined(slots, turns) == list(np.isfinite(xc))
    first = min(turns[i] for i, ok in enumerate(F.defined(slots, turns)) if ok)
    assert first == 21                       # 20 earlier action-reply pairs, the first action having none before it


def test_pools_are_fault_garage_s_registered_seeds_and_their_mirrors():
    entries = _record([3500, 903599, 3600, 903999, 4000, 2999], n=5)
    assert sorted(F.episodes_in(entries, "calibration")) == [3500, 903599]
    assert sorted(F.episodes_in(entries, "step0")) == [3600, 903999]


def test_labels_read_the_export_s_keys():
    ep = F.episodes_in(_record([3600]), "step0")[3600]
    lab = F.labels({"turns": Z.turns_of(ep)}, k=5)
    assert lab["ascents"] == list(ASC) and lab["broad"] == []
    assert lab["closures"] == [(60, LEN)]            # the repeats are (command, condition) pairs
    assert F.labels({"turns": Z.turns_of(ep)}, k=LEN + 1)["closures"] == []


def _scored(seed, X=None):
    e = F.labels_only(F.episodes_in(_record([seed]), "step0"))[seed]
    return e


def test_pairs_are_matched_on_position():
    """The closure window ends at turn 59; the ascents at 40 and 45 give pre-progress windows ending at 39 and 44, both
    more than 3 turns away, so with the position rule there is no same-episode control; without it, the nearest."""
    eps = {3600: _scored(3600)}
    free = Z.events_at(eps, 5, label_fn=F.labels)
    near = Z.events_at(eps, 5, label_fn=F.labels, within=F.WITHIN)
    assert len(free["closures"]) == 1 and free["closures"][0]["control"][-1] == 44
    assert near["closures"] == [] and len(near["no control"]) == 1
    # an episode in the pool whose ascent falls just after turn 59 supplies a matched control (cross-episode)
    eps[3601] = F.labels_only(F.episodes_in(_record([3601], ascents=(30, 61), onset=80), "step0"))[3601]
    near = Z.events_at(eps, 5, label_fn=F.labels, within=F.WITHIN)
    pair = [c for c in near["closures"] if c["seed"] == 3600][0]
    assert pair["control_seed"] == 3601 and abs(pair["control"][-1] - pair["rows"][-1]) <= F.WITHIN


def test_the_screen_counts_kept_events_and_projects_them():
    entries = _record(range(903500, 903504), stall=4) + _record(range(903600, 903604), ascents=(40, 58, 90))
    r = F.screen(entries)
    assert r["k"] == 5                             # from the calibration pool's resolved stalls of 4
    assert r["calibration episodes"] == 4 and r["Step 0 episodes"] == 4
    assert r["closures kept"] == 4 and r["projected closures"] == 400       # each closure paired in its own episode
    assert r["ascents kept"] == 4 * 3 and r["projected ascents"] == 1200    # post-ascent windows of 3 or more
    assert r["won"] == 1.0 and r["meets"] is True
    lost = _record(range(903500, 903504), stall=4, correct=False) + _record(range(903600, 903604), correct=False,
                                                                             ascents=(40, 58, 90))
    r = F.screen(lost)
    assert r["won"] == 0.0 and r["meets"] is False                          # under 10% diagnosed correctly


def test_score_reads_x_c_and_passes_on_a_planted_signal():
    """X_c low before each stall, baseline before progress: reading 1 well above the bar, decided on enough events."""
    rng = np.random.default_rng(1)
    entries = _record(range(3600, 3720), ascents=(40, 58, 90))
    eps = {}
    for s, ep in F.episodes_in(entries, "step0").items():
        n = len(ep["chunks"])
        slots, turns = [c["slot"] for c in ep["chunks"]], [c["turn"] for c in ep["chunks"]]
        ok = F.defined(slots, turns)
        xc = np.where(ok, rng.normal(1.0, 0.3, n), np.nan)
        for i, c in enumerate(ep["chunks"]):
            if ok[i] and c["slot"] == "self" and ONSET[0] - 10 <= c["turn"] < ONSET[0]:
                xc[i] -= 1.0
        sp = {"X_c": xc, "S_c": xc * 0}
        eps[s] = F.episode(ep, [sp, {"X_c": np.where(ok, rng.normal(1.0, 0.3, n), np.nan), "S_c": xc * 0}])
    r = Z.score(eps, 5, 1, placebo_draws=5, draws=50, label_fn=F.labels, within=F.WITHIN, variants=False)
    r1 = r["reading 1, closures"]
    assert r1["n"] == 120 and r1["AUC"] > 0.9 and r1["text AUC"] < 0.75
    assert all(abs(c["control"][-1] - c["rows"][-1]) <= F.WITHIN for c in Z.events_at(
        eps, 5, label_fn=F.labels, within=F.WITHIN)["closures"])
    assert "broad ascents" not in r["reported"]
