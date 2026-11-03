"""Step 3's reading (PLAN.md, "A cheap read of the source"), on planted numbers: plumbing, never a result."""
import json
from pathlib import Path

import numpy as np

from research import source_read as s

ROOT = Path(__file__).resolve().parents[1]


def test_the_key_needs_the_frame_and_reads_the_coupled_reply():
    assert s.key(False, True) is None and s.key(False, False) is None
    assert s.key(True, False) == "partner" and s.key(True, True) == "own"


def test_hits_and_the_first_hit():
    m = ["deposit", "damage"]
    assert s.hits("A deposit for damage, and a deposit.", m) == 3
    assert s.first_hit("Plan it. Then the deposit.", m) == 18
    assert s.first_hit("Nothing here.", m) is None


def test_the_frame_line_is_offered_not_instructed():
    msg = s.frame_messages("Plan it.", "the weather")[0]["content"]
    assert msg == "Person: Plan it.\nPartner: I keep thinking about the weather."


def test_folds_never_split_a_conversation():
    groups = ["a", "a", "b", "c", "c", "d", "e", "f"]
    f = s.folds(groups)
    for g in set(groups):
        assert len({f[i] for i, x in enumerate(groups) if x == g}) == 1
    assert len(set(f)) == 5


def test_a_separable_signal_is_read_out_of_fold_and_noise_is_not():
    rng = np.random.default_rng(0)
    groups = [str(i // 4) for i in range(160)]
    y = rng.random(160) < 0.5
    X = rng.normal(0, 1, (160, 50))
    X[:, 3] += 3 * y
    X += 5 * rng.normal(0, 1, (40, 50))[np.arange(160) // 4]        # a large topic offset per conversation
    assert s.auc(s.out_of_fold(X, y, groups, 20), y) > 0.8
    noise = rng.normal(0, 1, (160, 50))
    assert abs(s.auc(s.out_of_fold(noise, y, groups, 20), y) - 0.5) < 0.15


def test_the_verdict_reads_both_bars_and_the_baseline():
    assert s.read_verdict(0.72, 0.9, 0.01) == "holds"
    assert s.read_verdict(0.72, 0.9, -0.01) == "undecided"          # not better than the baseline
    assert s.read_verdict(0.4, 0.69, -0.2) == "no cheap read here"
    assert s.read_verdict(0.6, 0.8, 0.1) == "undecided"


def planted(n_conv=40, partner=True):
    rows = []
    for c in range(n_conv):
        samples = []
        for k in range(8):
            f, n = k < 6, k < 3
            samples.append({"F": {"carries": f}, "N": {"carries": n}, "key": s.key(f, n)})
        rows.append({"id": str(c), "samples": samples})
    return rows


def test_the_construction_counts_each_class_and_the_frames_effect():
    checks = s.construction(planted(), True)
    assert checks["enough"]["value"] == {"partner": 120, "own": 120} and s.holds(checks)
    assert checks["frame works"]["value"] == {"F": 0.75, "N": 0.375}
    assert not s.holds(s.construction(planted(7), True))             # 7 conversations: too few of each
    assert not s.holds(s.construction(planted(), False))              # the coupling did not redraw the same


def test_cluster_bootstrap_brackets_the_whole():
    rng = np.random.default_rng(2)
    groups = [str(i // 4) for i in range(160)]
    y = rng.random(160) < 0.5
    x = rng.normal(0, 1, 160) + 1.5 * y
    low, high = s.cluster_bootstrap(groups, lambda idx: s.auc(x[idx], y[idx]), n_boot=300)
    assert low <= s.auc(x, y) <= high


def test_the_frames_match_the_frozen_goals_one_to_one():
    frames = [json.loads(x) for x in (ROOT / s.FRAMES).read_text().splitlines()]
    goals = [json.loads(x)["id"] for x in (ROOT / "material/authorship_conversations.jsonl").read_text().splitlines()]
    assert [f["id"] for f in frames] == goals
    assert all(f["by"] == "model" and len(f["markers"]) == 3 and f["frame"] for f in frames)
    dry = [json.loads(x) for x in (ROOT / s.DRY).read_text().splitlines()]
    assert len(dry) == 2 and not {d["id"] for d in dry} & set(goals)


def test_centring_removes_each_conversations_mean_and_nothing_else():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 10.0]])
    assert np.allclose(s.centred(X, ["a", "a", "b"]), [[-1, -1], [1, 1], [0, 0]])
