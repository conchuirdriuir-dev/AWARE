"""The toy bed's plumbing. Whether the detectors pass is the experiment (`python -m product.toybed`), not a test."""
import numpy as np

from product import detectors, toybed
from product.compass import cell


def test_every_structure_generates_a_trajectory_with_its_externals():
    rng = np.random.default_rng(0)
    for kind in toybed.KINDS:
        run = toybed.generate(kind, rng, T=160, t0=80)
        assert run["s"].shape == run["E"].shape == (160, 32)
        assert np.isfinite(run["s"]).all()


def test_proliferation_externals_are_generated_from_the_entity():
    run = toybed.generate("closed loop (proliferation)", np.random.default_rng(0))
    t0 = run["t0"]
    assert np.linalg.norm(run["E"][t0 + 10] - run["s"][t0 + 10]) < 1.0
    assert np.linalg.norm(run["E"][t0 - 10] - run["s"][t0 - 10]) > 1.0


def test_point_contrast_sees_a_topic_shift_that_progress_does_not_make():
    rng = np.random.default_rng(1)
    topic = detectors.contrasts(toybed.generate("topic shift", rng), 1.0)["point persists"]
    calm = detectors.contrasts(toybed.generate("progress", rng), 1.0)["point persists"]
    assert topic > 2 * calm


def test_the_observer_reads_every_block_and_admits_into_both_codes():
    run = toybed.generate("progress", np.random.default_rng(2))
    blocks = detectors.observe(run["s"], run["E"], run["c"], 1.0)
    assert [b["start"] for b in blocks] == list(range(0, 240, 20))
    assert sum(b["growth"] for b in blocks) > 0
    assert all(b["margin"] == b["frame"] - b["self"] for b in blocks)


def test_round_one_compass_cells_follow_the_v9_table():
    from product.compass import cell_v9
    assert cell_v9(1.0, 1.0) == "on track"
    assert cell_v9(1.0, 3.0) == "progress"
    assert cell_v9(3.0, 3.0) == "lost"
    assert cell_v9(0.5, 3.0) == "closed loop"
    assert cell_v9(0.5, 1.0) == "mastery"


def test_auc_counts_ties_as_half():
    assert toybed.auc([2, 3], [1]) == 1.0 and toybed.auc([1], [1]) == 0.5 and toybed.auc([0], [1]) == 0.0


# round 2: v10's detectors

from product import coupling as cp  # noqa: E402
from product import round2  # noqa: E402


def test_echo_externals_repeat_the_entity_at_the_lag():
    run = toybed.generate("closed loop (proliferation)", np.random.default_rng(0), T=200, t0=100, lag=10)
    assert np.linalg.norm(run["E"][150] - run["s"][140]) < 1.0


def test_valid_inference_repeats_the_last_external_during_the_pause():
    run = toybed.generate("valid inference", np.random.default_rng(0), T=200, t0=100, pause=40)
    assert np.array_equal(run["E"][100], run["E"][139]) and not np.array_equal(run["E"][140], run["E"][99])


def test_coupling_reads_lower_for_an_echo_than_for_a_real_external():
    rng = np.random.default_rng(3)
    echo = [round2.window(round2.make("proliferation, lag 10", rng), 200)["coupling"] for _ in range(10)]
    real = [round2.window(round2.make("progress", rng), 200)["coupling"] for _ in range(10)]
    assert np.mean(real) > np.mean(echo) + 0.03


def test_admission_keeps_a_topic_candidate_and_rejects_a_noise_candidate():
    rng = np.random.default_rng(4)
    floor = 2.0
    assert cp.contrast(round2.make("topic shift", rng), floor)["admitted"]
    assert not cp.contrast(round2.make("noise spike", rng), floor)["admitted"]


def test_v10_compass_has_four_cells_and_no_band():
    assert cell(False, False) == "on track" and cell(False, True) == "closed loop"
    assert cell(True, False) == "progress" and cell(True, True) == "lost"


# round 3: sparse externals, at zero against the post-intake reference, the probe

from product import round3  # noqa: E402


def test_round_three_externals_are_sparse_after_the_intake_and_forced_during_the_probe():
    run = round3.make("progress", np.random.default_rng(0))
    m = run["m"]
    assert m[:round3.INTAKE].all() and m[round3.PROBE:round3.PROBE + round3.PROBE_LEN].all()
    assert 0.3 < m[round3.INTAKE:round3.PROBE].mean() < 0.5
    assert not run["E"][~m].any()                                  # the observer sees nothing where none arrived


def test_round_three_echoes_arrive_as_the_entity_at_the_lag():
    run = round3.make("proliferation, lag 3", np.random.default_rng(1))
    k = next(k for k in range(round3.T0 + 5, round3.T) if run["m"][k])
    assert np.linalg.norm(run["E"][k] - run["s"][k - 3]) < 1.0


def test_readings_follow_the_bounds_against_delta():
    assert round3.reading({"low": 0.2, "high": 0.4}, 0.1) == "open"
    assert round3.reading({"low": -0.1, "high": 0.05}, 0.1) == "at zero"
    assert round3.reading({"low": 0.05, "high": 0.2}, 0.1) == "undecided"
    assert round3.reading({"low": 0.2, "high": 0.4}, None) == "no reference"


def test_round_three_coupling_recovers_when_an_external_arrives_every_step(monkeypatch):
    monkeypatch.setattr(round3, "ARRIVAL", 1.0)
    rng = np.random.default_rng(2)
    healthy = [round3.coupling(round3.make("progress", rng), 300)["coupling"] for _ in range(5)]
    w = round3.coupling(round3.make("progress", rng), 300)
    assert np.median(healthy) > 0.05 and w["low"] <= w["coupling"] <= w["high"]


# round 4: event-locked coupling, news and gain

from product import round4  # noqa: E402


def test_round_four_news_is_small_for_an_echo_and_large_for_a_real_external():
    rng = np.random.default_rng(0)
    echo = round4.innovations(round4.make("proliferation, lag 15", rng))
    real = round4.innovations(round4.make("progress", rng))
    after = echo["t"] > round4.T0 + 5
    assert np.median(echo["news"][after]) < round4.FLOOR < np.median(real["news"])


def test_round_four_gain_is_near_zero_once_the_entity_stops_listening():
    rng = np.random.default_rng(1)
    run = round4.make("closed loop (dullness)", rng)
    inn = round4.innovations(run)
    late = np.flatnonzero((inn["t"] >= 300) & (inn["t"] + 4 <= 460))
    early = np.flatnonzero((inn["t"] >= 40) & (inn["t"] + 4 <= 200))
    assert abs(round4.gain(run, inn, late, 4)["gain"]) < 0.1 < round4.gain(run, inn, early, 4)["gain"]


def test_round_four_clusters_split_where_arrivals_are_k_apart():
    blocks = round4._clusters(np.array([10, 11, 12, 20, 21, 40]), 4)
    assert [list(b) for b in blocks] == [[0, 1, 2], [3, 4], [5]]


def test_round_four_reads_a_dull_entity_as_closed_and_progress_as_open():
    rng = np.random.default_rng(2)
    for name, want in (("closed loop (dullness)", "closed loop"), ("progress", "open")):
        run = round4.make(name, rng)
        inn = round4.innovations(run)
        ref = round4.reference(run, inn, 1)
        assert round4.read_point(run, inn, 460, 1, 4, 40, ref)["overall"] == want


# round 5: the event record

from product import round5  # noqa: E402


def test_round_five_expected_profile_peaks_at_lead_two():
    prof = round5.expected_profile()
    assert int(np.argmax(prof)) + 1 == 2 and np.isclose(prof[0], 0.5)


def test_round_five_surprise_is_large_on_a_noise_spike_and_explained_on_progress():
    rng = np.random.default_rng(3)
    spike = round4.make("noise spike", rng)
    calm = round4.make("progress", rng)
    ev_spike = round5.events(spike, round4.innovations(spike))
    ev_calm = round5.events(calm, round4.innovations(calm))
    assert ev_spike["onset surprise max"] > 5 * ev_calm["onset surprise max"]


def test_round_five_probe_is_read_by_its_own_responses():
    rng = np.random.default_rng(4)
    run = round4.make("healthy, declining", rng)
    inn = round4.innovations(run)
    ref = round4.reference(run, inn, 1)
    read = round5.probe_read(run, inn, 1, ref)
    assert read["news"] > round4.FLOOR and read["overall"] == "open"


# round 6: the residual descriptor

from product import round6  # noqa: E402


def test_round_six_residual_descriptor_sees_a_dynamics_shift_and_not_a_topic_shift():
    rng = np.random.default_rng(5)
    dyn = round4.make("dynamics shift", rng)
    top = round4.make("topic shift", rng)
    d_dyn = round6.observe(dyn, round4.innovations(dyn))["descriptor"]
    d_top = round6.observe(top, round4.innovations(top))["descriptor"]
    assert d_dyn > 4 * d_top


# round 7a: provenance and news

from product import round7a  # noqa: E402


def test_round_7a_long_news_is_low_for_replayed_content_and_short_news_is_not():
    rng = np.random.default_rng(6)
    run = round7a.make("replay", rng)
    inn = round7a.news(run)
    late = (inn["t"] >= 260) & (inn["t"] < 460)
    assert np.median(inn["long"][late]) < 2.0 < np.median(inn["news"][late])


def test_round_7a_confirmations_are_elicited_and_carry_little_short_news():
    rng = np.random.default_rng(7)
    run = round7a.make("confirmations", rng)
    inn = round7a.news(run)
    conf = inn["elicited"] & (inn["t"] < 460)
    assert conf.sum() > 20 and np.median(inn["news"][conf]) < 1.0


# round 7b: contingency and the reactive toy

from product import round7b  # noqa: E402


def test_round_7b_probe_displaces_the_entity_only_from_step_461():
    rng = np.random.default_rng(8)
    run = round7b.make("progress", rng)
    jump = run["s"][461] - run["s"][460]
    assert jump @ run["probe"] > 1.5


def test_round_7b_a_short_lag_mirror_follows_the_probe_and_a_replay_does_not():
    rng = np.random.default_rng(9)
    mirror = round7b.make("proliferation, lag 1", rng)
    stale = round7b.make("replay", rng)
    assert round7b.contingency(mirror, round7b.news(mirror)) > 0.3 > round7b.contingency(stale, round7b.news(stale))


def test_round_7b_attractor_loop_recurs_and_progress_does_not():
    rng = np.random.default_rng(10)
    loop, calm = round7b.make("attractor loop", rng), round7b.make("progress", rng)
    assert round7b.persistence(loop, round7b.news(loop))["recurrence"] > 0.3
    assert round7b.persistence(calm, round7b.news(calm))["recurrence"] < 0.05


# product step 4: labelled logs

from product import labelled_logs  # noqa: E402


def test_attention_indices_are_one_when_attention_depends_only_on_distance_and_scale_with_extra_attention():
    T = 40
    q, k = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    A = np.where(k <= q, np.exp(-(q - k) / 6.0), 0.0)[None]               # attention as a function of distance only
    spans = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40)]
    kinds = ["goal", "self", "external", "self", "external"]
    flat = labelled_logs.attention_indices(A, spans, kinds)
    assert abs(flat["G"] - 1) < 0.05 and abs(flat["S"] - 1) < 0.05
    boosted = A.copy()
    boosted[0, :, 1:8] *= 2.0                                             # twice the attention on the goal
    again = labelled_logs.attention_indices(boosted, spans, kinds)
    assert abs(again["G"] - 2) < 0.05 and abs(again["S"] - 1) < 0.05      # S is untouched by the goal's change


def test_self_attention_is_undefined_not_huge_when_no_earlier_self_chunk_exists():
    T = 40
    q, k = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    A = np.where(k <= q, np.exp(-(q - k) / 6.0), 0.0)[None]
    one = labelled_logs.attention_indices(A, [(0, 8), (8, 24), (24, 40)], ["goal", "external", "self"])
    assert one["S"] != one["S"] and abs(one["G"] - 1) < 0.05               # S is NaN; G still reads


def test_alignment_is_one_when_the_live_goal_is_nearest_and_zero_when_farthest():
    goals = np.array([[[0.0]], [[1.0]], [[5.0]]])                          # goals x layers x width
    near = np.array([[[0.1]]])
    assert labelled_logs.alignment(near, goals, 0) == 1.0
    assert labelled_logs.alignment(near, goals, 2) == 0.0
