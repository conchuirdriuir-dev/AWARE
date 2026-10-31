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
