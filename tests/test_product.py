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


def test_compass_cells_follow_the_v9_table():
    assert cell(1.0, 1.0) == "on track"
    assert cell(1.0, 3.0) == "progress"
    assert cell(3.0, 3.0) == "lost"
    assert cell(0.5, 3.0) == "closed loop"
    assert cell(0.5, 1.0) == "mastery"


def test_auc_counts_ties_as_half():
    assert toybed.auc([2, 3], [1]) == 1.0 and toybed.auc([1], [1]) == 0.5 and toybed.auc([0], [1]) == 0.0
