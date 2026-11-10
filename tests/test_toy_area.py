"""The toy-world checks behind "Meaning as Transport" (proposals/toy_area.py) hold on their closed forms."""
import numpy as np

from proposals import toy_area as T


def test_signature_matches_chen_on_two_segments():
    P = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    s1, s2 = T.signature(P, 2)
    assert np.allclose(s1, [1.0, 1.0])
    assert np.allclose(s2, [[0.5, 1.0], [0.0, 0.5]])         # x then y: S^{xy} = 1, S^{yx} = 0
    assert np.isclose(T.levy_area(P)[0, 1], 0.5)


def test_expm_matches_rotation():
    th = 0.7
    R = T.expm(np.array([[0.0, -th], [th, 0.0]]))
    assert np.allclose(R, [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])


def test_all_checks_pass_quick():
    assert T.run(quick=True)


def test_signature_of_one_segment_is_its_exponential():
    dx = np.array([0.3, -1.2])
    s1, s2, s3, s4 = T.signature(np.array([[0.0, 0.0], dx]), 4)
    assert np.allclose(s3, np.einsum("i,j,k->ijk", dx, dx, dx) / 6.0)
    assert np.allclose(s4, np.einsum("i,j,k,l->ijkl", dx, dx, dx, dx) / 24.0)


def test_stage_a_runs_and_area_reads_order(tmp_path):
    import json
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps({"mean_ratio": 0.05, "max_ratio": 0.05, "groups": []}))
    out = T.stage_a(str(noise), quick=True)
    assert out["toy"]["rho"][0] > 0.9                       # little noise: the loop is plain
    assert 0.2 < out["toy"]["boundary"][0] < 0.8            # same contents, same counts: boundary near chance


def test_visit_orders():
    rng = np.random.default_rng(0)
    d = T.visit_order("dither", rng, 4, 6)
    assert (np.bincount(d) == 6).all() and (d[1:] != d[:-1]).all()
    l = T.visit_order("loop", rng, 4, 6)
    assert (l[4:] == l[:-4]).all()


def test_stage_a2_runs_and_reports_both_bases():
    out = T.stage_a2(quick=True)
    assert {"window_R8", "as_run_R8", "boundary"} <= set(out["table"])
    assert out["pass"] == (out["bar_pass"] and out["toy_sees_failure"])
