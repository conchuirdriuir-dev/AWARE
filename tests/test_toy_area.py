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
