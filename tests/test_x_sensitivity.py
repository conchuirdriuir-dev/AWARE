"""The synthetic check of X runs, on a small space: the episode's layout, and the positive control (a strong topic
carried from the world's replies into the actions) separating the arms."""
import numpy as np

from probes import x_sensitivity as xs


def test_episodes_are_laid_out_as_the_calibration():
    X, srcs = xs.episode("category", True, 1.0, np.random.default_rng(0), d=64, turns=12)
    assert X.shape == (25, 64) and srcs[0] == "partner" and srcs[1::2] == ["self"] * 12 and srcs[2::2] == ["partner"] * 12


def test_the_contingent_action_carries_the_last_reply():
    rng = np.random.default_rng(1)
    X, _ = xs.episode("content", True, 3.0, rng, d=512, turns=6, means=(np.zeros(512), np.zeros(512)))
    reply, action = X[2], X[3]                               # turn 0's reply, turn 1's action
    other = X[4]                                             # turn 1's reply: a fresh vector
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    assert cos(reply, action) > 0.7 and abs(cos(other, action)) < 0.3


def test_a_strong_topic_separates_the_arms():
    r = xs.run(kinds=("topic",), strengths=(2.0,), worlds=6, d=256, turns=40, draws=100, estimators=("X",))
    assert r["X: topic, strength 2"]["AUC"] > 0.9


def test_x_c_sees_a_strong_one_step_copy_where_x_does_not_have_to():
    r = xs.run(kinds=("content",), strengths=(2.0,), worlds=6, d=256, turns=40, draws=100)
    assert r["X_c: content, strength 2"]["AUC"] > 0.9
