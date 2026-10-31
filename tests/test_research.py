import numpy as np

from research import gate1
from substrate.geometry import reflect, unit


def pairs(kind, rng, d=128, R=1.0, D=0.1, n=40):
    c = rng.normal(size=d)
    w = unit(rng.normal(size=d))
    X = c + R * (rng.normal(size=(n, 1)) * w + rng.normal(size=(n, d)) / np.sqrt(d) * 3)
    if kind == "mirror":
        Y = reflect(X, c, w)
    elif kind == "additive":
        Y = X + R * w
    else:
        Y = c + 0.3 * (X - c)
    jitter = lambda: rng.normal(scale=D / np.sqrt(d), size=(n, d))  # noqa: E731
    return X + jitter(), Y + jitter(), c


def test_each_planted_hypothesis_wins():
    rng = np.random.default_rng(0)
    for kind in gate1.HYPOTHESES:
        X, Y, c = pairs(kind, rng)
        assert gate1.verdict(X, Y, c, 0.1, 30)[0] == kind


def test_noise_as_large_as_the_structure_is_undecided_rather_guessed():
    X, Y, c = pairs("mirror", np.random.default_rng(0))
    assert gate1.verdict(X, Y, c, 10.0, 30)[0] == "undecided"
