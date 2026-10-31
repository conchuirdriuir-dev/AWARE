import numpy as np

import geometry as g
import toybed


def test_reflection_about_centroid_is_its_own_inverse_and_fixes_the_mirror():
    rng = np.random.default_rng(0)
    c, u, X = rng.normal(size=8), rng.normal(size=8), rng.normal(size=(5, 8))
    assert np.allclose(g.reflect(g.reflect(X, c, u), c, u), X)
    on_mirror = c + rng.normal(size=8)
    on_mirror -= ((on_mirror - c) @ g.unit(u)) * g.unit(u)
    assert np.allclose(g.reflect(on_mirror[None], c, u)[0], on_mirror)


def test_spread_recovers_the_planted_noise():
    rng = np.random.default_rng(0)
    sets = [rng.normal(size=256) + toybed.noise(rng, 50, 256, 0.3) for _ in range(20)]
    assert abs(g.spread(sets) - 0.3) < 0.03
    assert g.between(sets) > 10 * g.spread(sets)


def test_description_length_recovers_the_planted_axes_from_a_short_window():
    rng = np.random.default_rng(0)
    for D in (0.05, 0.2, 0.5):
        A = toybed.orthonormal(rng, 3, 512)
        X = rng.normal(size=(20, 3)) @ A + toybed.noise(rng, 20, 512, D)
        code = g.fit_code(X)
        assert len(code.axes) == 3
        assert g.alignment(code.axes, A) > 0.9


def test_alignment_is_one_for_the_same_space_and_zero_for_orthogonal_ones():
    rng = np.random.default_rng(0)
    Q = toybed.orthonormal(rng, 4, 16)
    rot = toybed.orthonormal(rng, 2, 2)
    assert np.isclose(g.alignment(Q[:2], rot @ Q[:2]), 1.0)
    assert np.isclose(g.alignment(Q[:2], Q[2:]), 0.0)


def test_mirror_and_additive_are_told_apart_when_noise_is_small():
    rng = np.random.default_rng(0)
    for kind in ("mirror", "additive"):
        c = toybed.landmarks(rng, 256, 0.1).mean(0)
        X, Y = toybed.pairs(kind, rng, c, 1.0, 0.1)
        assert g.mirror_or_additive(X, Y, c, 0.1, 30)[0] == kind


def test_noise_as_large_as_the_structure_is_undecided_rather_guessed():
    rng = np.random.default_rng(0)
    c = toybed.landmarks(rng, 256, 0.1).mean(0)
    X, Y = toybed.pairs("mirror", rng, c, 1.0, 0.1)
    assert g.mirror_or_additive(X, Y, c, 10.0, 30)[0] == "undecided"
