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


def free_pairs(kind, rng, d=128, n=200, topics=20, noise=0.02):
    """Pairs with no shared origin handed to the fit; the centre has to be found."""
    c, w = 5 * rng.normal(size=d), unit(rng.normal(size=d))
    X = c + rng.normal(size=(n, 1)) * w + rng.normal(size=(n, d)) * 3 / np.sqrt(d)
    Y = {"mirror": lambda: reflect(X, c, w),
         "additive": lambda: X + 3 * w,
         "contraction": lambda: c + 0.3 * (X - c)}[kind]()
    jitter = lambda: rng.normal(scale=noise / np.sqrt(d), size=(n, d))  # noqa: E731
    return X + jitter(), Y + jitter(), np.repeat(np.arange(topics), n // topics), c, w


def test_a_fitted_centre_finds_each_planted_centre():
    X, Y, g, c, w = free_pairs("contraction", np.random.default_rng(1))
    lam, v = gate1.fit_free(X, Y)["contraction"]
    assert abs(lam - 0.3) < 0.02
    assert np.linalg.norm(gate1.contraction_centre(lam, v) - c) < 0.2
    X, Y, g, c, w = free_pairs("mirror", np.random.default_rng(2))
    u, b = gate1.fit_free(X, Y)["mirror"]
    assert abs(u @ w) > 0.99
    assert gate1.distance_from_centre("mirror", (u, b), c) < 0.1


def test_no_finite_centre_near_lambda_one():
    assert gate1.contraction_centre(0.95, np.ones(3)) is None
    assert gate1.contraction_centre(0.5, np.ones(3)) is not None
    assert gate1.distance_from_centre("contraction", (0.95, np.ones(3)), np.zeros(3)) is None


def test_planted_contraction_and_mirror_win_held_out_by_topic():
    for seed, kind in enumerate(("mirror", "contraction")):
        X, Y, g, *_ = free_pairs(kind, np.random.default_rng(seed + 3))
        assert gate1.decide(gate1.held_out_by_topic(X, Y, g), 0.1) == kind


def test_a_fixed_wrong_centre_is_what_the_fitted_one_escapes():
    X, Y, g, c, w = free_pairs("contraction", np.random.default_rng(5))
    assert gate1.decide(gate1.held_out_by_topic(X, Y, g), 0.1) == "contraction"
    wrong = gate1.fixed_centre(c + 20 * unit(np.random.default_rng(6).normal(size=128)))
    assert gate1.decide(gate1.held_out_by_topic(X, Y, g, wrong), 0.1) != "contraction"


def test_identity_is_a_rival_too():
    errors = {"mirror": 1.0, "additive": 2.0, "contraction": 2.0, "identity": 1.5, "lam": 0.5}
    assert gate1.decide(errors, 0.4) == "mirror"
    assert gate1.decide(errors, 0.6) == "undecided"


def test_per_topic_scope_finds_what_global_cannot():
    rng = np.random.default_rng(7)
    d, n, topics = 128, 200, 20
    g = np.repeat(np.arange(topics), n // topics)
    shift = rng.normal(size=(topics, d))
    X = rng.normal(size=(n, d)) / np.sqrt(d)
    Y = X + shift[g] + rng.normal(scale=0.01 / np.sqrt(d), size=(n, d))
    per_topic = gate1.leave_one_out_per_topic(X, Y, g)
    across = gate1.held_out_by_topic(X, Y, g)
    assert per_topic["additive"] < 0.1 < 0.5 < across["additive"]


def test_scope_verdict_needs_three_of_four():
    assert gate1.scope_verdict(["mirror", "mirror", "mirror", "undecided"]) == "mirror"
    assert gate1.scope_verdict(["mirror", "mirror", "additive", "undecided"]) == "undecided"
    assert gate1.scope_verdict(["undecided"] * 4) == "undecided"


def test_shuffled_pairs_never_keep_a_topic_and_are_repeatable():
    g = np.repeat(np.arange(20), 10)
    p = gate1.shuffle_opposites(g)
    assert sorted(p.tolist()) == list(range(200))
    assert not (g == g[p]).any()
    assert (p == gate1.shuffle_opposites(g)).all()


def test_paraphrase_pairs_have_the_shape_of_the_negation_pairs():
    S = np.random.default_rng(8).normal(size=(20, 5, 16))
    X, Y, g = gate1.paraphrase_pairs(S)
    assert X.shape == Y.shape == (200, 16)
    assert np.bincount(g).tolist() == [10] * 20


def test_a_contraction_at_lambda_one_is_additive():
    """Contraction with lam = 1 is additive: as first written a planted additive could never clear the margin."""
    X, Y, g, *_ = free_pairs("additive", np.random.default_rng(9))
    errors = gate1.held_out_by_topic(X, Y, g)
    assert abs(errors["lam"] - 1) < 0.1 and abs(errors["additive"] - errors["contraction"]) < 0.01
    assert gate1.decide(errors, 0.1, nesting=False) == "undecided"
    assert gate1.decide(errors, 0.1) == "additive"


def test_a_contraction_away_from_lambda_one_stays_a_rival():
    errors = {"mirror": 5.0, "additive": 2.0, "contraction": 1.9, "identity": 5.0, "lam": 0.5}
    assert gate1.decide(errors, 1.0) == "undecided"
    errors["lam"] = 0.95
    assert gate1.decide(errors, 1.0) == "additive"
