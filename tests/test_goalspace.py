"""Goal-space T2's formula, MDL above D and scoring, on planted data (numpy only)."""
import math

import numpy as np

from goalspace import formula as F


def test_ppca_bits_match_a_full_gaussian():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 6)) @ np.diag([3, 2, 1, 0.5, 0.5, 0.5])
    fit = F.Fit(X, floor_var=0.01)
    k, lam_k, sigma2 = fit.params(2)
    C = fit.V[:k].T @ np.diag(lam_k - sigma2) @ fit.V[:k] + sigma2 * np.eye(6)
    x = rng.normal(size=6)
    r = x - fit.mu
    direct = 0.5 * (6 * math.log(2 * math.pi) + np.linalg.slogdet(C)[1] + r @ np.linalg.solve(C, r)) / math.log(2)
    assert abs(fit.bits(x, 2) - direct) < 1e-6


def test_mdl_recovers_a_planted_rank_and_the_floor_stops_it_coding_noise():
    rng = np.random.default_rng(1)
    d, n = 200, 150
    axes = np.linalg.qr(rng.normal(size=(d, 3)))[0].T
    X = rng.normal(size=(n, 3)) * [6, 5, 4] @ axes + rng.normal(scale=0.1, size=(n, d))
    assert F.Fit(X, floor_var=0.01).choose_k() == 3
    assert F.Fit(X, floor_var=100.0).choose_k() == 0          # everything below D is noise: nothing to code


def test_mdl_keeps_no_axes_for_noise_with_more_dimensions_than_states():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 1000))                          # diverse texts, no shared formula
    assert F.Fit(X, floor_var=1e-4).choose_k() <= 1


def test_prequential_gain_is_positive_for_a_shared_formula_and_not_for_noise():
    rng = np.random.default_rng(2)
    d, n = 100, 80
    axes = np.linalg.qr(rng.normal(size=(d, 2)))[0].T
    shared = rng.normal(size=(n, 2)) * [5, 4] @ axes + rng.normal(scale=0.3, size=(n, d))
    noise = rng.normal(scale=np.sqrt((25 + 16) / d + 0.09), size=(n, d))   # same total variance, no formula
    g_shared = F.prequential(shared, floor_var=0.05, start=20)["gain"][40:]
    g_noise = F.prequential(noise, floor_var=0.05, start=20)["gain"][40:]
    assert np.mean(g_shared > 0) > 0.9 and np.nanmean(g_noise) < np.nanmean(g_shared)


def test_surprisal_rises_where_the_formula_is_rewritten():
    rng = np.random.default_rng(3)
    d = 80
    a1, a2 = np.linalg.qr(rng.normal(size=(d, 4)))[0].T[:2], np.linalg.qr(rng.normal(size=(d, 4)))[0].T[2:]
    X = np.vstack([rng.normal(size=(60, 2)) * 5 @ a1, rng.normal(size=(20, 2)) * 5 @ a2])
    X += rng.normal(scale=0.3, size=X.shape)
    s = F.surprisal(F.prequential(X, floor_var=0.05, start=20)["Lk"])
    assert s[60] > np.nanpercentile(s[20:60], 95)


def test_tfidf_ignores_words_outside_its_vocabulary_and_normalises():
    tf = F.Tfidf(["the formula holds", "the goal holds"])
    V = tf(["formula goal", "unseen words only"])
    assert abs(np.linalg.norm(V[0]) - 1) < 1e-9 and not V[1].any()


def test_auc_and_adjusted_rand():
    assert F.auc([3, 4], [1, 2]) == 1.0 and F.auc([1, 2], [1, 2]) == 0.5
    assert abs(F.adjusted_rand([0, 0, 1, 1], [5, 5, 7, 7]) - 1) < 1e-12
    X = np.array([[0.0, 0], [0.1, 0], [5, 5], [5.1, 5]])
    assert abs(F.adjusted_rand(F.kmeans(X, 2), [0, 0, 1, 1]) - 1) < 1e-12


def test_axes_comparison_prefers_the_aims_own_axes_and_ties_when_axes_are_shared():
    rng = np.random.default_rng(5)
    d, n = 120, 90
    q = np.linalg.qr(rng.normal(size=(d, 6)))[0].T
    own, other = q[:3], q[3:]
    noise = lambda: rng.normal(scale=0.3, size=(n, d))  # noqa: E731
    aims = rng.normal(size=(n, 3)) * [5, 4, 3] @ own + noise()
    ctl_other = rng.normal(size=(n, 3)) * [5, 4, 3] @ other + noise()
    ctl_same = rng.normal(size=(n, 3)) * [5, 4, 3] @ own + noise()
    has = np.ones(n, dtype=bool)
    w_other = F.axes_comparison(aims, ctl_other, has, floor_var=0.05, start=40)["w"][40:]
    w_same = F.axes_comparison(aims, ctl_same, has, floor_var=0.05, start=40)["w"][40:]
    assert np.mean(w_other) > 0.9 and abs(np.mean(w_same) - 0.5) < 0.2
