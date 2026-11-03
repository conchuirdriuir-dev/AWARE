"""The body readout's progress by source (body/progress.py): MomentFit agrees with T2's Fit, and the three progress
readings behave as Step 0 registers them on toy streams."""
import numpy as np

from body.progress import GramFit, by_source, code, cost
from goalspace.formula import Fit


def test_gramfit_matches_fit_below_and_above_width():
    rng = np.random.default_rng(0)
    for n, d in ((30, 50), (200, 20)):
        X = rng.normal(size=(n + 1, d)) @ rng.normal(size=(d, d)) * 0.3 + rng.normal(size=d)
        a = Fit(X[:n], 1e-3)
        b = GramFit(X @ X.T, np.arange(n), d, 1e-3)
        assert a.choose_k() == b.choose_k()
        for k in (0, 1, 3, a.choose_k()):
            assert np.isclose(a.bits(X[n], k), b.bits(n, k), rtol=1e-7, atol=1e-6)
            assert np.isclose(a.code_length(k), b.code_length(k), rtol=1e-7, atol=1e-5)


def test_gramfit_on_a_subset_equals_fit_on_those_rows():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(81, 12))
    keep = np.setdiff1d(np.arange(80), [70, 72, 75])
    a = Fit(X[keep], 1e-3)
    assert np.isclose(code(X @ X.T, keep, 80, 12, 1e-3)[0], a.bits(X[80], a.choose_k()))


def _stream(rng, n, d, partner_teaches):
    """Self chunks drift along a direction the partner announces (or does not); tool chunks are noise."""
    u = np.zeros(d)
    u[0] = 1.0
    X, src = [], []
    for t in range(n):
        drift = 0.08 * t * u
        if t % 5 == 0:
            X.append(drift + 0.3 * u + 0.2 * rng.normal(size=d) if partner_teaches else rng.normal(size=d))
            src.append("partner")
        elif t % 5 == 4:
            X.append(rng.normal(size=d))
            src.append("tool")
        else:
            X.append(drift + 0.2 * rng.normal(size=d))
            src.append("self")
    return np.array(X), src


def test_cross_progress_higher_when_the_partner_carries_the_direction():
    rng = np.random.default_rng(2)
    a = by_source(*_stream(rng, 300, 8, True), floor_var=1e-3, burn_self=40)
    b = by_source(*_stream(rng, 300, 8, False), floor_var=1e-3, burn_self=40)
    assert np.nanmean(a["X"]) > np.nanmean(b["X"])


def test_no_partner_in_window_reads_zero_and_unscored_rows_are_nan():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 6))
    src = ["self"] * 120
    out = by_source(X, src, floor_var=1e-3, burn_self=40)
    scored = np.isfinite(out["L_now"])
    assert scored.sum() == 80 and not scored[:40].any()
    assert np.all(out["X"][scored] == 0.0) and np.all(out["T"][scored] == 0.0)
    assert np.all(np.isfinite(out["S"][scored]))


def test_cost_is_relative_to_earlier_scored_chunks():
    L = np.array([np.nan, 10.0, 12.0, np.nan, 8.0])
    c = cost(L)
    assert np.isnan(c[1]) and c[2] == 2.0 and c[4] == -3.0
