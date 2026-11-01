"""The formula in goal-space, and MDL above D (goalspace/t2_owner_formula.md): probabilistic PCA whose noise is
floored at D, its axes chosen by a two-part code priced within the span of the states, and prequential code lengths.

Plain numpy, so it is tested without a model. Code lengths are in bits; only their differences are read.
"""
from __future__ import annotations

import math

import numpy as np

LN2 = math.log(2.0)
KMAX = 20


class Fit:
    """PPCA on rows X with k axes: mean, top-k directions with their variances, and isotropic noise floored."""

    def __init__(self, X: np.ndarray, floor_var: float):
        X = np.asarray(X, dtype=np.float64)
        self.n, self.d = X.shape
        self.mu = X.mean(0)
        Z = X - self.mu
        _, s, vt = np.linalg.svd(Z, full_matrices=False)
        self.lam = s ** 2 / self.n                         # sample eigenvalues, largest first
        self.V = vt
        self.total = float(self.lam.sum())                 # total variance (trace of the sample covariance)
        self.floor = max(float(floor_var), 1e-12)

    @property
    def span(self) -> int:
        """The dimensions n states span: fewer states than dimensions span only n - 1 of them."""
        return max(1, min(self.d, self.n - 1))

    def params(self, k: int):
        """Noise is estimated within the span (DECISIONS.md 2026-10-30): the variance left after k axes, per spanned
        dimension, floored at D^2/d. Estimated over all d dimensions it is tiny, and every axis looks worth keeping."""
        k = min(k, len(self.lam), self.span - 1)
        rest = self.total - float(self.lam[:k].sum())
        sigma2 = max(rest / (self.span - k), self.floor)
        lam_k = np.maximum(self.lam[:k], sigma2)
        return k, lam_k, sigma2

    def loglik_bits(self, k: int) -> float:
        """Log-likelihood of the training rows within their span under the k-axis model, in bits."""
        k, lam_k, sigma2 = self.params(k)
        m = self.span
        logdet = float(np.log(lam_k).sum()) + (m - k) * math.log(sigma2)
        trace = float((self.lam[:k] / lam_k).sum()) + (self.total - float(self.lam[:k].sum())) / sigma2
        return -0.5 * self.n * (m * math.log(2 * math.pi) + logdet + trace) / LN2

    def code_length(self, k: int) -> float:
        """The two-part code: data under the model, plus the model's parameters priced within the span of the n
        states (DECISIONS.md 2026-10-30)."""
        return -self.loglik_bits(k) + 0.5 * (k * (self.n - 1) + 1) * math.log2(self.n)

    def choose_k(self, kmax: int = KMAX) -> int:
        top = max(0, min(kmax, self.n - 2, len(self.lam)))
        return int(np.argmin([self.code_length(k) for k in range(top + 1)]))

    def bits(self, x: np.ndarray, k: int) -> float:
        """Code length of a new row under the k-axis model: -log2 of its density."""
        k, lam_k, sigma2 = self.params(k)
        r = np.asarray(x, dtype=np.float64) - self.mu
        p = self.V[:k] @ r
        maha = float((p ** 2 / lam_k).sum()) + (float(r @ r) - float(p @ p)) / sigma2
        logdet = float(np.log(lam_k).sum()) + (self.d - k) * math.log(sigma2)
        return 0.5 * (self.d * math.log(2 * math.pi) + logdet + maha) / LN2

    def coords(self, X: np.ndarray, k: int) -> np.ndarray:
        return (np.asarray(X, dtype=np.float64) - self.mu) @ self.V[:k].T


def prequential(X: np.ndarray, floor_var: float, start: int, kmax: int = KMAX) -> dict:
    """Each row from `start` on, coded by the formula fitted on all rows before it: the mean-only code (k = 0), the
    chosen formula's code, and the k chosen. Rows before `start` are NaN."""
    n = len(X)
    L0, Lk, ks = np.full(n, np.nan), np.full(n, np.nan), np.full(n, -1)
    for t in range(start, n):
        fit = Fit(X[:t], floor_var)
        k = fit.choose_k(kmax)
        L0[t], Lk[t], ks[t] = fit.bits(X[t], 0), fit.bits(X[t], k), k
    return {"L0": L0, "Lk": Lk, "k": ks, "gain": L0 - Lk}


def surprisal(Lk: np.ndarray, window: int = 20) -> np.ndarray:
    """Each row's code length minus the median over the `window` rows before it (NaN where those are missing)."""
    s = np.full(len(Lk), np.nan)
    for t in range(window, len(Lk)):
        prev = Lk[t - window:t]
        if np.all(np.isfinite(prev)) and np.isfinite(Lk[t]):
            s[t] = Lk[t] - float(np.median(prev))
    return s


# --- the text baseline -----------------------------------------------------------------------------------------

def _words(text: str) -> list[str]:
    import re
    w = re.findall(r"[a-z0-9][a-z0-9'_\-]*", text.lower())
    return w + [a + " " + b for a, b in zip(w, w[1:])]


class Tfidf:
    """Word unigrams and bigrams, the vocabulary and idf fitted on `docs` only, rows L2-normalised."""

    def __init__(self, docs: list[str]):
        vocab, df = {}, {}
        for doc in docs:
            for term in set(_words(doc)):
                df[term] = df.get(term, 0) + 1
        for term in sorted(df):
            vocab[term] = len(vocab)
        self.vocab = vocab
        self.idf = np.array([math.log((1 + len(docs)) / (1 + df[t])) + 1.0 for t in sorted(df)])

    def __call__(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(self.vocab)))
        for i, text in enumerate(texts):
            for term in _words(text):
                j = self.vocab.get(term)
                if j is not None:
                    out[i, j] += 1.0
        out *= self.idf
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)


# --- scoring ---------------------------------------------------------------------------------------------------

def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    for v in np.unique(allv):                              # average ranks over ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), size=k, replace=False)]
    lab = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        new = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(2), axis=1)
        if np.array_equal(new, lab) and _ > 0:
            break
        lab = new
        for j in range(k):
            if np.any(lab == j):
                C[j] = X[lab == j].mean(0)
    return lab


def adjusted_rand(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    M = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(M, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2  # noqa: E731
    s_ij, s_a, s_b = comb(M).sum(), comb(M.sum(1)).sum(), comb(M.sum(0)).sum()
    expected = s_a * s_b / comb(len(a))
    top = (s_a + s_b) / 2
    return float((s_ij - expected) / (top - expected)) if top != expected else float("nan")


# --- reading 1, amended: the aims' own axes against the controls' axes, on the same next aim ---------------------

def bits_on_axes(X: np.ndarray, x: np.ndarray, V: np.ndarray, floor_var: float) -> float:
    """Code length of x under axes V (orthonormal rows), calibrated on the rows X: their mean, their variance along
    each axis, and what they leave off the axes per spanned dimension, floored."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    k = len(V)
    mu = X.mean(0)
    Z = X - mu
    total = float((Z ** 2).sum(1).mean())
    lam = (Z @ V.T) ** 2 if k else np.zeros((n, 0))
    lam = lam.mean(0)
    span = max(1, min(d, n - 1))
    sigma2 = max((total - float(lam.sum())) / max(span - k, 1), floor_var, 1e-12)
    lam = np.maximum(lam, sigma2)
    r = np.asarray(x, dtype=np.float64) - mu
    p = V @ r if k else np.zeros(0)
    maha = float((p ** 2 / lam).sum()) + (float(r @ r) - float(p @ p)) / sigma2
    logdet = float(np.log(lam).sum()) + (d - k) * math.log(sigma2)
    return 0.5 * (d * math.log(2 * math.pi) + logdet + maha) / LN2


def axes_comparison(A: np.ndarray, C: np.ndarray, has_c: np.ndarray, floor_var: float, start: int,
                    min_controls: int = 20, kmax: int = KMAX) -> dict:
    """For each aim t from `start`, k from MDL on the earlier aims; then, cross-fitted in two folds (earlier aims split
    by the parity of their position): the aims' top-k axes fitted on one fold's aims, the controls' top-k axes on the
    same fold's controls, and both calibrated on the other fold's aims, coding aim t; bits averaged over the folds.
    Without cross-fitting the aims' axes would be fitted on their own calibration data, an in-sample advantage.
    w = 1 if the aims' axes code aim t in fewer bits, 0 if more, 1/2 on a tie; NaN with fewer than `min_controls`
    earlier controls."""
    n = len(A)
    out = {"bits_aims_axes": np.full(n, np.nan), "bits_control_axes": np.full(n, np.nan), "w": np.full(n, np.nan),
           "k": np.full(n, -1)}
    for t in range(start, n):
        if np.count_nonzero(has_c[:t]) < min_controls:
            continue
        k = Fit(A[:t], floor_var).choose_k(kmax)
        ba = bc = 0.0
        for f in (0, 1):
            fit_idx = np.arange(f, t, 2)
            cal_idx = np.arange(1 - f, t, 2)
            ctl_idx = fit_idx[has_c[fit_idx]]
            kk = min(k, len(ctl_idx) - 1, len(fit_idx) - 1)
            Va = Fit(A[fit_idx], floor_var).V[:kk]
            Vc = Fit(C[ctl_idx], floor_var).V[:kk]
            ba += bits_on_axes(A[cal_idx], A[t], Va, floor_var) / 2
            bc += bits_on_axes(A[cal_idx], A[t], Vc, floor_var) / 2
        out["bits_aims_axes"][t], out["bits_control_axes"][t], out["k"][t] = ba, bc, k
        out["w"][t] = 1.0 if ba < bc else 0.0 if ba > bc else 0.5
    return out
