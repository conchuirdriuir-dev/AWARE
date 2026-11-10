"""Toy worlds for "Meaning as Transport": checks that bag, order, holonomy and flux are all signed area.

    python -m proposals.toy_area            # all four checks, prints a table, exits 1 if any fails
    python -m proposals.toy_area --quick    # smaller sizes (what the unit test runs)
    python -m proposals.toy_area --stage-a material/area_noise_8b.json [--states runs/area-noise-8b/states.npy]
                                            # area-sense.md's stage A: loop against dither at AWARE's size
    python -m proposals.toy_area --stage-a2 [--out material/area_stage_a2.json]
                                            # stage A2 (section 7): a toy that can fail the way stage B did

No language and no model: each check is built so its answer is known in closed form. It checks the math of the
doc's "One quantity: signed area" section before any reading on text relies on it. Nothing here is a sense or a
step's test; it decides nothing in PLAN.md.

  1. bag      The mean of a path (minus its start) is the (i, t) term of the depth-2 signature of (t, X).
              Shuffling the interior points (ends held) keeps that term exactly and changes the area.
  2. area     Level 2 of the log-signature is the Levy area; reversing a path flips it, and the symmetric part of
              level 2 is fixed by level 1 (shuffle identity), so it carries no order.
  3. holonomy A fixed connection A_1, A_2 carried round a circle: holonomy = exp(area * [A_1, A_2]) to leading
              order (error per unit area falls with the radius), and the depth-N signature expansion stays within
              the closed-form tail bound at depths 1-6 (transport is a function of the signature).
  4. flux     dX = -(k I + w J) X dt + s dW in 2-D. Mean area rate = -w s^2 / (2k) in closed form: zero for a pure
              landscape (w = 0) and proportional to the flux w otherwise. (Reversing a path flips its area: check 2.)
              The four w share one seed, so their errors are correlated, not independent confirmations.

Numpy only. Tolerances are set from the closed forms before running and printed beside each result.
"""
from __future__ import annotations

import argparse
import math
import sys

import numpy as np


# --------------------------------------------------------------------------------------------------------------
# Signatures of piecewise-linear paths (depth <= 3) via Chen's identity, and the Levy area
# --------------------------------------------------------------------------------------------------------------
def signature(path, depth=3):
    """Levels 1..depth of the signature of a piecewise-linear path, as a list of arrays (d,), (d, d), (d, d, d), ...
    Built segment by segment with Chen's identity: S <- S (x) exp(dx), where exp(dx) has level k = dx^(x)k / k!."""
    path = np.asarray(path, dtype=np.float64)
    d = path.shape[1]
    S = [np.ones(())] + [np.zeros((d,) * k) for k in range(1, depth + 1)]
    for dx in np.diff(path, axis=0):
        E = [np.ones(())]
        for k in range(1, depth + 1):
            E.append(np.multiply.outer(E[-1], dx) / k)
        S = [sum(np.multiply.outer(S[j], E[k - j]) for j in range(k + 1)) for k in range(depth + 1)]
    return S[1:]


def levy_area(path):
    """Antisymmetric part of level 2: A_ij = (S_ij - S_ji) / 2. Equals the level-2 log-signature coefficient."""
    s2 = signature(path, 2)[1]
    return (s2 - s2.T) / 2.0


def expm(M):
    """Matrix exponential by scaling and squaring with a Taylor series (numpy only; fine for small matrices)."""
    M = np.asarray(M, dtype=np.float64)
    n = max(0, int(np.ceil(np.log2(max(np.abs(M).sum(axis=1).max(), 1e-300)))) + 1)
    A = M / 2.0 ** n
    out, term = np.eye(len(M)), np.eye(len(M))
    for k in range(1, 20):
        term = term @ A / k
        out = out + term
    for _ in range(n):
        out = out @ out
    return out


# --------------------------------------------------------------------------------------------------------------
# The four checks. Each returns a list of (name, value, bar, passed) rows.
# --------------------------------------------------------------------------------------------------------------
def check_bag(seed=0, n=9, d=3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    t = np.linspace(0.0, 1.0, n)[:, None]
    s2 = signature(np.hstack([X, t]), 2)[1]                 # coordinates 0..d-1 are X, coordinate d is time
    term = s2[:d, d]                                         # S^{(i, t)} = int (X^i_u - X^i_0) du
    mean_pl = (((X[:-1] + X[1:]) / 2.0) * np.diff(t, axis=0)).sum(axis=0)   # mean of the piecewise-linear path
    err = float(np.abs(term - (mean_pl - X[0])).max())

    # Same points, other order, ends held: the time-weighted mean gives the two ends half weight, so only an
    # interior shuffle leaves the bag term itself unchanged (shuffling the ends moves it as much as the area).
    perm = np.r_[0, 1 + rng.permutation(n - 2), n - 1]
    Y = X[perm]
    bag_moved = float(np.abs(term - signature(np.hstack([Y, t]), 2)[1][:d, d]).max())
    area_moved = float(np.abs(levy_area(X) - levy_area(Y)).max())
    return [
        ("bag: S^(i,t) = mean - start, max error", err, "< 1e-12", err < 1e-12),
        ("bag: interior reorder moves S^(i,t) by", bag_moved, "< 1e-12", bag_moved < 1e-12),
        ("bag: interior reorder moves the Levy area by", area_moved, "> 0.1", area_moved > 0.1),
    ]


def check_area(seed=1, n=12, d=4):
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.normal(size=(n, d)), axis=0)
    s1, s2 = signature(X, 2)
    A, Ar = levy_area(X), levy_area(X[::-1])
    flip = float(np.abs(A + Ar).max())                       # reversal flips the area
    sym = float(np.abs((s2 + s2.T) / 2.0 - np.outer(s1, s1) / 2.0).max())   # symmetric part = level 1 squared
    return [
        ("area: reversed path has area -A, max error", flip, "< 1e-9", flip < 1e-9),
        ("area: sym. part of level 2 = (level 1)^2 / 2", sym, "< 1e-9", sym < 1e-9),
    ]


def _transport(A1, A2, P):
    """Exact transport of a fixed linear connection along a polyline P in the plane."""
    H = np.eye(len(A1))
    for d in np.diff(P, axis=0):
        H = expm(A1 * d[0] + A2 * d[1]) @ H
    return H


def _rho(sig, mats):
    """Holonomy from the signature: sum over words of S^{word} * A_{word} (later letters act on the left)."""
    m = len(mats[0])
    out = np.eye(m)
    for level, S in enumerate(sig, start=1):
        for idx in np.ndindex(*S.shape):
            M = np.eye(m)
            for i in idx:
                M = mats[i] @ M
            out = out + S[idx] * M
    return out


def check_holonomy(seed=2, m=3, radii=(0.3, 0.1, 0.03), steps=2000):
    rng = np.random.default_rng(seed)
    A1, A2 = rng.normal(size=(2, m, m))
    F = A2 @ A1 - A1 @ A2                                    # curvature of a fixed connection, in this ordering
    rows, ratios = [], []
    th = np.linspace(0.0, 2 * np.pi, steps + 1)
    for r in radii:
        P = r * np.c_[np.cos(th), np.sin(th)]
        H = _transport(A1, A2, P)
        area = levy_area(P)[0, 1]                            # = pi r^2, counter-clockwise
        ratio = float(np.abs(H - expm(area * F)).max() / area)
        ratios.append(ratio)
        rows.append((f"holonomy: |H - exp(area F)| / area, r = {r}", ratio, "falls with r", None))
    falls = all(b < a * 0.6 for a, b in zip(ratios, ratios[1:]))
    rows = [(n, v, b, falls) for n, v, b, _ in rows]

    # Transport is a function of the signature: the depth-N expansion converges to the exact transport. Bar, fixed
    # from the closed form: |H - rho(S_N)| <= sum_{k > N} (L a)^k / k!, with L the path's l1 length and
    # a = max_i ||A_i||_2. The path is scaled so that L a = 2, where the bound is informative.
    P = np.cumsum(rng.normal(size=(30, 2)), axis=0)
    a = max(np.linalg.norm(A1, 2), np.linalg.norm(A2, 2))
    P = P * 2.0 / (np.abs(np.diff(P, axis=0)).sum() * a)
    H = _transport(A1, A2, P)
    depths = range(1, 7)
    sig = signature(P, max(depths))
    for N in depths:
        e = float(np.abs(H - _rho(sig[:N], (A1, A2))).max())
        tail = float(sum(2.0 ** k / math.factorial(k) for k in range(N + 1, 40)))
        rows.append((f"holonomy: |H - rho(S_depth{N})| on an open path", e, f"<= tail bound {tail:.2g}", e <= tail))
    return rows


def area_rate(k, w, s=1.0, chains=64, T=400.0, dt=0.01, burn=20.0, seed=3, reverse=False):
    """Mean Levy-area rate of dX = -(k I + w J) X dt + s dW, J = [[0, -1], [1, 0]], over parallel chains."""
    rng = np.random.default_rng(seed)
    B = k * np.eye(2) + w * np.array([[0.0, -1.0], [1.0, 0.0]])
    x = rng.normal(scale=s / np.sqrt(2 * k), size=(chains, 2))   # start at the stationary spread
    n_burn, n = int(burn / dt), int(T / dt)
    xs = np.empty((n + 1, chains, 2))
    for i in range(n_burn + n):
        x = x - dt * x @ B.T + s * np.sqrt(dt) * rng.normal(size=(chains, 2))
        if i >= n_burn - 1:
            xs[i - n_burn + 1] = x
    if reverse:
        xs = xs[::-1]
    dx = np.diff(xs, axis=0)
    mid = (xs[:-1] + xs[1:]) / 2.0                           # Stratonovich midpoint, the area of the polyline
    a = 0.5 * (mid[..., 0] * dx[..., 1] - mid[..., 1] * dx[..., 0]).sum(axis=0) / T
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(chains))


def check_flux(k=1.0, s=1.0, ws=(0.0, 0.25, 0.5, 1.0), quick=False):
    kw = dict(chains=32, T=150.0) if quick else {}
    rows = []
    for w in ws:
        est, se = area_rate(k, w, s, **kw)
        theory = -w * s * s / (2 * k)
        ok = abs(est - theory) < 4 * se + 0.02
        rows.append((f"flux: area rate, w = {w} (theory {theory:+.3f})", est, f"theory +/- 4se ({se:.3f})", ok))
    return rows


# --------------------------------------------------------------------------------------------------------------
# Stage A of proposals/area-sense.md (section 4): can area part a loop from a dither in one 24-turn window?
# Accepted by the owner 2026-11-09; the bar and every setting below are fixed before the noise is measured.
# --------------------------------------------------------------------------------------------------------------
STAGE_A = dict(d=8, contents=4, rounds=6, burn=10, R=8, episodes=40, draws=2000, bar=0.90, seed=50000)


def area_rho(path):
    """The sense on a path already in its basis: (sigma_max(A), ||A||_F), each over the sum of squared steps."""
    A = levy_area(path)
    steps = float((np.diff(path, axis=0) ** 2).sum())
    return float(np.linalg.svd(A, compute_uv=False)[0]) / steps, float(np.linalg.norm(A)) / steps


def basis(X, R):
    """The episode's own R principal directions of the rows of X (as rate's X_c)."""
    mu = X.mean(0)
    Vt = np.linalg.svd(X - mu, full_matrices=False)[2]
    return mu, Vt[: min(R, Vt.shape[0])].T


def visit_order(kind, rng, m, rounds):
    """Loop: one fixed cycle, `rounds` times. Dither: each content `rounds` times, shuffled, never twice running."""
    if kind == "loop":
        return np.tile(rng.permutation(m), rounds)
    while True:
        o = rng.permutation(np.repeat(np.arange(m), rounds))
        if (o[1:] != o[:-1]).all():
            return o


def nearest_mean(W, start):
    """Boundary's analogue: each window state's distance to its nearest earlier state, averaged over the window."""
    return float(np.mean([np.linalg.norm(W[:i] - W[i], axis=1).min() for i in range(start, len(W))]))


def toy_episode(kind, ratio, rng, d, contents, rounds, burn, R):
    """Four contents at median distance 1 in d dimensions; burn replies are other contents; each visit adds isotropic
    noise of expected norm `ratio`. Commands are fixed points (the same text, the same state)."""
    pts = rng.normal(size=(contents + burn + contents, d))
    iu = np.triu_indices(contents, 1)
    scale = np.median(np.linalg.norm(pts[:contents, None] - pts[None, :contents], axis=-1)[iu])
    pts = pts / scale
    C, B, K = pts[:contents], pts[contents:contents + burn], pts[contents + burn:]
    noise = lambda n: rng.normal(size=(n, d)) * ratio / np.sqrt(d)
    o = visit_order(kind, rng, contents, rounds)
    world = np.vstack([B + noise(burn), C[o] + noise(len(o))])
    mu, V = basis(world, R)
    rho, frob = area_rho((world[burn:] - mu) @ V)
    inter = np.empty((2 * len(o), d))
    inter[0::2], inter[1::2] = K[o], world[burn:]
    mu2, V2 = basis(np.vstack([world[:burn], inter]), R)
    return {"rho": rho, "frob": frob, "boundary": nearest_mean(world, burn),
            "inter": area_rho((inter - mu2) @ V2)[0]}


def real_episode(kind, S, groups, rng, contents, rounds, burn, R):
    """Reported only: the 8B's own paraphrase states (texts x layers x width). Contents are four sets' means; a visit
    adds a residual (a paraphrase less its set's mean) from the other sets; burn replies are paraphrases of other
    sets. Read at full width through the episode's R-direction basis, per layer, then the mean over layers."""
    ids = rng.permutation(np.unique(groups))
    cs, others = ids[:contents], ids[contents:]
    o = visit_order(kind, rng, contents, rounds)
    out = {"rho": [], "frob": [], "boundary": []}
    pick_b = [rng.choice(np.flatnonzero(groups == g)) for g in rng.choice(others, burn, replace=False)]
    pool = np.flatnonzero(np.isin(groups, others))
    pick_r = rng.choice(pool, len(o))
    for j in range(S.shape[1]):
        X = S[:, j].astype(np.float64)
        mean = {g: X[groups == g].mean(0) for g in np.unique(groups)}
        res = X[pick_r] - np.stack([mean[g] for g in groups[pick_r]])
        world = np.vstack([X[pick_b], np.stack([mean[cs[i]] for i in o]) + res])
        mu, V = basis(world, R)
        rho, frob = area_rho((world[burn:] - mu) @ V)
        out["rho"].append(rho)
        out["frob"].append(frob)
        out["boundary"].append(nearest_mean(world, burn))
    return {k: float(np.mean(v)) for k, v in out.items()}


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def auc_interval(pos, neg, rng, draws):
    pos, neg = np.asarray(pos), np.asarray(neg)
    a = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(draws)]
    return float(np.percentile(a, 5)), float(np.percentile(a, 95))


def stage_a(noise_file, states_file=None, quick=False):
    cfg = dict(STAGE_A)
    if quick:
        cfg.update(episodes=10, draws=200)
    noise = __import__("json").loads(open(noise_file).read())
    keys = ("d", "contents", "rounds", "burn", "R")

    def pool(make):
        eps = {k: [] for k in ("loop", "dither")}
        for i in range(2 * cfg["episodes"]):
            kind = "loop" if i % 2 == 0 else "dither"
            eps[kind].append(make(kind, np.random.default_rng(cfg["seed"] + i)))
        return eps

    def table(eps, fields):
        rng = np.random.default_rng(cfg["seed"] - 1)
        return {f: (auc([e[f] for e in eps["loop"]], [e[f] for e in eps["dither"]]),
                    *auc_interval([e[f] for e in eps["loop"]], [e[f] for e in eps["dither"]], rng, cfg["draws"]))
                for f in fields}

    out = {"config": cfg, "noise": {"mean_ratio": noise["mean_ratio"], "max_ratio": noise["max_ratio"]}}
    toy = lambda r: pool(lambda kind, rng: toy_episode(kind, r, rng, *(cfg[k] for k in keys)))
    out["toy"] = table(toy(noise["mean_ratio"]), ("rho", "frob", "boundary", "inter"))
    out["toy_max_ratio"] = table(toy(noise["max_ratio"]), ("rho", "boundary"))
    if states_file:
        S, groups = np.load(states_file), np.array(noise["groups"])
        out["real"] = table(pool(lambda kind, rng: real_episode(kind, S, groups, rng, *(cfg[k] for k in keys[1:]))),
                            ("rho", "frob", "boundary"))
    out["pass"] = out["toy"]["rho"][0] >= cfg["bar"]

    print(f"noise ratio (within / between): mean over layers {noise['mean_ratio']:.3f}, "
          f"max {noise['max_ratio']:.3f}")
    names = {"rho": "area, sigma_max (the sense)", "frob": "area, Frobenius", "boundary": "boundary's analogue",
             "inter": "area on the interleaved path"}
    for part, label in (("toy", "toy at the mean ratio"), ("toy_max_ratio", "toy at the largest layer's ratio"),
                        ("real", "the 8B's own paraphrase states (reported)")):
        if part in out:
            print(f"\n{label}: AUC loop over dither [90%]")
            for f, (a, lo, hi) in out[part].items():
                print(f"  {names[f]:<30} {a:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"\nBar: the sense's AUC on the toy at the mean ratio >= {cfg['bar']}: "
          f"{'PASS' if out['pass'] else 'FAIL'} ({out['toy']['rho'][0]:.3f})")
    return out


# --------------------------------------------------------------------------------------------------------------
# Stage A2 (area-sense.md section 7, fixed 2026-11-10 before this code): the toy with what stage A lacked, so it
# can fail the way stage B did. More dimensions than the basis keeps, four contents of unequal spread, and replies
# before the window more spread than the contents. The bar is on the window's basis; the as-run basis must read
# lower, or the toy still cannot see the failure.
# --------------------------------------------------------------------------------------------------------------
STAGE_A2 = dict(d=64, shares=(0.76, 0.17, 0.08), before=11, before_spread=1.5, ratio=0.73, contents=4, rounds=6,
                R=8, episodes=40, draws=2000, seed=51000, bar=0.90, gap=0.05)


def toy_episode_a2(kind, rng, d, shares, before, before_spread, ratio, contents, rounds, R_list):
    """One episode; returns rho under the window's basis and the as-run basis for each R, and boundary's analogue."""
    Q = np.linalg.qr(rng.normal(size=(d, len(shares))))[0]                 # 3 random orthonormal directions in d
    C = (rng.normal(size=(contents, len(shares))) * np.sqrt(shares)) @ Q.T
    iu = np.triu_indices(contents, 1)
    C = C / np.median(np.linalg.norm(C[:, None] - C[None], axis=-1)[iu])
    c_rms = np.sqrt(((C - C.mean(0)) ** 2).sum(1).mean())
    B = rng.normal(size=(before, d))
    B = B - B.mean(0)
    B = C.mean(0) + B * (before_spread * c_rms / np.sqrt((B ** 2).sum(1).mean()))
    noise = lambda n: rng.normal(size=(n, d)) * ratio / np.sqrt(d)
    o = visit_order(kind, rng, contents, rounds)
    world = np.vstack([B + noise(before), C[o] + noise(len(o))])
    W = world[before:]
    out = {"boundary": nearest_mean(world, before)}
    for R in R_list:
        for name, fit in (("window", W), ("as_run", world)):
            mu, V = basis(fit, R)
            out[f"{name}_R{R}"] = area_rho((W - mu) @ V)[0]
    return out


def stage_a2(quick=False):
    cfg = dict(STAGE_A2)
    if quick:
        cfg.update(episodes=10, draws=200)
    R_list = (cfg["R"], 4, 12, 16)
    eps = {"loop": [], "dither": []}
    for i in range(2 * cfg["episodes"]):
        kind = "loop" if i % 2 == 0 else "dither"
        eps[kind].append(toy_episode_a2(kind, np.random.default_rng(cfg["seed"] + i), cfg["d"], cfg["shares"],
                                        cfg["before"], cfg["before_spread"], cfg["ratio"], cfg["contents"],
                                        cfg["rounds"], R_list))
    rng = np.random.default_rng(cfg["seed"] - 1)
    fields = [f"{n}_R{R}" for R in R_list for n in ("window", "as_run")] + ["boundary"]
    table = {}
    for f in fields:
        pos, neg = [e[f] for e in eps["loop"]], [e[f] for e in eps["dither"]]
        table[f] = (auc(pos, neg), *auc_interval(pos, neg, rng, cfg["draws"]))
    win, run_ = table[f"window_R{cfg['R']}"][0], table[f"as_run_R{cfg['R']}"][0]
    out = {"config": cfg, "table": table, "bar_pass": win >= cfg["bar"], "toy_sees_failure": win - run_ >= cfg["gap"]}
    out["pass"] = out["bar_pass"] and out["toy_sees_failure"]
    print("stage A2: AUC loop over dither [90%]")
    for f, (a, lo, hi) in table.items():
        print(f"  {f:<14} {a:.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"\nBar: window basis, R = {cfg['R']}, >= {cfg['bar']}: {'PASS' if out['bar_pass'] else 'FAIL'} ({win:.3f})")
    print(f"Toy check: as-run at least {cfg['gap']} lower: {'PASS' if out['toy_sees_failure'] else 'FAIL'} "
          f"({run_:.3f}, gap {win - run_:+.3f})")
    return out


def run(quick=False):
    rows = check_bag() + check_area() + check_holonomy() + check_flux(quick=quick)
    width = max(len(r[0]) for r in rows)
    for name, value, bar, ok in rows:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {value:>11.4g}   {bar}")
    return all(r[3] for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="smaller sizes")
    ap.add_argument("--stage-a", metavar="NOISE_JSON", help="area-sense.md's stage A, with the measured noise")
    ap.add_argument("--stage-a2", action="store_true", help="area-sense.md's stage A2 (section 7)")
    ap.add_argument("--states", help="with --stage-a: the 8B's paraphrase states, for the reported real variant")
    ap.add_argument("--out", help="with --stage-a: write the result here as JSON")
    a = ap.parse_args()
    if a.stage_a2:
        out = stage_a2(quick=a.quick)
        if a.out:
            open(a.out, "w").write(__import__("json").dumps(out, indent=1))
        sys.exit(0)
    if a.stage_a:
        out = stage_a(a.stage_a, a.states, quick=a.quick)
        if a.out:
            open(a.out, "w").write(__import__("json").dumps(out, indent=1))
        sys.exit(0)
    ok = run(quick=a.quick)
    print("all checks pass" if ok else "SOME CHECKS FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
