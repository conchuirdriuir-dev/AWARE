"""Toy bed round 7a: provenance and news (product/rounds/round7a.md).

    python -m product.round7a                        # 200 trials a structure, seed 0
    python -m product.round7a --out runs/round7a.json

Each external carries its arrival cause, spontaneous or elicited, and news is measured short (against the last 21
states and the held external) and long (against the session's record). The partner-side loop is read on long news over
spontaneous arrivals. The blind structure's expected reading is used only to score; it is not printed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np

from product import round4 as r4
from product import round5 as r5
from product import round6 as r6
from product.toybed import auc, orthonormal, rotation

STRUCTURES = {**r4.STRUCTURES, "replay": ("replay", 0), "confirmations": ("confirmations", 0)}
LOOPS = r4.LOOPS + ("replay",)


def generate(kind: str, rng, lag: int = 0, D: float = 1.0) -> dict:
    """Round 4's generator with an arrival-cause marker, plus replay and confirmations."""
    d, T, T0, RAMP = r4.D_DIM, r4.T, r4.T0, r4.RAMP
    c0 = rng.normal(scale=10.0, size=d)
    t = np.arange(T)
    novelty = np.ones(T)
    if kind == "declining":
        novelty = 1 - 0.85 * np.clip((t - T0) / RAMP, 0, 1)
    novelty[r4.PROBE:r4.PROBE + r4.PROBE_LEN] = 1.0
    walk = np.cumsum(rng.normal(scale=0.3 * D / np.sqrt(d), size=(T, d)) * novelty[:, None], axis=0)
    if kind == "topic shift":
        walk[T0:] += 6 * D * orthonormal(rng, 1, d)[0]
    trend = np.zeros(d)
    if kind == "valid inference":
        trend = 0.3 * D * orthonormal(rng, 1, d)[0]
        walk += t[:, None] * trend
    content = c0 + walk + rng.normal(size=(T, d)) / np.sqrt(d) * (r4.JUMP * D * novelty)[:, None]
    noise_scale = np.where(t < r4.INTAKE, 2 * D, D * novelty)
    fresh = content + rng.normal(size=(T, d)) * (noise_scale / np.sqrt(d))[:, None]
    m = (rng.random(T) < r4.ARRIVAL) | (t < r4.INTAKE)
    if kind == "valid inference":
        m[T0:T0 + r4.PAUSE] = False
    m[r4.PROBE:r4.PROBE + r4.PROBE_LEN] = True
    elicited = np.zeros(T, dtype=bool)
    elicited[r4.PROBE:r4.PROBE + r4.PROBE_LEN] = True
    confirm = (rng.random(T) < 0.6) & (t >= T0) & (t < r4.PROBE)
    replay_src = [j for j in range(60, T0) if m[j]]
    I = np.eye(d)
    rot = 0.5 * I
    if kind == "dynamics shift":
        Q = orthonormal(rng, 4, d)
        R = np.zeros((4, 4))
        R[:2, :2], R[2:, 2:] = rotation(0.6, 0.9), rotation(0.6, 0.9)
        rot = 0.5 * I + Q.T @ (R - 0.5 * np.eye(4)) @ Q
    u = orthonormal(rng, 1, d)[0]
    s, E, Hd = np.zeros((T, d)), np.zeros((T, d)), np.zeros((T, d))
    s[0] = held = fresh[0]
    cycle = 0
    for k in range(T):
        if m[k]:
            if kind == "proliferation" and k >= T0:
                E[k] = s[k - lag] + rng.normal(scale=0.3 * D / np.sqrt(d), size=d)
            elif kind == "replay" and k >= T0:
                E[k] = content[replay_src[cycle % len(replay_src)]] + rng.normal(scale=D / np.sqrt(d), size=d)
                cycle += 1
            elif kind == "confirmations" and confirm[k]:
                E[k] = s[k] + rng.normal(scale=0.3 * D / np.sqrt(d), size=d)
                elicited[k] = True
            else:
                E[k] = fresh[k]
            held = E[k]
        Hd[k] = held
        if k == T - 1:
            break
        f = min(max((k - T0) / RAMP, 0.0), 1.0)
        A, beta, sigma, drift = 0.5 * I, 1.0, D, np.zeros(d)
        if kind == "dynamics shift" and k >= T0:
            A = rot
        elif kind == "dullness":
            A, beta, drift = (0.5 + 0.48 * f) * I, 1.0 - f, f * 0.3 * D * u
        elif kind == "mastery":
            A, sigma = (0.5 - 0.3 * f) * I, D * (1 - 0.7 * f)
        elif kind == "noise spike" and T0 <= k < T0 + 3:
            sigma = 20 * D
        drive = held - c0
        if kind == "valid inference" and T0 <= k < T0 + r4.PAUSE:
            drive = walk[T0 - 1] + trend * (k - T0 + 1)
        s[k + 1] = c0 + A @ (s[k] - c0) + beta * (I - A) @ drive + drift + rng.normal(scale=sigma / np.sqrt(d), size=d)
    E_in = np.where(m[:, None], 0.5 * E + 0.5 * s, 0.0)
    return {"s": s, "E": E, "m": m, "elicited": elicited, "E_in": E_in, "H": Hd, "c": fresh[:10].mean(0),
            "t0": T0, "kind": kind}


def make(name: str, rng) -> dict:
    kind, lag = STRUCTURES[name]
    return generate(kind, rng, lag)


def blind_module():
    spec = importlib.util.spec_from_file_location("blind_round7a", r5.BLIND / "round7a_structure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def news(run: dict) -> dict:
    """Short innovations (round 4's) with, for the same arrivals, long news and the arrival cause."""
    inn = r4.innovations(run)
    s, E, m = run["s"], run["E"], run["m"]
    arrived = np.flatnonzero(m)
    long = np.zeros(len(inn["t"]))
    for i, t in enumerate(inn["t"]):
        earlier = arrived[arrived < t]
        candidates = np.vstack([s[:t + 1], E[earlier]])
        long[i] = np.min(np.linalg.norm(candidates - E[t], axis=1))
    return {**inn, "long": long, "elicited": run["elicited"][inn["t"]]}


def news_at(inn: dict, point: int, N: int, use_cause: bool = True) -> dict:
    keep = (inn["t"] <= point) & (~inn["elicited"] if use_cause else True)
    long, short = inn["long"][keep][-N:], inn["news"][keep][-N:]
    upper = long.mean() + r4.Z * long.std(ddof=1) / np.sqrt(len(long))
    return {"long": float(long.mean()), "short": float(short.mean()), "at zero": bool(upper < r4.FLOOR),
            "re-presentation": bool(upper < r4.FLOOR and short.mean() > r4.FLOOR)}


def overall(news_zero: bool, gain_reading: str) -> str:
    if news_zero or gain_reading == "at zero":
        return "closed loop"
    if gain_reading == "open":
        return "open"
    return "no reference" if gain_reading == "no reference" else "undecided"


def gain_reading(g: dict, ref: dict | None, need_budget: bool = True) -> str:
    if ref is None:
        return "no reference"
    if need_budget and not g["met"]:
        return "undecided"
    if g["gain"] - r4.Z * g["se"] > ref["delta"]:
        return "open"
    if g["gain"] + r4.Z * g["se"] < ref["delta"]:
        return "at zero"
    return "undecided"


def read_point(run, inn, point, K, N, M, ref, use_cause=True) -> dict:
    nw = news_at(inn, point, N, use_cause)
    g = r4.gain_at(run, inn, point, K, M, ref)
    gr = gain_reading(g, ref)
    return {"overall": overall(nw["at zero"], gr), "gain reading": gr, "gain": g["gain"], "long news": nw["long"],
            "short news": nw["short"], "news at zero": nw["at zero"], "re-presentation": nw["re-presentation"]}


def probe_read(run, inn, K, ref) -> dict:
    idx = np.flatnonzero((inn["t"] >= r4.PROBE) & (inn["t"] < r4.PROBE + r4.PROBE_LEN))
    long = inn["long"][idx]
    news_zero = bool(long.mean() + r4.Z * long.std(ddof=1) / np.sqrt(len(long)) < r4.FLOOR)
    g = {**r4.gain(run, inn, idx, K), "met": True}
    gr = gain_reading(g, ref, need_budget=False)
    return {"overall": overall(news_zero, gr), "gain reading": gr, "long news": float(long.mean())}


def tune(n: int, seed: int, K: int) -> dict:
    """Round 4's rules for N and M, with news at zero now on long news over spontaneous arrivals."""
    rng = np.random.default_rng(seed + 50_000)
    prog = [make("progress", rng) for _ in range(n)]
    mast = [make("mastery", rng) for _ in range(n)]
    prog_inn, mast_inn = [news(r) for r in prog], [news(r) for r in mast]
    n_sweep = {N: float(np.mean([news_at(i, r4.READ, N)["at zero"] for i in prog_inn])) for N in r4.NS}
    N = next((N for N in r4.NS if n_sweep[N] <= 0.1), r4.NS[-1])
    m_sweep = {M: [float(np.mean([r4.gain_at(r, i, r4.READ, K, M, r4.reference(r, i, K))["met"]
                                  for r, i in zip(runs, inns)])) for runs, inns in ((prog, prog_inn), (mast, mast_inn))]
               for M in r4.MS}
    M = next((M for M in r4.MS if min(m_sweep[M]) >= 0.9), r4.MS[-1])
    return {"N": N, "M": M, "N sweep": n_sweep, "M sweep": m_sweep}


def calibrate_separation(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 10_000)
    ob = []
    for _ in range(n):
        run = make("progress", rng)
        ob.append(r6.observe(run, r4.innovations(run)))
    return {"onset news": float(np.percentile([o["onset news max"] for o in ob], 99)),
            "onset surprise": float(np.percentile([o["onset surprise max"] for o in ob], 99)),
            "later surprise": float(np.percentile([o["later surprise median"] for o in ob], 99)),
            "descriptor": float(np.percentile([o["descriptor"] for o in ob], 99))}


def read(run: dict, name: str, cal: dict) -> dict:
    K, N, M, th = cal["K"], cal["N"], cal["M"], cal["thresholds"]
    inn = news(run)
    ref = r4.reference(run, inn, K)
    first = read_point(run, inn, r4.READ, K, N, M, ref)
    probed = first["overall"] == "undecided"
    final = probe_read(run, inn, K, ref) if probed else first
    ob = r6.observe(run, inn)
    news_spike = ob["onset news max"] > th["onset news"]
    if news_spike and first["gain reading"] != "at zero":
        label = "topic shift"
    elif (not news_spike and ob["onset surprise max"] > th["onset surprise"]
          and ob["later surprise median"] <= th["later surprise"]):
        label = "noise spike"
    else:
        label = "progress"
    out = {"label": label, "first": first, "final": final, "probed": probed,
           "descriptor fires": ob["descriptor"] > th["descriptor"],
           "confirmations": int(np.sum(inn["elicited"] & (inn["news"] < r4.FLOOR) & (inn["t"] < r4.PROBE)))}
    if name == "confirmations":
        blind_first = read_point(run, inn, r4.READ, K, N, M, ref, use_cause=False)
        out["ignoring cause"] = {"first": blind_first["overall"],
                                 "final": probe_read(run, inn, K, ref)["overall"] if blind_first["overall"] == "undecided"
                                 else blind_first["overall"]}
    if name in ("progress", "mastery", "healthy, declining", "topic shift"):
        healthy = (inn["t"] >= 260) & (inn["t"] < r4.PROBE) & ~inn["elicited"]
        out["short-long correlation"] = float(np.corrcoef(inn["news"][healthy], inn["long"][healthy])[0, 1])
    if name == "progress" and ref is not None:
        idx = r5.ref_idx(inn, K)
        rng = np.random.default_rng(int(abs(ref["gain"]) * 1e9) % (2 ** 32))
        clusters = r4._clusters(inn["t"][idx], K)
        boot = [r4.gain(run, inn, np.sort(np.concatenate([idx[clusters[j]] for j in
                                                           rng.integers(len(clusters), size=len(clusters))])), K)["gain"]
                for _ in range(200)]
        out["interval"] = {"gain": ref["gain"], "se": ref["se"], "bootstrap se": float(np.std(boot))}
    if name == "valid inference":
        out["valid points"] = [read_point(run, inn, p, K, N, M, ref)["overall"] for p in r4.VALID_POINTS]
    return out


def evaluate(trials: int = 200, seed: int = 0) -> dict:
    kcal = r5.choose_K(200, seed)
    cal = {**kcal, **tune(200, seed, kcal["K"]), "thresholds": calibrate_separation(500, seed)}
    rng = np.random.default_rng(seed)
    res = {name: [read(make(name, rng), name, cal) for _ in range(trials)] for name in STRUCTURES}
    blind = blind_module()
    blind_rng = np.random.default_rng(seed + 60_000)
    res["blind"] = [read(blind.generate(blind_rng), "blind", cal) for _ in range(trials)]

    def share(name, f):
        return float(np.mean([f(r) for r in res[name]]))

    separation = {n: share(n, lambda r, n=n: r["label"] == n) for n in ("topic shift", "noise spike", "progress")}
    dynamics_ok = share("dynamics shift", lambda r: r["first"]["overall"] != "closed loop" and r["label"] != "topic shift")
    gain_auc = auc([r["first"]["gain"] for r in res["mastery"]], [r["first"]["gain"] for r in res["closed loop (dullness)"]])
    news_auc = {n: auc([r["first"]["long news"] for r in res["progress"]], [r["first"]["long news"] for r in res[n]])
                for n in r4.ECHOES}
    valid = float(np.mean([p not in ("closed loop", "no reference") for r in res["valid inference"] for p in r["valid points"]]))
    healthy_closed = {n: share(n, lambda r: r["first"]["overall"] in ("closed loop", "no reference")) for n in r4.HEALTHY}
    loops_final = {n: share(n, lambda r: r["final"]["overall"] == "closed loop") for n in LOOPS}
    declining_final = share("healthy, declining", lambda r: r["final"]["overall"] not in ("closed loop", "no reference"))
    confirm_ok = share("confirmations", lambda r: r["first"]["overall"] != "closed loop" and r["final"]["overall"] != "closed loop")
    confirm_ignoring = share("confirmations", lambda r: r["ignoring cause"]["first"] != "closed loop"
                             and r["ignoring cause"]["final"] != "closed loop")
    iv = [r["interval"] for r in res["progress"] if "interval" in r]
    mean_gain = float(np.mean([x["gain"] for x in iv]))
    honesty = {"lower bound above the mean": float(np.mean([x["gain"] - r4.Z * x["se"] > mean_gain for x in iv])),
               "upper bound below the mean": float(np.mean([x["gain"] + r4.Z * x["se"] < mean_gain for x in iv])),
               "median analytic / bootstrap se": float(np.median([x["se"] / x["bootstrap se"] for x in iv]))}
    expected = json.loads((r5.BLIND / "round7a_expected.json").read_text())["final reading"]   # the key: scoring only
    blind_match = share("blind", lambda r: r["final"]["overall"] == expected)
    passed = {
        "1 separation": all(v >= 0.9 for v in separation.values()) and dynamics_ok >= 0.9,
        "2 gain: mastery over dullness": gain_auc >= 0.8,
        "3 long news: progress over echoes": all(v >= 0.8 for v in news_auc.values()),
        "4 valid inference": valid >= 0.9,
        "5 healthy not closed loop": all(v <= 0.1 for v in healthy_closed.values()),
        "6 final reading": all(v >= 0.8 for v in loops_final.values()) and declining_final >= 0.9,
        "7 response profile": cal["peak lead"] == r5.EXPECTED_PEAK and cal["profile correlation, leads 1-8"] >= 0.95,
        "8 intervals honest": honesty["lower bound above the mean"] <= 0.15 and honesty["upper bound below the mean"] <= 0.15
        and 0.75 <= honesty["median analytic / bootstrap se"] <= 1.25,
        "9 blind structure": blind_match >= 0.8,
        "10 confirmations not capture": confirm_ok >= 0.9}
    described = {n: {"labels": {k: share(n, lambda r, k=k: r["label"] == k) for k in ("topic shift", "noise spike", "progress")},
                     "first": {k: share(n, lambda r, k=k: r["first"]["overall"] == k) for k in ("open", "undecided", "closed loop", "no reference")},
                     "final": {k: share(n, lambda r, k=k: r["final"]["overall"] == k) for k in ("open", "undecided", "closed loop", "no reference")},
                     "re-presentation at 460": share(n, lambda r: r["first"]["re-presentation"]),
                     "long news, median": float(np.median([r["first"]["long news"] for r in res[n]])),
                     "short news, median": float(np.median([r["first"]["short news"] for r in res[n]])),
                     "descriptor fires": share(n, lambda r: r["descriptor fires"])} for n in res}
    correlation = {n: float(np.nanmedian([r["short-long correlation"] for r in res[n]]))
                   for n in ("progress", "mastery", "healthy, declining", "topic shift")}
    return {"trials": trials, "seed": seed, "calibration": cal, "separation": separation,
            "dynamics shift not closed loop or topic": dynamics_ok, "gain AUC, mastery over dullness": gain_auc,
            "long news AUC, progress over echo": news_auc, "valid inference not closed loop": valid,
            "healthy read closed loop at 460": healthy_closed, "loops closed loop, final": loops_final,
            "declining not closed loop, final": declining_final, "interval honesty": honesty,
            "blind structure: final reading matches": blind_match,
            "confirmations not closed loop": confirm_ok, "confirmations not closed loop, ignoring cause": confirm_ignoring,
            "short-long news correlation, healthy (median)": correlation, "passed": passed, "described": described}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the full report as JSON")
    a = p.parse_args(argv)
    rep = evaluate(a.trials, a.seed)

    def show(v):
        if isinstance(v, dict):
            return {k: show(x) for k, x in v.items()}
        if isinstance(v, list):
            return [round(x, 3) for x in v]
        return round(v, 3) if isinstance(v, float) else v
    for key, v in rep.items():
        if key not in ("trials", "seed", "described", "calibration"):
            print(f"{key}: {show(v)}")
    c = rep["calibration"]
    print(f"calibration: K {c['K']}, N {c['N']}, M {c['M']}, peak lead {c['peak lead']}, "
          f"profile correlation {c['profile correlation, leads 1-8']:.3f}, thresholds {show(c['thresholds'])}")
    for name, row in rep["described"].items():
        print(f"  {name}: {show(row)}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
