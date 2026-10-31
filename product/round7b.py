"""Toy bed round 7b: contingency and the reactive toy, with round 7a's fixes (product/rounds/round7b.md).

    python -m product.round7b                        # 200 trials a structure, seed 0
    python -m product.round7b --out runs/round7b.json

The news floor is sqrt(2) D; only low-news elicited arrivals are confirmations; a partner-side closed loop is probed
before it is final, and the probe is a contingency test (mirror or stale). Entity captured, persistence and
recurrence are new readings. The blind structure's expected reading is used only to score; it is not printed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np

from product import round4 as r4
from product import round5 as r5
from product import round6 as r6
from product import round7a as r7a
from product.toybed import auc, orthonormal, rotation

F = np.sqrt(2.0)                                    # the news floor, in D
PROBE_LEN_D = 3.0
STRUCTURES = {**r7a.STRUCTURES, "confirmations, heavy": ("confirmations heavy", 0),
              "active sensing": ("active sensing", 0), "half echo": ("half echo", 3),
              "saturating": ("saturating", 0), "attractor loop": ("attractor", 0),
              "captured entity": ("captured", 0)}
HEALTHY = r4.HEALTHY + ("saturating", "active sensing", "confirmations, heavy")
LOOPS = r4.LOOPS + ("replay", "attractor loop")


def generate(kind: str, rng, lag: int = 0, D: float = 1.0) -> dict:
    """Round 7a's generator with the probe's displacement and six reactive or nonlinear conditions."""
    d, T, T0, RAMP, PROBE = r4.D_DIM, r4.T, r4.T0, r4.RAMP, r4.PROBE
    c0 = rng.normal(scale=10.0, size=d)
    t = np.arange(T)
    novelty = np.ones(T)
    if kind == "declining":
        novelty = 1 - 0.85 * np.clip((t - T0) / RAMP, 0, 1)
    novelty[PROBE:PROBE + r4.PROBE_LEN] = 1.0
    walk = np.cumsum(rng.normal(scale=0.3 * D / np.sqrt(d), size=(T, d)) * novelty[:, None], axis=0)
    if kind == "topic shift":
        walk[T0:] += 6 * D * orthonormal(rng, 1, d)[0]
    trend = np.zeros(d)
    if kind == "valid inference":
        trend = 0.3 * D * orthonormal(rng, 1, d)[0]
        walk += t[:, None] * trend
    if kind == "captured":
        away = orthonormal(rng, 1, d)[0]
        walk += np.clip(t - T0, 0, None)[:, None] * 0.3 * D * away
    content = c0 + walk + rng.normal(size=(T, d)) / np.sqrt(d) * (r4.JUMP * D * novelty)[:, None]
    noise_scale = np.where(t < r4.INTAKE, 2 * D, D * novelty)
    fresh = content + rng.normal(size=(T, d)) * (noise_scale / np.sqrt(d))[:, None]
    m = (rng.random(T) < r4.ARRIVAL) | (t < r4.INTAKE)
    if kind == "valid inference":
        m[T0:T0 + r4.PAUSE] = False
    m[PROBE:PROBE + r4.PROBE_LEN] = True
    elicited = np.zeros(T, dtype=bool)
    elicited[PROBE:PROBE + r4.PROBE_LEN] = True
    coin = rng.random(T)
    replay_src = [j for j in range(60, T0) if m[j]]
    probe = orthonormal(rng, 1, d)[0]
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
    cycle, ack, cyc = 0, None, None

    def small():
        return rng.normal(scale=0.3 * D / np.sqrt(d), size=d)
    for k in range(T):
        if m[k]:
            post = k >= T0
            in_probe = PROBE <= k < PROBE + r4.PROBE_LEN
            if kind == "proliferation" and post:
                E[k] = s[k - lag] + small()
            elif kind == "half echo" and post and coin[k] < 0.5:
                E[k] = s[k - lag] + small()
            elif kind == "replay" and post:
                E[k] = content[replay_src[cycle % len(replay_src)]] + rng.normal(scale=D / np.sqrt(d), size=d)
                cycle += 1
            elif kind in ("confirmations", "confirmations heavy") and post and not in_probe \
                    and coin[k] < (0.6 if kind == "confirmations" else 0.9):
                E[k] = s[k] + small()
                elicited[k] = True
            elif kind == "active sensing" and post and not in_probe:
                if coin[k] < 0.6:
                    E[k] = fresh[k]
                    elicited[k] = True
                else:
                    ack = c0 + walk[T0] if ack is None else ack
                    E[k] = ack + small()
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
        noise = rng.normal(scale=sigma / np.sqrt(d), size=d)
        drive = held - c0
        if kind == "valid inference" and T0 <= k < T0 + r4.PAUSE:
            drive = walk[T0 - 1] + trend * (k - T0 + 1)
        if kind == "saturating":
            n = np.linalg.norm(drive)
            drive = drive / n * 3.0 * D * np.tanh(n / (3.0 * D)) if n > 0 else drive
        if kind == "attractor" and k >= T0:
            cyc = s[194:200].copy() if cyc is None else cyc
            target = cyc[(k - T0) % len(cyc)]
            s[k + 1] = target + (0.5 * (E[k] - target) if m[k] else 0.0) + noise
        elif kind == "captured" and k >= T0:
            s[k + 1] = held + noise
        else:
            s[k + 1] = c0 + A @ (s[k] - c0) + beta * (I - A) @ drive + drift + noise
        if k == PROBE:
            s[k + 1] += PROBE_LEN_D * D * probe
    E_in = np.where(m[:, None], 0.5 * E + 0.5 * s, 0.0)
    return {"s": s, "E": E, "m": m, "elicited": elicited, "probe": probe, "E_in": E_in, "H": Hd,
            "c": fresh[:10].mean(0), "t0": T0, "kind": kind}


def make(name: str, rng) -> dict:
    kind, lag = STRUCTURES[name]
    return generate(kind, rng, lag)


def blind_module():
    spec = importlib.util.spec_from_file_location("blind_round7b", r5.BLIND / "round7b_structure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def news(run: dict) -> dict:
    inn = r7a.news(run)
    return {**inn, "confirmation": inn["elicited"] & (inn["news"] < F)}


def news_at(inn: dict, point: int, N: int, use_cause: bool = True) -> dict:
    keep = (inn["t"] <= point) & (~inn["confirmation"] if use_cause else True)
    long = inn["long"][keep][-N:]
    if len(long) < 2:
        return {"long": float("nan"), "at zero": False}
    upper = long.mean() + r4.Z * long.std(ddof=1) / np.sqrt(len(long))
    return {"long": float(long.mean()), "at zero": bool(upper < F)}


def persistence(run: dict, inn: dict) -> dict:
    idx = np.flatnonzero((inn["t"] >= 260) & (inn["t"] <= 440) & (inn["t"] + 20 <= r4.READ))
    pg = r4.gain(run, inn, idx, 1, leads=list(range(12, 21)))["gain"] if len(idx) >= 3 else float("nan")
    s = run["s"]
    hits = [np.min(np.linalg.norm(s[:t - 21] - s[t], axis=1)) < F for t in range(300, r4.READ)]
    return {"persistent gain": float(pg), "recurrence": float(np.mean(hits))}


def contingency(run: dict, inn: dict) -> float:
    p, E = run["probe"], run["E"]
    resp = inn["t"][(inn["t"] >= r4.PROBE) & (inn["t"] < r4.PROBE + r4.PROBE_LEN)]
    before = inn["t"][(inn["t"] < r4.PROBE) & ~inn["confirmation"]][-10:]
    return float(((E[resp] @ p).mean() - (E[before] @ p).mean()) / PROBE_LEN_D)


def distance(run: dict) -> float:
    return float(np.linalg.norm(run["s"][400:r4.READ].mean(0) - run["c"]))


def calibrate(n: int, seed: int, K: int) -> dict:
    rng = np.random.default_rng(seed + 10_000)
    ob, extra = [], []
    for _ in range(n):
        run = make("progress", rng)
        inn = news(run)
        ob.append(r6.observe(run, inn))
        extra.append({**persistence(run, inn), "contingency": contingency(run, inn), "distance": distance(run)})
    return {"onset news": float(np.percentile([o["onset news max"] for o in ob], 99)),
            "onset surprise": float(np.percentile([o["onset surprise max"] for o in ob], 99)),
            "later surprise": float(np.percentile([o["later surprise median"] for o in ob], 99)),
            "descriptor": float(np.percentile([o["descriptor"] for o in ob], 99)),
            "persistent gain": float(np.nanpercentile([e["persistent gain"] for e in extra], 1)),
            "recurrence": float(np.percentile([e["recurrence"] for e in extra], 99)),
            "contingency": float(np.percentile([e["contingency"] for e in extra], 99)),
            "distance": float(np.percentile([e["distance"] for e in extra], 99))}


def tune(n: int, seed: int, K: int) -> dict:
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


def at_460(run, inn, K, N, M, ref, th, use_cause=True) -> dict:
    nw = news_at(inn, r4.READ, N, use_cause)
    g = r4.gain_at(run, inn, r4.READ, K, M, ref)
    gr = r7a.gain_reading(g, ref)
    pers = persistence(run, inn)
    loop_pers = pers["persistent gain"] < th["persistent gain"] and pers["recurrence"] > th["recurrence"]
    captured = ref is not None and g["gain"] - r4.Z * g["se"] > 1.5 * ref["gain"] and distance(run) > th["distance"]
    if gr == "at zero" or loop_pers:
        reading, probe = "closed loop", False
    elif captured:
        reading, probe = "entity captured", False
    elif nw["at zero"]:
        reading, probe = "closed loop", True
    elif gr == "open":
        reading, probe = "open", False
    else:
        reading, probe = ("no reference" if gr == "no reference" else "undecided"), True
    return {"reading": reading, "probe": probe, "gain": g["gain"], "gain reading": gr, "long news": nw["long"],
            "news at zero": nw["at zero"], **pers, "captured": captured}


def after_probe(run, inn, K, ref, th, use_cause=True) -> dict:
    idx = np.flatnonzero((inn["t"] >= r4.PROBE) & (inn["t"] < r4.PROBE + r4.PROBE_LEN))
    keep = idx[~inn["confirmation"][idx]] if use_cause else idx
    long = inn["long"][keep]
    news_zero = len(long) < 2 or bool(long.mean() + r4.Z * long.std(ddof=1) / np.sqrt(len(long)) < F)
    kappa = contingency(run, inn)
    if news_zero:
        return {"reading": "closed loop", "kind": "mirror" if kappa > th["contingency"] else "stale", "contingency": kappa}
    g = {**r4.gain(run, inn, idx, K), "met": True}
    gr = r7a.gain_reading(g, ref, need_budget=False)
    reading = "closed loop" if gr == "at zero" else "open" if gr == "open" else (
        "no reference" if gr == "no reference" else "undecided")
    return {"reading": reading, "kind": "entity side" if gr == "at zero" else None, "contingency": kappa}


def read(run: dict, name: str, cal: dict) -> dict:
    K, N, M, th = cal["K"], cal["N"], cal["M"], cal["thresholds"]
    inn = news(run)
    ref = r4.reference(run, inn, K)
    first = at_460(run, inn, K, N, M, ref, th)
    final = after_probe(run, inn, K, ref, th) if first["probe"] else {"reading": first["reading"], "kind": None,
                                                                      "contingency": contingency(run, inn)}
    ob = r6.observe(run, inn)
    news_spike = ob["onset news max"] > th["onset news"]
    if news_spike and first["gain reading"] != "at zero":
        label = "topic shift"
    elif (not news_spike and ob["onset surprise max"] > th["onset surprise"]
          and ob["later surprise median"] <= th["later surprise"]):
        label = "noise spike"
    else:
        label = "progress"
    out = {"label": label, "first": first, "final": final, "descriptor fires": ob["descriptor"] > th["descriptor"]}
    if name == "confirmations, heavy":
        f2 = at_460(run, inn, K, N, M, ref, th, use_cause=False)
        out["ignoring cause"] = after_probe(run, inn, K, ref, th, use_cause=False)["reading"] if f2["probe"] else f2["reading"]
    if name == "progress" and ref is not None:
        idx = r5.ref_idx(inn, K)
        rng = np.random.default_rng(int(abs(ref["gain"]) * 1e9) % (2 ** 32))
        clusters = r4._clusters(inn["t"][idx], K)
        boot = [r4.gain(run, inn, np.sort(np.concatenate([idx[clusters[j]] for j in
                                                           rng.integers(len(clusters), size=len(clusters))])), K)["gain"]
                for _ in range(200)]
        out["interval"] = {"gain": ref["gain"], "se": ref["se"], "bootstrap se": float(np.std(boot))}
    if name == "valid inference":
        out["valid points"] = [r7a.read_point(run, inn, p, K, N, M, ref)["overall"] for p in r4.VALID_POINTS]
    return out


def evaluate(trials: int = 200, seed: int = 0) -> dict:
    kcal = r5.choose_K(200, seed)
    cal = {**kcal, **tune(200, seed, kcal["K"]), "thresholds": calibrate(500, seed, kcal["K"])}
    rng = np.random.default_rng(seed)
    res = {name: [read(make(name, rng), name, cal) for _ in range(trials)] for name in STRUCTURES}
    blind = blind_module()
    blind_rng = np.random.default_rng(seed + 60_000)
    res["blind"] = [read(blind.generate(blind_rng), "blind", cal) for _ in range(trials)]

    def share(name, f):
        return float(np.mean([f(r) for r in res[name]]))

    separation = {n: share(n, lambda r, n=n: r["label"] == n) for n in ("topic shift", "noise spike", "progress")}
    dynamics_ok = share("dynamics shift", lambda r: r["first"]["reading"] != "closed loop" and r["label"] != "topic shift")
    gain_auc = auc([r["first"]["gain"] for r in res["mastery"]], [r["first"]["gain"] for r in res["closed loop (dullness)"]])
    news_auc = {n: auc([r["first"]["long news"] for r in res["progress"]], [r["first"]["long news"] for r in res[n]])
                for n in r4.ECHOES}
    valid = float(np.mean([p not in ("closed loop", "no reference") for r in res["valid inference"] for p in r["valid points"]]))
    healthy_final = {n: share(n, lambda r: r["final"]["reading"] not in ("closed loop", "entity captured", "no reference"))
                     for n in HEALTHY}
    loops_final = {n: share(n, lambda r: r["final"]["reading"] == "closed loop") for n in LOOPS}
    iv = [r["interval"] for r in res["progress"] if "interval" in r]
    mean_gain = float(np.mean([x["gain"] for x in iv]))
    honesty = {"lower bound above the mean": float(np.mean([x["gain"] - r4.Z * x["se"] > mean_gain for x in iv])),
               "upper bound below the mean": float(np.mean([x["gain"] + r4.Z * x["se"] < mean_gain for x in iv])),
               "median analytic / bootstrap se": float(np.median([x["se"] / x["bootstrap se"] for x in iv]))}
    expected = json.loads((r5.BLIND / "round7b_expected.json").read_text())["final reading"]   # the key: scoring only
    blind_match = share("blind", lambda r: r["final"]["reading"] == expected)
    cause_with = share("confirmations, heavy", lambda r: r["final"]["reading"] != "closed loop")
    cause_without = share("confirmations, heavy", lambda r: r["ignoring cause"] == "closed loop")
    kappa = {n: [r["final"]["contingency"] for r in res[n]] for n in r4.ECHOES + ("replay",)}
    kappa_auc = {n: auc(kappa[n], kappa["replay"]) for n in r4.ECHOES}
    mirror = {n: float(np.mean([r["final"].get("kind") == "mirror" for r in res[n] if r["final"]["reading"] == "closed loop"] or [0]))
              for n in r4.ECHOES}
    stale = float(np.mean([r["final"].get("kind") == "stale" for r in res["replay"] if r["final"]["reading"] == "closed loop"] or [0]))
    captured_hit = share("captured entity", lambda r: r["final"]["reading"] == "entity captured")
    captured_mastery = share("mastery", lambda r: r["final"]["reading"] == "entity captured")
    passed = {
        "1 separation": all(v >= 0.9 for v in separation.values()) and dynamics_ok >= 0.9,
        "2 gain: mastery over dullness": gain_auc >= 0.8,
        "3 long news: progress over echoes": all(v >= 0.8 for v in news_auc.values()),
        "4 valid inference": valid >= 0.9,
        "5 healthy, final": all(v >= 0.9 for v in healthy_final.values()),
        "6 loops, final": all(v >= 0.8 for v in loops_final.values()),
        "7 response profile": cal["peak lead"] == r5.EXPECTED_PEAK and cal["profile correlation, leads 1-8"] >= 0.95,
        "8 intervals honest": honesty["lower bound above the mean"] <= 0.15 and honesty["upper bound below the mean"] <= 0.15
        and 0.75 <= honesty["median analytic / bootstrap se"] <= 1.25,
        "9 blind structure": blind_match >= 0.8,
        "10 arrival cause does work": cause_with >= 0.9 and cause_without >= 0.5,
        "11 capture is contingency": all(v >= 0.8 for v in kappa_auc.values()) and all(v >= 0.8 for v in mirror.values())
        and stale >= 0.8,
        "12 entity captured": captured_hit >= 0.8 and captured_mastery <= 0.1}
    described = {n: {"labels": {k: share(n, lambda r, k=k: r["label"] == k) for k in ("topic shift", "noise spike", "progress")},
                     "first": {k: share(n, lambda r, k=k: r["first"]["reading"] == k)
                               for k in ("open", "undecided", "closed loop", "entity captured", "no reference")},
                     "probed": share(n, lambda r: r["first"]["probe"]),
                     "final": {k: share(n, lambda r, k=k: r["final"]["reading"] == k)
                               for k in ("open", "undecided", "closed loop", "entity captured", "no reference")},
                     "final kind": {k: share(n, lambda r, k=k: r["final"].get("kind") == k) for k in ("mirror", "stale", "entity side")},
                     "contingency, median": float(np.median([r["final"]["contingency"] for r in res[n]])),
                     "persistent gain, median": float(np.nanmedian([r["first"]["persistent gain"] for r in res[n]])),
                     "recurrence, median": float(np.median([r["first"]["recurrence"] for r in res[n]])),
                     "descriptor fires": share(n, lambda r: r["descriptor fires"])} for n in res}
    return {"trials": trials, "seed": seed, "calibration": cal, "separation": separation,
            "dynamics shift not closed loop or topic": dynamics_ok, "gain AUC, mastery over dullness": gain_auc,
            "long news AUC, progress over echo": news_auc, "valid inference not closed loop": valid,
            "healthy not closed, final": healthy_final, "loops closed, final": loops_final, "interval honesty": honesty,
            "blind structure: final reading matches": blind_match,
            "confirmations heavy: open with the field / closed without it": [cause_with, cause_without],
            "contingency AUC, echo over replay": kappa_auc, "echo loops called mirror": mirror,
            "replay loops called stale": stale, "entity captured: captured entity / mastery": [captured_hit, captured_mastery],
            "passed": passed, "described": described}


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
