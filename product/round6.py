"""Toy bed round 6 (product/rounds/round6.md): round 5 with the news-spike rule on per-run maxima, the product's
separation narrowed to what the product needs, and the residual descriptor scored apart as discovery's own test.

    python -m product.round6                         # 200 trials a structure, seed 0
    python -m product.round6 --out runs/round6.json

The blind structure's expected reading is used only to score; it is not printed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np

from product import round4 as r4
from product import round5 as r5
from product.toybed import auc

REF_WINDOW, LATER_WINDOW, AXES, RIDGE = (40, 200), (260, 440), 4, 1.0


def blind_module():
    spec = importlib.util.spec_from_file_location("blind_round6", r5.BLIND / "round6_structure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def surprise_vectors(run: dict, inn: dict, g: np.ndarray) -> np.ndarray:
    """The step the entity took minus what the innovations it received account for, as vectors."""
    s = run["s"]
    step = s[1:] - s[:-1]
    explained = np.zeros_like(step)
    inc = np.diff(g)
    for t_i, u in zip(inn["t"], inn["u"]):
        stop = min(t_i + r5.KMAX, len(step))
        explained[t_i:stop] += inc[:stop - t_i, None] * u
    return step - explained


def descriptor(V: np.ndarray) -> float:
    """How far the step-to-step dynamics of unexplained surprise moved between the reference window and the later
    stretch, in the top principal axes of the two together."""
    a, b = V[REF_WINDOW[0]:REF_WINDOW[1]], V[LATER_WINDOW[0]:LATER_WINDOW[1]]
    a, b = a - a.mean(0), b - b.mean(0)
    basis = np.linalg.svd(np.vstack([a, b]), full_matrices=False)[2][:AXES]

    def modes(W):
        Z = W @ basis.T
        X, Y = Z[:-1], Z[1:]
        A = np.linalg.solve(X.T @ X + RIDGE * np.eye(AXES), X.T @ Y).T
        return np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    return float(np.linalg.norm(modes(a) - modes(b)))


def observe(run: dict, inn: dict) -> dict:
    g = r5.reference_profile(run, inn)
    V = surprise_vectors(run, inn, g)
    sur = np.linalg.norm(V, axis=1)
    onset = (inn["t"] >= r5.ONSET[0]) & (inn["t"] < r5.ONSET[1])
    return {"onset news max": float(inn["news"][onset].max()) if onset.any() else 0.0,
            "onset surprise max": float(sur[r5.ONSET[0]:r5.ONSET[1]].max()),
            "later surprise median": float(np.median(sur[r5.LATER[0]:r5.LATER[1]])),
            "descriptor": descriptor(V)}


def calibrate(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 10_000)
    ob = []
    for _ in range(n):
        run = r4.make("progress", rng)
        ob.append(observe(run, r4.innovations(run)))
    return {"onset news": float(np.percentile([o["onset news max"] for o in ob], 99)),
            "onset surprise": float(np.percentile([o["onset surprise max"] for o in ob], 99)),
            "later surprise": float(np.percentile([o["later surprise median"] for o in ob], 95)),
            "descriptor": float(np.percentile([o["descriptor"] for o in ob], 99))}


def read(run: dict, name: str, cal: dict) -> dict:
    K, N, M, th = cal["K"], cal["N"], cal["M"], cal["thresholds"]
    inn = r4.innovations(run)
    ref = r4.reference(run, inn, K)
    first = r4.read_point(run, inn, r4.READ, K, N, M, ref)
    probed = first["overall"] == "undecided"
    final = r5.probe_read(run, inn, K, ref) if probed else first
    ob = observe(run, inn)
    news_spike = ob["onset news max"] > th["onset news"]
    if news_spike and first["gain reading"] != "at zero":
        label = "topic shift"
    elif (not news_spike and ob["onset surprise max"] > th["onset surprise"]
          and ob["later surprise median"] <= th["later surprise"]):
        label = "noise spike"
    else:
        label = "progress"
    out = {"label": label, "first": first, "final": final, "probed": probed,
           "descriptor fires": ob["descriptor"] > th["descriptor"], "descriptor": ob["descriptor"]}
    if name == "progress" and ref is not None:
        idx = r5.ref_idx(inn, K)
        rng = np.random.default_rng(int(abs(ref["gain"]) * 1e9) % (2 ** 32))
        clusters = r4._clusters(inn["t"][idx], K)
        boot = [r4.gain(run, inn, np.sort(np.concatenate([idx[clusters[j]] for j in
                                                           rng.integers(len(clusters), size=len(clusters))])), K)["gain"]
                for _ in range(200)]
        out["interval"] = {"gain": ref["gain"], "se": ref["se"], "bootstrap se": float(np.std(boot))}
    if name == "valid inference":
        out["valid points"] = [r4.read_point(run, inn, p, K, N, M, ref)["overall"] for p in r4.VALID_POINTS]
    return out


def evaluate(trials: int = 200, seed: int = 0) -> dict:
    kcal = r5.choose_K(200, seed)
    tuned = r4.tune(200, seed, kcal["K"])
    cal = {**kcal, **tuned, "thresholds": calibrate(500, seed)}
    rng = np.random.default_rng(seed)
    res = {name: [read(r4.make(name, rng), name, cal) for _ in range(trials)] for name in r4.STRUCTURES}
    blind = blind_module()
    blind_rng = np.random.default_rng(seed + 60_000)
    res["blind"] = [read(blind.generate(blind_rng), "blind", cal) for _ in range(trials)]

    def share(name, f):
        return float(np.mean([f(r) for r in res[name]]))

    separation = {n: share(n, lambda r, n=n: r["label"] == n) for n in ("topic shift", "noise spike", "progress")}
    dynamics_ok = share("dynamics shift", lambda r: r["first"]["overall"] != "closed loop" and r["label"] != "topic shift")
    gain_auc = auc([r["first"]["gain"] for r in res["mastery"]], [r["first"]["gain"] for r in res["closed loop (dullness)"]])
    news_auc = {n: auc([r["first"]["news"] for r in res["progress"]], [r["first"]["news"] for r in res[n]]) for n in r4.ECHOES}
    valid = float(np.mean([p not in ("closed loop", "no reference") for r in res["valid inference"] for p in r["valid points"]]))
    healthy_closed = {n: share(n, lambda r: r["first"]["overall"] in ("closed loop", "no reference")) for n in r4.HEALTHY}
    loops_final = {n: share(n, lambda r: r["final"]["overall"] == "closed loop") for n in r4.LOOPS}
    declining_final = share("healthy, declining", lambda r: r["final"]["overall"] not in ("closed loop", "no reference"))
    iv = [r["interval"] for r in res["progress"] if "interval" in r]
    mean_gain = float(np.mean([x["gain"] for x in iv]))
    honesty = {"lower bound above the mean": float(np.mean([x["gain"] - r4.Z * x["se"] > mean_gain for x in iv])),
               "upper bound below the mean": float(np.mean([x["gain"] + r4.Z * x["se"] < mean_gain for x in iv])),
               "median analytic / bootstrap se": float(np.median([x["se"] / x["bootstrap se"] for x in iv]))}
    expected = json.loads((r5.BLIND / "round6_expected.json").read_text())["final reading"]   # the key: scoring only
    blind_match = share("blind", lambda r: r["final"]["overall"] == expected)
    fires = {n: share(n, lambda r: r["descriptor fires"]) for n in res}
    passed = {
        "1 separation": all(v >= 0.9 for v in separation.values()) and dynamics_ok >= 0.9,
        "2 gain: mastery over dullness": gain_auc >= 0.8,
        "3 news: progress over echoes": all(v >= 0.8 for v in news_auc.values()),
        "4 valid inference": valid >= 0.9,
        "5 healthy not closed loop": all(v <= 0.1 for v in healthy_closed.values()),
        "6 final reading": all(v >= 0.8 for v in loops_final.values()) and declining_final >= 0.9,
        "7 response profile": cal["peak lead"] == r5.EXPECTED_PEAK and cal["profile correlation, leads 1-8"] >= 0.95,
        "8 intervals honest": honesty["lower bound above the mean"] <= 0.15 and honesty["upper bound below the mean"] <= 0.15
        and 0.75 <= honesty["median analytic / bootstrap se"] <= 1.25,
        "9 blind structure": blind_match >= 0.8}
    discovery = {"D residual descriptor": fires["dynamics shift"] >= 0.9 and fires["topic shift"] <= 0.1
                 and fires["healthy, declining"] <= 0.1}
    described = {n: {"labels": {k: share(n, lambda r, k=k: r["label"] == k) for k in ("topic shift", "noise spike", "progress")},
                     "first": {k: share(n, lambda r, k=k: r["first"]["overall"] == k) for k in ("open", "undecided", "closed loop", "no reference")},
                     "probed": share(n, lambda r: r["probed"]),
                     "final": {k: share(n, lambda r, k=k: r["final"]["overall"] == k) for k in ("open", "undecided", "closed loop", "no reference")},
                     "descriptor fires": fires[n]} for n in res}
    return {"trials": trials, "seed": seed, "calibration": cal, "separation": separation,
            "dynamics shift not closed loop or topic": dynamics_ok, "gain AUC, mastery over dullness": gain_auc,
            "news AUC, progress over echo": news_auc, "valid inference not closed loop": valid,
            "healthy read closed loop at 460": healthy_closed, "loops closed loop, final": loops_final,
            "declining not closed loop, final": declining_final, "interval honesty": honesty,
            "blind structure: final reading matches": blind_match, "descriptor fires": fires,
            "passed": passed, "discovery": discovery, "described": described}


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
