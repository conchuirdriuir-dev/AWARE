"""Toy bed round 5: the observer as an event record (product/rounds/round5.md).

    python -m product.round5                         # 200 trials a structure, seed 0
    python -m product.round5 --out runs/round5.json

Round 4's generator and readings, with separation read from events (news spikes, unexplained surprise, the gain
profile's shape), the probe read by its own responses, the detection horizon chosen by rule, and one blind structure
written before this code. The blind structure's expected reading is used only to score; it is not printed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from product import round4 as r4
from product.toybed import auc

BLIND = Path(__file__).parent / "blind"
KMAX, ONSET, LATER = 20, (200, 230), (260, 440)
EXPECTED_PEAK = 2


def expected_profile(kmax: int = 8) -> np.ndarray:
    """This generator's response at lead k: the external drives the entity until replaced (probability 0.4 a step)."""
    return np.array([sum(0.5 ** (k - 1 - i) * 0.5 * 0.6 ** i for i in range(k)) for k in range(1, kmax + 1)])


def blind_module():
    spec = importlib.util.spec_from_file_location("blind_round5", BLIND / "round5_structure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ref_idx(inn: dict, lead: int) -> np.ndarray:
    return np.flatnonzero((inn["t"] >= r4.REFERENCE[0]) & (inn["t"] + lead <= r4.REFERENCE[1]))


def choose_K(n: int, seed: int) -> dict:
    """The horizon whose averaged gain is detected most surely, and the measured response profile."""
    rng = np.random.default_rng(seed + 40_000)
    z = np.zeros((n, KMAX))
    blocks = np.zeros((n, KMAX))
    prof = []
    for j in range(n):
        run = r4.make("progress", rng)
        inn = r4.innovations(run)
        prof.append(r4.profile(run, inn, KMAX))
        for k in range(1, KMAX + 1):
            idx = ref_idx(inn, k)
            g = r4.gain(run, inn, idx, k)
            z[j, k - 1] = g["gain"] / g["se"] if g["se"] > 0 else 0.0
            blocks[j, k - 1] = len(r4._clusters(inn["t"][idx], k))
    med = np.median(z, axis=0)
    defined = np.median(blocks, axis=0) >= 10            # amendment: the error needs enough overlap blocks
    mean_prof = np.mean(prof, axis=0)
    return {"K": int(np.argmax(np.where(defined, med, -np.inf)) + 1), "median z by horizon": med.tolist(),
            "median blocks by horizon": np.median(blocks, axis=0).tolist(), "profile": mean_prof.tolist(),
            "peak lead": int(np.argmax(mean_prof) + 1),
            "profile correlation, leads 1-8": float(np.corrcoef(mean_prof[:8], expected_profile())[0, 1])}


def reference_profile(run: dict, inn: dict) -> np.ndarray:
    """The session's own gain at leads 0..KMAX from reference-window arrivals, g(0) = 0."""
    idx = ref_idx(inn, KMAX)
    if len(idx) < 3:
        return np.zeros(KMAX + 1)
    return np.concatenate([[0.0], [r4.gain(run, inn, idx, 1, leads=[k])["gain"] for k in range(1, KMAX + 1)]])


def surprise(run: dict, inn: dict, g: np.ndarray) -> np.ndarray:
    """|s(t+1) - s(t) - what the innovations received account for|, for every step."""
    s = run["s"]
    step = s[1:] - s[:-1]
    explained = np.zeros_like(step)
    inc = np.diff(g)                                   # g(j+1) - g(j), j = 0..KMAX-1
    for t_i, u in zip(inn["t"], inn["u"]):
        stop = min(t_i + KMAX, len(step))
        explained[t_i:stop] += inc[:stop - t_i, None] * u
    return np.linalg.norm(step - explained, axis=1)


def later_profile(run: dict, inn: dict, lo: int, hi: int) -> np.ndarray:
    idx = np.flatnonzero((inn["t"] >= lo) & (inn["t"] < hi) & (inn["t"] + 8 <= r4.READ))
    if len(idx) < 3:
        return np.full(8, np.nan)
    return np.array([r4.gain(run, inn, idx, 1, leads=[k])["gain"] for k in range(1, 9)])


def events(run: dict, inn: dict) -> dict:
    """What separation reads: the largest news at onset, surprise at onset and later, the gain profile's change."""
    g = reference_profile(run, inn)
    sur = surprise(run, inn, g)
    onset = (inn["t"] >= ONSET[0]) & (inn["t"] < ONSET[1])
    before = later_profile(run, inn, r4.REFERENCE[0], r4.REFERENCE[1] - 8)
    after = later_profile(run, inn, *LATER)
    return {"onset news": inn["news"][onset].tolist(),
            "onset surprise max": float(sur[ONSET[0]:ONSET[1]].max()),
            "later surprise median": float(np.median(sur[LATER[0]:LATER[1]])),
            "profile change": float(np.linalg.norm(after - before)),
            "surprise median": float(np.median(sur[r4.REFERENCE[0]:r4.READ]))}


def calibrate_events(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 10_000)
    ev = []
    for _ in range(n):
        run = r4.make("progress", rng)
        ev.append(events(run, r4.innovations(run)))
    return {"onset news": float(np.percentile([x for e in ev for x in e["onset news"]], 99)),
            "onset surprise": float(np.percentile([e["onset surprise max"] for e in ev], 99)),
            "later surprise": float(np.percentile([e["later surprise median"] for e in ev], 95)),
            "profile change": float(np.nanpercentile([e["profile change"] for e in ev], 99))}


def probe_read(run: dict, inn: dict, K: int, ref: dict | None) -> dict:
    """The probe read by its own responses: news and gain over the arrivals at 460-479 only."""
    idx = np.flatnonzero((inn["t"] >= r4.PROBE) & (inn["t"] < r4.PROBE + r4.PROBE_LEN))
    news = inn["news"][idx]
    upper = news.mean() + r4.Z * news.std(ddof=1) / np.sqrt(len(news))
    news_zero = bool(upper < r4.FLOOR)
    g = r4.gain(run, inn, idx, K)
    if ref is None:
        gain_reading = "no reference"
    elif g["gain"] - r4.Z * g["se"] > ref["delta"]:
        gain_reading = "open"
    elif g["gain"] + r4.Z * g["se"] < ref["delta"]:
        gain_reading = "at zero"
    else:
        gain_reading = "undecided"
    if news_zero or gain_reading == "at zero":
        overall = "closed loop"
    elif gain_reading == "open":
        overall = "open"
    else:
        overall = "no reference" if gain_reading == "no reference" else "undecided"
    return {"overall": overall, "news": float(news.mean()), "news at zero": news_zero, "gain": g["gain"],
            "gain reading": gain_reading}


def read(run: dict, name: str, cal: dict) -> dict:
    K, N, M, th = cal["K"], cal["N"], cal["M"], cal["events"]
    inn = r4.innovations(run)
    ref = r4.reference(run, inn, K)
    first = r4.read_point(run, inn, r4.READ, K, N, M, ref)
    probed = first["overall"] == "undecided"
    final = probe_read(run, inn, K, ref) if probed else first
    ev = events(run, inn)
    news_spike = any(x > th["onset news"] for x in ev["onset news"])
    if news_spike and first["gain reading"] != "at zero":
        label = "topic shift"
    elif (not news_spike and ev["onset surprise max"] > th["onset surprise"]
          and ev["later surprise median"] <= th["later surprise"]):
        label = "noise spike"
    elif ev["profile change"] > th["profile change"]:
        label = "closed loop" if first["overall"] == "closed loop" else (
            "no reference" if ref is None else "dynamics shift")
    else:
        label = "progress"
    out = {"label": label, "reference": ref is not None, "first": first, "final": final, "probed": probed,
           "surprise median": ev["surprise median"]}
    if name == "progress" and ref is not None:
        idx = ref_idx(inn, K)
        rng = np.random.default_rng(int(abs(ref["gain"]) * 1e9) % (2 ** 32))
        clusters = r4._clusters(inn["t"][idx], K)
        boot = [r4.gain(run, inn, np.sort(np.concatenate([idx[clusters[j]] for j in
                                                           rng.integers(len(clusters), size=len(clusters))])), K)["gain"]
                for _ in range(200)]
        out["interval"] = {"gain": ref["gain"], "se": ref["se"], "bootstrap se": float(np.std(boot))}
    if name == "progress" or name in r4.ECHOES:
        inn_in = r4.innovations(run, "E_in")
        out["in context"] = r4.read_point(run, inn_in, r4.READ, K, N, M, r4.reference(run, inn_in, K))
    if name == "valid inference":
        out["valid points"] = [r4.read_point(run, inn, p, K, N, M, ref)["overall"] for p in r4.VALID_POINTS]
    return out


def evaluate(trials: int = 200, seed: int = 0) -> dict:
    kcal = choose_K(200, seed)
    tuned = r4.tune(200, seed, kcal["K"])
    cal = {**kcal, **tuned, "events": calibrate_events(500, seed)}
    rng = np.random.default_rng(seed)
    res = {name: [read(r4.make(name, rng), name, cal) for _ in range(trials)] for name in r4.STRUCTURES}
    blind = blind_module()
    blind_rng = np.random.default_rng(seed + 60_000)
    res["blind"] = [read(blind.generate(blind_rng), "blind", cal) for _ in range(trials)]

    def share(name, f):
        return float(np.mean([f(r) for r in res[name]]))

    separation = {n: share(n, lambda r, n=n: r["label"] == n) for n in ("topic shift", "dynamics shift", "noise spike", "progress")}
    loops_as_dynamics = {n: share(n, lambda r: r["label"] in ("dynamics shift", "no reference")) for n in r4.LOOPS}
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
               "median analytic / bootstrap se": float(np.median([x["se"] / x["bootstrap se"] for x in iv])),
               "runs": len(iv)}
    expected = json.loads((BLIND / "round5_expected.json").read_text())["final reading"]   # the key: scoring only
    blind_match = share("blind", lambda r: r["final"]["overall"] == expected)
    passed = {
        "1 separation": all(v >= 0.9 for v in separation.values()) and all(v <= 0.1 for v in loops_as_dynamics.values()),
        "2 gain: mastery over dullness": gain_auc >= 0.8,
        "3 news: progress over echoes": all(v >= 0.8 for v in news_auc.values()),
        "4 valid inference": valid >= 0.9,
        "5 healthy not closed loop": all(v <= 0.1 for v in healthy_closed.values()),
        "6 final reading": all(v >= 0.8 for v in loops_final.values()) and declining_final >= 0.9,
        "7 response profile": cal["peak lead"] == EXPECTED_PEAK and cal["profile correlation, leads 1-8"] >= 0.95,
        "8 intervals honest": honesty["lower bound above the mean"] <= 0.15 and honesty["upper bound below the mean"] <= 0.15
        and 0.75 <= honesty["median analytic / bootstrap se"] <= 1.25,
        "9 blind structure": blind_match >= 0.8}
    described = {n: {"labels": {k: share(n, lambda r, k=k: r["label"] == k) for k in
                                ("topic shift", "dynamics shift", "noise spike", "progress", "closed loop", "no reference")},
                     "first": {k: share(n, lambda r, k=k: r["first"]["overall"] == k) for k in ("open", "undecided", "closed loop", "no reference")},
                     "probed": share(n, lambda r: r["probed"]),
                     "final": {k: share(n, lambda r, k=k: r["final"]["overall"] == k) for k in ("open", "undecided", "closed loop", "no reference")},
                     "news, median": float(np.median([r["first"]["news"] for r in res[n]])),
                     "gain, median": float(np.nanmedian([r["first"]["gain"] for r in res[n]])),
                     "surprise, median": float(np.median([r["surprise median"] for r in res[n]]))} for n in res}
    framing = {"progress news at zero": {"alone": share("progress", lambda r: r["first"]["news at zero"]),
                                         "in context": share("progress", lambda r: r["in context"]["news at zero"])}}
    return {"trials": trials, "seed": seed, "calibration": cal, "separation": separation,
            "loops called dynamics shift (no reference counted)": loops_as_dynamics,
            "gain AUC, mastery over dullness": gain_auc, "news AUC, progress over echo": news_auc,
            "valid inference not closed loop": valid, "healthy read closed loop at 460": healthy_closed,
            "loops closed loop, final": loops_final, "declining not closed loop, final": declining_final,
            "interval honesty": honesty, "blind structure: final reading matches": blind_match,
            "framing": framing, "passed": passed, "described": described}


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
        if key not in ("trials", "seed", "described"):
            print(f"{key}: {show(v)}")
    for name, row in rep["described"].items():
        print(f"  {name}: {show(row)}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
