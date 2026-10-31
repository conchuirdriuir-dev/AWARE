"""Toy bed round 2: v10's detectors against v10's bars (PLAN.md, product step 3, round 2).

    python -m product.round2                         # 200 trials a structure, thresholds from 500 progress runs
    python -m product.round2 --trials 50 --out runs/round2.json

Round 1 (v9's detectors) stays runnable as `python -m product.toybed`; its results are the record in PLAN.md.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from product import coupling as cp
from product.compass import cell
from product.toybed import auc, generate

T, T0 = 360, 120
LATE = 200
STRUCTURES = {                                  # name: (generator kind, echo lag)
    "topic shift": ("topic shift", 0), "dynamics shift": ("dynamics shift", 0),
    "closed loop (dullness)": ("closed loop (dullness)", 0),
    "proliferation, lag 1": ("closed loop (proliferation)", 1),
    "proliferation, lag 3": ("closed loop (proliferation)", 3),
    "proliferation, lag 10": ("closed loop (proliferation)", 10),
    "mastery": ("mastery", 0), "noise spike": ("noise spike", 0), "progress": ("progress", 0),
    "valid inference": ("valid inference", 0)}
LOOPS = ("closed loop (dullness)", "proliferation, lag 1", "proliferation, lag 3", "proliferation, lag 10")
AFTER_RESUME = (160, 180, 200)
WHOLE_RUN = tuple(range(0, T - cp.COUPLING_WINDOW + 1, 20))


def make(name: str, rng, D: float = 1.0) -> dict:
    kind, lag = STRUCTURES[name]
    return generate(kind, rng, D=D, T=T, t0=T0, lag=lag)


def window(run: dict, start: int) -> dict:
    stop = start + cp.COUPLING_WINDOW
    return cp.coupling(run["s"][start:stop], run["E"][start:stop], run["c"])


def calibrate(n: int, seed: int, D: float = 1.0) -> dict:
    rng = np.random.default_rng(seed + 10_000)
    runs = [make("progress", rng, D) for _ in range(n)]
    floor = float(np.median([cp.contrast(r)["empty residual"] for r in runs]))
    late = [window(r, LATE) for r in runs]
    return {"floor": floor,
            "point contrast": float(np.percentile([cp.contrast(r)["statistic"] for r in runs], 99)),
            "descriptor": float(np.percentile([cp.descriptor(r) for r in runs], 99)),
            "coupling at zero": float(np.percentile([w["coupling"] for w in late], 5)),
            "residual at floor": float(np.percentile([w["residual"] for w in late], 95))}


def flagged(w: dict, th: dict) -> bool:
    """Closed loop: coupling at zero while the residual sits at the floor."""
    return w["coupling"] < th["coupling at zero"] and w["residual"] < th["residual at floor"]


def read(run: dict, th: dict) -> dict:
    con = cp.contrast(run, th["floor"])
    late = window(run, LATE)
    point, desc = con["statistic"] > th["point contrast"], cp.descriptor(run) > th["descriptor"]
    open_ = late["coupling"] >= th["coupling at zero"]
    if point:
        label = "topic shift" if con["admitted"] else "noise spike"
    elif desc:
        label = "dynamics shift" if open_ else "closed loop"
    else:
        label = "progress"
    return {"label": label, "point fires": point, "admitted": con["admitted"], "descriptor fires": desc,
            "coupling": late["coupling"], "residual": late["residual"],
            "cell": cell(late["residual"] > th["residual at floor"], not open_)}


def evaluate(trials: int = 200, calibration: int = 500, seed: int = 0, D: float = 1.0) -> dict:
    th = calibrate(calibration, seed, D)
    rng = np.random.default_rng(seed)
    res = {name: [read(make(name, rng, D), th) | {} for _ in range(trials)] for name in STRUCTURES}
    vi_rng = np.random.default_rng(seed + 20_000)
    vi_after, vi_whole = [], []
    for _ in range(trials):
        run = make("valid inference", vi_rng, D)
        vi_after += [flagged(window(run, a), th) for a in AFTER_RESUME]
        vi_whole += [flagged(window(run, a), th) for a in WHOLE_RUN]

    def hit(name, label):
        return float(np.mean([r["label"] == label for r in res[name]]))

    def coup(name):
        return [r["coupling"] for r in res[name]]

    separation = {n: hit(n, n) for n in ("topic shift", "dynamics shift", "noise spike", "progress")}
    loops_as_dynamics = {n: hit(n, "dynamics shift") for n in LOOPS}
    mastery = {n: auc(coup("mastery"), coup(n)) for n in LOOPS}
    progress = {n: auc(coup("progress"), coup(n)) for n in LOOPS if n.startswith("proliferation")}
    valid = {"not flagged, after resume": 1 - float(np.mean(vi_after)),
             "not flagged, whole run": 1 - float(np.mean(vi_whole))}
    passed = {"separation": all(v >= 0.9 for v in separation.values())
              and all(v <= 0.1 for v in loops_as_dynamics.values()),
              "loop vs mastery": all(v >= 0.8 for v in mastery.values()),
              "proliferation vs progress": all(v >= 0.8 for v in progress.values()),
              "valid inference": valid["not flagged, after resume"] >= 0.9}
    described = {}
    for name, rs in res.items():
        labels = [r["label"] for r in rs]
        cells = [r["cell"] for r in rs]
        described[name] = {"commonest label": max(set(labels), key=labels.count),
                           "point fires": float(np.mean([r["point fires"] for r in rs])),
                           "admitted": float(np.mean([r["admitted"] for r in rs])),
                           "descriptor fires": float(np.mean([r["descriptor fires"] for r in rs])),
                           "coupling, median": float(np.median(coup(name))),
                           "residual, median": float(np.median([r["residual"] for r in rs])),
                           "commonest cell": max(set(cells), key=cells.count)}
    return {"trials": trials, "calibration": calibration, "seed": seed, "thresholds": th,
            "separation": separation, "loops called dynamics shift": loops_as_dynamics,
            "coupling AUC, mastery over loop": mastery, "coupling AUC, progress over proliferation": progress,
            "valid inference": valid, "passed": passed, "described": described}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--calibration", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the full report as JSON")
    a = p.parse_args(argv)
    rep = evaluate(a.trials, a.calibration, a.seed)
    for key in ("separation", "loops called dynamics shift", "coupling AUC, mastery over loop",
                "coupling AUC, progress over proliferation", "valid inference"):
        print(f"{key}:", {k: round(v, 2) for k, v in rep[key].items()})
    print("passed:", rep["passed"])
    for name, row in rep["described"].items():
        print(f"  {name}: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
