"""Toy bed round 4: event-locked coupling, read as news and gain (product/rounds/round4.md).

    python -m product.round4                         # 200 trials a structure, seed 0
    python -m product.round4 --trials 50 --out runs/round4.json

Each arrival's innovation is the external minus the nearest of the entity's last H states and the external held
before it; its length is news. Gain is how much of an innovation appears in the entity over the next K steps, pooled
over arrivals by regression. K is measured, N and M are tuned, each on calibration runs kept apart from the test.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from product import round3 as r3
from product.toybed import auc, orthonormal, rotation

D_DIM, T, T0, RAMP, INTAKE, ARRIVAL = 32, 500, 200, 60, 40, 0.4
PROBE, PROBE_LEN, PAUSE, JUMP = 460, 20, 40, 3.0
READ, REREAD = 460, 495
REFERENCE = (40, 200)
VALID_POINTS = (260, 280, 300, 320)
H, Z, W, FLOOR = 21, 1.2816, 0.25, 1.0
NS, MS = (4, 8, 16, 32), (10, 20, 40, 80)
STRUCTURES = {
    "topic shift": ("topic shift", 0), "dynamics shift": ("dynamics shift", 0),
    "closed loop (dullness)": ("dullness", 0),
    **{f"proliferation, lag {L}": ("proliferation", L) for L in (1, 3, 10, 15, 20)},
    "mastery": ("mastery", 0), "noise spike": ("noise spike", 0), "progress": ("progress", 0),
    "valid inference": ("valid inference", 0), "healthy, declining": ("declining", 0)}
ECHOES = tuple(n for n in STRUCTURES if n.startswith("proliferation"))
LOOPS = ("closed loop (dullness)",) + ECHOES
HEALTHY = ("progress", "mastery", "topic shift", "dynamics shift", "healthy, declining")


def generate(kind: str, rng, lag: int = 0, D: float = 1.0) -> dict:
    """Round 3's generator, with a fresh content jump on every arriving external and the probe always generated at
    460-479 (readings at 460 never see it; it is used only where the rule fires it)."""
    d = D_DIM
    c0 = rng.normal(scale=10.0, size=d)
    t = np.arange(T)
    novelty = np.ones(T)
    if kind == "declining":
        novelty = 1 - 0.85 * np.clip((t - T0) / RAMP, 0, 1)
    novelty[PROBE:PROBE + PROBE_LEN] = 1.0
    walk = np.cumsum(rng.normal(scale=0.3 * D / np.sqrt(d), size=(T, d)) * novelty[:, None], axis=0)
    if kind == "topic shift":
        walk[T0:] += 6 * D * orthonormal(rng, 1, d)[0]
    trend = np.zeros(d)
    if kind == "valid inference":
        trend = 0.3 * D * orthonormal(rng, 1, d)[0]
        walk += t[:, None] * trend
    jumps = rng.normal(size=(T, d)) / np.sqrt(d) * (JUMP * D * novelty)[:, None]
    noise_scale = np.where(t < INTAKE, 2 * D, D * novelty)
    fresh = c0 + walk + jumps + rng.normal(size=(T, d)) * (noise_scale / np.sqrt(d))[:, None]
    m = (rng.random(T) < ARRIVAL) | (t < INTAKE)
    if kind == "valid inference":
        m[T0:T0 + PAUSE] = False
    m[PROBE:PROBE + PROBE_LEN] = True
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
    for k in range(T):
        if m[k]:
            if kind == "proliferation" and k >= T0:
                E[k] = s[k - lag] + rng.normal(scale=0.3 * D / np.sqrt(d), size=d)
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
        if kind == "valid inference" and T0 <= k < T0 + PAUSE:
            drive = walk[T0 - 1] + trend * (k - T0 + 1)
        s[k + 1] = c0 + A @ (s[k] - c0) + beta * (I - A) @ drive + drift + rng.normal(scale=sigma / np.sqrt(d), size=d)
    E_in = np.where(m[:, None], 0.5 * E + 0.5 * s, 0.0)
    return {"s": s, "E": E, "m": m, "E_in": E_in, "H": Hd, "c": fresh[:10].mean(0), "t0": T0, "kind": kind}


def make(name: str, rng, D: float = 1.0) -> dict:
    kind, lag = STRUCTURES[name]
    return generate(kind, rng, lag, D)


def innovations(run: dict, external: str = "E") -> dict:
    """For every arrival from step H on: its innovation u and its news |u|."""
    s, X, Hd, m = run["s"], run[external], run["H"], run["m"]
    times = [t for t in np.flatnonzero(m) if t >= H]
    u = np.zeros((len(times), s.shape[1]))
    for i, t in enumerate(times):
        candidates = np.vstack([s[t - H + 1:t + 1], Hd[t - 1][None]])
        nearest = candidates[np.argmin(np.linalg.norm(candidates - X[t], axis=1))]
        u[i] = X[t] - nearest
    return {"t": np.array(times), "u": u, "news": np.linalg.norm(u, axis=1)}


def _clusters(t: np.ndarray, K: int) -> list[np.ndarray]:
    """Blocks of arrivals less than K steps apart: their responses overlap."""
    cuts = np.flatnonzero(np.diff(t) >= K) + 1
    return np.split(np.arange(len(t)), cuts)


def gain(run: dict, inn: dict, idx: np.ndarray, K: int, leads=None) -> dict:
    """Gain over the arrivals `idx`: mean over leads 1..K (or `leads`) of sum<s(t+k)-s(t), u> / sum|u|^2, with a
    cluster-robust standard error."""
    s = run["s"]
    leads = leads if leads is not None else range(1, K + 1)
    t, u = inn["t"][idx], inn["u"][idx]
    delta = np.mean([s[t + k] - s[t] for k in leads], axis=0)
    b = np.sum(u * u, axis=1)
    g = float(np.sum(np.sum(delta * u, axis=1)) / np.sum(b))
    psi = np.sum((delta - g * u) * u, axis=1)
    var = sum(np.sum(psi[c]) ** 2 for c in _clusters(t, K)) / np.sum(b) ** 2
    return {"gain": g, "se": float(np.sqrt(var)), "n": len(idx)}


def profile(run: dict, inn: dict, kmax: int = 20) -> np.ndarray:
    idx = np.flatnonzero((inn["t"] >= REFERENCE[0]) & (inn["t"] + kmax < READ))
    return np.array([gain(run, inn, idx, 1, leads=[k])["gain"] for k in range(1, kmax + 1)])


def measure_K(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 40_000)
    prof = []
    for _ in range(n):
        run = make("progress", rng)
        prof.append(profile(run, innovations(run)))
    mean = np.mean(prof, axis=0)
    plateau = mean[14:20].mean()
    K = int(np.argmax(mean >= 0.9 * plateau) + 1)
    return {"K": K, "profile": mean.tolist(), "plateau": float(plateau)}


def reference(run: dict, inn: dict, K: int) -> dict | None:
    idx = np.flatnonzero((inn["t"] >= REFERENCE[0]) & (inn["t"] + K <= REFERENCE[1]))
    if len(idx) < 3:
        return None
    g = gain(run, inn, idx, K)
    if g["gain"] - Z * g["se"] <= 0:
        return None
    return {**g, "delta": 0.5 * g["gain"], "news": float(inn["news"][idx].mean())}


def news_at(inn: dict, point: int, N: int) -> dict:
    news = inn["news"][inn["t"] <= point][-N:]
    upper = news.mean() + Z * news.std(ddof=1) / np.sqrt(len(news))
    return {"news": float(news.mean()), "at zero": bool(upper < FLOOR)}


def gain_at(run: dict, inn: dict, point: int, K: int, M: int, ref: dict | None) -> dict:
    """Pool backwards from `point` until the budget is met or M arrivals are used."""
    avail = np.flatnonzero(inn["t"] + K <= point)[::-1]
    g = None
    for n in range(3, min(M, len(avail)) + 1):
        g = gain(run, inn, np.sort(avail[:n]), K)
        if ref is not None and Z * g["se"] <= W * ref["gain"]:
            return {**g, "met": True}
    return {**(g or {"gain": float("nan"), "se": float("nan"), "n": 0}), "met": False}


def read_point(run: dict, inn: dict, point: int, K: int, N: int, M: int, ref: dict | None) -> dict:
    news = news_at(inn, point, N)
    g = gain_at(run, inn, point, K, M, ref)
    if ref is None:
        gain_reading = "no reference"
    elif not g["met"]:
        gain_reading = "undecided"
    elif g["gain"] - Z * g["se"] > ref["delta"]:
        gain_reading = "open"
    elif g["gain"] + Z * g["se"] < ref["delta"]:
        gain_reading = "at zero"
    else:
        gain_reading = "undecided"
    if news["at zero"] or gain_reading == "at zero":
        overall = "closed loop"
    elif gain_reading == "open":
        overall = "open"
    elif gain_reading == "no reference":
        overall = "no reference"
    else:
        overall = "undecided"
    return {"overall": overall, "gain reading": gain_reading, "news": news["news"], "news at zero": news["at zero"],
            "gain": g["gain"], "met": g["met"]}


def tune(n: int, seed: int, K: int) -> dict:
    rng = np.random.default_rng(seed + 50_000)
    prog = [make("progress", rng) for _ in range(n)]
    mast = [make("mastery", rng) for _ in range(n)]
    prog_inn = [innovations(r) for r in prog]
    mast_inn = [innovations(r) for r in mast]
    n_sweep = {N: float(np.mean([news_at(i, READ, N)["at zero"] for i in prog_inn])) for N in NS}
    N = next((N for N in NS if n_sweep[N] <= 0.1), NS[-1])
    m_sweep = {}
    for M in MS:
        met = []
        for runs, inns in ((prog, prog_inn), (mast, mast_inn)):
            met.append(float(np.mean([gain_at(r, i, READ, K, M, reference(r, i, K))["met"] for r, i in zip(runs, inns)])))
        m_sweep[M] = met
    M = next((M for M in MS if min(m_sweep[M]) >= 0.9), MS[-1])
    return {"N": N, "M": M, "N sweep (progress news at zero)": n_sweep,
            "M sweep (progress, mastery budget met)": m_sweep}


def calibrate_contrast(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 10_000)
    runs = [make("progress", rng) for _ in range(n)]
    con = [r3.contrast(r) for r in runs]
    return {"floor": float(np.median([x["empty residual"] for x in con])),
            "point contrast": float(np.percentile([x["statistic"] for x in con], 99)),
            "descriptor": float(np.percentile([r3.descriptor(r) for r in runs], 99))}


def read(run: dict, name: str, cal: dict) -> dict:
    K, N, M = cal["K"], cal["N"], cal["M"]
    inn = innovations(run)
    ref = reference(run, inn, K)
    first = read_point(run, inn, READ, K, N, M, ref)
    probed = first["overall"] == "undecided"
    final = read_point(run, inn, REREAD, K, N, M, ref) if probed else first
    con = r3.contrast(run, cal["floor"])
    point, desc = con["statistic"] > cal["point contrast"], r3.descriptor(run) > cal["descriptor"]
    if point:
        label = "topic shift" if con["admitted"] else "noise spike"
    elif desc:
        label = "no reference" if ref is None else ("dynamics shift" if first["overall"] != "closed loop" else "closed loop")
    else:
        label = "progress"
    out = {"label": label, "reference": ref is not None, "first": first, "final": final, "probed": probed,
           "lead 1 gain": gain_at(run, inn, READ, 1, M, None)["gain"] if ref else float("nan")}
    if name == "progress" and ref is not None:
        idx = np.flatnonzero((inn["t"] >= REFERENCE[0]) & (inn["t"] + K <= REFERENCE[1]))
        rng = np.random.default_rng(int(abs(ref["gain"]) * 1e9) % (2 ** 32))
        clusters = _clusters(inn["t"][idx], K)
        boot = []
        for _ in range(200):
            pick = np.concatenate([idx[clusters[j]] for j in rng.integers(len(clusters), size=len(clusters))])
            boot.append(gain(run, inn, np.sort(pick), K)["gain"])
        out["interval"] = {"gain": ref["gain"], "se": ref["se"], "bootstrap se": float(np.std(boot))}
    if name == "progress" or name in ECHOES:
        inn_in = innovations(run, "E_in")
        ref_in = reference(run, inn_in, K)
        out["in context"] = read_point(run, inn_in, READ, K, N, M, ref_in)
    if name == "valid inference":
        out["valid points"] = [read_point(run, inn, p, K, N, M, ref)["overall"] for p in VALID_POINTS]
    return out


def evaluate(trials: int = 200, seed: int = 0) -> dict:
    kcal = measure_K(200, seed)
    tuned = tune(200, seed, kcal["K"])
    cal = {**kcal, **tuned, **calibrate_contrast(500, seed)}
    rng = np.random.default_rng(seed)
    res = {name: [read(make(name, rng), name, cal) for _ in range(trials)] for name in STRUCTURES}

    def share(name, f):
        return float(np.mean([f(r) for r in res[name]]))

    separation = {n: share(n, lambda r, n=n: r["label"] == n) for n in ("topic shift", "dynamics shift", "noise spike", "progress")}
    loops_as_dynamics = {n: share(n, lambda r: r["label"] in ("dynamics shift", "no reference")) for n in LOOPS}
    gain_auc = auc([r["first"]["gain"] for r in res["mastery"]], [r["first"]["gain"] for r in res["closed loop (dullness)"]])
    news_auc = {n: auc([r["first"]["news"] for r in res["progress"]], [r["first"]["news"] for r in res[n]]) for n in ECHOES}
    valid = float(np.mean([p not in ("closed loop", "no reference") for r in res["valid inference"] for p in r["valid points"]]))
    healthy_closed = {n: share(n, lambda r: r["first"]["overall"] in ("closed loop", "no reference")) for n in HEALTHY}
    loops_final = {n: share(n, lambda r: r["final"]["overall"] == "closed loop") for n in LOOPS}
    declining_final = share("healthy, declining", lambda r: r["final"]["overall"] not in ("closed loop", "no reference"))
    iv = [r["interval"] for r in res["progress"] if "interval" in r]
    mean_gain = float(np.mean([x["gain"] for x in iv]))
    honesty = {"lower bound above the mean": float(np.mean([x["gain"] - Z * x["se"] > mean_gain for x in iv])),
               "upper bound below the mean": float(np.mean([x["gain"] + Z * x["se"] < mean_gain for x in iv])),
               "median analytic / bootstrap se": float(np.median([x["se"] / x["bootstrap se"] for x in iv])),
               "runs": len(iv)}
    framing = {n: {"alone": news_auc[n],
                   "in context": auc([r["in context"]["news"] for r in res["progress"]], [r["in context"]["news"] for r in res[n]])}
               for n in ECHOES}
    framing["progress news at zero"] = {"alone": share("progress", lambda r: r["first"]["news at zero"]),
                                        "in context": share("progress", lambda r: r["in context"]["news at zero"])}
    passed = {
        "1 separation": all(v >= 0.9 for v in separation.values()) and all(v <= 0.1 for v in loops_as_dynamics.values()),
        "2 gain: mastery over dullness": gain_auc >= 0.8,
        "3 news: progress over echoes": all(v >= 0.8 for v in news_auc.values()),
        "4 valid inference": valid >= 0.9,
        "5 healthy not closed loop": all(v <= 0.1 for v in healthy_closed.values()),
        "6 after the probe": all(v >= 0.8 for v in loops_final.values()) and declining_final >= 0.9,
        "7 K recovered": abs(cal["K"] - 4) <= 1,
        "8 intervals honest": honesty["lower bound above the mean"] <= 0.15 and honesty["upper bound below the mean"] <= 0.15
        and 0.75 <= honesty["median analytic / bootstrap se"] <= 1.25}
    described = {n: {"reference": share(n, lambda r: r["reference"]),
                     "first": {k: share(n, lambda r, k=k: r["first"]["overall"] == k) for k in ("open", "undecided", "closed loop", "no reference")},
                     "probed": share(n, lambda r: r["probed"]),
                     "final": {k: share(n, lambda r, k=k: r["final"]["overall"] == k) for k in ("open", "undecided", "closed loop", "no reference")},
                     "news, median": float(np.median([r["first"]["news"] for r in res[n]])),
                     "gain, median": float(np.nanmedian([r["first"]["gain"] for r in res[n]])),
                     "lead 1 gain, median": float(np.nanmedian([r["lead 1 gain"] for r in res[n]]))} for n in STRUCTURES}
    return {"trials": trials, "seed": seed, "calibration": cal, "separation": separation,
            "loops called dynamics shift (no reference counted)": loops_as_dynamics,
            "gain AUC, mastery over dullness": gain_auc, "news AUC, progress over echo": news_auc,
            "valid inference not closed loop": valid, "healthy read closed loop at 460": healthy_closed,
            "loops closed loop, final": loops_final, "declining not closed loop, final": declining_final,
            "interval honesty": honesty, "framing": framing, "passed": passed, "described": described}


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
