"""Report-only diagnostics on area's stage B, after its deciding result (proposals/area-sense.md, section 7).

    python -m proposals.area_stage_b_diag --states runs/area-b-8b/states [--out material/area_stage_b_diag.json]
    python -m proposals.area_stage_b_diag --packed runs/area-b-8b/states_packed_1.npz

Reads the states the sense received (one <seed>.npz an episode: S chunks x layers x width, with world, fit_world
and the chunks' slots; kinds.json; meta.json), or their packed copy (each layer's chunks as coordinates in their own
span, which keeps every distance and projection inside an episode). It decides nothing: stage B's result stands as
fixed. The list of readings was fixed before the states were opened:

  1. the result as run (basis on every world reply up to the window's end, R = 8), to check the copy
  2. how much of the window's variance lies inside the basis
  3. the basis fitted on the window's replies only
  4. the basis fitted on the four contents' means (a ceiling: it uses which reply answers which command)
  5. R = 4, 12, 16
  6. each by layer

and, after those were read, how much of each content direction either basis keeps. kinds.json is the answer key:
only this scoring reads it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from body.area import basis, window_rho


def load(states=None, packed=None):
    """{seed: {"S": chunks x layers x r, "world", "fit_world"}}, kinds, meta."""
    if packed:
        z = np.load(packed, allow_pickle=False)
        kinds, meta = json.loads(str(z["file__kinds.json"])), json.loads(str(z["file__meta.json"]))
        eps = {s: {"S": z[f"{s}__C"].astype(np.float64), "world": z[f"{s}__world"], "fit_world": z[f"{s}__fit_world"]}
               for s in kinds}
    else:
        d = Path(states)
        kinds, meta = json.loads((d / "kinds.json").read_text()), json.loads((d / "meta.json").read_text())
        eps = {}
        for s in kinds:
            z = np.load(d / f"{s}.npz", allow_pickle=False)
            eps[s] = {"S": z["S"].astype(np.float64), "world": z["world"], "fit_world": z["fit_world"]}
    return eps, kinds, meta


def contents(e):
    """Which command each window reply answers: the self chunk before it. Commands are fixed texts read on their
    own, so one command gives one state."""
    cmd = e["S"][e["world"] - 1, 0]
    return np.unique(np.round(cmd, 3), axis=0, return_inverse=True)[1].ravel()


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum())
                 / (len(pos) * len(neg)))


def interval(v, y, draws=2000, seed=0):
    rng = np.random.default_rng(seed)
    p, n = v[y], v[~y]
    b = [auc(p[rng.integers(0, len(p), len(p))], n[rng.integers(0, len(n), len(n))]) for _ in range(draws)]
    return [auc(p, n), float(np.percentile(b, 5)), float(np.percentile(b, 95))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states")
    ap.add_argument("--packed")
    ap.add_argument("--out")
    a = ap.parse_args()
    eps, kinds, meta = load(a.states, a.packed)
    seeds = sorted(kinds)
    y = np.array([kinds[s] == "loop" for s in seeds])
    nl = eps[seeds[0]]["S"].shape[1]
    fit_all = lambda e, j: e["S"][e["fit_world"], j]
    fit_win = lambda e, j: e["S"][e["world"], j]

    def fit_means(e, j):
        lab, W = contents(e), e["S"][e["world"], j]
        return np.stack([W[lab == g].mean(0) for g in np.unique(lab)])

    def reading(fit, r):
        M = np.array([[window_rho(eps[s]["S"][eps[s]["world"], j], fit(eps[s], j), r)[0] for j in range(nl)]
                      for s in seeds])
        return {"auc": interval(M.mean(1), y), "by_layer": [auc(M[y, j], M[~y, j]) for j in range(nl)]}

    out = {"episodes": len(seeds), "loops": int(y.sum()), "layers": meta.get("layers"),
           "source": a.states or a.packed, "readings": {}}
    rows = [("as run: all world replies, R=8", fit_all, 8), ("the window's replies only, R=8", fit_win, 8),
            ("the four contents' means, R=3", fit_means, 3)]
    rows += [(f"as run, R={r}", fit_all, r) for r in (4, 12, 16)]
    rows += [(f"window only, R={r}", fit_win, r) for r in (4, 12, 16)]
    print(f"{'basis':<34} {'AUC loop over dither [90%]':<28} by layer")
    for name, fit, r in rows:
        res = out["readings"][name] = reading(fit, r)
        u, lo, hi = res["auc"]
        by = res["by_layer"]
        print(f"{name:<34} {u:.3f} [{lo:.3f}, {hi:.3f}]         {min(by):.2f}-{max(by):.2f}")

    share = {"as run": [], "window only": []}
    keep = {"as run": [], "window only": []}
    split = []
    for s in seeds:
        e = eps[s]
        lab = contents(e)
        for j in range(nl):
            W = e["S"][e["world"], j]
            X = W - W.mean(0)
            M = np.stack([W[lab == g].mean(0) for g in np.unique(lab)])
            sg, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)[1:]
            split.append(sg[:3] ** 2 / (sg[:3] ** 2).sum())
            for name, fit in (("as run", fit_all), ("window only", fit_win)):
                V = basis(fit(e, j), 8)[1]
                share[name].append(float(((X @ V) ** 2).sum() / (X ** 2).sum()))
                keep[name].append([float(((Vt[i] @ V) ** 2).sum()) for i in range(3)])
    out["window_variance_in_basis"] = {k: float(np.median(v)) for k, v in share.items()}
    out["content_variance_by_direction"] = [float(x) for x in np.median(split, 0)]
    out["content_direction_kept"] = {k: {"median": [float(x) for x in np.median(v, 0)],
                                         "p10": [float(x) for x in np.percentile(v, 10, 0)]} for k, v in keep.items()}
    print("\nwindow variance inside the basis (median):",
          {k: round(v, 2) for k, v in out["window_variance_in_basis"].items()})
    print("between-content variance over its three directions (median):",
          [round(x, 2) for x in out["content_variance_by_direction"]])
    for k, v in out["content_direction_kept"].items():
        print(f"share of each content direction inside the basis, {k}: median {[round(x, 2) for x in v['median']]}, "
              f"10th percentile {[round(x, 2) for x in v['p10']]}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
