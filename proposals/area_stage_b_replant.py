"""Report-only: stage B's real replies replanted as new loops and dithers (proposals/area-sense.md, section 7,
"Why the toy missed"; the numbers reported on PR #72).

    python -m proposals.area_stage_b_replant --packed runs/area-b-8b/states_packed_1.npz [--out material/area_stage_b_replant.json]
    python -m proposals.area_stage_b_replant --states runs/area-b-8b/states

Exploratory, on the first read's 80 episodes (seeds 5000-5079), after stage A2 failed: it asks why the toy missed,
and decides nothing. Each episode's own window replies are dealt out again in a fresh loop or dither order, so the
kind is planted here and the answer key's kind is not read (the key is read only to list the seeds). The replies
before the window stay as they were. Four materials:

  real             the real replies: each visit to a content takes one of that content's six real replies
  resid_shuffled   the real content means, with the real rewording residuals handed out without regard to content
  iso64, iso1024   the real content means and earlier replies, with isotropic noise of the real size

Each is read with the window's basis and with the as-run basis (earlier replies included), R = 8, rho averaged over
the layers. Then what the real rewording noise looks like, and how flat stage A2's drawn contents were.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from body.area import window_rho
from proposals.area_stage_b_diag import auc, contents, load
from proposals.toy_area import STAGE_A2

R, REPEATS, WIDTH = 8, 3, 1024
MATERIALS = (("real replies (real means, real rewording)", "real"),
             ("real means, real residuals handed out across contents", "resid_shuffled"),
             ("real means, isotropic noise in 64 dimensions", "iso64"),
             ("real means, isotropic noise in 1024 dimensions", "iso1024"))


def order(kind, rng, m=4, rounds=6):
    if kind == "loop":
        return np.tile(rng.permutation(m), rounds)
    while True:
        o = rng.permutation(np.repeat(np.arange(m), rounds))
        if (o[1:] != o[:-1]).all():
            return o


def episode(e, kind, rng, variant):
    """rho under (the window's basis, the as-run basis), averaged over the layers, for one replanted episode."""
    S, w, lab = e["S"], e["world"], contents(e)
    pre = np.setdiff1d(e["fit_world"], w)
    groups = np.unique(lab)
    o = order(kind, rng, len(groups), len(w) // len(groups))
    take = {g: list(rng.permutation(np.flatnonzero(lab == g))) for g in groups}
    idx = np.array([take[g].pop() for g in o])
    perm = rng.permutation(len(w))
    zs = rng.normal(size=(S.shape[1], len(w), WIDTH))
    out = np.empty((S.shape[1], 2))
    for j in range(S.shape[1]):
        W, P = S[w, j], S[pre, j]
        M = np.stack([W[lab == g].mean(0) for g in groups])
        res = W - M[lab]
        if variant == "real":
            X, B = W[idx], P
        elif variant == "resid_shuffled":
            X, B = M[o] + res[perm], P
        else:
            d = int(variant[3:])
            n = len(w) // len(groups)
            sig = np.sqrt((res ** 2).sum(1).mean() * n / (n - 1))
            base = np.vstack([P, M])
            U, sg, _ = np.linalg.svd(base - base.mean(0), full_matrices=False)
            co = np.zeros((len(base), d))
            co[:, :len(sg)] = U * sg
            B, X = co[:len(P)], co[len(P):][o] + zs[j, :, :d] * sig / np.sqrt(d)
        out[j] = [window_rho(X, fit, R)[0] for fit in (X, np.vstack([B, X]))]
    return out.mean(0)


def replant(eps, seeds, variant, repeats=REPEATS):
    res = []
    for r in range(repeats):
        v = {"loop": [], "dither": []}
        for i, s in enumerate(seeds):
            for kind in ("loop", "dither"):
                v[kind].append(episode(eps[s], kind, np.random.default_rng(1000 * r + 2 * i + (kind == "loop")),
                                       variant))
        L, D = np.array(v["loop"]), np.array(v["dither"])
        res.append((auc(L[:, 0], D[:, 0]), auc(L[:, 1], D[:, 1])))
    return [float(x) for x in np.mean(res, 0)]


def noise_shape(eps, seeds):
    """The real rewording residuals inside an episode: how many directions they fill, and where."""
    pr, top5, inside = [], [], []
    for s in seeds:
        e = eps[s]
        lab = contents(e)
        for j in range(e["S"].shape[1]):
            W = e["S"][e["world"], j]
            M = np.stack([W[lab == g].mean(0) for g in np.unique(lab)])
            res = W - M[lab]
            ev = np.linalg.svd(res, compute_uv=False) ** 2
            pr.append(ev.sum() ** 2 / (ev ** 2).sum())
            top5.append(ev[:5].sum() / ev.sum())
            Vc = np.linalg.svd(M - M.mean(0), full_matrices=False)[2][:3]
            inside.append(((res @ Vc.T) ** 2).sum() / (res ** 2).sum())
    return {"participation_ratio": float(np.median(pr)), "share_in_top_5": float(np.median(top5)),
            "share_in_content_directions": float(np.median(inside))}


def third_share(eps, seeds, draws=4000, seed=0):
    """The third content direction's share of between-content variance: real episodes against stage A2's draws."""
    real = []
    for s in seeds:
        e = eps[s]
        lab = contents(e)
        for j in range(e["S"].shape[1]):
            W = e["S"][e["world"], j]
            M = np.stack([W[lab == g].mean(0) for g in np.unique(lab)])
            sg = np.linalg.svd(M - M.mean(0), compute_uv=False)[:3] ** 2
            real.append(sg[2] / sg.sum())
    rng, toy = np.random.default_rng(seed), []
    for _ in range(draws):
        C = rng.normal(size=(STAGE_A2["contents"], 3)) * np.sqrt(STAGE_A2["shares"])
        sg = np.linalg.svd(C - C.mean(0), compute_uv=False) ** 2
        toy.append(sg[2] / sg.sum())
    return {"real": float(np.median(real)), "stage_a2_draws": float(np.median(toy))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states")
    ap.add_argument("--packed")
    ap.add_argument("--out")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    a = ap.parse_args()
    eps, kinds, _ = load(a.states, a.packed)
    seeds = sorted(kinds)
    out = {"episodes": len(seeds), "source": a.states or a.packed, "repeats": a.repeats, "R": R, "replanted": {}}
    print(f"{'material':<56} {'window basis':>12} {'as-run basis':>13} {'gap':>7}")
    for name, v in MATERIALS:
        win, run = replant(eps, seeds, v, a.repeats)
        out["replanted"][name] = {"window": win, "as_run": run}
        print(f"{name:<56} {win:>12.3f} {run:>13.3f} {win - run:>+7.3f}")
    out["rewording_noise"] = noise_shape(eps, seeds)
    out["third_content_direction_share"] = third_share(eps, seeds)
    print("\nreal rewording noise (medians):", {k: round(v, 3) for k, v in out["rewording_noise"].items()})
    print("third content direction's share (median):",
          {k: round(v, 3) for k, v in out["third_content_direction_share"].items()})
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
