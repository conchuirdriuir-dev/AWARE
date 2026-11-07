"""Choosing X_c's defaults (R, ridge λ, earlier pairs before scoring) before the deciding run: a development probe,
deciding nothing about rate (PLAN.md, "Rate, conditional"; the owner, 2026-11-07).

Material, and only this (the owner's rule, via PR #49): synthetic episodes made here, and the calibration's mirror
seeds 907000–907039 played by both scripts (`runs/rate-cal-dev/`, or any play of those seeds: the scripts are
deterministic), read in the text space (TF-IDF, no model). Never 7000–7039 (spent) or 7100–7139 (the deciding seeds).

**The rule, written before any candidate was scored (2026-11-07):**
- Candidates: R in {4, 8, 16, 32} x λ in {0.1, 1, 10}, with 20 earlier pairs; then, at the chosen R and λ, the
  earlier pairs in {10, 20, 30}.
- Synthetic cells, the same episodes for every candidate (d = 1,024, unit noise, 20 pairs of episodes, as
  `probes/x_sensitivity.py` lays them out): **category** at strength 0.5 and 1 (the calibration's kind), **content** at
  1; and a **null** cell, category at 1 with both arms blind.
- Development: the AUC of contingent over blind on 907000–907039 in the text space, as `score` computes it (main).
- A candidate is **eligible** if the null cell's 90% interval contains 0.5. Among the eligible, the **score** is the
  mean of four AUCs: the three synthetic sensitivity cells and the development main AUC. The highest score wins;
  within 0.01 of the best, the current default (16, 1, 20) is kept if it is there, else the smaller R, then the
  larger λ. The same rule picks the earlier pairs in the second stage.
- Reported beside it, deciding nothing: the development can't-go check, and each cell's interval.

    OMP_NUM_THREADS=1 python -m probes.x_c_defaults --dev runs/rate-cal-dev --out x_c_defaults.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

RS, LAMBDAS, PAIRS = (4, 8, 16, 32), (0.1, 1.0, 10.0), (10, 20, 30)
DEFAULT = (16, 1.0, 20)
CELLS = (("category", 0.5, True), ("category", 1.0, True), ("content", 1.0, True), ("category", 1.0, False))
D_SYN, WORLDS = 1024, 20
TIE = 0.01


def _slots_turns(srcs):
    return ["self" if x == "self" else "world" for x in srcs], [-1] + [i // 2 for i in range(len(srcs) - 1)]


def synthetic_cell(kind: str, strength: float, contingent_arm: bool, r: int, lam: float, pairs: int,
                   worlds: int = WORLDS, d: int = D_SYN) -> dict:
    """The AUC of the 'contingent' arm's mean X_c over the blind arm's, on episodes drawn from a fixed seed (the same for
    every candidate). With contingent_arm False, both arms are blind: the null cell."""
    from body.conditional import conditional
    from body.rate_calibration import compare
    from probes.x_sensitivity import KINDS, episode

    rng = np.random.default_rng([7, KINDS.index(kind), int(strength * 1000), int(contingent_arm)])
    by_world = {}
    for w in range(worlds):
        by_world[w] = {}
        for arm, cont in (("contingent", contingent_arm), ("blind", False)):
            X, srcs = episode(kind, cont, strength, rng, d=d)
            slots, turns = _slots_turns(srcs)
            x = conditional(X, slots, turns, 1.0, r=r, lam=lam, min_pairs=pairs)["X_c"]
            by_world[w][arm] = float(np.nanmean(x))
    return compare(by_world, 2000)


def dev_texts(dev: Path) -> dict:
    """arm -> seed -> the episode, from the development play (both arms)."""
    from body.stepm1 import episodes
    from substrate.record import read

    return {arm: {s: ep for (s, _), ep in episodes(read(dev / arm / "record.jsonl")).items()}
            for arm in ("contingent", "blind")}


def dev_scores(eps: dict, r: int, lam: float, pairs: int, sets_path: str) -> dict:
    """The development main AUC and can't-go check in the text space, as `score` computes them (texts uncut: the
    256-token cap needs the 8B's tokenizer and touches almost no chunk of these scripts)."""
    from body.conditional import conditional
    from body.rate_calibration import ARMS, MIN_CLEAN_TURNS, clean_rows, compare
    from body.step0_habitat import TEXT_VOCAB
    from goalspace.formula import Tfidf
    from substrate.geometry import spread
    from substrate.measure_d import load_sets

    sets = load_sets(sets_path)
    main, clean = {}, {}
    for arm in ARMS:
        for seed, ep in eps[arm].items():
            texts = [c["text"] for c in ep["chunks"]]
            tf = Tfidf(texts[:TEXT_VOCAB])
            T = tf(texts)
            d_text = spread([tf(s["texts"]) for s in sets])
            slots, turns = [c["slot"] for c in ep["chunks"]], [c["turn"] for c in ep["chunks"]]
            x = conditional(T, slots, turns, d_text ** 2 / max(T.shape[1], 1), r=r, lam=lam, min_pairs=pairs)["X_c"]
            main.setdefault(seed, {})[arm] = float(np.nanmean(x))
            rows = [i for i in clean_rows(ep) if np.isfinite(x[i])]
            clean.setdefault(seed, {})[arm] = (float(np.mean(x[rows])) if rows else np.nan, len(rows))
    ok = [w for w, v in clean.items() if len(v) == 2 and all(n >= MIN_CLEAN_TURNS for _, n in v.values())]
    return {"main": compare(main, 2000),
            "can't-go check": compare({w: {a: clean[w][a][0] for a in ARMS} for w in ok}, 2000)}


def evaluate(args) -> dict:
    (r, lam, pairs), dev, sets_path = args
    out = {"R": r, "lambda": lam, "pairs": pairs, "cells": {}}
    for kind, s, cont in CELLS:
        out["cells"][f"{kind} {s:g}" + ("" if cont else " null")] = synthetic_cell(kind, s, cont, r, lam, pairs)
    out["dev"] = dev_scores(dev_texts(Path(dev)), r, lam, pairs, sets_path) if dev else None
    sens = [out["cells"][k]["AUC"] for k in ("category 0.5", "category 1", "content 1")]
    if out["dev"]:
        sens.append(out["dev"]["main"]["AUC"])
    null = out["cells"]["category 1 null"]["90% interval"]
    out["eligible"] = bool(null[0] <= 0.5 <= null[1])
    out["score"] = float(np.mean(sens))
    return out


def choose(results: list[dict]) -> dict:
    """The rule in the module's docstring."""
    ok = [r for r in results if r["eligible"]]
    if not ok:
        return {"chosen": None, "why": "no candidate kept the null cell at chance"}
    best = max(r["score"] for r in ok)
    near = [r for r in ok if r["score"] >= best - TIE]
    default = [r for r in near if (r["R"], r["lambda"], r["pairs"]) == DEFAULT]
    pick = default[0] if default else sorted(near, key=lambda r: (r["R"], -r["lambda"], r["pairs"]))[0]
    return {"chosen": {"R": pick["R"], "lambda": pick["lambda"], "pairs": pick["pairs"]}, "best score": best,
            "within the tie": [(r["R"], r["lambda"], r["pairs"]) for r in near]}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dev", help="the development play of 907000-907039 (contingent/ and blind/ records)")
    p.add_argument("--sets", default="material/paraphrases.jsonl")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--out")
    a = p.parse_args(argv)
    with ProcessPoolExecutor(a.workers) as pool:
        stage1 = list(pool.map(evaluate, [((r, lam, 20), a.dev, a.sets) for r, lam in itertools.product(RS, LAMBDAS)]))
        first = choose(stage1)
        r, lam = first["chosen"]["R"], first["chosen"]["lambda"]
        done = {(x["R"], x["lambda"], x["pairs"]): x for x in stage1}
        stage2 = [done[(r, lam, n)] if (r, lam, n) in done else None for n in PAIRS]
        todo = [((r, lam, n), a.dev, a.sets) for n, x in zip(PAIRS, stage2) if x is None]
        new = iter(pool.map(evaluate, todo))
        stage2 = [x if x is not None else next(new) for x in stage2]
    second = choose(stage2)
    result = {"stage 1": stage1, "stage 1 choice": first, "stage 2": stage2, "stage 2 choice": second}
    for x in stage1 + [y for y in stage2 if y not in stage1]:
        c = x["cells"]
        print(f"R {x['R']:2d} λ {x['lambda']:<4g} pairs {x['pairs']}: score {x['score']:.3f} "
              f"{'ok ' if x['eligible'] else 'NULL'} | cat .5 {c['category 0.5']['AUC']:.2f} cat 1 "
              f"{c['category 1']['AUC']:.2f} content 1 {c['content 1']['AUC']:.2f} null {c['category 1 null']['AUC']:.2f}"
              + (f" | dev {x['dev']['main']['AUC']:.3f} can't-go {x['dev']['can' + chr(39) + 't-go check'].get('AUC', float('nan')):.3f}"
                 if x["dev"] else ""), flush=True)
    print("stage 1:", json.dumps(first), "\nstage 2:", json.dumps(second))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
