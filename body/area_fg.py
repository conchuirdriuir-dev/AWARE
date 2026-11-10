"""Area's stage B: the loop-dither dissociation in Fault Garage (proposals/area-sense.md section 3; owner, 2026-11-09).

Fault Garage's `export_aware.py --area pool|mirror` writes the record (its `faultgarage/area_plant.py`): level 2, a
burn of 10 turns, then a window of 24 turns of four state-keeping commands, a loop or a dither, every visit reworded.
The reader reads every chunk; scoring alone reads the kind.

    python -m body.area_fg read --model Qwen/Qwen3-0.6B --run runs/area-b-0.6b       # the rehearsal (mirrors)
    python -m body.area_fg read --model Qwen/Qwen3-8B --dtype bfloat16 --run runs/area-b-8b
    python -m body.area_fg score --run runs/area-b-8b

**The sense** (body/area.py): rho on the window's world replies, R = 8, the basis fitted on the episode's world
chunks up to the window's end, mean over the read layers (the last third). **Boundary** is Step -1's news
(body/stepm1.news) on the same states, as the window's mean news, in units of D (the paraphrase spread under the same
reader). Sink rule "norm", the substrate's; the plant is new material, with no TextWorld record to rebuild.

**Pass, both** (fixed before any reading): 1. rho's AUC loop over dither >= 0.80, its 90% lower bound > 0.70;
2. the 90% interval of (rho's AUC - boundary's AUC) on the same windows entirely above zero. 80 episodes, 2,000
bootstrap draws resampling episodes. The 0.6B rehearsal on the mirrors decides nothing; the 8B in bf16 decides.
Reported, deciding nothing: area given boundary; rho on the interleaved path; the Frobenius form; rho and boundary
by layer; the within/between ratio of the window's replies (stage A's noise measure, on Fault Garage's phrasings).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from body.area import R, spread, window_rho
from body.machine import describe, record_machine
from body.stepm1 import episodes, news
from substrate.record import Record, read

BURN, WINDOW, CONTENTS, ROUNDS = 10, 24, 4, 6                  # Fault Garage's area_plant, fixed before play
POOL, MIRROR = (5000, 5199), (905000, 905199)                  # its registry
BARS = {"auc": 0.80, "lower": 0.70, "per_kind": 40, "draws": 2000}


# --- the reading: chunks only ------------------------------------------------------------------------------------------

def window_of(chunks: list[dict]) -> dict:
    """Indices of the window's world replies (in order), of all its chunks (the interleaved path), of the chunks a
    basis may be fitted on (everything up to the window's end), and each reply's command as typed (the self chunk of
    its turn): the plant's structure, read from the chunks alone."""
    turns = [c["turn"] for c in chunks]
    last = BURN + WINDOW - 1
    in_win = [i for i, t in enumerate(turns) if BURN <= t <= last]
    world = [i for i in in_win if chunks[i]["slot"] == "world"]
    command = {c["turn"]: c["text"] for c in chunks if c["slot"] == "self"}
    out = {"world": world, "all": in_win,
           "fit_world": [i for i, c in enumerate(chunks) if c["slot"] == "world" and c["turn"] <= last],
           "fit_all": [i for i, t in enumerate(turns) if t <= last],
           "commands": [command[turns[i]] for i in world]}
    if len(world) != WINDOW or len(set(out["commands"])) != CONTENTS:
        raise ValueError(f"not the plant's window: {len(world)} replies, {len(set(out['commands']))} commands")
    return out


def senses(S: np.ndarray, chunks: list[dict], unit: np.ndarray) -> dict:
    """Everything scoring needs from one episode's states (chunks x layers x width), per layer."""
    w = window_of(chunks)
    N = news(S, [c["slot"] for c in chunks], unit)
    groups = np.array(w["commands"])
    out = {k: np.full(S.shape[1], np.nan) for k in ("rho", "rho_frob", "rho_inter", "boundary", "within", "between",
                                                     "ratio")}
    for j in range(S.shape[1]):
        X = S[:, j].astype(np.float64)
        out["rho"][j], out["rho_frob"][j] = window_rho(X[w["world"]], X[w["fit_world"]], R)
        out["rho_inter"][j] = window_rho(X[w["all"]], X[w["fit_all"]], R)[0]
        out["boundary"][j] = float(np.nanmean(N[w["world"], j]))
        sp = spread(X[w["world"]], groups)
        out["within"][j], out["between"][j], out["ratio"][j] = sp["within"], sp["between"], sp["ratio"]
    out["news"] = N
    return out


def cmd_read(a) -> None:
    from substrate.echo_floor import d_resumable, session_resumable
    from substrate.measure_d import load_sets
    from substrate.states import Reader

    machine = describe()
    if machine["commit"] is None or machine["uncommitted_changes"]:
        raise SystemExit("read from a clean, committed AWARE checkout: the record must carry AWARE's commit")
    run = Path(a.run)
    record = Record(run / "record.jsonl")
    eps = episodes(read(record.path))
    if not eps:
        raise SystemExit("no complete episode in this record")
    started = time.time()
    reader = Reader(a.model, revision=a.revision, device=a.device, dtype=a.dtype, gpu_memory=a.gpu_memory, sink="norm")
    if reader.info["commit"] is None and a.revision is None:
        raise SystemExit("the model's revision is unknown: pass --revision")
    record_machine(record, "area_fg read")
    n = reader.info["layers"]
    layers = list(range(math.ceil(2 * n / 3), n))
    D = d_resumable(reader, load_sets(a.sets), layers, record, False, True)
    record.append("read", "code", step="area_fg", **reader.info, read_layers=layers, D=[float(x) for x in D], R=R)
    for (seed, _), ep in sorted(eps.items()):
        def compute(ep=ep, seed=seed):
            S = reader.states([c["text"] for c in ep["chunks"]], layers)
            return senses(S, ep["chunks"], np.asarray(D, dtype=np.float64)), {"seed": seed, "layers": layers,
                                                                              "chunks": len(ep["chunks"])}
        session_resumable(record, f"areab-{seed}-a{ep['attempts']}", compute)
    record.append("read_end", "code", step="area_fg", episodes=len(eps), seconds=round(time.time() - started, 1))


# --- scoring: the only reader of keys -------------------------------------------------------------------------------

def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def kinds(eps: dict) -> dict:
    """seed -> "loop" or "dither", from the keys; checks that the keys mark the plant's window where reading took it."""
    out = {}
    for (seed, _), ep in eps.items():
        keys = ep["keys"]
        ks = {k.get("area_kind") for k in keys.values()}
        if len(ks) != 1 or None in ks:
            raise SystemExit(f"episode {seed}: no single area_kind in its keys")
        window = sorted(t for t, k in keys.items() if k.get("phase") == "window")
        if window != list(range(BURN, BURN + WINDOW)):
            raise SystemExit(f"episode {seed}: the keys' window is not turns {BURN}-{BURN + WINDOW - 1}")
        out[seed] = ks.pop()
    return out


def score(per: dict, kind: dict, draws: int = BARS["draws"], seed: int = 0) -> dict:
    """per: seed -> senses' per-layer arrays; kind: seed -> loop/dither. The bars and the reported readings."""
    loop = sorted(s for s in per if kind[s] == "loop")
    dith = sorted(s for s in per if kind[s] == "dither")
    m = lambda f, s: float(np.nanmean(per[s][f]))                                   # noqa: E731
    val = {f: ({s: m(f, s) for s in per}) for f in ("rho", "rho_frob", "rho_inter", "boundary", "ratio")}
    rng = np.random.default_rng(seed)
    boot = {"rho": [], "boundary": [], "diff": []}
    lp, dp = np.array([val["rho"][s] for s in loop]), np.array([val["rho"][s] for s in dith])
    lb, db = np.array([val["boundary"][s] for s in loop]), np.array([val["boundary"][s] for s in dith])
    for _ in range(draws):
        i, j = rng.integers(len(loop), size=len(loop)), rng.integers(len(dith), size=len(dith))
        a_r, a_b = auc(lp[i], dp[j]), auc(lb[i], db[j])
        boot["rho"].append(a_r); boot["boundary"].append(a_b); boot["diff"].append(a_r - a_b)
    ci = {k: [float(np.percentile(v, 5)), float(np.percentile(v, 95))] for k, v in boot.items()}
    a_rho, a_bnd = auc(lp, dp), auc(lb, db)
    # area given boundary: rho's residual after a line on boundary, over every episode (reported)
    seeds = loop + dith
    x, y = np.array([val["boundary"][s] for s in seeds]), np.array([val["rho"][s] for s in seeds])
    resid = y - np.polyval(np.polyfit(x, y, 1), x)
    n_layers = len(next(iter(per.values()))["rho"])
    out = {
        "episodes": {"loop": len(loop), "dither": len(dith)},
        "bar1_area": {"auc": a_rho, "ci90": ci["rho"], "pass": bool(a_rho >= BARS["auc"] and ci["rho"][0] > BARS["lower"])},
        "bar2_apart_from_boundary": {"boundary_auc": a_bnd, "boundary_ci90": ci["boundary"], "diff": a_rho - a_bnd,
                                     "diff_ci90": ci["diff"], "pass": bool(ci["diff"][0] > 0)},
        "reported": {
            "area_given_boundary_auc": auc(resid[:len(loop)], resid[len(loop):]),
            "rho_interleaved_auc": auc([val["rho_inter"][s] for s in loop], [val["rho_inter"][s] for s in dith]),
            "rho_frobenius_auc": auc([val["rho_frob"][s] for s in loop], [val["rho_frob"][s] for s in dith]),
            "rho_by_layer_auc": [auc([per[s]["rho"][j] for s in loop], [per[s]["rho"][j] for s in dith])
                                 for j in range(n_layers)],
            "boundary_by_layer_auc": [auc([per[s]["boundary"][j] for s in loop], [per[s]["boundary"][j] for s in dith])
                                      for j in range(n_layers)],
            "noise_ratio_mean": float(np.mean([val["ratio"][s] for s in seeds])),
            "noise_ratio_by_layer": [float(np.mean([per[s]["ratio"][j] for s in seeds])) for j in range(n_layers)],
        },
    }
    out["pass"] = bool(out["bar1_area"]["pass"] and out["bar2_apart_from_boundary"]["pass"])
    return out


def cmd_score(a) -> None:
    run = Path(a.run)
    record = Record(run / "record.jsonl")
    entries = read(record.path)
    eps = episodes(entries)
    reads = [e for e in entries if e["kind"] == "read" and e.get("step") == "area_fg"]
    if not reads:
        raise SystemExit("no area_fg reading in this record: run read first")
    kind = kinds(eps)
    seeds = sorted(kind)
    mirror = all(MIRROR[0] <= s <= MIRROR[1] for s in seeds)
    if not mirror and not all(POOL[0] <= s <= POOL[1] for s in seeds):
        raise SystemExit("seeds outside area's registered pool and its mirrors")
    counts = {k: sum(v == k for v in kind.values()) for k in ("loop", "dither")}
    if counts != {"loop": BARS["per_kind"], "dither": BARS["per_kind"]}:
        raise SystemExit(f"the pool is {counts}, not {BARS['per_kind']} of each kind")
    per = {}
    for (seed, _), ep in eps.items():
        path = run / "checkpoints" / f"areab-{seed}-a{ep['attempts']}.npz"
        if not path.exists():
            raise SystemExit(f"episode {seed} has no reading: run read first")
        data = np.load(path)
        per[seed] = {k: data[k] for k in data.files}
    out = score(per, kind)
    out.update(rehearsal=mirror, deciding=not mirror and "8B" in reads[-1].get("model", ""),
               model=reads[-1].get("model"), read_layers=reads[-1].get("read_layers"), bars=BARS)
    record.append("score", "code", step="area_fg", **out)
    (run / "area_b.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read")
    r.add_argument("--model", required=True)
    r.add_argument("--revision")
    r.add_argument("--run", required=True)
    r.add_argument("--sets", default="material/paraphrases.jsonl")
    r.add_argument("--device", default="auto")
    r.add_argument("--dtype", default="bfloat16")
    r.add_argument("--gpu-memory", type=float)
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    a = p.parse_args(argv)
    {"read": cmd_read, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
