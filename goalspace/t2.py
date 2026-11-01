"""T2: the owner's formula, inferred from the owner's aims (goalspace/t2_owner_formula.md).

    python -m goalspace.t2 --model Qwen/Qwen3-8B --corpus .../dara-session-transcripts --run runs/t2-8b \\
        --device auto --gpu-memory 6.5 --draft
    python -m goalspace.t2 --model Qwen/Qwen3-0.6B --corpus .../dara-session-transcripts --run runs/t2-06 --draft

The corpus is private: no text from it is recorded, only references and numbers. States are checkpointed in batches,
so an interrupted run resumes when the same command is run again. The confirmation labels and the reader's lines are
read only by `score`, after every code length is computed.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np

from goalspace.formula import Fit, Tfidf, adjusted_rand, auc, kmeans, prequential, surprisal
from substrate.geometry import spread
from substrate.record import Record

BURN, CAP, WINDOW, MIN_CONTROL_CHARS, QUOTE_PREFIX, BATCH = 40, 256, 20, 40, 60, 40
DRAWS, MIN_EVENTS = 2000, 12


# --- material ----------------------------------------------------------------------------------------------------

def material(root: str, seed: int = 0) -> dict:
    """The aims in time order and one control per aim (an owner turn of the same session, not an aim, nearest in time
    and not already taken). Texts stay in memory only."""
    from product.labelled_logs import load_corpus, ts

    R = json.load(open(Path(root) / "study" / "restorations_reading_v0_1.json"))
    sessions = load_corpus(root)
    aims = sorted(R["aims"], key=lambda a: (a["time"], a["id"]))
    prefixes = {a["quote"][:QUOTE_PREFIX].strip() for a in aims}
    rng = random.Random(seed)
    taken, controls = set(), []
    for a in aims:
        s = sessions.get(a["ord"])
        best = None
        if s is not None:
            t0 = ts(a["time"])
            cands = [(abs((s["times"][i] - t0).total_seconds()), rng.random(), i) for i, c in enumerate(s["chunks"])
                     if c["role"] == "human" and len(c["text"]) >= MIN_CONTROL_CHARS
                     and (a["ord"], i) not in taken and not any(p and p in c["text"] for p in prefixes)]
            if cands:
                best = min(cands)[2]
                taken.add((a["ord"], best))
        controls.append(None if best is None else {"text": s["chunks"][best]["text"], "ref": s["chunks"][best]["ref"]})
    return {"aims": aims, "controls": controls, "reading": R}


# --- states ------------------------------------------------------------------------------------------------------

def read_states(reader, texts: list[str], layers: list[int], record: Record, key: str) -> np.ndarray:
    from substrate.echo_floor import session_resumable

    out = []
    for b in range(0, len(texts), BATCH):
        def compute(b=b):
            return {"S": reader.states(texts[b:b + BATCH], layers)}, {"from": b, "n": len(texts[b:b + BATCH])}
        arrays, _ = session_resumable(record, f"{key}-{b:04d}", compute)
        out.append(arrays["S"])
    return np.concatenate(out)                                   # texts x layers x width


# --- the readings ------------------------------------------------------------------------------------------------

def streams(A: np.ndarray, C: np.ndarray, has_c: np.ndarray, floor_var: float) -> dict:
    """Prequential code lengths for the aims' stream and the controls' stream (each fitted on its own earlier texts);
    a control's position is its aim's."""
    aims = prequential(A, floor_var, start=WINDOW)
    idx = np.flatnonzero(has_c)
    ctl = prequential(C[idx], floor_var, start=min(WINDOW, len(idx)))
    gain_c = np.full(len(A), np.nan)
    gain_c[idx] = ctl["gain"]
    return {"gain": aims["gain"], "Lk": aims["Lk"], "k": aims["k"], "gain_control": gain_c,
            "surprisal": surprisal(aims["Lk"], WINDOW)}


def per_space(X_aims: np.ndarray, X_ctl: np.ndarray, has_c: np.ndarray, D: float) -> dict:
    floor_var = D ** 2 / X_aims.shape[1]
    return streams(X_aims, X_ctl, has_c, floor_var)


# --- scoring (the only place the keys are read) -------------------------------------------------------------------

def events(aims: list[dict], R: dict, labels: dict, select) -> list[int]:
    """Index of the first aim at or after each selected departure, in the same session, after burn-in; distinct."""
    out = set()
    for d in R["departures"]:
        lab = labels.get(d["id"])
        if not lab or lab["real"] != ["yes"] or not select(lab):
            continue
        hit = [i for i, a in enumerate(aims) if a["ord"] == d["ord"] and a["time"] >= d["time"]]
        if hit and hit[0] >= BURN:
            out.add(hit[0])
    return sorted(out)


def win_rate(spaces: list[dict], scored: np.ndarray) -> np.ndarray:
    """Per scored aim, the share of spaces (layers) where its gain beats its control's."""
    w = np.mean([sp["gain"][scored] > sp["gain_control"][scored] for sp in spaces], axis=0)
    return w.astype(float)


def score(root: str, aims: list[dict], R: dict, layer_spaces: list[dict], text_space: dict, all_fit_coords) -> dict:
    labels = json.load(open(Path(root) / "study" / "labels_confirmed.json"))
    n = len(aims)
    rng = np.random.default_rng(0)
    scored_all = np.arange(BURN, n)
    paired = scored_all[np.isfinite(layer_spaces[0]["gain_control"][scored_all])]

    # reading 1
    w = win_rate(layer_spaces, paired)
    w_text = win_rate([text_space], paired)
    sess = np.array([aims[i]["ord"] for i in paired])
    uniq = np.unique(sess)
    boot = []
    for _ in range(DRAWS):
        pick = rng.choice(uniq, size=len(uniq))
        m = np.concatenate([np.flatnonzero(sess == s) for s in pick])
        boot.append(w[m].mean())
    lo, hi = np.percentile(boot, [5, 95])
    r1 = {"W": float(w.mean()), "90% interval": [float(lo), float(hi)], "text W": float(w_text.mean()),
          "n": int(len(paired)), "aims without a control": int(len(scored_all) - len(paired)),
          "per layer W": [float(np.mean(sp["gain"][paired] > sp["gain_control"][paired])) for sp in layer_spaces],
          "bits saved per aim, median by layer": [float(np.nanmedian(sp["gain"][paired])) for sp in layer_spaces],
          "bits saved per control, median by layer": [float(np.nanmedian(sp["gain_control"][paired])) for sp in layer_spaces]}
    r1["passed"] = bool(lo > 0.5 and r1["W"] > r1["text W"])

    # reading 2 and the reported event sets
    def event_auc(ev: list[int]) -> dict:
        others = np.array([i for i in scored_all if i not in set(ev)])
        ev = np.array(ev, dtype=int)
        if len(ev) == 0:
            return {"n": 0, "AUC": float("nan")}
        per = [auc(sp["surprisal"][ev], sp["surprisal"][others]) for sp in layer_spaces]
        boot = []
        for _ in range(DRAWS):
            e, o = rng.choice(ev, len(ev)), rng.choice(others, len(others))
            boot.append(np.mean([auc(sp["surprisal"][e], sp["surprisal"][o]) for sp in layer_spaces]))
        lo, hi = np.percentile(boot, [5, 95])
        return {"n": int(len(ev)), "AUC": float(np.mean(per)), "90% interval": [float(lo), float(hi)],
                "text AUC": auc(text_space["surprisal"][ev], text_space["surprisal"][others]),
                "per layer AUC": per}

    asc = events(aims, R, labels, lambda l: "ascent" in l["kind"] and l["who"] == ["owner"])
    r2 = event_auc(asc)
    if r2["n"] < MIN_EVENTS:
        r2["passed"] = None
        r2["note"] = f"too few events to decide (fewer than {MIN_EVENTS})"
    else:
        r2["passed"] = bool(r2["AUC"] >= 0.7 and r2["90% interval"][0] > 0.5 and r2["AUC"] > r2["text AUC"])
    reported = {
        "agent closures": event_auc(events(aims, R, labels, lambda l: "closure" in l["kind"] and l["who"] == ["agent"])),
        "undecidable-then": event_auc(events(aims, R, labels, lambda l: "undecidable-then" in l["kind"]))}

    # the reader's lines (a model's key)
    lined = [i for i, a in enumerate(aims) if a.get("line")]
    lines = np.array([aims[i]["line"] for i in lined])
    line_rows = []
    for coords in all_fit_coords:
        Z = coords[lined]
        if Z.shape[1] == 0:
            line_rows.append({"k": 0, "pair AUC": float("nan"), "adjusted Rand": float("nan")})
            continue
        dist = np.linalg.norm(Z[:, None] - Z[None], axis=2)
        iu = np.triu_indices(len(lined), 1)
        same = lines[:, None] == lines[None]
        line_rows.append({"k": int(Z.shape[1]),
                          "pair AUC": auc(dist[iu][~same[iu]], dist[iu][same[iu]]),
                          "adjusted Rand": adjusted_rand(kmeans(Z, len(set(lines))), lines)})
    reported["reader's lines (a model's key)"] = line_rows
    return {"reading 1": r1, "reading 2": r2, "reported": reported,
            "passed": {"1 the formula regenerates later aims": r1["passed"],
                       "2 the formula registers ascents": r2["passed"]}}


# --- the run -----------------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True, help="the dara-session-transcripts folder (with sessions/ and study/)")
    p.add_argument("--run", required=True)
    p.add_argument("--sets", default="material/paraphrases.jsonl")
    p.add_argument("--draft", action="store_true", help="allow unfrozen paraphrase sets; recorded as a draft run")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"])
    p.add_argument("--gpu-memory", type=float)
    p.add_argument("--dtype", choices=["bfloat16", "float32"])
    a = p.parse_args(argv)

    from substrate.echo_floor import d_resumable
    from substrate.measure_d import load_sets
    from substrate.states import Reader

    reader = Reader(a.model, device=a.device, dtype=a.dtype, gpu_memory=a.gpu_memory)
    n_layers = reader.info["layers"]
    layers = list(range(math.ceil(2 * n_layers / 3), n_layers))
    record = Record(Path(a.run) / "record.jsonl")
    mat = material(a.corpus)
    aims, controls = mat["aims"], mat["controls"]
    record.append("run", "code", **reader.info, read_layers=layers, draft=a.draft, burn=BURN, cap=CAP,
                  aims=[{"id": x["id"], "ord": x["ord"], "time": x["time"]} for x in aims],
                  controls=[None if c is None else c["ref"] for c in controls])
    tok = reader.tokenizer

    def capped(t):
        return tok.decode(tok(t, add_special_tokens=False)["input_ids"][:CAP])
    aim_texts = [capped(x["quote"]) for x in aims]
    has_c = np.array([c is not None for c in controls])
    ctl_texts = [capped(c["text"]) if c is not None else "" for c in controls]

    sets = load_sets(a.sets)
    D = d_resumable(reader, sets, layers, record, a.draft, True)
    SA = read_states(reader, aim_texts, layers, record, "aims")
    SC = read_states(reader, [t for t, h in zip(ctl_texts, has_c) if h], layers, record, "controls")
    C_full = np.zeros_like(SA)
    C_full[has_c] = SC

    layer_spaces, all_fit_coords = [], []
    for j, layer in enumerate(layers):
        sp = per_space(SA[:, j].astype(float), C_full[:, j].astype(float), has_c, float(D[j]))
        layer_spaces.append(sp)
        fit = Fit(SA[:, j].astype(float), float(D[j]) ** 2 / SA.shape[2])
        all_fit_coords.append(fit.coords(SA[:, j], fit.choose_k()))
        record.append("layer", "code", layer=layer, D=float(D[j]), k=[int(k) for k in sp["k"]],
                      gain=[None if not np.isfinite(g) else float(g) for g in sp["gain"]],
                      gain_control=[None if not np.isfinite(g) else float(g) for g in sp["gain_control"]],
                      surprisal=[None if not np.isfinite(g) else float(g) for g in sp["surprisal"]])

    burn_docs = aim_texts[:BURN] + [t for t, h in zip(ctl_texts[:BURN], has_c[:BURN]) if h]
    tf = Tfidf(burn_docs)
    D_text = spread([tf(s["texts"]) for s in sets])
    text_space = per_space(tf(aim_texts), tf(ctl_texts), has_c, D_text)
    record.append("text baseline", "code", vocabulary=len(tf.vocab), D=D_text,
                  k=[int(k) for k in text_space["k"]])

    result = score(a.corpus, aims, mat["reading"], layer_spaces, text_space, all_fit_coords)
    report = record.append("report", "code", **result)
    r1, r2 = report["reading 1"], report["reading 2"]
    print(f"{reader.info['model']}{'  DRAFT sets' if a.draft else ''}; layers {layers[0]}-{layers[-1]}")
    print(f"  reading 1: W {r1['W']:.3f} interval {np.round(r1['90% interval'], 3).tolist()} text W {r1['text W']:.3f}"
          f" (n {r1['n']}, without a control {r1['aims without a control']})")
    print(f"  reading 2: AUC {r2['AUC']:.3f} interval {np.round(r2.get('90% interval', [np.nan] * 2), 3).tolist()}"
          f" text AUC {r2.get('text AUC', float('nan')):.3f} (events {r2['n']})")
    for name, v in report["reported"].items():
        if isinstance(v, dict):
            print(f"  reported, {name}: AUC {v['AUC']:.3f} (n {v['n']})")
    print("  passed:", report["passed"])


if __name__ == "__main__":
    main()
