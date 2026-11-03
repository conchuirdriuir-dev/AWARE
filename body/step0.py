"""Body readout, Step 0: does cross-progress read the agent's closures? (PLAN.md, "Body readout", Step 0.)

    python -m body.step0 --model Qwen/Qwen3-8B --corpus .../dara-session-transcripts --run runs/body0-8b \\
        --device auto --gpu-memory 6.5 --draft
    python -m body.step0 --model Qwen/Qwen3-0.6B --corpus ... --run runs/body0-06 --draft --sessions S14 S22

The corpus is private: no text from it is recorded, only references and numbers. States and code lengths are
checkpointed per session, so an interrupted run resumes when the same command is run again. The confirmation labels
and first-return times are read only by `score`, after every code length is computed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from body.progress import by_source, cost
from goalspace.formula import Tfidf, auc
from substrate.corpus import load_session, sessions as session_paths
from substrate.geometry import spread
from substrate.record import Record

CAP, BURN_SELF, MIN_PARTNER_CHARS, BATCH = 256, 40, 20, 40
EVENT_CHUNKS, MIN_EVENT_CHUNKS, GAP_MINUTES = 10, 3, 60
AUC_BAR, MIN_CLOSURES, MIN_ASCENTS, DRAWS = 0.70, 15, 12, 2000
D021 = ("S14", "2026-10-12T15:44", "2026-10-12T23:56")        # the spec's Step 1 case, read here as one case only
SYSTEM_PREFIXES = ("[SYSTEM", "<system-reminder>", "[Artifact comment", "<task-notification>",
                   "This session is being continued from a previous conversation")


def ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "")[:26])


# --- material ----------------------------------------------------------------------------------------------------

def source(chunk: dict) -> str | None:
    """partner, self or tool, as registered; None for what is left out (tool calls, system text, short turns)."""
    if chunk["provenance"] == "self":
        return None if "tool" in chunk else "self"
    if chunk["cause"] == "elicited":
        return "tool"
    text = chunk["text"]
    if text.startswith(SYSTEM_PREFIXES) or len(text) < MIN_PARTNER_CHARS:
        return None
    return "partner"


def material(root: str, only: list[str] | None = None) -> dict:
    """Per session: its kept chunks in time order, each {"text", "source", "ref", "at"}. Texts stay in memory only."""
    out = {}
    for path in session_paths(Path(root) / "sessions"):
        key = path.name.split(".")[0]
        if only and key not in only:
            continue
        kept = []
        for c in load_session(path):
            s = source(c)
            if s is not None:
                kept.append({"text": c["text"], "source": s, "ref": c["ref"], "at": c["at"]})
        out[key] = kept
    return out


# --- states ------------------------------------------------------------------------------------------------------

def read_states(reader, texts: list[str], layers: list[int], record: Record, key: str) -> np.ndarray:
    from substrate.echo_floor import session_resumable

    out = []
    for b in range(0, len(texts), BATCH):
        def compute(b=b):
            return {"S": reader.states(texts[b:b + BATCH], layers)}, {"from": b, "n": len(texts[b:b + BATCH])}
        arrays, _ = session_resumable(record, f"{key}-{b:05d}", compute)
        out.append(arrays["S"])
    return np.concatenate(out)                                   # texts x layers x width


def session_codes(record: Record, key: str, X: np.ndarray, sources: list[str], floor_var: float, every: int) -> dict:
    from substrate.echo_floor import session_resumable

    def compute():
        r = by_source(X, sources, floor_var, burn_self=BURN_SELF, every=every)
        r["C"] = cost(r["L_now"])
        return {k: np.asarray(v, dtype=float) for k, v in r.items()}, {"n": len(X), "every": every}
    arrays, _ = session_resumable(record, key, compute)
    return arrays


# --- scoring (the only place the keys are read) -------------------------------------------------------------------

def keys(root: str) -> list[dict]:
    """Each departure with its owner's labels and its first-return time (None if none recorded)."""
    root = Path(root) / "study"
    R = json.load(open(root / "restorations_reading_v0_1.json"))
    L = json.load(open(root / "labels_confirmed.json"))
    md = open(root / "departures_confirmed.md").read()
    ret = {}
    for part in re.split(r"\n## ", md)[1:]:
        m = re.search(r"\*\*First return \(([^)]+)\)", part)
        ret[part[:4]] = m.group(1) if m else None
    return [{"id": d["id"], "ord": d["ord"], "time": d["time"], "return": ret.get(d["id"]), "label": L.get(d["id"])}
            for d in R["departures"]]


def scored_index(sess: dict) -> list[int]:
    return [i for i in range(len(sess["chunks"])) if np.isfinite(sess["codes"][0]["L_now"][i])]


def windows(sessions: dict, deps: list[dict]) -> dict:
    """Closure and ascent windows, and one author-matched control per event (same session, outside every departure's
    span, more than GAP_MINUTES from any departure, nearest in time, never reused)."""
    gap = dt.timedelta(minutes=GAP_MINUTES)
    events = []
    for d in deps:
        lab = d["label"]
        if not lab or lab["real"] != ["yes"] or d["ord"] not in sessions:
            continue
        s = sessions[d["ord"]]
        idx = scored_index(s)
        times = [ts(s["chunks"][i]["at"]) for i in idx]
        if "closure" in lab["kind"] and lab["who"] == ["agent"]:
            if not d["return"]:
                continue
            before = [i for i, t in zip(idx, times) if t < ts(d["return"])][-EVENT_CHUNKS:]
            events.append({"id": d["id"], "kind": "closure", "ord": d["ord"], "rows": before})
        if "ascent" in lab["kind"] and lab["who"] == ["owner"]:
            after = [i for i, t in zip(idx, times) if t > ts(d["time"])][:EVENT_CHUNKS]
            events.append({"id": d["id"], "kind": "ascent", "ord": d["ord"], "rows": after})
    dropped = [e["id"] + "/" + e["kind"] for e in events if len(e["rows"]) < MIN_EVENT_CHUNKS]
    events = [e for e in events if len(e["rows"]) >= MIN_EVENT_CHUNKS]

    taken = {k: set() for k in sessions}
    no_control = []
    for e in events:
        s = sessions[e["ord"]]
        idx = scored_index(s)
        spans = []
        for d in deps:
            if d["ord"] != e["ord"]:
                continue
            a = ts(d["time"])
            b = ts(d["return"]) if d["return"] else a
            spans.append((a - gap, max(a, b) + gap))
        ok = [i for i in idx if i not in taken[e["ord"]]
              and not any(lo <= ts(s["chunks"][i]["at"]) <= hi for lo, hi in spans)]
        n = len(e["rows"])
        centre = ts(s["chunks"][e["rows"][n // 2]]["at"])
        best = None
        pos = {i: p for p, i in enumerate(idx)}
        for start in range(len(ok) - n + 1):
            run = ok[start:start + n]
            if pos[run[-1]] - pos[run[0]] != n - 1:          # contiguous among the scored chunks
                continue
            dist = abs((ts(s["chunks"][run[n // 2]]["at"]) - centre).total_seconds())
            if best is None or dist < best[0]:
                best = (dist, run)
        if best is None:
            no_control.append(e["id"] + "/" + e["kind"])
            e["control"] = None
        else:
            e["control"] = best[1]
            taken[e["ord"]].update(best[1])
    return {"events": [e for e in events if e["control"] is not None], "dropped (fewer than 3 chunks)": dropped,
            "no control": no_control}


def _stat(sessions, e, rows, space, field):
    return float(np.nanmean(sessions[e["ord"]]["codes"][space][field][rows]))


def reading(sessions: dict, evs: list[dict], spaces: list[int], text_space: int, lower: bool, rng) -> dict:
    """AUC per layer of the events' window mean X against their controls' (closures lower, ascents higher), its mean
    over layers, a 90% interval resampling events with their controls, and the text baseline's AUC."""
    if not evs:
        return {"n": 0, "AUC": float("nan")}

    def a(space, pick):
        ev = [_stat(sessions, evs[i], evs[i]["rows"], space, "X") for i in pick]
        ct = [_stat(sessions, evs[i], evs[i]["control"], space, "X") for i in pick]
        return auc(ct, ev) if lower else auc(ev, ct)
    allp = list(range(len(evs)))
    per = [a(sp, allp) for sp in spaces]
    boot = [np.mean([a(sp, rng.choice(len(evs), len(evs))) for sp in spaces]) for _ in range(DRAWS)]
    lo, hi = np.percentile(boot, [5, 95])
    return {"n": len(evs), "AUC": float(np.mean(per)), "90% interval": [float(lo), float(hi)],
            "text AUC": a(text_space, allp), "per layer AUC": per}


def score(root: str, sessions: dict, n_layers: int) -> dict:
    deps = keys(root)
    w = windows(sessions, deps)
    rng = np.random.default_rng(0)
    spaces, text = (list(range(n_layers)), n_layers) if n_layers else ([0], 0)   # text-only dry run: one space
    clo = [e for e in w["events"] if e["kind"] == "closure"]
    asc = [e for e in w["events"] if e["kind"] == "ascent"]
    r1 = reading(sessions, clo, spaces, text, lower=True, rng=rng)
    r2 = reading(sessions, asc, spaces, text, lower=False, rng=rng)
    if r1["n"] < MIN_CLOSURES:
        r1["passed"], r1["note"] = None, f"too few events to decide (fewer than {MIN_CLOSURES})"
    else:
        r1["passed"] = bool(r1["AUC"] >= AUC_BAR and r1["90% interval"][0] > 0.5 and r1["AUC"] > r1["text AUC"])
    if r2["n"] < MIN_ASCENTS:
        r2["passed"], r2["note"] = None, f"too few events to decide (fewer than {MIN_ASCENTS})"
    else:
        r2["passed"] = bool(r2["90% interval"][0] > 0.5)

    def med(evs, field, which):
        vals = [np.mean([_stat(sessions, e, e[which], sp, field) for sp in spaces]) for e in evs]
        return float(np.median(vals)) if vals else float("nan")
    reported = {}
    for name, evs in (("closures", clo), ("ascents", asc)):
        reported[name] = {f: {"events": med(evs, f, "rows"), "controls": med(evs, f, "control")}
                          for f in ("X", "T", "S", "C")}
        tail = []
        for e in evs:
            r = e["rows"]
            if len(r) >= 6:
                tail.append(np.mean([_stat(sessions, e, r[-3:], sp, "T") - _stat(sessions, e, r[-6:-3], sp, "T")
                                     for sp in spaces]))
        reported[name]["T, last 3 minus the 3 before"] = float(np.median(tail)) if tail else float("nan")
    ctl_X = [np.mean([_stat(sessions, e, e["control"], sp, "X") for sp in spaces]) for e in w["events"]]
    split = float(np.median(ctl_X)) if ctl_X else float("nan")
    quad = {}
    for name, evs, which in (("closures", clo, "rows"), ("ascents", asc, "rows"), ("controls", w["events"], "control")):
        cells = {"cheap, X above": 0, "cheap, X below": 0, "costly, X above": 0, "costly, X below": 0}
        for e in evs:
            c = np.mean([_stat(sessions, e, e[which], sp, "C") for sp in spaces])
            x = np.mean([_stat(sessions, e, e[which], sp, "X") for sp in spaces])
            if np.isfinite(c) and np.isfinite(x):
                cells[f"{'cheap' if c < 0 else 'costly'}, X {'above' if x >= split else 'below'}"] += 1
        quad[name] = cells
    reported["quadrants (split at the controls' median X)"] = quad
    reported["dropped (fewer than 3 chunks)"] = w["dropped (fewer than 3 chunks)"]
    reported["no control"] = w["no control"]
    key, lo, hi = D021
    if key in sessions:
        s = sessions[key]
        rows = [i for i in scored_index(s) if ts(lo) <= ts(s["chunks"][i]["at"]) <= ts(hi)]
        reported["the D021 window, chunk by chunk (mean over layers)"] = [
            {"at": s["chunks"][i]["at"][:16], "ref": s["chunks"][i]["ref"],
             **{f: float(np.mean([s["codes"][sp][f][i] for sp in spaces])) for f in ("X", "T", "S", "C")}}
            for i in rows]
    reported["not yet computed in this code"] = ["coupling on the same windows", "proxy B", "proxy C",
                                                 "X with W of 10 and 40 chunks"]
    return {"reading 1, closures": r1, "reading 2, ascents": r2, "reported": reported,
            "passed": {"1 closures": r1["passed"], "2 ascents": r2["passed"]}}


# --- the run -----------------------------------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True, help="the dara-session-transcripts folder (with sessions/ and study/)")
    p.add_argument("--run", required=True)
    p.add_argument("--sets", default="material/paraphrases.jsonl")
    p.add_argument("--draft", action="store_true", help="allow unfrozen paraphrase sets; recorded as a draft run")
    p.add_argument("--sessions", nargs="*", help="a subset of sessions (rehearsal only; the result reads all)")
    p.add_argument("--every", type=int, default=1, help="score every n-th self chunk (the registered thinning)")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"])
    p.add_argument("--gpu-memory", type=float)
    p.add_argument("--dtype", choices=["bfloat16", "float32"])
    p.add_argument("--text-only", action="store_true",
                   help="plumbing dry run without a model: the TF-IDF space alone, texts cut at 1,000 characters")
    a = p.parse_args(argv)

    from substrate.echo_floor import d_resumable
    from substrate.measure_d import load_sets
    from substrate.states import Reader

    t_start = time.time()
    record = Record(Path(a.run) / "record.jsonl")
    mat = material(a.corpus, a.sessions)
    counts = {k: {s: sum(c["source"] == s for c in v) for s in ("partner", "self", "tool")} for k, v in mat.items()}
    sets = load_sets(a.sets)
    if a.text_only:
        reader, layers, info = None, [], {"model": "none (text-only dry run)"}

        def capped(t):
            return t[:1000]
        D = []
    else:
        reader = Reader(a.model, device=a.device, dtype=a.dtype, gpu_memory=a.gpu_memory,
                        sink="first")  # T2's registered sink rule, as Step 0 registers
        n_layers = reader.info["layers"]
        layers = list(range(math.ceil(2 * n_layers / 3), n_layers))
        info = reader.info
        tok = reader.tokenizer

        def capped(t):
            return tok.decode(tok(t, add_special_tokens=False)["input_ids"][:CAP])
        D = d_resumable(reader, sets, layers, record, a.draft, True)
    record.append("run", "code", **info, read_layers=layers, draft=a.draft, cap=CAP, burn_self=BURN_SELF,
                  every=a.every, text_only=a.text_only, sessions=counts)
    sessions, timing = {}, {}
    for key, chunks in mat.items():
        t0 = time.time()
        texts = [capped(c["text"]) for c in chunks]
        S = read_states(reader, texts, layers, record, f"states-{key}") if layers else None
        t1 = time.time()
        srcs = [c["source"] for c in chunks]
        codes = []
        for j, layer in enumerate(layers):
            codes.append(session_codes(record, f"codes-{key}-L{layer}", S[:, j].astype(float), srcs,
                                       float(D[j]) ** 2 / S.shape[2], a.every))
        tf = Tfidf(texts[:BURN_SELF])
        D_text = spread([tf(s["texts"]) for s in sets])
        T = tf(texts)
        codes.append(session_codes(record, f"codes-{key}-text", T, srcs, D_text ** 2 / max(T.shape[1], 1), a.every))
        sessions[key] = {"chunks": chunks, "codes": codes}
        timing[key] = {"chunks": len(chunks), "read seconds": round(t1 - t0, 1),
                       "code seconds": round(time.time() - t1, 1)}
        record.append("session", "code", session=key, **timing[key],
                      scored=int(np.isfinite(codes[0]["L_now"]).sum()))
        print(f"  {key}: {timing[key]}", flush=True)

    result = score(a.corpus, sessions, len(layers))
    result["timing"] = {"sessions": timing, "total seconds": round(time.time() - t_start, 1)}
    report = record.append("report", "code", **result)
    print(f"{info['model']}{'  DRAFT sets' if a.draft else ''}; layers {layers}; sessions {len(sessions)}")
    for name in ("reading 1, closures", "reading 2, ascents"):
        r = report[name]
        print(f"  {name}: AUC {r['AUC']:.3f} interval {r.get('90% interval')} text {r.get('text AUC')} "
              f"(events {r['n']}) passed {r['passed']}")
    print("  reported:", json.dumps(report["reported"], indent=1))


if __name__ == "__main__":
    main()
