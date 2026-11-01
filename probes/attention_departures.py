"""Attention to the owner before a departure, against matched controls (a probe beside `attention_gap.py`, not a
step; neither track's criterion).

`attention_gap.py` found that the agent's attention to the owner's messages falls as tokens pass since the owner
spoke, and that the owner's first message (the origin) is under-weighted throughout. This asks whether either is
lower just before a confirmed departure than at matched moments that are not departures.

Written 2026-11-01, before its code.
- **Departures:** the owner's confirmed labels (`study/labels_confirmed.json` and the Restorations reading's
  departures, on the transcripts repository's default branch; read from a directory given by `--study`), real = yes,
  grouped as the study's `pilot5.py` does: closure if marked closure, else accommodation, else ascent, else
  undecidable (set aside). Closure and accommodation are the compass's targets; ascent is what it must not break.
- **Anchor:** an owner departure is the owner's message with the departure's event id. An agent or joint departure
  refers to the turn's end, so its anchor is the agent's last prose chunk at or before the departure's time.
- **Too early to read:** a departure whose anchor is among its session's first 40 chunks has too little before it
  and is left out, as controls are.
- **Controls:** for each departure, two chunks of the same author class (agent prose, or owner message) in the same
  session, at least 200 characters long, at least 100 chunks from every departure in the session and not among its
  first 40 chunks; seeded.
- **Window,** the same for departures and controls: the session's first chunk if it is an owner message (the origin;
  sessions whose archive begins mid-session have none, and their origin measure is left out), then the chunks just
  before the anchor, up to W tokens in all, anchor excluded (W = 8,192 on Qwen3-0.6B here). A constructed context:
  the origin kept, the recent stretch as it was. ChatML per chunk with true roles, chunks capped at 256 tokens.
- **Queries:** up to 32 evenly spaced tokens of each agent-prose chunk starting in the last 2,048 tokens before the
  anchor. Attention rows recomputed per layer, head mean, as `attention_gap.py`; the same two lifts over the base rate
  among content keys: toward all owner messages, and toward the origin.
- **Adjusted for the gap:** the gap curve (mean lift per gap bin and layer, `attention_gap.py`'s bins) is taken from
  the controls' queries; each query's lift minus its bin's control mean is its residual. A window's score is its
  queries' mean residual over the middle layers (6-26 of 28, scaled). The gap at the anchor (tokens since the owner
  last spoke) is reported on its own: it needs no model.
  The gap is measured in the whole session (capped chunks), not the window; inside the window, a query whose stretch
  holds no owner message has its gap counted from the origin; a window with no owner message at all (no origin, and
  the owner silent through the stretch) scores nothing (amended after a crash on one, before any result was read).
- **How it is read,** per kind with at least five departures scored: attention to the owner (or to the origin) is
  weaker before that kind if the AUC of control scores over departure scores is 0.70 or more; 0.30 or less means
  stronger; between, no difference at this n. Pilot scale: no claim beyond it.

    python probes/attention_departures.py --model Qwen/Qwen3-0.6B --corpus ../dara-session-transcripts/sessions \\
        --study /tmp/study --run runs/attn-dep-qwen06 --window 8192
"""
from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from substrate.corpus import load_session  # noqa: E402
from substrate.record import Record  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gap = _load("attention_gap")
long, share = gap.long, gap.share
SELF, EXT = share.SELF, share.EXT
PRE = 2048
KINDS = ("closure", "accommodation", "ascent")


def departures(study: Path):
    reading = json.loads((study / "restorations_reading_v0_1.json").read_text())
    labels = json.loads((study / "labels_confirmed.json").read_text())
    out = []
    for d in reading["departures"]:
        lab = labels.get(d["id"])
        if not lab or lab["real"] != ["yes"]:
            continue
        k = lab["kind"]
        kind = next((x for x in KINDS if x in k), "undecidable")
        if kind == "undecidable":
            continue
        who = "owner" if lab["who"] == ["owner"] else "agent"
        out.append({"id": d["id"], "session": d["ord"], "time": d["time"], "ref": d["ref"], "kind": kind, "who": who})
    return out


def author(ch):
    c = share.kind(ch, "prose")
    return "agent" if c == SELF else "owner" if c == EXT else None


def anchor(chunks, dep):
    if dep["who"] == "owner":
        hit = [i for i, c in enumerate(chunks) if c["ref"][1] == dep["ref"]]
        if hit:
            return hit[0]
    ats = [c["at"][:19] for c in chunks]
    j = bisect.bisect_right(ats, dep["time"][:19]) - 1
    while j >= 0 and author(chunks[j]) != dep["who"]:
        j -= 1
    return j if j >= 0 else None


def build(tok, chunks, at, W, cap):
    """The origin (if the session opens with the owner), then the chunks just before `at`, up to W tokens."""
    def cost(c):
        return min(len(tok(c["text"], add_special_tokens=False)["input_ids"]), cap) + 6
    has_origin = author(chunks[0]) == "owner" and at > 0
    budget = W - 2 - (cost(chunks[0]) if has_origin else 0)
    lo = at
    while lo - 1 >= (1 if has_origin else 0) and budget - cost(chunks[lo - 1]) >= 0:
        lo -= 1
        budget -= cost(chunks[lo])
    picked = ([chunks[0]] if has_origin else []) + chunks[lo:at]
    ids, cls, _, owner, kept = share.window(tok, picked, W, cap, match="prose")
    return ids, cls, owner, kept, has_origin


def true_gap(tok, chunks, at, cap):
    """Tokens (capped chunks, wrappers included) between the owner's last message before `at` and `at`, in the
    whole session; -1 if the owner has not spoken."""
    n = 0
    for j in range(at - 1, -1, -1):
        if author(chunks[j]) == "owner":
            return n
        n += min(len(tok(chunks[j]["text"], add_special_tokens=False)["input_ids"]), cap) + 6
    return -1


def pre_queries(owner, cls, kept, T):
    out = []
    for i in range(len(kept)):
        pos = np.flatnonzero(owner == i)
        if len(pos) and cls[pos[0]] == SELF and pos[0] >= T - PRE:
            out.extend(pos[np.linspace(0, len(pos) - 1, min(long.PER_CHUNK, len(pos))).round().astype(int)].tolist())
    return np.unique(np.array(out, dtype=int))


def run(model_name, corpus, study, run_dir, W, device, dtype, revision=None, seed=0, cap=256, per=2, limit=None):
    model, tok = long.load(model_name, device, dtype, revision)
    L = len(model.model.layers)
    deps = departures(Path(study))
    record = Record(Path(run_dir) / "record.jsonl")
    record.append("attention departures, run", "code", model=model_name, revision=revision,
                  commit=getattr(model.config, "_commit_hash", None), layers=L, window=W, cap=cap, pre=PRE,
                  controls_per=per, seed=seed, device=device, dtype=dtype,
                  departures=[{k: d[k] for k in ("id", "session", "kind", "who")} for d in deps])
    rng = np.random.default_rng(seed)
    loaded, windows = {}, []
    for s in sorted({d["session"] for d in deps}):
        path = Path(corpus) / f"{s}.tar.gz"
        if path.exists():
            loaded[s] = load_session(path)
    for d in deps:
        ch = loaded.get(d["session"])
        a = anchor(ch, d) if ch else None
        if a is None or a < 40:
            print(f"{d['id']}: {'no anchor' if a is None else f'anchor at chunk {a}, under 40'}", flush=True)
            continue
        windows.append({"role": "departure", "dep": d, "at": a})
    for d in [w["dep"] for w in windows]:
        ch = loaded[d["session"]]
        near = [w["at"] for w in windows if w["dep"]["session"] == d["session"]]
        pool = [i for i, c in enumerate(ch) if i >= 40 and author(c) == d["who"] and len(c["text"]) >= 200
                and all(abs(i - j) >= 100 for j in near)]
        for i in rng.permutation(len(pool))[:per]:
            windows.append({"role": "control", "dep": d, "at": int(pool[i])})
    rows = []
    for n, w in enumerate(windows[:limit]):
        ch = loaded[w["dep"]["session"]]
        ids, cls, owner, kept, has_origin = build(tok, ch, w["at"], W, cap)
        qpos = pre_queries(owner, cls, kept, len(ids))
        anchor_gap = true_gap(tok, ch, w["at"], cap)
        entry = {"role": w["role"], "id": w["dep"]["id"], "kind": w["dep"]["kind"], "who": w["dep"]["who"],
                 "session": w["dep"]["session"], "at": w["at"], "tokens": int(len(ids)), "queries": int(len(qpos)),
                 "origin": has_origin, "anchor gap": anchor_gap}
        cache = Path(run_dir) / "windows" / f"{n:03d}.npz"                   # derived; lets a stopped run resume
        if len(qpos) and cache.exists():
            z = np.load(cache)
            entry["gap"], entry["owner"] = z["gap"], z["owner"]
            entry["origin lift"] = z["origin"] if has_origin else None
        elif len(qpos):
            g = gap.gaps(qpos, cls)
            origin = 0 if has_origin else -99
            red = gap.read_window(model, ids, qpos, cls, owner, origin)
            entry["gap"] = g
            entry["owner"] = np.stack([gap.lift(r["owner"], r["content"], r["n owner"], r["n content"]) for r in red], 1)
            entry["origin lift"] = (np.stack([gap.lift(r["origin"], r["content"], r["n origin"], r["n content"])
                                              for r in red], 1) if has_origin else None)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, gap=g, owner=entry["owner"],
                     origin=entry["origin lift"] if has_origin else np.zeros(0))
        rows.append(entry)
        record.append("attention departures, window", "code", **{("departure kind" if k == "kind" else k): v
                                                                  for k, v in entry.items()
                                                                  if k not in ("gap", "owner", "origin lift")},
                      chunk_refs=[c["ref"] for c in kept], anchor_ref=ch[w["at"]]["ref"])
        print(f"{n + 1}/{len(windows)} {w['role']} {w['dep']['id']} ({w['dep']['kind']}): {len(ids)} tokens, "
              f"{len(qpos)} queries, anchor gap {anchor_gap}", flush=True)
    return summarise(rows, L, record, run_dir)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    return float(np.mean((pos[:, None] > neg[None]) + 0.5 * (pos[:, None] == neg[None])))


def summarise(rows, L, record, run_dir):
    mid = list(range(round(6 * L / 28), round(26 * L / 28) + 1))
    edges = [lo for lo, _ in gap.GAPS[1:]]

    def binned(g):
        return np.searchsorted(edges, g, side="right")

    curve = {}
    for key, field in (("owner", "owner"), ("origin", "origin lift")):
        acc = [[] for _ in gap.GAPS]
        for r in rows:
            if r["role"] == "control" and r.get(field) is not None:
                for g, v in zip(r["gap"], r[field]):
                    if g >= 0:
                        acc[binned(g)].append(v)
        curve[key] = np.array([np.nanmean(a, 0) if a else np.full(L, np.nan) for a in acc])
    for r in rows:
        for key, field in (("owner", "owner"), ("origin", "origin lift")):
            ok = (r["gap"] >= 0) if r.get(field) is not None else None
            if ok is None or not ok.any():
                r[key + " score"] = float("nan")
                continue
            res = r[field][ok] - curve[key][binned(r["gap"][ok])]
            r[key + " score"] = float(np.nanmean(res[:, mid])) if np.isfinite(res[:, mid]).any() else float("nan")
            r[key + " raw"] = float(np.nanmean(r[field][ok][:, mid]))
    report = {"middle layers": [mid[0], mid[-1]], "kinds": {}}
    for kind in KINDS:
        dep = [r for r in rows if r["role"] == "departure" and r["kind"] == kind]
        ctl = [r for r in rows if r["role"] == "control" and r["kind"] == kind]
        k = {"departures": len(dep), "controls": len(ctl)}
        for key in ("owner", "origin"):
            ds = [r[key + " score"] for r in dep if np.isfinite(r.get(key + " score", np.nan))]
            cs = [r[key + " score"] for r in ctl if np.isfinite(r.get(key + " score", np.nan))]
            a = auc(cs, ds)
            k[key] = {"scored": [len(ds), len(cs)], "departure mean": float(np.mean(ds)) if ds else float("nan"),
                      "control mean": float(np.mean(cs)) if cs else float("nan"), "AUC control over departure": a,
                      "reading": ("too few" if len(ds) < 5 else "weaker before" if a >= 0.7 else
                                  "stronger before" if a <= 0.3 else "no difference")}
        dg = [r["anchor gap"] for r in dep if r["anchor gap"] >= 0]
        cg = [r["anchor gap"] for r in ctl if r["anchor gap"] >= 0]
        k["anchor gap"] = {"departure median": float(np.median(dg)) if dg else float("nan"),
                           "control median": float(np.median(cg)) if cg else float("nan"),
                           "AUC departure over control": auc(dg, cg)}
        report["kinds"][kind] = k
    record.append("attention departures, report", "code", **report,
                  windows=[{k: r[k] for k in ("role", "id", "kind", "who", "session", "at", "anchor gap",
                                               "owner score", "origin score") if k in r} for r in rows])
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--corpus", required=True)
    p.add_argument("--study", required=True, help="directory holding labels_confirmed.json and the reading")
    p.add_argument("--run", required=True)
    p.add_argument("--window", type=int, default=8192)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--limit", type=int, help="read only the first N windows (a smoke test)")
    a = p.parse_args(argv)
    rep = run(a.model, a.corpus, a.study, a.run, a.window, a.device, a.dtype, a.revision, limit=a.limit)
    print(f"middle layers {rep['middle layers']}; scores are gap-adjusted lifts (control gap curve), middle-layer mean")
    for kind, k in rep["kinds"].items():
        print(f"{kind}: {k['departures']} departures, {k['controls']} controls")
        for key in ("owner", "origin"):
            v = k[key]
            print(f"   toward {key:<6}: departure {v['departure mean']:+.2f}, control {v['control mean']:+.2f}, "
                  f"AUC control>departure {v['AUC control over departure']:.2f} (n {v['scored']}) -> {v['reading']}")
        g = k["anchor gap"]
        print(f"   tokens since the owner spoke, at the anchor: departure median {g['departure median']:.0f}, "
              f"control {g['control median']:.0f}, AUC departure>control {g['AUC departure over control']:.2f}")


if __name__ == "__main__":
    main()
