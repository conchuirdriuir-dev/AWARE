"""Body readout v0.2, Step 0's habitat form: does the world part of rate flatten before a stall, and not after an
ascent? (PLAN.md, "Body readout", "The habitat", "Step 0, habitat form".)

Three commands, in order. `play` appends every turn of Step 0's pool as it happens (the harness of Step -1, no plants
in this pool); `read` derives states from the recorded text alone, with no graft, and turns them into code lengths,
checkpointed per episode and never seeing a key; `score` alone reads the keys (the engine's labels and k).

    python -m body.step0_habitat play --model Qwen/Qwen3-8B --run runs/step0h-8b
    python -m body.step0_habitat read --model Qwen/Qwen3-8B --run runs/step0h-8b
    python -m body.step0_habitat score --run runs/step0h-8b --k K

K is the k fixed in PLAN.md from the calibration pool. --limit and --rehearsal play part of the pool, or its mirror
seeds (new worlds on the same plan, deciding nothing).

The reading is Step 0's formula and fit unchanged (`body/progress.py`): each turn's action coded on everything before
it, and again with the window's world parts left out. The world's replies fill the partner slot (owner's decision 2,
2026-11-04), so world cross-progress is by_source's X.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from body.progress import by_source, cost
from goalspace.formula import Tfidf, auc
from substrate.record import Record, read

W, BURN, MIN_ROWS = 10, 10, 3                # window and burn-in in turns; an event needs 3 scored turns
AUC_BAR, MIN_CLOSURES, MIN_ASCENTS, DRAWS, PLACEBO_DRAWS = 0.70, 100, 50, 2000, 200
CAP, TEXT_VOCAB, VARIANTS = 256, 40, (5, 20)
FIELDS = ("L_now", "X", "S", "C") + tuple(f"X_w{v}" for v in VARIANTS)


# --- material: the record, by episode --------------------------------------------------------------------------

def sources(chunks: list[dict]) -> list[str]:
    """Self for the agent's actions; partner for the world's text (the opening, replies, any event)."""
    return ["self" if c["slot"] == "self" else "partner" for c in chunks]


def codes(X: np.ndarray, srcs: list[str], floor_var: float, variants: bool = True) -> dict:
    """The code lengths and readings for one space of one episode, label-blind: W = 10 turns, burn-in 10 turns;
    reported only, X at W = 5 and 20."""
    r = by_source(X, srcs, floor_var, burn_self=BURN, window_self=W)
    out = {"L_now": r["L_now"], "X": r["X"], "S": r["S"], "C": cost(r["L_now"])}
    for v in VARIANTS:
        out[f"X_w{v}"] = (by_source(X, srcs, floor_var, burn_self=BURN, window_self=v)["X"] if variants
                          else np.full(len(X), np.nan))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def turns_of(ep: dict) -> list[dict]:
    """Each turn's keys with its action: what the labels read. Keys only: scoring calls this, nothing else."""
    acts = {c["turn"]: c["text"] for c in ep["chunks"] if c["slot"] == "self"}
    return [dict(ep["keys"][t], action=acts[t]) for t in sorted(ep["keys"])]


def episode(ep: dict, spaces: list[dict]) -> dict:
    """What scoring works on: per turn, its action's row; the scored turns (those with a code length); per space, the
    readings by turn."""
    row = {c["turn"]: i for i, c in enumerate(ep["chunks"]) if c["slot"] == "self"}
    scored = [t for t in sorted(row) if np.isfinite(spaces[0]["L_now"][row[t]])]
    by_turn = [{f: {t: float(sp[f][row[t]]) for t in scored} for f in FIELDS} for sp in spaces]
    return {"seed": ep["info"]["seed"], "params": ep["info"].get("pool_params"), "scored": scored,
            "spaces": by_turn, "turns": turns_of(ep), "start": ep["info"].get("quest_left")}


# --- labels and windows (keys; scoring only) ---------------------------------------------------------------------

def labels(e: dict, k: int) -> dict:
    """Closures at k, ascents, and broad ascents (reported only: a room entered for the first time, or a verb first
    used on a turn that made a new state, not a move; the agent's reading of the spec's two extra kinds)."""
    from body.habitat import ascents, closures

    turns = e["turns"]
    rooms, verbs, broad = set(), set(), []
    for i, t in enumerate(turns):
        verb = (t["action"].strip().lower().split() or [""])[0]
        new_room = t["room"] is not None and t["room"] not in rooms and i > 0
        new_verb = t["new_state"] and verb != "go" and verb not in verbs
        if new_room or new_verb:
            broad.append(i)
        if t["room"] is not None:
            rooms.add(t["room"])
        if new_verb:
            verbs.add(verb)
    return {"closures": closures(turns, k), "ascents": ascents(turns, e["start"]), "broad": broad}


def windows(e: dict, lab: dict, which: str = "ascents") -> tuple[list[dict], list[str]]:
    """Closure windows: the last W scored turns before the onset (the onset's own action is part of the label, so the
    window ends before it). Ascent windows: the first W scored turns after the ascent. Fewer than 3: dropped."""
    out, dropped = [], []
    for onset, n in lab["closures"]:
        rows = [t for t in e["scored"] if t < onset][-W:]
        ev = {"kind": "closure", "seed": e["seed"], "at": onset, "rows": rows, "stall": list(range(onset, onset + n))}
        (out if len(rows) >= MIN_ROWS else dropped).append(ev)
    for a in lab[which]:
        rows = [t for t in e["scored"] if t > a][:W]
        ev = {"kind": "ascent", "seed": e["seed"], "at": a, "rows": rows}
        (out if len(rows) >= MIN_ROWS else dropped).append(ev)
    return out, [f"{d['seed']}/{d['kind']}@{d['at']}" for d in dropped]


def inside(lab: dict) -> set:
    return {t for onset, n in lab["closures"] for t in range(onset, onset + n)}


def pre_progress(e: dict, lab: dict) -> list[tuple[int, list[int]]]:
    """(ascent, window) for each ascent: the last W scored turns before it, at least 3, none inside a closure
    (amendment 2)."""
    out, stuck = [], inside(lab)
    for a in lab["ascents"]:
        rows = [t for t in e["scored"] if t < a][-W:]
        if len(rows) >= MIN_ROWS and not stuck & set(rows):
            out.append((a, rows))
    return out


def runs_of(e: dict, ok: list[int], n: int) -> list[list[int]]:
    """Every run of n turns of `ok` contiguous among the episode's scored turns."""
    pos = {t: p for p, t in enumerate(e["scored"])}
    return [ok[a:a + n] for a in range(len(ok) - n + 1) if pos[ok[a + n - 1]] - pos[ok[a]] == n - 1]


def match(eps: dict, labs: dict, events: list[dict], taken: dict) -> dict:
    """Reading 1's pairs (amendment 2): each closure window with one pre-progress window, cut to its last n turns, in
    the same episode, nearest the closure window's middle turn, never reused (`taken`: seed -> ascents used); where the
    episode has none, the nearest world by seed in the pool (amendment 3: every world shares the pool's generator)."""

    def control_in(seed, n, centre):
        best = None
        for a, rows in pre_progress(eps[seed], labs[seed]):
            if a in taken[seed] or len(rows) < n:
                continue
            run = rows[-n:]
            d = abs(run[n // 2] - centre)
            if best is None or d < best[0]:
                best = (d, a, run)
        return best

    kept, none, cross = [], [], []
    for ev in events:
        n, seed = len(ev["rows"]), ev["seed"]
        best, where = control_in(seed, n, ev["rows"][n // 2]), seed
        if best is None:
            same = sorted((s for s in eps if s != seed), key=lambda s: abs(s - seed))   # amendment 3: the pool
            for s in same:
                best = control_in(s, n, ev["rows"][n // 2])
                if best is not None:
                    where = s
                    cross.append(f"{seed}/{ev['kind']}@{ev['at']} from {s}")
                    break
        if best is None:
            none.append(f"{seed}/{ev['kind']}@{ev['at']}")
            continue
        taken[where].add(best[1])
        kept.append({**ev, "control": best[2], "control_seed": where, "control_ascent": best[1]})
    return {"events": kept, "no control": none, "cross-episode controls": cross}


def events_at(eps: dict, k: int, which: str = "ascents") -> dict:
    """Closure windows paired with pre-progress windows (reading 1), and the post-ascent windows (reading 2; `which`
    picks the ascents or, reported only, the broad ascents)."""
    labs = {s: labels(e, k) for s, e in eps.items()}
    clo, asc, dropped = [], [], []
    for s in sorted(eps):
        w, d = windows(eps[s], labs[s], which)
        clo += [x for x in w if x["kind"] == "closure"]
        asc += [x for x in w if x["kind"] == "ascent"]
        dropped += d
    m = match(eps, labs, clo, {s: set() for s in eps})
    return {"closures": m["events"], "pre-stall": clo, "ascents": asc, "no control": m["no control"],
            "cross-episode controls": m["cross-episode controls"], "dropped (fewer than 3 turns)": dropped,
            "labels": labs}


# --- the readings ------------------------------------------------------------------------------------------------

def stat(eps: dict, seed: int, rows: list[int], space: int, field: str = "X") -> float:
    vals = [eps[seed]["spaces"][space][field].get(t, np.nan) for t in rows]
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def _auc(eps, evs, space, lower, field="X"):
    ev = [stat(eps, e["seed"], e["rows"], space, field) for e in evs]
    ct = [stat(eps, e["control_seed"], e["control"], space, field) for e in evs]
    return auc(ct, ev) if lower else auc(ev, ct)


def _boot(by_ep: dict, value, draws: int, seed: int) -> list[float]:
    keys, rng, out = sorted(by_ep), np.random.default_rng(seed), []
    for _ in range(draws):
        pick = [by_ep[keys[i]] for i in rng.integers(0, len(keys), len(keys))]
        out.append(value(pick))
    return out


def reading(eps: dict, evs: list[dict], spaces: list[int], text: int, lower: bool, seed: int = 0,
            draws: int = DRAWS, field: str = "X") -> dict:
    """Paired: the AUC per layer of event windows' mean X against their controls' (`lower`: events lower), its mean
    over layers, a 90% interval from `draws` resamplings of episodes (each with all its events), and the text
    baseline's AUC."""
    if not evs:
        return {"n": 0, "AUC": float("nan")}
    per = [_auc(eps, evs, sp, lower, field) for sp in spaces]
    by_ep = defaultdict(list)
    for e in evs:
        by_ep[e["seed"]].append(e)
    boot = _boot(by_ep, lambda pick: np.mean([_auc(eps, [x for p in pick for x in p], sp, lower, field)
                                              for sp in spaces]), draws, seed)
    lo, hi = np.nanpercentile(boot, [5, 95]) if boot else (np.nan, np.nan)
    return {"n": len(evs), "episodes": len(by_ep), "AUC": float(np.mean(per)), "90% interval": [float(lo), float(hi)],
            "text AUC": _auc(eps, evs, text, lower, field), "per layer AUC": per}


def unpaired(eps: dict, pos: list[dict], neg: list[dict], spaces: list[int], text: int, seed: int = 0,
             draws: int = DRAWS, field: str = "X") -> dict:
    """Reading 2 (amendment 2): the AUC of `pos` windows' mean X over `neg` windows', per layer and its mean, with a
    90% interval resampling episodes (each with all its windows of both kinds), and the text baseline's."""
    if not pos or not neg:
        return {"n": len(pos), "against": len(neg), "AUC": float("nan")}

    def a(P, N, sp):
        return auc([stat(eps, w["seed"], w["rows"], sp, field) for w in P],
                   [stat(eps, w["seed"], w["rows"], sp, field) for w in N])
    per = [a(pos, neg, sp) for sp in spaces]
    by_ep = defaultdict(lambda: ([], []))
    for w in pos:
        by_ep[w["seed"]][0].append(w)
    for w in neg:
        by_ep[w["seed"]][1].append(w)
    by_ep = dict(by_ep)

    def value(pick):
        P = [w for p, _ in pick for w in p]
        N = [w for _, n in pick for w in n]
        return np.mean([a(P, N, sp) for sp in spaces]) if P and N else np.nan
    boot = [v for v in _boot(by_ep, value, draws, seed) if np.isfinite(v)]
    lo, hi = np.percentile(boot, [5, 95]) if boot else (np.nan, np.nan)
    return {"n": len(pos), "against": len(neg), "episodes": len(by_ep), "AUC": float(np.mean(per)),
            "90% interval": [float(lo), float(hi)], "text AUC": a(pos, neg, text), "per layer AUC": per}


def decide(r1: dict, r2: dict) -> dict:
    """Both readings must pass; fewer than 100 closures or 50 ascents is too few to decide."""
    if r1["n"] < MIN_CLOSURES or r2["n"] < MIN_ASCENTS:
        return {"passed": None, "note": f"too few to decide ({r1['n']} closures, {r2['n']} ascents; "
                                        f"needs {MIN_CLOSURES} and {MIN_ASCENTS})"}
    p1 = r1["AUC"] >= AUC_BAR and r1["90% interval"][0] > 0.5 and r1["AUC"] > r1["text AUC"]
    p2 = r2["90% interval"][0] > 0.5
    return {"passed": bool(p1 and p2), "1 closures": bool(p1), "2 ascents": bool(p2)}


def placebo(eps: dict, w: dict, spaces: list[int], text: int, draws: int = PLACEBO_DRAWS) -> dict:
    """For each event, a pretend window as long at a random run of its episode outside every closure, read by the same
    rules: pretend closures paired with pre-progress windows, pretend ascents against the real pre-stall windows. The
    AUCs show what the pairing and the windows' placing give with no event in them."""
    rng, out, labs = np.random.default_rng(5), defaultdict(list), w["labels"]
    for _ in range(draws):
        fake = {"closure": [], "ascent": []}
        for ev in w["pre-stall"] + w["ascents"]:
            e = eps[ev["seed"]]
            ok = [t for t in e["scored"] if t not in inside(labs[ev["seed"]])]
            runs = runs_of(e, ok, len(ev["rows"]))
            if runs:
                run = runs[rng.integers(len(runs))]
                fake[ev["kind"]].append({"kind": ev["kind"], "seed": ev["seed"], "at": run[0], "rows": run})
        m = match(eps, labs, fake["closure"], {s: set() for s in eps})["events"]
        out["n closures"].append(len(m))
        out["n ascents"].append(len(fake["ascent"]))
        if m:
            out["closures"].append(float(np.mean([_auc(eps, m, sp, True) for sp in spaces])))
            out["closures, text"].append(_auc(eps, m, text, True))
        if fake["ascent"] and w["pre-stall"]:
            u = unpaired(eps, fake["ascent"], w["pre-stall"], spaces, text, draws=0)
            out["ascents"].append(u["AUC"])
            out["ascents, text"].append(u["text AUC"])
    summary = {"draws": draws}
    for k, v in out.items():
        summary[k] = (float(np.median(v)) if k.startswith("n ") else
                      {"mean": float(np.mean(v)), "5th-95th": [float(x) for x in np.percentile(v, [5, 95])]})
    return summary


def score(eps: dict, k: int, n_layers: int, placebo_draws: int = PLACEBO_DRAWS, draws: int = DRAWS) -> dict:
    """Step 0's habitat form (amendment 2) at the frozen k; reported at k/2 and 2k (rounded up), with the scoring's
    checks. Reading 1: pre-progress X over pre-stall X, paired. Reading 2: post-ascent X over pre-stall X."""
    spaces, text = list(range(n_layers)), n_layers

    def at(kk):
        w = events_at(eps, kk)
        r1 = reading(eps, w["closures"], spaces, text, True, draws=draws)
        r2 = unpaired(eps, w["ascents"], w["pre-stall"], spaces, text, draws=draws)
        return w, r1, r2

    w, r1, r2 = at(k)
    clo, asc = w["closures"], w["ascents"]
    out = {"k": k, "reading 1, closures": r1, "reading 2, ascents": r2, "decision": decide(r1, r2), "reported": {}}
    rep = out["reported"]
    for kk in sorted({math.ceil(k / 2), 2 * k} - {k}):
        _, a1, a2 = at(kk)
        rep[f"at k = {kk}"] = {"closures": {x: a1.get(x) for x in ("n", "AUC", "90% interval", "text AUC")},
                               "ascents": {x: a2.get(x) for x in ("n", "AUC", "90% interval", "text AUC")},
                               "decision": decide(a1, a2)}
    same = [e for e in clo if e["control_seed"] == e["seed"]]
    rep["same-episode pairs only (reading 1)"] = reading(eps, same, spaces, text, True, draws=draws)
    rep["placebo windows"] = placebo(eps, w, spaces, text, placebo_draws)

    def med(wins, field):
        vals = [np.nanmean([stat(eps, x["seed"], x["rows"], sp, field) for sp in spaces]) for x in wins]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")
    progress = [{"seed": e["control_seed"], "rows": e["control"]} for e in clo]
    rep["medians: pre-stall, pre-progress, post-ascent"] = {
        f: {"pre-stall": med(clo, f), "pre-progress": med(progress, f), "post-ascent": med(asc, f)}
        for f in ("X", "S", "C")}
    stall = [{"seed": e["seed"], "rows": [t for t in e["stall"] if t in eps[e["seed"]]["spaces"][0]["X"]]}
             for e in clo]
    rep["X during the stall (median; flat by construction, not evidence)"] = med([x for x in stall if x["rows"]], "X")
    wb = events_at(eps, k, which="broad")
    rep["broad ascents"] = unpaired(eps, wb["ascents"], wb["pre-stall"], spaces, text, draws=draws)
    for v in VARIANTS:
        f = f"X_w{v}"
        rep[f"X with W = {v}"] = {
            "closures": float(np.mean([_auc(eps, clo, sp, True, f) for sp in spaces])) if clo else float("nan"),
            "ascents": unpaired(eps, asc, w["pre-stall"], spaces, text, draws=0, field=f)["AUC"]}
    rep["dropped (fewer than 3 turns)"] = w["dropped (fewer than 3 turns)"]
    rep["no control"] = w["no control"]
    rep["cross-episode controls"] = w["cross-episode controls"]
    rep["episodes"] = len(eps)
    return out


# --- commands ----------------------------------------------------------------------------------------------------

def _episodes(run: Path) -> dict:
    from body.habitat import mirrored, pool
    from body.stepm1 import episodes

    mine = set(pool("step0_habitat"))
    return {k: v for k, v in episodes(read(run / "record.jsonl")).items()
            if k[1] is None and mirrored(k[0]) in mine}


def cmd_play(a) -> None:
    from body.habitat import REHEARSAL, Agent, play, pool

    run = Path(a.run)
    record = Record(run / "record.jsonl")
    done = set(_episodes(run)) if record.path.exists() else set()
    if (a.nf4 or a.context or a.generator) and not a.rehearsal:
        raise SystemExit("--nf4, --context and --generator change the player or worlds: rehearsals only")
    gen = json.loads(a.generator) if a.generator else None
    agent = Agent(a.model, device=a.device, dtype=a.dtype, cache=not a.no_cache, nf4=a.nf4, forms=a.forms,
                  **({"context": a.context} if a.context else {}))
    for seed in [s + REHEARSAL * a.rehearsal for s in pool("step0_habitat")[: a.limit or None]]:
        if (seed, None) not in done:
            play(agent, record, seed, run / "games", generator=gen)


def cmd_read(a) -> None:
    from substrate.echo_floor import d_resumable, session_resumable
    from substrate.geometry import spread
    from substrate.measure_d import load_sets
    from substrate.states import Reader

    run = Path(a.run)
    record = Record(run / "record.jsonl")
    eps = _episodes(run)
    sets = load_sets(a.sets)
    reader = Reader(a.model, device=a.device, dtype=a.dtype, gpu_memory=a.gpu_memory, sink="first")
    n = reader.info["layers"]
    layers = list(range(math.ceil(2 * n / 3), n))
    D = d_resumable(reader, sets, layers, record, a.draft, True)
    tok = reader.tokenizer
    record.append("read", "code", step="step0_habitat", **reader.info, read_layers=layers,
                  D=[float(x) for x in D], cap=CAP, window=W, burn_in=BURN, draft=a.draft)
    for (seed, _), ep in sorted(eps.items()):
        def compute(ep=ep):
            texts = [tok.decode(tok(c["text"], add_special_tokens=False)["input_ids"][:CAP]) for c in ep["chunks"]]
            srcs = sources(ep["chunks"])
            S = reader.states(texts, layers)
            arrays = {}
            for j, layer in enumerate(layers):
                X = S[:, j].astype(float)
                for f, v in codes(X, srcs, float(D[j]) ** 2 / X.shape[1], not a.no_variants).items():
                    arrays[f"L{layer}:{f}"] = v
            tf = Tfidf(texts[:TEXT_VOCAB])
            T = tf(texts)
            d_text = spread([tf(s["texts"]) for s in sets])
            for f, v in codes(T, srcs, d_text ** 2 / max(T.shape[1], 1), not a.no_variants).items():
                arrays[f"text:{f}"] = v
            return arrays, {"seed": seed, "chunks": len(texts), "layers": layers, "attempt": ep["attempts"]}
        session_resumable(record, f"step0h-{seed}-a{ep['attempts']}", compute)


def load(run: Path) -> tuple[dict, int]:
    """Every complete episode with its checkpointed code lengths, as scoring works on them."""
    eps, out, n_layers = _episodes(run), {}, None
    for (seed, _), ep in sorted(eps.items()):
        path = run / "checkpoints" / f"step0h-{seed}-a{ep['attempts']}.npz"
        if not path.exists():
            raise SystemExit(f"episode {seed} has no code lengths: run read first")
        data = np.load(path)
        names = sorted({k.split(":")[0] for k in data.files if k != "text" and not k.startswith("text:")},
                       key=lambda s: int(s[1:]))
        spaces = [{f: data[f"{nm}:{f}"] for f in FIELDS} for nm in names + ["text"]]
        n_layers = len(names)
        out[seed] = episode(ep, spaces)
    return out, n_layers or 0


def cmd_score(a) -> None:
    run = Path(a.run)
    eps, n_layers = load(run)
    result = score(eps, a.k, n_layers)
    Record(run / "record.jsonl").append("score", "code", step="step0_habitat", **result)
    print(json.dumps(result, indent=1, default=float))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("play", "read", "score"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        if name != "score":
            s.add_argument("--model", required=True)
            s.add_argument("--device", default="auto")
            s.add_argument("--dtype", default="bfloat16")
        if name == "play":
            s.add_argument("--limit", type=int, help="only the first n seeds of the pool (a rehearsal)")
            s.add_argument("--rehearsal", action="store_true", help="the pool's mirror seeds (+900000)")
            s.add_argument("--no-cache", action="store_true")
            s.add_argument("--nf4", action="store_true", help="a 4-bit model: rehearsals only, recorded on every episode")
            s.add_argument("--forms", action="store_true", help="Stage 2's player: told TextWorld's command forms")
            s.add_argument("--context", type=int, help="a shorter context in tokens: rehearsals only")
            s.add_argument("--generator", help='candidate world settings as JSON, e.g. {"rooms": [10, 20], ...}: rehearsals only')
        if name == "read":
            s.add_argument("--sets", default="material/paraphrases.jsonl")
            s.add_argument("--gpu-memory", type=float)
            s.add_argument("--draft", action="store_true")
            s.add_argument("--no-variants", action="store_true", help="skip X at W = 5 and 20 (reported only)")
        if name == "score":
            s.add_argument("--k", type=int, required=True, help="the k fixed in PLAN.md from the calibration pool")
    a = p.parse_args(argv)
    {"play": cmd_play, "read": cmd_read, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
