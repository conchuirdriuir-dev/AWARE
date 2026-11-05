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

W, BURN, MIN_ROWS, GAP = 10, 10, 3, 10       # window and burn-in in turns; an event needs 3 scored turns
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


def eligible(e: dict, lab: dict, taken: set) -> list[int]:
    """Scored turns outside every closure, at least GAP turns from any closure's onset and any ascent, not taken."""
    inside = {t for onset, n in lab["closures"] for t in range(onset, onset + n)}
    marks = [onset for onset, _ in lab["closures"]] + list(lab["ascents"])
    return [t for t in e["scored"] if t not in inside and t not in taken and all(abs(t - m) >= GAP for m in marks)]


def runs_of(e: dict, ok: list[int], n: int) -> list[list[int]]:
    """Every run of n turns of `ok` contiguous among the episode's scored turns."""
    pos = {t: p for p, t in enumerate(e["scored"])}
    return [ok[a:a + n] for a in range(len(ok) - n + 1) if pos[ok[a + n - 1]] - pos[ok[a]] == n - 1]


def match(eps: dict, labs: dict, events: list[dict], taken: dict) -> dict:
    """One control per event, as long as it: in the same episode, nearest in time to the event's middle turn, never
    reused; where the episode has none, the nearest world by seed with the same generator settings (cross-episode)."""

    def control_in(seed, n, centre):
        best = None
        for run in runs_of(eps[seed], eligible(eps[seed], labs[seed], taken[seed]), n):
            d = abs(run[n // 2] - centre)
            if best is None or d < best[0]:
                best = (d, run)
        return None if best is None else best[1]

    kept, none, cross = [], [], []
    for ev in events:
        n, seed = len(ev["rows"]), ev["seed"]
        run, where = control_in(seed, n, ev["rows"][n // 2]), seed
        if run is None:
            same = sorted((s for s in eps if s != seed and eps[s]["params"] == eps[seed]["params"]),
                          key=lambda s: abs(s - seed))
            for s in same:
                run = control_in(s, n, ev["rows"][n // 2])
                if run is not None:
                    where = s
                    cross.append(f"{seed}/{ev['kind']}@{ev['at']} from {s}")
                    break
        if run is None:
            none.append(f"{seed}/{ev['kind']}@{ev['at']}")
            continue
        taken[where].update(run)
        kept.append({**ev, "control": run, "control_seed": where})
    return {"events": kept, "no control": none, "cross-episode controls": cross}


def events_at(eps: dict, k: int, which: str = "ascents") -> dict:
    labs = {s: labels(e, k) for s, e in eps.items()}
    evs, dropped = [], []
    for s in sorted(eps):
        w, d = windows(eps[s], labs[s], which)
        evs += w
        dropped += d
    m = match(eps, labs, evs, {s: set() for s in eps})
    return {**m, "dropped (fewer than 3 turns)": dropped, "labels": labs}


# --- the readings ------------------------------------------------------------------------------------------------

def stat(eps: dict, seed: int, rows: list[int], space: int, field: str = "X") -> float:
    vals = [eps[seed]["spaces"][space][field].get(t, np.nan) for t in rows]
    return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")


def _auc(eps, evs, space, lower, field="X"):
    ev = [stat(eps, e["seed"], e["rows"], space, field) for e in evs]
    ct = [stat(eps, e["control_seed"], e["control"], space, field) for e in evs]
    return auc(ct, ev) if lower else auc(ev, ct)


def reading(eps: dict, evs: list[dict], spaces: list[int], text: int, lower: bool, seed: int = 0,
            draws: int = DRAWS, field: str = "X") -> dict:
    """The AUC per layer of event windows' mean X against their controls' (closures lower, ascents higher), its mean
    over layers, a 90% interval from `draws` resamplings of episodes (each with all its events), and the text
    baseline's AUC."""
    if not evs:
        return {"n": 0, "AUC": float("nan")}
    per = [_auc(eps, evs, sp, lower, field) for sp in spaces]
    by_ep = defaultdict(list)
    for e in evs:
        by_ep[e["seed"]].append(e)
    keys, rng, boot = sorted(by_ep), np.random.default_rng(seed), []
    for _ in range(draws):
        pick = [x for i in rng.integers(0, len(keys), len(keys)) for x in by_ep[keys[i]]]
        boot.append(np.mean([_auc(eps, pick, sp, lower, field) for sp in spaces]))
    lo, hi = np.nanpercentile(boot, [5, 95])
    return {"n": len(evs), "episodes": len(keys), "AUC": float(np.mean(per)), "90% interval": [float(lo), float(hi)],
            "text AUC": _auc(eps, evs, text, lower, field), "per layer AUC": per}


def decide(r1: dict, r2: dict) -> dict:
    """Both readings must pass; fewer than 100 closures or 50 ascents is too few to decide."""
    if r1["n"] < MIN_CLOSURES or r2["n"] < MIN_ASCENTS:
        return {"passed": None, "note": f"too few to decide ({r1['n']} closures, {r2['n']} ascents; "
                                        f"needs {MIN_CLOSURES} and {MIN_ASCENTS})"}
    p1 = r1["AUC"] >= AUC_BAR and r1["90% interval"][0] > 0.5 and r1["AUC"] > r1["text AUC"]
    p2 = r2["90% interval"][0] > 0.5
    return {"passed": bool(p1 and p2), "1 closures": bool(p1), "2 ascents": bool(p2)}


def placebo(eps: dict, labs: dict, events: list[dict], spaces: list[int], text: int,
            draws: int = PLACEBO_DRAWS) -> dict:
    """For each event, a pretend event as long, at a random eligible run of its episode, its control matched by the
    same rules; no closure or ascent is in it, so the AUCs should sit at 0.5 unless the matching moves them."""
    rng, out = np.random.default_rng(5), defaultdict(list)
    for _ in range(draws):
        taken, fake = {s: set() for s in eps}, []
        for ev in events:
            runs = runs_of(eps[ev["seed"]], eligible(eps[ev["seed"]], labs[ev["seed"]], taken[ev["seed"]]),
                           len(ev["rows"]))
            if runs:
                run = runs[rng.integers(len(runs))]
                taken[ev["seed"]].update(run)
                fake.append({"kind": ev["kind"], "seed": ev["seed"], "at": run[0], "rows": run})
        m = match(eps, labs, fake, taken)["events"]
        for name, lower in (("closures", True), ("ascents", False)):
            evs = [e for e in m if e["kind"] == name[:-1]]
            out[f"n {name}"].append(len(evs))
            if evs:
                out[name].append(float(np.mean([_auc(eps, evs, sp, lower) for sp in spaces])))
                out[f"{name}, text"].append(_auc(eps, evs, text, lower))
    summary = {"draws": draws}
    for k, v in out.items():
        summary[k] = (float(np.median(v)) if k.startswith("n ") else
                      {"mean": float(np.mean(v)), "5th-95th": [float(x) for x in np.percentile(v, [5, 95])]})
    return summary


def score(eps: dict, k: int, n_layers: int, placebo_draws: int = PLACEBO_DRAWS, draws: int = DRAWS) -> dict:
    """Step 0's habitat form at the frozen k; reported at k/2 and 2k (rounded up), with the scoring's checks."""
    spaces, text = list(range(n_layers)), n_layers

    def at(kk):
        w = events_at(eps, kk)
        clo = [e for e in w["events"] if e["kind"] == "closure"]
        asc = [e for e in w["events"] if e["kind"] == "ascent"]
        r1 = reading(eps, clo, spaces, text, True, draws=draws)
        r2 = reading(eps, asc, spaces, text, False, draws=draws)
        return w, clo, asc, r1, r2

    w, clo, asc, r1, r2 = at(k)
    out = {"k": k, "reading 1, closures": r1, "reading 2, ascents": r2, "decision": decide(r1, r2), "reported": {}}
    rep = out["reported"]
    for kk in sorted({math.ceil(k / 2), 2 * k} - {k}):
        _, _, _, a1, a2 = at(kk)
        rep[f"at k = {kk}"] = {"closures": {x: a1.get(x) for x in ("n", "AUC", "90% interval", "text AUC")},
                               "ascents": {x: a2.get(x) for x in ("n", "AUC", "90% interval", "text AUC")},
                               "decision": decide(a1, a2)}
    same = lambda evs: [e for e in evs if e["control_seed"] == e["seed"]]  # noqa: E731
    rep["same-episode pairs only"] = {"closures": reading(eps, same(clo), spaces, text, True, draws=draws),
                                      "ascents": reading(eps, same(asc), spaces, text, False, draws=draws)}
    rep["placebo windows"] = placebo(eps, w["labels"], clo + asc, spaces, text, placebo_draws)

    def med(evs, field, which):
        vals = [np.nanmean([stat(eps, e["seed" if which == "rows" else "control_seed"], e[which], sp, field)
                            for sp in spaces]) for e in evs]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")
    for name, evs in (("closures", clo), ("ascents", asc)):
        rep[f"{name}: medians, events and controls"] = {
            f: {"events": med(evs, f, "rows"), "controls": med(evs, f, "control")} for f in ("X", "S", "C")}
    stall = [np.nanmean([stat(eps, e["seed"], [t for t in e["stall"] if t in eps[e["seed"]]["spaces"][sp]["X"]],
                              sp) for sp in spaces]) for e in clo]
    stall = [v for v in stall if np.isfinite(v)]
    rep["X during the stall (median; flat by construction, not evidence)"] = (float(np.median(stall)) if stall
                                                                              else float("nan"))
    wb = events_at(eps, k, which="broad")
    rep["broad ascents"] = reading(eps, [e for e in wb["events"] if e["kind"] == "ascent"], spaces, text, False,
                                   draws=draws)
    for v in VARIANTS:
        rep[f"X with W = {v}"] = {
            name: float(np.mean([_auc(eps, evs, sp, lower, f"X_w{v}") for sp in spaces])) if evs else float("nan")
            for name, evs, lower in (("closures", clo, True), ("ascents", asc, False))}
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
    agent = Agent(a.model, device=a.device, dtype=a.dtype, cache=not a.no_cache, nf4=a.nf4,
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
