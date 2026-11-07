"""Rate's calibration in Stage 1 (PLAN.md, "Two stages", "Rate's calibration"; owner, 2026-11-05, PR #32 Q3; its form
under "Rate's calibration: the form"): two scripts that differ only in contingency, on the same worlds. The contingent
script takes its next action from the reply (a direction the reply names); the blind script replays, move for move,
what the contingent script did in another world, so its actions follow the same law and never depend on what it is
told. World rate (Step 0's X: the bits the window's world parts save in coding the next action) should be higher under
the contingent script. A known answer by construction: it calibrates the gauge, never lead.

It gates (owner, 2026-11-07, Step 0 amendment 7's sequel): rate enters the sense tokens only if the AUC of contingent
over blind has its 90% lower bound above 0.5, and so does the can't-go check, the same comparison on the scored turns
whose window holds no "You can't go that way." reply, in either arm (the blind arm hears that reply far more often).
The check needs 10 worlds where both episodes have at least 3 such turns, or it is too few to decide and rate stays
out. A pass says the gauge sees a dependence built in; rate's predictive validity stays untested.

    python -m body.rate_calibration play  --run runs/rate-cal
    python -m body.rate_calibration read  --model Qwen/Qwen3-8B --run runs/rate-cal
    python -m body.rate_calibration score --run runs/rate-cal

`play` needs no GPU. Each arm has its own record (`<run>/contingent/`, `<run>/blind/`), since both play every world;
`read` and `score` write to `<run>/record.jsonl`. `read` sees only the recorded text, never which arm it reads: the
arm is the run folder's name, read by `score` alone.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from body.conditional import conditional
from body.machine import record_machine
from goalspace.formula import Tfidf, auc
from substrate.record import Record, read

SEEDS = (7000, 7039)          # 20 pairs of worlds, outside every pool in the three world files (the agent's choice)
# The forms: v1, X as first built (registered, run 2026-11-07: failed); conditional, X_c (body/conditional.py; drafted
# 2026-11-07, PLAN.md "Repairs"), read on fresh seeds. The statistic, the can't-go check and the gate are the same.
FORMS = {"v1": {"seeds": SEEDS, "field": "X", "self": "S"},
         "conditional": {"seeds": (7100, 7299), "field": "X_c", "self": "S_c"}}   # 100 pairs (owner, 2026-11-07)
SCRIPT_SEED = 710000          # the contingent script's draws: default_rng(seed + 710000)
ARMS = ("contingent", "blind")
DIRECTIONS = ("north", "south", "east", "west")
NAMED = re.compile(r"\b(north|south|east|west)\b", re.I)
DRAWS = 2000
# The gate (owner, 2026-11-07): rate enters the sense tokens only if both lower bounds are above 0.5
BAR = 0.5
CANT = "You can't go that way"   # the can't-go check: scored turns whose window holds no such reply, in either arm
MIN_CLEAN_TURNS, MIN_CLEAN_WORLDS = 3, 10   # the agent's choice, 2026-11-07, before any play of the seeds
W = 10                                       # the window in turns, as Step 0's reading (body/step0_habitat.py)


def seeds(rehearsal: bool = False, limit: int | None = None, form: str = "v1") -> list[int]:
    from body.habitat import REHEARSAL

    lo, hi = FORMS[form]["seeds"]
    out = list(range(lo, hi + 1))[: limit or None]
    if limit and len(out) % 2:
        out = out[:-1]                                   # whole pairs only
    return [s + REHEARSAL * rehearsal for s in out]


def partner(seed: int) -> int:
    """The other world of the seed's pair: 7000 with 7001, 7002 with 7003, and so on."""
    return seed + 1 if seed % 2 == 0 else seed - 1


def named(text: str) -> list[str]:
    """The directions a reply names, in the order it names them, once each."""
    out = []
    for m in NAMED.finditer(text):
        d = m.group(1).lower()
        if d not in out:
            out.append(d)
    return out


def _final(engine) -> str | None:
    """The quest's last command when one step is left: never issued, so no episode ends early (as Stage 1's script)."""
    walk = list(engine.state.policy_commands or [])
    return walk[0].strip().lower() if len(walk) == 1 else None


def world_text(engine, first: bool) -> str:
    """The last thing the world said, as the player sees it: the opening without its banner, or the reply."""
    from body.habitat import clean, unbanner

    text = clean(engine.state.feedback)
    return unbanner(text) if first else text


class Contingent:
    """Each turn, a direction the last thing the world said names, drawn uniformly from default_rng(seed + 710000);
    `look` when it names none. Its actions are the code's."""

    name, revision, scripted = "contingent-v1", None, True

    def start(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed + SCRIPT_SEED)
        self.first = True

    def act(self, engine) -> str:
        self.last_phase = "contingent"
        dirs = named(world_text(engine, self.first))
        self.first = False
        cmd = f"go {self.rng.choice(dirs)}" if dirs else "look"
        return "look" if cmd == _final(engine) else cmd


class Blind:
    """The contingent script's actions in the partner world, in order, whatever this world says; `look` once they run
    out. Its actions are the code's."""

    name, revision, scripted = "blind-v1", None, True

    def __init__(self, actions: dict):
        self.actions = actions                           # partner seed -> its contingent actions, in order

    def start(self, seed: int) -> None:
        self.queue, self.t = self.actions[partner(seed)], 0

    def act(self, engine) -> str:
        self.last_phase = "blind"
        cmd = self.queue[self.t] if self.t < len(self.queue) else "look"
        self.t += 1
        return "look" if cmd == _final(engine) else cmd


def actions_by_seed(record_path: Path) -> dict:
    from body.stepm1 import episodes

    eps = episodes(read(record_path)) if record_path.exists() else {}
    return {seed: [c["text"] for c in ep["chunks"] if c["slot"] == "self"] for (seed, _), ep in eps.items()}


def _arm(run: Path, arm: str) -> Record:
    return Record(run / arm / "record.jsonl")


def cmd_play(a) -> None:
    from body import habitat
    from body.stepm1 import episodes

    run = Path(a.run)
    todo = seeds(a.rehearsal, a.limit, getattr(a, "form", "v1"))
    for arm in ARMS:
        record = _arm(run, arm)
        entries = read(record.path) if record.path.exists() else []
        done = {s for s, _ in episodes(entries)} | {e["seed"] for e in entries if e["kind"] == "skip"}
        record_machine(record, f"rate_calibration play ({arm})")
        if arm == "contingent":
            agent = Contingent()
        else:
            agent = Blind(actions_by_seed(_arm(run, "contingent").path))
        for seed in todo:
            if seed in done:
                continue
            if arm == "blind" and partner(seed) not in agent.actions:
                record.append("skip", "code", seed=seed, reason="the partner world has no contingent episode")
                continue
            habitat.play_or_skip(agent, record, seed, run / arm / "games", kind=None)   # no plant; the engine capped


# --- reading: Step 0's formula and fit unchanged, label-blind ----------------------------------------------------

def cmd_read(a) -> None:
    from body.stepm1 import episodes
    from body.step0_habitat import CAP, TEXT_VOCAB, codes, sources
    from substrate.echo_floor import d_resumable, session_resumable
    from substrate.geometry import spread
    from substrate.measure_d import load_sets
    from substrate.states import Reader

    run = Path(a.run)
    record = Record(run / "record.jsonl")
    record_machine(record, "rate_calibration read")
    started = time.time()
    sets = load_sets(a.sets)
    reader = Reader(a.model, device=a.device, dtype=a.dtype, gpu_memory=a.gpu_memory, sink="first")
    n = reader.info["layers"]
    layers = list(range(math.ceil(2 * n / 3), n))
    D = d_resumable(reader, sets, layers, record, a.draft, True)
    tok = reader.tokenizer
    record.append("read", "code", step="rate_calibration", **reader.info, read_layers=layers,
                  D=[float(x) for x in D], cap=CAP, draft=a.draft)
    total = 0
    for arm in ARMS:                                     # the same computation for both: the arm only names the file
        for (seed, _), ep in sorted(episodes(read(run / arm / "record.jsonl")).items()):
            def compute(ep=ep, seed=seed):
                texts = [tok.decode(tok(c["text"], add_special_tokens=False)["input_ids"][:CAP]) for c in ep["chunks"]]
                srcs = sources(ep["chunks"])
                S = reader.states(texts, layers)
                arrays = {}
                slots, turns = [c["slot"] for c in ep["chunks"]], [c["turn"] for c in ep["chunks"]]
                for j, layer in enumerate(layers):
                    X = S[:, j].astype(float)
                    fv = float(D[j]) ** 2 / X.shape[1]
                    for f, v in codes(X, srcs, fv, not a.no_variants).items():
                        arrays[f"L{layer}:{f}"] = v
                    for f, v in conditional(X, slots, turns, fv).items():     # both forms, from one reading
                        arrays[f"L{layer}:{f}"] = v
                tf = Tfidf(texts[:TEXT_VOCAB])
                T = tf(texts)
                d_text = spread([tf(s["texts"]) for s in sets])
                for f, v in codes(T, srcs, d_text ** 2 / max(T.shape[1], 1), not a.no_variants).items():
                    arrays[f"text:{f}"] = v
                for f, v in conditional(T, slots, turns, d_text ** 2 / max(T.shape[1], 1)).items():
                    arrays[f"text:{f}"] = v
                return arrays, {"seed": seed, "chunks": len(texts), "layers": layers, "attempt": ep["attempts"]}
            session_resumable(record, f"{arm}-{seed}-a{ep['attempts']}", compute)
            total += len(ep["chunks"])
    record.append("read_end", "code", step="rate_calibration", chunks=total, seconds=round(time.time() - started, 1))


# --- scoring: the only reader of the arm and the keys --------------------------------------------------------------

def episode_means(data, field: str = "X") -> dict:
    """Per space (each read layer, and "text"): the mean of `field` over the episode's scored turns."""
    names = sorted({k.split(":")[0] for k in data.files}, key=lambda s: (s == "text", int(s[1:]) if s != "text" else 0))
    out = {}
    for nm in names:
        v = np.asarray(data[f"{nm}:{field}"], dtype=float)
        out[nm] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
    return out


def compare(by_world: dict, draws: int = DRAWS, seed: int = 0) -> dict:
    """by_world: world seed -> {"contingent": value, "blind": value}. The AUC of contingent over blind, with a 90%
    interval resampling worlds (each drawn world brings both its episodes); the paired difference's median and the
    share of worlds where contingent is higher."""
    worlds = sorted(w for w, v in by_world.items() if all(np.isfinite(v.get(a, np.nan)) for a in ARMS))
    if not worlds:
        return {"worlds": 0}
    c = [by_world[w]["contingent"] for w in worlds]
    b = [by_world[w]["blind"] for w in worlds]
    rng, vals = np.random.default_rng(seed), []
    for _ in range(draws):
        pick = rng.integers(0, len(worlds), len(worlds))
        vals.append(auc([c[i] for i in pick], [b[i] for i in pick]))
    lo, hi = np.percentile(vals, [5, 95]) if vals else (math.nan, math.nan)
    diff = np.array(c) - np.array(b)
    return {"worlds": len(worlds), "AUC": auc(c, b), "90% interval": [float(lo), float(hi)],
            "median paired difference (bits)": float(np.median(diff)),
            "worlds contingent higher": float(np.mean(diff > 0)),
            "lower bound above 0.5": bool(lo > BAR)}


def clean_rows(ep: dict) -> list[int]:
    """The chunk rows of the episode's actions whose window (the replies to the W turns before) holds no can't-go
    reply. Scoring only: it reads the world's text, never the reading."""
    bad = {c["turn"] for c in ep["chunks"] if c["slot"] == "world" and c["turn"] >= 0 and CANT in c["text"]}
    return [i for i, c in enumerate(ep["chunks"])
            if c["slot"] == "self" and not bad & set(range(c["turn"] - W, c["turn"]))]


def clean_mean(data, rows: list[int], spaces: list[str], field: str = "X") -> tuple[float, int]:
    """The mean over `spaces` of each space's mean `field` over the scored turns among `rows`, and how many there
    are."""
    per, n = [], 0
    for sp in spaces:
        v = np.asarray(data[f"{sp}:{field}"], dtype=float)[rows] if rows else np.array([])
        v = v[np.isfinite(v)]
        n = len(v)
        per.append(float(v.mean()) if len(v) else np.nan)
    return (float(np.mean(per)) if per and np.isfinite(per).all() else float("nan")), n


def score(run: Path, draws: int = DRAWS, form: str = "v1") -> dict:
    from body.stepm1 import episodes

    F, SF = FORMS[form]["field"], FORMS[form]["self"]

    eps = {arm: episodes(read(run / arm / "record.jsonl")) for arm in ARMS}
    means = {arm: {} for arm in ARMS}
    clean = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for (seed, _), ep in eps[arm].items():
            path = run / "checkpoints" / f"{arm}-{seed}-a{ep['attempts']}.npz"
            if not path.exists():
                raise SystemExit(f"{arm} episode {seed} has no code lengths: run read first")
            data = np.load(path)
            fields = (F, SF) + (("L_now", "X_w5", "X_w20") if form == "v1" else ())
            means[arm][seed] = {f: episode_means(data, f) for f in fields}
            layer_names = [nm for nm in means[arm][seed][F] if nm != "text"]
            clean[arm][seed] = clean_mean(data, clean_rows(ep), layer_names, F)
    worlds = sorted(set(means["contingent"]) & set(means["blind"]))
    layers = [nm for nm in means["contingent"][worlds[0]][F] if nm != "text"] if worlds else []

    def table(field: str, spaces: list[str]) -> dict:
        return {w: {arm: float(np.nanmean([means[arm][w][field][sp] for sp in spaces])) for arm in ARMS}
                for w in worlds}

    main = compare(table(F, layers), draws)
    ok = [w for w in worlds if all(clean[arm][w][1] >= MIN_CLEAN_TURNS for arm in ARMS)]
    check = compare({w: {arm: clean[arm][w][0] for arm in ARMS} for w in ok}, draws)
    check["clean scored turns, per arm"] = {arm: int(sum(clean[arm][w][1] for w in worlds)) for arm in ARMS}
    too_few = check.get("worlds", 0) < MIN_CLEAN_WORLDS
    main_ok = bool(main.get("90% interval", [np.nan])[0] > BAR)
    check_ok = (not too_few) and bool(check["90% interval"][0] > BAR)
    out = {"form": form, "worlds played by both": len(worlds),
           f"{F}, mean over the read layers": main,
           "the can't-go check": {**check, "too few to decide": too_few},
           "gate": {"main, lower bound above 0.5": main_ok,
                    "can't-go check, lower bound above 0.5": None if too_few else check_ok,
                    "pass": main_ok and check_ok,
                    "meaning": ("rate may enter the sense tokens (calibrated; its predictive validity untested)"
                                if main_ok and check_ok else "rate stays out of the sense tokens")},
           "reported": {}}
    rep = out["reported"]
    rep[f"{F}, text baseline (TF-IDF)"] = compare(table(F, ["text"]), draws)
    rep[f"{F} per layer"] = {sp: compare(table(F, [sp]), 0).get("AUC") for sp in layers}
    if form == "v1":
        rep["X with W = 5, 20"] = {f: compare(table(f, layers), 0).get("AUC") for f in ("X_w5", "X_w20")}
    rep[f"self-progress {SF} (no direction expected)"] = compare(table(SF, layers), 0)
    rep[f"median {F} per arm (bits)"] = {arm: float(np.median([np.nanmean([means[arm][w][F][sp] for sp in layers])
                                                               for w in worlds])) for arm in ARMS} if worlds else {}
    # what the worlds said, per arm (keys and text; reported only)
    for arm in ARMS:
        new, moved, looks, turns = 0, 0, 0, 0
        for (seed, _), ep in eps[arm].items():
            acts = [c["text"] for c in ep["chunks"] if c["slot"] == "self"]
            turns += len(acts)
            looks += sum(x == "look" for x in acts)
            new += sum(k["new_state"] for k in ep["keys"].values())
            rooms = [k["room"] for _, k in sorted(ep["keys"].items())]
            moved += sum(r != q for r, q in zip(rooms, rooms[1:]))
        rep[f"{arm}: turns, share new state, share moved, share look"] = [
            turns, new / max(turns, 1), moved / max(turns, 1), looks / max(turns, 1)]
    skips = {arm: [e["seed"] for e in read(run / arm / "record.jsonl") if e["kind"] == "skip"] for arm in ARMS}
    rep["skipped"] = skips
    return out


def cmd_score(a) -> None:
    run = Path(a.run)
    result = score(run, form=getattr(a, "form", "v1"))
    Record(run / "record.jsonl").append("score", "code", step="rate_calibration", **result)
    print(json.dumps(result, indent=1, default=float))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("play", "read", "score"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        if name in ("play", "score"):
            s.add_argument("--form", choices=tuple(FORMS), default="v1",
                           help="v1: X as first built (registered, run); conditional: X_c on fresh seeds")
        if name == "play":
            s.add_argument("--limit", type=int, help="only the first n seeds, in whole pairs (a rehearsal)")
            s.add_argument("--rehearsal", action="store_true", help="the mirror seeds (+900000), deciding nothing")
        if name == "read":
            s.add_argument("--model", required=True)
            s.add_argument("--device", default="auto")
            s.add_argument("--dtype", default="bfloat16")
            s.add_argument("--gpu-memory", type=float)
            s.add_argument("--sets", default="material/paraphrases.jsonl")
            s.add_argument("--draft", action="store_true")
            s.add_argument("--no-variants", action="store_true", help="skip X at W = 5 and 20 (reported only)")
    a = p.parse_args(argv)
    {"play": cmd_play, "read": cmd_read, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
