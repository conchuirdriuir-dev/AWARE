"""Body readout v0.2, Step -1C: does shape read fit to learned structure, apart from boundary's novelty? (PLAN.md,
"Body readout", "Step -1C"; proposals/staged-schedule.md sec. 4a.)

Three commands, in order, as Step -1's. `play` draws the pool's seeds in order, the script playing, until 80 episodes
of each kind are usable, and appends every turn as it happens; `read` derives states from the recorded text alone,
with no graft, exactly as Step -1's `read` does (news, shape and token counts per chunk); `score` alone reads the
keys (each episode's kind and the turn it entered the planted room).

    python -m body.stepm1c play  --run runs/stepm1c-8b
    python -m body.stepm1c read  --model Qwen/Qwen3-8B --run runs/stepm1c-8b
    python -m body.stepm1c score --run runs/stepm1c-8b

The rehearsal (0.6B, deciding nothing) plays mirror seeds and reads with the 0.6B's own floor:

    python -m body.stepm1c play  --run runs/stepm1c-06 --rehearsal
    python -m body.stepm1c read  --model Qwen/Qwen3-0.6B --run runs/stepm1c-06 --floor runs/echo-floor-06-first/floor.json --draft
    python -m body.stepm1c score --run runs/stepm1c-06

The player is always the script (Stage 1's, unchanged); no model plays and the play needs no GPU. The length match
needs the Qwen3 tokenizer (`body/offtheme.py`).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from goalspace.formula import auc
from substrate.record import Record, read

WORLDS = Path(__file__).resolve().parent.parent / "material" / "habitat_worlds_stepm1c.json"
POOL = "step_minus_1c"
SHAPE_AUC, SHAPE_LOWER, DRAWS = 0.80, 0.70, 2000          # the owner's bars, 2026-11-05 (PLAN.md, fixed in advance)


# --- the pool and each seed's plan (keys: read by play's harness and by scoring only) ----------------------------

def spec() -> dict:
    return json.loads(WORLDS.read_text())


def seeds(rehearsal: bool = False) -> list[int]:
    from body.habitat import REHEARSAL

    lo, hi = spec()["pools"][POOL]["seeds"]
    return [s + REHEARSAL * rehearsal for s in range(lo, hi + 1)]


def plan(seed: int) -> dict:
    """The seed's plant: one extra room, unlocked at a turn drawn from 15-35 as Step -1's plants are, on-structure on
    an even seed and off-structure on an odd one. A rehearsal seed takes the kind of the seed it mirrors."""
    from body.habitat import mirrored

    turn = int(np.random.default_rng(seed + 500000).integers(15, 36))
    return {"kind": "new_room", "turn": turn, "vault": "on" if mirrored(seed) % 2 == 0 else "off"}


def draw(pool: list[int], play_one, per_kind: int, done: dict | None = None) -> dict:
    """Seeds in order until `per_kind` episodes of each kind are usable (the owner's rule, 2026-11-05): a seed is
    skipped, never loosened, when its world cannot be built, its off-structure description finds no length match,
    the script never enters the planted room, or its play runs past the caps (owner, 2026-11-05: TextWorld's quest
    tracking can explode mid-episode). A seed of a kind already full is not played. `play_one(seed)` returns how the
    episode ended ("unbuildable", "unmatched", "unplayable", "entered", "played"); `done` holds outcomes already
    in the record, which are counted and not played again. Returns the counts and the seeds drawn, by kind."""
    done = done or {}
    usable, counts, drawn = Counter(), {k: Counter() for k in ("on", "off")}, {"on": [], "off": []}
    for seed in pool:
        kind = plan(seed)["vault"]
        if usable[kind] >= per_kind:
            if all(usable[k] >= per_kind for k in counts):
                break
            continue
        how = done[seed] if seed in done else play_one(seed)
        counts[kind][how] += 1
        drawn[kind].append(seed)
        usable[kind] += how == "entered"
    return {"per_kind": per_kind, "usable": dict(usable), "full": all(usable[k] >= per_kind for k in counts),
            "counts": {k: {"entered": v["entered"], "unbuildable": v["unbuildable"], "no length match": v["unmatched"],
                           "unplayable": v["unplayable"], "never entered": v["played"]} for k, v in counts.items()},
            "seeds drawn": drawn}


def outcomes(entries: list[dict]) -> dict:
    """Seed -> how its episode ended, from the record: skips and complete episodes. A cut-off episode is not here,
    so play plays it again."""
    out = {}
    for e in entries:
        if e["kind"] == "skip":
            out[e["seed"]] = ("unmatched" if e["reason"] == "no length match"
                              else "unplayable" if e["reason"].startswith("unplayable") else "unbuildable")
        elif e["kind"] == "end" and "entered" in e:
            out[e["seed"]] = "entered" if e["entered"] else "played"
    return out


PLAY_MEMORY = 8 * 2**30      # as a build's cap (Step 0 amendment 4)
PLAY_SECONDS = 600


def play_capped(record: Record, seed: int, games: Path, memory: int = PLAY_MEMORY, seconds: float = PLAY_SECONDS) -> str:
    """One scripted episode in a subprocess capped at `memory` bytes of address space and `seconds`, appending to the
    same record. TextWorld's quest tracking (`GameProgression.update`) can explode on some states: seed 906402's play
    reached 28 GB at turn 76 in the rehearsal of 2026-11-05 and the machine killed it. Past either cap the episode is
    left cut off in the record (never scored: `stepm1.episodes` keeps complete episodes only) and the seed is skipped
    as unplayable, counted with the other skips before any reading (owner, 2026-11-05). The script's actions do not
    depend on the kind, so a seed that cannot be played is lost to both kinds alike."""
    import os
    import resource
    import signal
    import subprocess
    import sys

    from body.habitat import ROOT

    def cap():
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))

    code = ("import sys, warnings; warnings.filterwarnings('ignore'); from pathlib import Path; "
            "from body.habitat import Script, play; from body.stepm1c import plan, WORLDS; "
            "from substrate.record import Record; s = int(sys.argv[1]); "
            "print('PLAYED ' + play(Script(), Record(sys.argv[2]), s, Path(sys.argv[3]), worlds_path=WORLDS, "
            "plan=plan(s)))")
    p = subprocess.Popen([sys.executable, "-c", code, str(seed), str(Path(record.path).resolve()), str(Path(games).resolve())], cwd=ROOT,
                         env={**os.environ, "PYTHONPATH": str(ROOT)}, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, preexec_fn=cap, start_new_session=True)
    try:
        out, err = p.communicate(timeout=seconds)
        why = None if p.returncode == 0 else f"exit {p.returncode}: {(err.strip().splitlines() or ['no output'])[-1][:200]}"
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)     # the build it may have started goes with it
        p.communicate()
        out, why = "", f"over {seconds:.0f} s"
    record.seq = len(read(record.path))      # the subprocess appended; number on from its last entry
    line = next((x for x in out.splitlines() if x.startswith("PLAYED ")), None)
    if why or line is None:
        record.append("skip", "code", seed=seed, reason=f"unplayable: {why or 'no outcome'}")
        return "unplayable"
    return line[7:]


def cut_off(entries: list[dict]) -> list[int]:
    """Seeds whose latest episode began but never ended (the run stopped mid-episode), in record order."""
    open_, out = None, []
    for e in entries:
        if e["kind"] == "episode":
            if open_ is not None:
                out.append(open_)
            open_ = e["seed"]
        elif e["kind"] == "end":
            open_ = None
    if open_ is not None:
        out.append(open_)
    ended = {e["seed"] for e in entries if e["kind"] in ("end", "skip")}
    resumed = {s for e in entries if e["kind"] == "resume" for s in e["cut_off"]}
    return [s for s in dict.fromkeys(out) if s not in ended and s not in resumed]


# --- scoring (the only reader of keys) ----------------------------------------------------------------------------

def scored(ep: dict) -> int | None:
    """The chunk scored in an episode: the reply on the turn the script entered the planted room (its first
    description). None if it never entered."""
    turn = next((t for t, k in ep["keys"].items() if k.get("entered")), None)
    if turn is None:
        return None
    return next(i for i, c in enumerate(ep["chunks"]) if c["slot"] == "world" and c["turn"] == turn
                and c.get("cause", "elicited") == "elicited")


def _aucs(rows: list[dict], field: str) -> float:
    return auc([r[field] for r in rows if r["kind"] == "off"], [r[field] for r in rows if r["kind"] == "on"])


def score(eps: dict, news_by: dict, shape_by: dict, tokens_by: dict, draws: int = DRAWS, seed: int = 0) -> dict:
    """Step -1C's two bars and its precondition, from the episodes and their readings (episode -> chunks x layers).
    Every episode gives one chunk, so resampling episodes resamples chunks."""
    from body.stepm1 import BINS, MIN_BIN

    rows = []
    for key, ep in sorted(eps.items()):
        i = scored(ep)
        if i is None or key not in shape_by:
            continue
        rows.append({"seed": key[0], "kind": ep["info"]["plan"]["vault"], "shape": float(np.mean(shape_by[key][i])),
                     "news": float(np.mean(news_by[key][i])), "tokens": int(tokens_by[key][i]),
                     "shape_layers": shape_by[key][i], "news_layers": news_by[key][i]})
    bad = [r["seed"] for r in rows if not (np.isfinite(r["shape"]) and np.isfinite(r["news"]))]
    if bad:
        raise ValueError(f"no shape or news on the scored chunk of {bad}: run read again")
    n = Counter(r["kind"] for r in rows)
    out: dict = {"episodes": dict(n), "reported": {}}

    a_shape, a_news = _aucs(rows, "shape"), _aucs(rows, "news")
    rng, boot_shape, boot_diff = np.random.default_rng(seed), [], []
    for _ in range(draws):
        pick = [rows[j] for j in rng.integers(0, len(rows), len(rows))]
        if len({r["kind"] for r in pick}) < 2:
            continue
        s, b = _aucs(pick, "shape"), _aucs(pick, "news")
        boot_shape.append(s)
        boot_diff.append(s - b)
    lo, hi = (np.percentile(boot_shape, [5, 95]) if boot_shape else (math.nan, math.nan))
    dlo, dhi = (np.percentile(boot_diff, [5, 95]) if boot_diff else (math.nan, math.nan))

    # 1. shape: off-structure over on-structure
    out["shape"] = {"auc": a_shape, "interval_90": [float(lo), float(hi)], "bar": SHAPE_AUC, "lower_bar": SHAPE_LOWER,
                    "pass": bool(a_shape >= SHAPE_AUC and lo > SHAPE_LOWER)}
    # 2. apart from boundary, one-sided (owner, 2026-11-05, amendment 1, before any reading): shape's AUC minus
    #    boundary's on the same chunks, its 90% interval entirely above zero
    out["apart_from_boundary"] = {"shape_auc": a_shape, "boundary_auc": a_news, "margin": a_shape - a_news,
                                  "interval_90": [float(dlo), float(dhi)],
                                  "direction": "shape above boundary" if a_shape > a_news else "boundary above shape",
                                  "pass": bool(dlo > 0)}
    # precondition, reported: boundary fires (news above the floor) on both kinds, since both rooms are unseen
    out["precondition"] = {k: f"{sum(r['news'] > 1 for r in rows if r['kind'] == k)} of {n[k]} above the floor"
                           for k in ("on", "off")}
    out["pass"] = out["shape"]["pass"] and out["apart_from_boundary"]["pass"]

    # reported, deciding nothing (amendment 1): shape given boundary, which a ceiling on both AUCs does not hide
    out["reported"]["shape given boundary"] = given_boundary(rows)

    # reported, deciding nothing: per layer, length beside every comparison, length-binned AUCs
    L = len(rows[0]["shape_layers"]) if rows else 0
    out["reported"]["per_layer"] = {
        f"layer {j}": {"shape_auc": auc([r["shape_layers"][j] for r in rows if r["kind"] == "off"],
                                        [r["shape_layers"][j] for r in rows if r["kind"] == "on"]),
                       "boundary_auc": auc([r["news_layers"][j] for r in rows if r["kind"] == "off"],
                                           [r["news_layers"][j] for r in rows if r["kind"] == "on"])}
        for j in range(L)}
    out["reported"]["length"] = {
        "median tokens": {k: float(np.median([r["tokens"] for r in rows if r["kind"] == k])) if n[k] else None
                          for k in ("on", "off")},
        "tokens auc, off over on": _aucs(rows, "tokens")}
    bins = defaultdict(list)
    for r in rows:
        b = next((f"[{lo_}, {hi_})" for lo_, hi_ in zip(BINS, BINS[1:]) if lo_ <= r["tokens"] < hi_), None)
        if b is not None:
            bins[b].append(r)
    out["reported"]["by length bin"] = {
        b: ({"on": sum(r["kind"] == "on" for r in v), "off": sum(r["kind"] == "off" for r in v),
             "shape_auc": _aucs(v, "shape"), "boundary_auc": _aucs(v, "news")}
            if min(sum(r["kind"] == k for r in v) for k in ("on", "off")) >= MIN_BIN
            else {"on": sum(r["kind"] == "on" for r in v), "off": sum(r["kind"] == "off" for r in v),
                  "note": f"too few (under {MIN_BIN} of a kind)"})
        for b, v in sorted(bins.items())}
    out["reported"]["medians"] = {
        k: {"shape bits": float(np.median([r["shape"] for r in rows if r["kind"] == k])) if n[k] else None,
            "news": float(np.median([r["news"] for r in rows if r["kind"] == k])) if n[k] else None}
        for k in ("on", "off")}
    return out


def given_boundary(rows: list[dict]) -> dict:
    """Does shape still part the kinds once boundary's part is taken out? Reported, deciding nothing (owner,
    2026-11-05, amendment 1): shape less its least-squares line on news, fitted on all chunks blind to the kind, then
    its AUC off over on. Conservative where boundary itself parts the kinds: the line on news then takes up part of
    the kind's effect on shape too, so a real effect reads low, never high. (Strata of news were tried on toy material
    and dropped, written before any run: when boundary parts the kinds, each stratum's off chunks still sit higher in
    news, and a shape that only tracks news read 0.97.)"""
    if not rows:
        return {}
    news = np.array([r["news"] for r in rows])
    sh = np.array([r["shape"] for r in rows])
    off = np.array([r["kind"] == "off" for r in rows])
    slope, icept = np.polyfit(news, sh, 1) if len(set(news)) > 1 else (0.0, float(np.mean(sh)))
    resid = sh - (slope * news + icept)
    return {"slope": float(slope), "shape_auc": auc(resid[off], resid[~off]),
            "reads": "conservative where boundary parts the kinds"}


# --- commands -----------------------------------------------------------------------------------------------------

def cmd_play(a) -> None:
    from body.habitat import Script, play

    run = Path(a.run)
    record = Record(run / "record.jsonl")
    entries = read(record.path) if record.path.exists() else []
    done = outcomes(entries)
    cut = cut_off(entries)
    if cut:                                  # said in the record before the seed is played again from its start
        record.append("resume", "code", cut_off=cut, note="episodes begun but never ended; each is played again, "
                      "and the last attempt is the one kept (stepm1.episodes)")
    per_kind = a.limit or spec()["pools"][POOL]["per_kind"]

    def play_one(seed):
        return play_capped(record, seed, run / "games")

    result = draw(seeds(a.rehearsal), play_one, per_kind, done)
    record.append("drawn", "code", rehearsal=a.rehearsal, **result)
    print(json.dumps({k: v for k, v in result.items() if k != "seeds drawn"}, indent=2))
    if not result["full"]:
        print("the pool ran out before every kind was full: for the owner")


def cmd_read(a) -> None:
    from body.stepm1 import cmd_read as read_as_step_minus_1

    read_as_step_minus_1(a)


def cmd_score(a) -> None:
    from body.stepm1 import episodes

    run = Path(a.run)
    entries = read(run / "record.jsonl")
    eps = {k: v for k, v in episodes(entries).items() if "plan" in v["info"]}
    load = lambda k, s: np.load(run / "news" / f"{k[0]}-world{s}.npy")
    news_by = {k: load(k, "") for k in eps}
    stale = [k for k in eps if news_by[k].shape[0] != len(eps[k]["chunks"])]
    if stale:
        raise SystemExit(f"news does not match the record for {stale}: run read again")
    shape_by = {k: load(k, "-shape") for k in eps}
    tokens_by = {k: load(k, "-tokens") for k in eps}
    result = score(eps, news_by, shape_by, tokens_by)
    drawn = [e for e in entries if e["kind"] == "drawn"]
    if drawn:
        result["drawn"] = {k: v for k, v in drawn[-1].items() if k in ("usable", "full", "counts", "rehearsal")}
    result["reported"]["chunks cut at 256 tokens"] = int(sum((t > 256).sum() for t in tokens_by.values()))
    Record(run / "record.jsonl").append("score", "code", **{k: v for k, v in result.items()})
    print(json.dumps(result, indent=2, default=float))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("play", "read", "score"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        if name == "play":
            s.add_argument("--rehearsal", action="store_true", help="mirror seeds: new worlds on the pool's plan")
            s.add_argument("--limit", type=int, help="usable episodes per kind (a dry run); the registered run is 80")
        if name == "read":
            s.add_argument("--model", required=True)
            s.add_argument("--device", default="auto")
            s.add_argument("--dtype", default="bfloat16")
            s.add_argument("--floor", default="material/echo_floor_8b.json")
            s.add_argument("--sets", default="material/paraphrases.jsonl")
            s.add_argument("--gpu-memory", type=float)
            s.add_argument("--draft", action="store_true")
    a = p.parse_args(argv)
    {"play": cmd_play, "read": cmd_read, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
