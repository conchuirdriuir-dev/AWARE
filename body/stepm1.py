"""Body readout v0.2, Step -1: does boundary hear the world and not itself? (PLAN.md, "Body readout", "The habitat".)

Three commands, in order. The record comes first: `play` appends every turn as it happens; `read` derives states from the
recorded text alone, with no graft, and checkpoints them; `score` alone reads the keys.

    python -m body.stepm1 play --model Qwen/Qwen3-8B --run runs/stepm1-8b --pool calibration
    python -m body.stepm1 play --model Qwen/Qwen3-8B --run runs/stepm1-8b --pool step_minus_1
    python -m body.stepm1 play --model Qwen/Qwen3-8B --run runs/stepm1-8b --pool self_talk
    python -m body.stepm1 read --model Qwen/Qwen3-8B --run runs/stepm1-8b
    python -m body.stepm1 score --run runs/stepm1-8b

The floor file holds the echo floor per layer of the last third, in units of D, measured with `read`'s own settings by
`body/floor_file.py`; for Qwen3-8B it is `material/echo_floor_8b.json` (the default). Another model needs its own:
the 0.6B dry run makes one with `python -m body.floor_file --model Qwen/Qwen3-0.6B ...` and passes it as --floor. D is
measured on the paraphrase sets in the same run, as Step 0 does. A dry run passes --limit to play a few worlds of each
pool;
--rehearsal plays new worlds on the pool's plan instead, spending none of its registered seeds.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from goalspace.formula import auc
from substrate.record import Record, read

SELF_TALK_BARS = {"exact": 0.01, "restated": 0.05}
FIRE_TOTAL, FIRE_EACH, AUC_BAR, AUC_LOWER, DRAWS = 36, 17, 0.90, 0.80, 2000
FLOOR_CHECK = 0.95
POOLS = ("calibration", "step_minus_1", "self_talk")


# --- material: the record, grouped into episodes -----------------------------------------------------------------

def episodes(entries: list[dict], complete_only: bool = True) -> dict:
    """Episode key -> its chunks in order (the opening, then each turn's action and reply), its keys by turn, and
    what the episode entry says. The readout side uses only `chunks`.

    An episode is complete when its `end` entry follows. One cut off (the run stopped mid-episode) stays in the record
    but is left out here, and `play` plays it again from the start; the last attempt is the one kept, and `attempts`
    counts them. A record written before `end` entries existed has none, and every episode in it counts as complete."""
    out: dict = {}
    current, attempts = None, defaultdict(int)
    legacy = not any(e["kind"] == "end" for e in entries)
    for e in entries:
        if e["kind"] == "episode":
            current = (e["seed"], e.get("self_talk"))
            attempts[current] += 1
            out[current] = {"info": e, "chunks": [], "keys": {}, "complete": legacy, "attempts": attempts[current]}
        elif current is None:
            continue
        elif e["kind"] == "chunk":
            out[current]["chunks"].append(e)
        elif e["kind"] == "key":
            out[current]["keys"][e["turn"]] = e
        elif e["kind"] == "end":
            out[current]["complete"] = True
            current = None
    return {k: v for k, v in out.items() if v["complete"] or not complete_only}


def pool_of(key, spec) -> str:
    """The pool an episode is scored in; a rehearsal seed is scored in the pool of the seed it mirrors."""
    from body.habitat import mirrored, pool

    seed, variant = key
    if variant:
        return "self_talk"
    for name in ("calibration", "step_minus_1", "step0_habitat", "behaviour_test"):
        if mirrored(seed) in pool(name, spec):
            return name
    return "babbling"


# --- the sense: boundary (news) -----------------------------------------------------------------------------------

def news(S: np.ndarray, slots: list[str], unit: np.ndarray) -> np.ndarray:
    """For each chunk in the world slot after the opening, its nearest distance to every earlier chunk of the episode
    (actions and replies, this turn's action included), per layer, in echo-floor units. S is chunks x layers x width;
    unit is the floor per layer in raw distance. NaN where no news is read (self chunks, the opening)."""
    out = np.full((len(slots), S.shape[1]), np.nan)
    for i, slot in enumerate(slots):
        if slot == "world" and i > 0:
            out[i] = np.min(np.linalg.norm(S[:i] - S[i][None], axis=2), axis=0) / unit
    return out


# --- scoring (the only reader of keys) ----------------------------------------------------------------------------

def _boot(pos_by, neg_by, draws=DRAWS, seed=0) -> tuple[float, float, float]:
    """AUC of pos over neg, with a 90% interval from resampling episodes (each with its own pos and neg)."""
    eps = sorted(set(pos_by) | set(neg_by))
    point = auc([v for e in eps for v in pos_by.get(e, [])], [v for e in eps for v in neg_by.get(e, [])])
    rng, vals = np.random.default_rng(seed), []
    for _ in range(draws):
        pick = [eps[i] for i in rng.integers(0, len(eps), len(eps))]
        p = [v for e in pick for v in pos_by.get(e, [])]
        n = [v for e in pick for v in neg_by.get(e, [])]
        if p and n:
            vals.append(auc(p, n))
    lo, hi = np.percentile(vals, [5, 95]) if vals else (math.nan, math.nan)
    return point, float(lo), float(hi)


def turn_news(ep: dict, N: np.ndarray, cause: str = "elicited") -> dict:
    """Turn -> the mean over layers of its reply's news; with cause "spontaneous", of the planted event's own chunk
    (Step -1 amendment 3). Self-talk replies carry no cause and count as replies."""
    return {c["turn"]: float(np.mean(N[i])) for i, c in enumerate(ep["chunks"])
            if c["slot"] == "world" and c["turn"] >= 0 and c.get("cause", "elicited") == cause}


def joined(ep: dict) -> list[tuple[int, str]]:
    """For each planted event's chunk i: the text its turn's reply and event make together, as the agent reads them.
    Its news is read against every chunk before the reply (reported, deciding nothing)."""
    ch = ep["chunks"]
    return [(i, ch[i - 1]["text"] + "\n\n" + c["text"]) for i, c in enumerate(ch)
            if c.get("cause") == "spontaneous" and c["turn"] >= 0]      # the opening is spontaneous too


def score(eps: dict, news_by: dict, spec: dict, joined_by: dict | None = None) -> dict:
    """Step -1's three bars and the floor's habitat check, from episodes and their news (episode -> chunks x layers)."""
    out: dict = {"reported": {}}
    by_pool = defaultdict(list)
    for key in eps:
        by_pool[pool_of(key, spec)].append(key)

    # the floor's habitat check: replies identical to an earlier reply of the same calibration episode
    below = []
    for key in by_pool["calibration"]:
        seen, tn = set(), turn_news(eps[key], news_by[key])
        for c in eps[key]["chunks"]:
            if c["slot"] == "world" and c["turn"] >= 0 and c.get("cause", "elicited") == "elicited":
                if c["text"] in seen:
                    below.append(tn[c["turn"]] < 1)
                seen.add(c["text"])
    share = float(np.mean(below)) if below else math.nan
    out["floor_check"] = {"repeats": len(below), "below_floor": share, "bar": FLOOR_CHECK,
                          "pass": bool(below) and share >= FLOOR_CHECK}

    # k, from the calibration episodes' resolved stalls; fixed in PLAN.md before Step 0, not used here
    from body.habitat import choose_k

    flags = [[eps[key]["keys"][t]["new_state"] for t in sorted(eps[key]["keys"])] for key in by_pool["calibration"]]
    try:
        out["k"] = {"k": choose_k(flags), "episodes": len(flags)}
    except ValueError as err:
        out["k"] = {"k": None, "episodes": len(flags), "why": str(err)}

    # 1. self-talk
    st = {}
    for variant, bar in SELF_TALK_BARS.items():
        vals = [v for key in by_pool["self_talk"] if key[1] == variant
                for v in turn_news(eps[key], news_by[key]).values()]
        above = float(np.mean([v > 1 for v in vals])) if vals else math.nan
        st[variant] = {"arrivals": len(vals), "above_floor": above, "bar": bar, "pass": bool(vals) and above <= bar}
    out["self_talk"] = st

    # 2 and 3. planted novelty; a plant that did not happen counts as not firing
    fired, pos, neg, natural_new, natural_flat = defaultdict(list), {}, {}, {}, {}
    collided = defaultdict(list)   # delivered plants by whether the agent's own action also made a new state there
    appended = defaultdict(list)   # delivered plants read as before amendment 3: the event inside its turn's reply
    for key in by_pool["step_minus_1"]:
        ep, tn = eps[key], turn_news(eps[key], news_by[key])
        kind = ep["info"].get("planted_kind")
        ev = turn_news(ep, news_by[key], "spontaneous")               # the plant's own chunk, where recorded apart
        tn = {**tn, **ev}
        jn = turn_news(ep, joined_by[key], "spontaneous") if joined_by and key in joined_by else {}
        here = [t for t, k in ep["keys"].items() if k.get("planted_here")]
        delivered = [t for t in here if ep["keys"][t].get("switched", True) and
                     (kind != "rule_change" or ep["keys"][t].get("blocked"))]
        if kind:
            fired[kind].append(bool(delivered) and tn.get(delivered[0], 0) > 1)
            if delivered and delivered[0] in jn:
                appended[kind].append(jn[delivered[0]] > 1)
            if delivered:
                natural = ep["keys"][delivered[0]].get("natural_new")
                collided[{True: "with", False: "clean", None: "unknown"}[natural]].append(fired[kind][-1])
            if delivered:
                pos[key] = [tn[delivered[0]]]
        flat = [tn[t] for t, k in ep["keys"].items() if not k["new_state"] and t not in here and t in tn]
        new = [tn[t] for t, k in ep["keys"].items() if k["new_state"] and t not in here and t in tn]
        if kind:
            neg[key] = flat
        natural_new[key], natural_flat[key] = new, flat
    total = sum(sum(v) for v in fired.values())
    events = sum(len(v) for v in fired.values())
    out["fires"] = {"fired": total, "events": events, "by_kind": {k: f"{sum(v)} of {len(v)}" for k, v in fired.items()},
                    "pass": total >= FIRE_TOTAL and all(sum(fired[k]) >= FIRE_EACH for k in ("new_room", "rule_change"))}
    a, lo, hi = _boot(pos, neg)
    out["above_ordinary"] = {"auc": a, "interval_90": [lo, hi], "bar": AUC_BAR, "lower_bar": AUC_LOWER,
                             "pass": a >= AUC_BAR and lo > AUC_LOWER}
    out["reported"]["plants_on_natural_novelty"] = {
        "deciding": "nothing", **{k: f"{sum(v)} of {len(v)} fired" for k, v in sorted(collided.items())}}
    out["reported"]["plants_read_appended"] = {
        "deciding": "nothing", **{k: f"{sum(v)} of {len(v)} fired" for k, v in sorted(appended.items())}}
    out["reported"]["natural_new_state"] = dict(zip(("auc", "lo", "hi"), _boot(natural_new, natural_flat)))
    out["pass"] = all(v["pass"] for v in st.values()) and out["fires"]["pass"] and out["above_ordinary"]["pass"]
    return out


# --- commands -----------------------------------------------------------------------------------------------------

def cmd_play(a) -> None:
    from body.habitat import REHEARSAL, Agent, play, pool, self_talk, worlds

    spec = worlds()
    record = Record(Path(a.run) / "record.jsonl")
    done = set(episodes(read(record.path)).keys()) if record.path.exists() else set()
    if (a.nf4 or a.context or a.generator) and not a.rehearsal:
        raise SystemExit("--nf4, --context and --generator change the player or worlds: rehearsals only")
    gen = json.loads(a.generator) if a.generator else None
    agent = Agent(a.model, device=a.device, dtype=a.dtype, cache=not a.no_cache, nf4=a.nf4,
                  **({"context": a.context} if a.context else {}))
    seeds = [s + REHEARSAL * a.rehearsal for s in pool(a.pool, spec)[: a.limit or None]]
    for seed in seeds:
        if a.pool == "self_talk":
            for variant in ("exact", "restated"):
                if (seed, variant) not in done:
                    self_talk(agent, record, seed, variant, spec["pools"]["step_minus_1"]["self_talk"]["turns"])
        elif (seed, None) not in done:
            play(agent, record, seed, Path(a.run) / "games", generator=gen)


def cmd_read(a) -> None:
    from substrate.echo_floor import d_resumable
    from substrate.measure_d import load_sets
    from substrate.states import Reader

    run = Path(a.run)
    record = Record(run / "record.jsonl")
    eps = episodes(read(record.path))
    floor = {int(k): float(v) for k, v in json.loads(Path(a.floor).read_text()).items()}
    reader = Reader(a.model, device=a.device, dtype=a.dtype, gpu_memory=a.gpu_memory, sink="first")
    n = reader.info["layers"]
    layers = list(range(math.ceil(2 * n / 3), n))
    missing = [l for l in layers if l not in floor]
    if missing:
        raise SystemExit(f"the floor file has no floor for layers {missing}")
    D = d_resumable(reader, load_sets(a.sets), layers, record, a.draft, True)
    unit = np.array([floor[l] * float(D[j]) for j, l in enumerate(layers)])
    record.append("read", "code", **reader.info, read_layers=layers, floor_D=[floor[l] for l in layers],
                  D=[float(x) for x in D], draft=a.draft)
    out = run / "news"
    out.mkdir(exist_ok=True)
    for (seed, variant), ep in eps.items():
        path = out / f"{seed}-{variant or 'world'}.npy"
        if path.exists() and np.load(path).shape[0] == len(ep["chunks"]):
            continue                                      # read already; a replayed episode is read again
        S = reader.states([c["text"] for c in ep["chunks"]], layers)
        np.save(path, news(S, [c["slot"] for c in ep["chunks"]], unit))
        pairs = joined(ep)
        if pairs:                                         # reported only: the event read inside its turn's reply
            J = np.full((len(ep["chunks"]), len(layers)), np.nan)
            T = reader.states([text for _, text in pairs], layers)
            for (i, _), state in zip(pairs, T):
                J[i] = np.min(np.linalg.norm(S[: i - 1] - state[None], axis=2), axis=0) / unit
            np.save(out / f"{seed}-{variant or 'world'}-joined.npy", J)


def cmd_score(a) -> None:
    from body.habitat import worlds

    run = Path(a.run)
    eps = episodes(read(run / "record.jsonl"))
    news_by = {k: np.load(run / "news" / f"{k[0]}-{k[1] or 'world'}.npy") for k in eps}
    stale = [k for k in eps if news_by[k].shape[0] != len(eps[k]["chunks"])]
    if stale:
        raise SystemExit(f"news does not match the record for {stale}: run read again")
    joined_by = {k: np.load(p) for k in eps if (p := run / "news" / f"{k[0]}-{k[1] or 'world'}-joined.npy").exists()}
    result = score(eps, news_by, worlds(), joined_by)
    Record(run / "record.jsonl").append("score", "code", **result)
    print(json.dumps(result, indent=2))


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
            s.add_argument("--pool", choices=POOLS, required=True)
            s.add_argument("--limit", type=int, help="the dry run: only the first n seeds of the pool")
            s.add_argument("--no-cache", action="store_true", help="re-read the whole prompt every turn")
            s.add_argument("--nf4", action="store_true", help="a 4-bit model: rehearsals only, recorded on every episode")
            s.add_argument("--context", type=int, help="a shorter context in tokens: rehearsals only")
            s.add_argument("--generator", help='candidate world settings as JSON, e.g. {"rooms": [10, 20], ...}: rehearsals only')
            s.add_argument("--rehearsal", action="store_true",
                           help="play the pool's mirror seeds (+900000): new worlds on the same plan, no registered seed spent")
        if name == "read":
            s.add_argument("--floor", default="material/echo_floor_8b.json")
            s.add_argument("--sets", default="material/paraphrases.jsonl")
            s.add_argument("--gpu-memory", type=float)
            s.add_argument("--draft", action="store_true")
    a = p.parse_args(argv)
    {"play": cmd_play, "read": cmd_read, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
