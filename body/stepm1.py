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
import time
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
            if "drop_from_turn" in e:            # cut by the engine's cap (Step 0 amendment 6): kept to its last full turn
                cut = e["drop_from_turn"]
                out[current]["chunks"] = [c for c in out[current]["chunks"] if c["turn"] < cut]
                out[current]["keys"] = {t: k for t, k in out[current]["keys"].items() if t < cut}
                out[current]["cut"] = e["cut"]
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


def shape_report(eps: dict, shape_by: dict, tokens_by: dict, spec: dict) -> dict:
    """Shape by pool and source (self, world, planted event), pooled too: the median over scored chunks of the mean over
    layers, per-layer medians, and the chunks' median token length beside it. Per chunk, the files hold both."""
    groups = defaultdict(lambda: {"shape": [], "layers": [], "tokens": []})
    for key, ep in eps.items():
        if key not in shape_by:
            continue
        Sh, Tk, name = shape_by[key], tokens_by.get(key), pool_of(key, spec)
        for i, c in enumerate(ep["chunks"]):
            if not np.isfinite(Sh[i]).all():
                continue
            src = "self" if c["slot"] == "self" else ("planted event" if c.get("cause") == "spontaneous" else "world")
            for g in (f"{name}: {src}", f"{name}: pooled"):
                groups[g]["shape"].append(float(np.mean(Sh[i])))
                groups[g]["layers"].append(Sh[i])
                if Tk is not None:
                    groups[g]["tokens"].append(int(Tk[i]))
    return {g: {"chunks": len(v["shape"]), "median bits": float(np.median(v["shape"])),
                "median bits per layer": [float(x) for x in np.median(np.array(v["layers"]), axis=0)],
                "median tokens": float(np.median(v["tokens"])) if v["tokens"] else None}
            for g, v in sorted(groups.items())}


def score(eps: dict, news_by: dict, spec: dict, joined_by: dict | None = None, shape_by: dict | None = None,
          tokens_by: dict | None = None) -> dict:
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
        out["k"] = {"k": choose_k(flags), "episodes": len(flags),
                "note": "reported only: k is set from Stage 2's player on the calibration pool"}
    except ValueError as err:
        out["k"] = {"k": None, "episodes": len(flags), "why": str(err)}

    # 1. self-talk
    st = {}
    for variant, bar in SELF_TALK_BARS.items():
        vals = [v for key in by_pool["self_talk"] if key[1] == variant
                for v in turn_news(eps[key], news_by[key]).values()]
        above = float(np.mean([v > 1 for v in vals])) if vals else math.nan
        st[variant] = {"arrivals": len(vals), "above_floor": above, "bar": bar, "pass": bool(vals) and above <= bar}
    # Amendment 4 (owner, 2026-11-05, before any 8B run): the restated part is reported in Stage 1 and gates at the
    # tickle test after babbling; the exact part gates Stage 1 unchanged.
    st["exact"]["gates"] = "Stage 1"
    st["restated"]["gates"] = "the tickle test after babbling (amendment 4): reported here, deciding nothing"
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
    # per layer (registered, reported): the third bar's AUC and the plants fired, layer by layer
    per_layer = {}
    n_layers = next(iter(news_by.values())).shape[1] if news_by else 0
    for j in range(n_layers):
        p_j, n_j, f_j = {}, {}, 0
        for key in by_pool["step_minus_1"]:
            ep, N = eps[key], news_by[key]
            rows = {c["turn"]: i for i, c in enumerate(ep["chunks"]) if c["slot"] == "world" and c["turn"] >= 0}
            ev = {c["turn"]: i for i, c in enumerate(ep["chunks"]) if c.get("cause") == "spontaneous" and c["turn"] >= 0}
            here = [t for t, k in ep["keys"].items() if k.get("planted_here")]
            if here and ep["info"].get("planted_kind"):
                i = ev.get(here[0], rows.get(here[0]))
                if i is not None:
                    p_j[key] = [float(N[i, j])]
                    f_j += N[i, j] > 1
            n_j[key] = [float(N[rows[t], j]) for t, k in ep["keys"].items()
                        if not k["new_state"] and t not in here and t in rows]
        a_j = auc([v for x in p_j.values() for v in x], [v for x in n_j.values() for v in x])
        per_layer[f"layer {j}"] = {"above_ordinary_auc": a_j, "plants_fired": int(f_j)}
    out["reported"]["per_layer"] = per_layer
    # throughput and turns dropped from context (registered, reported), from the record's own times and fields
    play = defaultdict(lambda: {"turns": 0, "seconds": 0.0, "turns_with_dropped": 0})
    for key, ep in eps.items():
        name = pool_of(key, spec)
        ats = [c["at"] for c in ep["chunks"] if c.get("at")]
        acts = [c for c in ep["chunks"] if c["slot"] == "self"]
        play[name]["turns"] += len(acts)
        if len(ats) > 1:
            from datetime import datetime
            play[name]["seconds"] += (datetime.fromisoformat(ats[-1]) - datetime.fromisoformat(ats[0])).total_seconds()
        play[name]["turns_with_dropped"] += sum(bool(c.get("dropped")) for c in acts)
    out["reported"]["play"] = {k: {**v, "turns_per_hour": v["turns"] / v["seconds"] * 3600 if v["seconds"] else None}
                               for k, v in play.items()}
    if shape_by:
        out["reported"]["shape"] = shape_report(eps, shape_by, tokens_by, spec)
    out["reported"]["not produced"] = ("coupling and rate per turn: registered as reported, not yet computed"
                                       + ("" if shape_by else "; shape: no shape files (run read again)"))
    out["reported"]["natural_new_state"] = dict(zip(("auc", "lo", "hi"), _boot(natural_new, natural_flat)))
    out["pass"] = st["exact"]["pass"] and out["fires"]["pass"] and out["above_ordinary"]["pass"]   # amendment 4
    return out


# --- commands -----------------------------------------------------------------------------------------------------

def cmd_play(a) -> None:
    from body.habitat import REHEARSAL, Agent, Script, play, pool, self_talk, worlds

    spec = worlds()
    record = Record(Path(a.run) / "record.jsonl")
    done = set(episodes(read(record.path)).keys()) if record.path.exists() else set()
    done |= {(e["seed"], None) for e in read(record.path) if e["kind"] == "skip"} if record.path.exists() else set()
    if (a.nf4 or a.context or a.generator) and not a.rehearsal:
        raise SystemExit("--nf4, --context and --generator change the player or worlds: rehearsals only")
    gen = json.loads(a.generator) if a.generator else None
    if not a.script and not a.model:
        raise SystemExit("play needs --model, or --script for Stage 1's scripted player")
    if a.script and a.pool == "self_talk":
        raise SystemExit("self-talk is the perceiver's own words: the 8B writes it, not the script")
    agent = Script() if a.script else Agent(a.model, device=a.device, dtype=a.dtype, cache=not a.no_cache, nf4=a.nf4, forms=a.forms,
                                            **({"context": a.context} if a.context else {}))
    record_machine(record, f"stepm1 play --pool {a.pool}")
    seeds = [s + REHEARSAL * a.rehearsal for s in pool(a.pool, spec)[: a.limit or None]]
    for seed in seeds:
        if a.pool == "self_talk":
            for variant in ("exact", "restated"):
                if (seed, variant) not in done:
                    self_talk(agent, record, seed, variant, spec["pools"]["step_minus_1"]["self_talk"]["turns"])
        elif (seed, None) not in done:
            play(agent, record, seed, Path(a.run) / "games", generator=gen)


SHAPE_BURN, SHAPE_CAP = 10, 256          # shape (owner, 2026-11-05, PR #32 sec. 4a): turns that fit and never score


def shape(S: np.ndarray, turns: list[int], floor_var: list[float]) -> np.ndarray:
    """Shape per chunk and layer (Spec v0.2; reported, deciding nothing): the bits of each chunk under the formula
    fitted on every earlier chunk of the episode, both sources (Lift v0: `goalspace.formula.prequential`'s Lk,
    computed from the Gram matrix as Step 0 does). Chunks in the first SHAPE_BURN turns fit and never score: NaN."""
    from body.machine import record_machine
from body.progress import code

    n, L, d = S.shape
    out = np.full((n, L), np.nan)
    start = next((i for i, t in enumerate(turns) if t >= SHAPE_BURN), n)
    for j in range(L):
        X = S[:, j].astype(np.float64)
        G = X @ X.T
        for t in range(max(start, 2), n):
            out[t, j] = code(G, np.arange(t), t, d, floor_var[j])[0]
    return out


def cmd_read(a) -> None:
    from substrate.echo_floor import d_resumable
    from substrate.measure_d import load_sets
    from substrate.states import Reader

    run = Path(a.run)
    record = Record(run / "record.jsonl")
    eps = episodes(read(record.path))
    record_machine(record, "stepm1 read")
    started, counts = time.time(), {"read": 0, "already_read": 0}
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
        done = [path, out / f"{seed}-{variant or 'world'}-shape.npy", out / f"{seed}-{variant or 'world'}-tokens.npy"]
        if all(p.exists() and np.load(p).shape[0] == len(ep["chunks"]) for p in done):
            counts["already_read"] += 1
            continue                                      # read already; a replayed episode is read again
        counts["read"] += 1
        S = reader.states([c["text"] for c in ep["chunks"]], layers)
        np.save(path, news(S, [c["slot"] for c in ep["chunks"]], unit))
        # shape on Step 0's readings: each chunk cut at 256 tokens, so only longer chunks are read again
        tok = reader.tokenizer
        ids = [tok(c["text"], add_special_tokens=False)["input_ids"] for c in ep["chunks"]]
        long_ = [i for i, x in enumerate(ids) if len(x) > SHAPE_CAP]
        Sc = S.copy()
        if long_:
            Sc[long_] = reader.states([tok.decode(ids[i][:SHAPE_CAP]) for i in long_], layers)
        np.save(out / f"{seed}-{variant or 'world'}-shape.npy",
                shape(Sc, [c["turn"] for c in ep["chunks"]], [float(x) ** 2 / S.shape[2] for x in D]))
        np.save(out / f"{seed}-{variant or 'world'}-tokens.npy", np.array([len(x) for x in ids]))
        pairs = joined(ep)
        if pairs:                                         # reported only: the event read inside its turn's reply
            J = np.full((len(ep["chunks"]), len(layers)), np.nan)
            T = reader.states([text for _, text in pairs], layers)
            for (i, _), state in zip(pairs, T):
                J[i] = np.min(np.linalg.norm(S[: i - 1] - state[None], axis=2), axis=0) / unit
            np.save(out / f"{seed}-{variant or 'world'}-joined.npy", J)
    record.append("read_end", "code", episodes=len(eps), **counts, seconds=round(time.time() - started, 1),
                  chunks=sum(len(ep["chunks"]) for ep in eps.values()))


BINS = (1, 8, 16, 32, 64, 257)              # chunk token lengths (the agent's choice, PLAN.md, before any run)
MIN_BIN = 20


def phase_comparison(eps: dict, shape_by: dict, news_by: dict, tokens_by: dict, phases: dict) -> dict:
    """Shape and news in the script's loop phases against its explore phases (owner, PR #32 sec. 4a; reported,
    deciding nothing): by source (action, reply) within token-length bins, the medians and the AUC of loop over
    explore, each as the mean over layers. `phases`: episode key -> turn -> phase."""
    cells = defaultdict(lambda: {"loop": [], "explore": []})
    for key, ep in eps.items():
        if key not in phases or key not in shape_by:
            continue
        for i, c in enumerate(ep["chunks"]):
            ph = phases[key].get(c["turn"])
            if ph is None or c.get("cause") == "spontaneous" or not np.isfinite(shape_by[key][i]).all():
                continue
            n = int(tokens_by[key][i])
            b = next((f"[{lo}, {hi})" for lo, hi in zip(BINS, BINS[1:]) if lo <= n < hi), None)
            if b is None:
                continue
            src = "action" if c["slot"] == "self" else "reply"
            nv = float(np.mean(news_by[key][i])) if np.isfinite(news_by[key][i]).all() else np.nan
            cells[(src, b)][ph].append((float(np.mean(shape_by[key][i])), nv))
    out = {}
    for (src, b), v in sorted(cells.items()):
        lo, ex = v["loop"], v["explore"]
        row = {"loop chunks": len(lo), "explore chunks": len(ex)}
        if len(lo) < MIN_BIN or len(ex) < MIN_BIN:
            row["note"] = f"too few (under {MIN_BIN} of a phase)"
        else:
            for j, name in ((0, "shape"), (1, "news")):
                L = [x[j] for x in lo if np.isfinite(x[j])]
                E = [x[j] for x in ex if np.isfinite(x[j])]
                if L and E:
                    row[name] = {"loop median": float(np.median(L)), "explore median": float(np.median(E)),
                                 "AUC loop over explore": auc(L, E)}
        out[f"{src}, tokens {b}"] = row
    return out


def phases_of(run: Path, eps: dict) -> dict:
    """Each scripted episode's phase by turn: from its keys where play recorded them, else from the derived labels
    `phases` wrote (kept only where the replay matched every action)."""
    derived = run / "derived" / "phases.json"
    saved = json.loads(derived.read_text()) if derived.exists() else {}
    out = {}
    for key, ep in eps.items():
        if any("phase" in k for k in ep["keys"].values()):
            out[key] = {t: k.get("phase") for t, k in ep["keys"].items()}
        elif str(key[0]) in saved and not key[1]:
            out[key] = {int(t): p for t, p in saved[str(key[0])].items()}
    return out


def cmd_phases(a) -> None:
    """Derive the script's phases for a record played before play recorded them (owner, 2026-11-05): replay the
    deterministic script from each seed in a scratch directory and keep the labels only if every action matches."""
    import tempfile

    from body.habitat import Script, play

    run = Path(a.run)
    eps = episodes(read(run / "record.jsonl"))
    out, report = {}, {"matched": 0, "mismatched": [], "skipped": 0}
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Record(Path(tmp) / "replay.jsonl")
        for (seed, variant), ep in sorted(eps.items()):
            if variant or ep["info"].get("model") != Script.name or any("phase" in k for k in ep["keys"].values()):
                report["skipped"] += 1
                continue
            play(Script(), scratch, seed, Path(tmp) / "games", max_turns=len(ep["keys"]),
                 generator=ep["info"].get("generator"))
            again = episodes(read(scratch.path))[(seed, None)]
            mine = [c["text"] for c in ep["chunks"] if c["slot"] == "self"]
            theirs = [c["text"] for c in again["chunks"] if c["slot"] == "self"]
            if mine == theirs:
                out[str(seed)] = {str(t): k["phase"] for t, k in again["keys"].items()}
                report["matched"] += 1
            else:
                report["mismatched"].append(seed)
    path = run / "derived" / "phases.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out))
    Record(run / "record.jsonl").append("derived", "code", what="script phases, by replay", path=str(path), **report)
    print(json.dumps(report))


def cmd_score(a) -> None:
    from body.habitat import worlds

    run = Path(a.run)
    eps = episodes(read(run / "record.jsonl"))
    news_by = {k: np.load(run / "news" / f"{k[0]}-{k[1] or 'world'}.npy") for k in eps}
    stale = [k for k in eps if news_by[k].shape[0] != len(eps[k]["chunks"])]
    if stale:
        raise SystemExit(f"news does not match the record for {stale}: run read again")
    joined_by = {k: np.load(p) for k in eps if (p := run / "news" / f"{k[0]}-{k[1] or 'world'}-joined.npy").exists()}
    shape_by = {k: np.load(p) for k in eps if (p := run / "news" / f"{k[0]}-{k[1] or 'world'}-shape.npy").exists()}
    tokens_by = {k: np.load(p) for k in eps if (p := run / "news" / f"{k[0]}-{k[1] or 'world'}-tokens.npy").exists()}
    result = score(eps, news_by, worlds(), joined_by, shape_by, tokens_by)
    if tokens_by:
        result["reported"]["chunks cut at 256 tokens"] = int(sum((t > SHAPE_CAP).sum() for t in tokens_by.values()))
    ph = phases_of(run, eps)
    if ph and shape_by:
        result["reported"]["shape and news, loop against explore"] = phase_comparison(eps, shape_by, news_by,
                                                                                      tokens_by, ph)
    Record(run / "record.jsonl").append("score", "code", **result)
    print(json.dumps(result, indent=2))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("play", "read", "score", "phases"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True)
        if name not in ("score", "phases"):
            s.add_argument("--model", required=name == "read", help="the player (play) or the perceiver (read)")
            s.add_argument("--device", default="auto")
            s.add_argument("--dtype", default="bfloat16")
        if name == "play":
            s.add_argument("--pool", choices=POOLS, required=True)
            s.add_argument("--limit", type=int, help="the dry run: only the first n seeds of the pool")
            s.add_argument("--no-cache", action="store_true", help="re-read the whole prompt every turn")
            s.add_argument("--nf4", action="store_true", help="a 4-bit model: rehearsals only, recorded on every episode")
            s.add_argument("--forms", action="store_true", help="Stage 2's player: told TextWorld's command forms")
            s.add_argument("--script", action="store_true", help="Stage 1's scripted player (no model, no GPU)")
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
    {"play": cmd_play, "read": cmd_read, "score": cmd_score, "phases": cmd_phases}[a.cmd](a)


if __name__ == "__main__":
    main()
