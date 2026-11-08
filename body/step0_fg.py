"""Rate's predictive test on Fault Garage: Step 0's question and bars, unchanged, in Stage 2's form (owner,
2026-11-08; DECISIONS.md, proposals/fault-garage-rate.md). A weaker Qwen3 plays Fault Garage there; Fault Garage's
`export_aware.py --run` writes the record, with the oracle's Step 0 labels in its key entries; the 8B reads here.

    python -m body.step0_fg screen --run runs/fg-rate-4b-screen         # labels only, no state read: the player screen
    python -m body.step0_fg k --run runs/fg-rate-4b                     # k from the calibration pool, for PLAN.md
    python -m body.step0_fg read --model Qwen/Qwen3-8B --run runs/fg-rate-4b
    python -m body.step0_fg score --run runs/fg-rate-4b --k K

What differs from TextWorld's form (`body/step0_habitat.py`), all fixed by the owner on 2026-11-08:

- **The reading is X_c** (`body/conditional.py`, unchanged: R = 8, lambda = 10, 20 earlier pairs, burn-in 10), in
  place of X. A turn is scored once X_c is defined there.
- **The labels are Fault Garage's**, written by its export from the oracle (`faultgarage/step0.py` there): new_state,
  room (the car's condition), ascent. Closures and k are AWARE's (`body/habitat.py`), on (command, condition) pairs.
- **Pairs are matched on position:** a pre-progress window ends within WITHIN turns of the pre-stall window it is
  paired with (`step0_habitat.match`).
- The pools are Fault Garage's registered seeds, level 2: calibration 3500-3599, Step 0 3600-3999; the screen plays
  their mirrors (+900000).

Windows, bars, draws, the text baseline and the checks are Step 0's (`step0_habitat.score`). Only `k`, `screen` and
`score` read keys; `read` sees the chunks alone.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from body.conditional import MIN_PAIRS, conditional, pairs
from body.habitat import choose_k, closures, mirrored
from body.machine import describe, record_machine
from body.step0_habitat import CAP, FIELDS, TEXT_VOCAB, events_at, score, turns_of
from goalspace.formula import Tfidf
from substrate.record import Record, read

POOLS = {"calibration": (3500, 3599), "step0": (3600, 3999)}     # Fault Garage's registry (its CLAUDE.md)
WITHIN = 3                                                       # pairs within +-3 turns (owner, 2026-11-08)
BURN = 10                                                        # conditional()'s default, as rate's calibration
SCREEN = {"closures": 150, "ascents": 75, "won": 0.10, "pool": 400}


# --- material ------------------------------------------------------------------------------------------------------

def in_pool(seed: int, pool: str) -> bool:
    lo, hi = POOLS[pool]
    return lo <= mirrored(seed) <= hi


def _entries(run: Path) -> list[dict]:
    return read(Path(run) / "record.jsonl")


def episodes_in(entries: list[dict], pool: str) -> dict:
    """The record's complete Fault Garage episodes in `pool` (registered seeds or their mirrors), by seed."""
    from body.stepm1 import episodes

    out = {}
    for (seed, _), ep in episodes(entries).items():
        if ep["info"].get("environment") == "fault_garage" and in_pool(seed, pool):
            out[seed] = ep
    return out


def outcomes(entries: list[dict]) -> dict:
    """seed -> whether the player's diagnosis was correct (the end entry's; a key, read by the screen only)."""
    return {e["seed"]: bool(e.get("correct")) for e in entries if e["kind"] == "end"}


def defined(slots: list[str], turns: list[int], burn_turns: int = BURN, min_pairs: int = MIN_PAIRS) -> list[bool]:
    """Per chunk, whether conditional() gives it an X_c: the rule it scores by, without the reading."""
    out = [False] * len(slots)
    for k, (t, _, _) in enumerate(p for p in pairs(slots, turns) if p[2]):
        out[t] = k >= min_pairs and turns[t] >= burn_turns
    return out


# --- labels (keys; scoring only) -------------------------------------------------------------------------------------

def labels(e: dict, k: int) -> dict:
    """Closures at k on (command, condition) pairs; ascents as the export marked them; no broad ascents."""
    turns = e["turns"]
    assert [t["turn"] for t in turns] == list(range(len(turns))), "a turn without its key"
    as_typed = [dict(t, action=t.get("command") or "") for t in turns]
    return {"closures": closures(as_typed, k), "ascents": [i for i, t in enumerate(turns) if t.get("ascent")],
            "broad": []}


def labels_only(eps: dict) -> dict:
    """Episodes as scoring sees them, with no code lengths: what the screen needs. A turn counts as scored where X_c
    would be defined."""
    out = {}
    for seed, ep in eps.items():
        slots, turns = [c["slot"] for c in ep["chunks"]], [c["turn"] for c in ep["chunks"]]
        ok = defined(slots, turns)
        out[seed] = {"seed": seed, "turns": turns_of(ep), "spaces": [{f: {} for f in FIELDS}],
                     "scored": [turns[i] for i, c in enumerate(ep["chunks"]) if c["slot"] == "self" and ok[i]]}
    return out


def k_of(eps: dict) -> int:
    return choose_k([[ep["keys"][t]["new_state"] for t in sorted(ep["keys"])] for ep in eps.values()])


# --- the screen (Stage 2's criteria, from the labels alone) ------------------------------------------------------------

def screen(entries: list[dict]) -> dict:
    """k from the calibration play; closures (paired) and ascents kept under Step 0's rules, projected to the pool's
    400 worlds; the share of episodes diagnosed correctly. Bars: 150 closures, 75 ascents, 10% correct."""
    cal, s0 = episodes_in(entries, "calibration"), episodes_in(entries, "step0")
    if not cal or not s0:
        raise SystemExit(f"the screen needs both pools played ({len(cal)} calibration, {len(s0)} Step 0 episodes)")
    k = k_of(cal)
    w = events_at(labels_only(s0), k, label_fn=labels, within=WITHIN)
    scale = SCREEN["pool"] / len(s0)
    done = outcomes(entries)
    won = [done.get(s, False) for s in list(cal) + list(s0)]
    out = {"k": k, "calibration episodes": len(cal), "Step 0 episodes": len(s0),
           "closures kept": len(w["closures"]), "ascents kept": len(w["ascents"]),
           "pre-stall windows": len(w["pre-stall"]), "no control": len(w["no control"]),
           "projected closures": len(w["closures"]) * scale, "projected ascents": len(w["ascents"]) * scale,
           "won": float(np.mean(won))}
    out["meets"] = bool(out["projected closures"] >= SCREEN["closures"] and out["projected ascents"] >= SCREEN["ascents"]
                        and out["won"] >= SCREEN["won"])
    return out


# --- the reading ------------------------------------------------------------------------------------------------------

def readings(X: np.ndarray, slots: list[str], turns: list[int], floor_var: float) -> dict:
    """X_c and S_c for one space of one episode: rate's calibration's call, unchanged."""
    return conditional(X, slots, turns, floor_var)


def cmd_read(a) -> None:
    from substrate.echo_floor import d_resumable, session_resumable
    from substrate.geometry import spread
    from substrate.measure_d import load_sets
    from substrate.states import Reader

    machine = describe()
    if machine["commit"] is None or machine["uncommitted_changes"]:
        raise SystemExit("read from a clean, committed AWARE checkout: the record must carry AWARE's commit")
    run = Path(a.run)
    record = Record(run / "record.jsonl")
    entries = _entries(run)
    eps = {**episodes_in(entries, "calibration"), **episodes_in(entries, "step0")}
    if not eps:
        raise SystemExit("no Fault Garage episode of rate's pools in this record")
    started = time.time()
    sets = load_sets(a.sets)
    reader = Reader(a.model, revision=a.revision, device=a.device, dtype=a.dtype, gpu_memory=a.gpu_memory,
                    sink="first")
    if reader.info["commit"] is None and a.revision is None:
        raise SystemExit("the model's revision is unknown: pass --revision")
    record_machine(record, "step0_fg read")
    n = reader.info["layers"]
    layers = list(range(math.ceil(2 * n / 3), n))
    D = d_resumable(reader, sets, layers, record, a.draft, True)
    tok = reader.tokenizer
    record.append("read", "code", step="step0_fg", **reader.info, read_layers=layers, D=[float(x) for x in D],
                  cap=CAP, draft=a.draft)
    for seed, ep in sorted(eps.items()):
        def compute(ep=ep, seed=seed):
            texts = [tok.decode(tok(c["text"], add_special_tokens=False)["input_ids"][:CAP]) for c in ep["chunks"]]
            slots, turns = [c["slot"] for c in ep["chunks"]], [c["turn"] for c in ep["chunks"]]
            S = reader.states(texts, layers)
            arrays = {}
            for j, layer in enumerate(layers):
                X = S[:, j].astype(float)
                for f, v in readings(X, slots, turns, float(D[j]) ** 2 / X.shape[1]).items():
                    arrays[f"L{layer}:{f}"] = v
            tf = Tfidf(texts[:TEXT_VOCAB])
            T = tf(texts)
            d_text = spread([tf(s["texts"]) for s in sets])
            for f, v in readings(T, slots, turns, d_text ** 2 / max(T.shape[1], 1)).items():
                arrays[f"text:{f}"] = v
            return arrays, {"seed": seed, "chunks": len(texts), "layers": layers, "attempt": ep["attempts"]}
        session_resumable(record, f"step0fg-{seed}-a{ep['attempts']}", compute)
    record.append("read_end", "code", step="step0_fg", episodes=len(eps), seconds=round(time.time() - started, 1),
                  chunks=sum(len(ep["chunks"]) for ep in eps.values()))


def episode(ep: dict, spaces: list[dict]) -> dict:
    """What scoring works on, from one episode's X_c and S_c per space: X is X_c, S is S_c; Step 0's other fields
    have no reading here."""
    rows = {c["turn"]: i for i, c in enumerate(ep["chunks"]) if c["slot"] == "self"}
    scored = [t for t in sorted(rows) if np.isfinite(spaces[0]["X_c"][rows[t]])]
    by_turn = []
    for sp in spaces:
        d = {f: {} for f in FIELDS}
        d["X"] = {t: float(sp["X_c"][rows[t]]) for t in scored}
        d["S"] = {t: float(sp["S_c"][rows[t]]) for t in scored}
        by_turn.append(d)
    return {"seed": ep["info"]["seed"], "scored": scored, "spaces": by_turn, "turns": turns_of(ep)}


def load(run: Path) -> tuple[dict, int]:
    """Step 0's pool with its checkpointed X_c, as scoring works on it."""
    eps, out, n_layers = episodes_in(_entries(run), "step0"), {}, 0
    for seed, ep in sorted(eps.items()):
        path = run / "checkpoints" / f"step0fg-{seed}-a{ep['attempts']}.npz"
        if not path.exists():
            raise SystemExit(f"episode {seed} has no readings: run read first")
        data = np.load(path)
        names = sorted({k.split(":")[0] for k in data.files if not k.startswith("text:")}, key=lambda s: int(s[1:]))
        spaces = [{f: data[f"{nm}:{f}"] for f in ("X_c", "S_c")} for nm in names + ["text"]]
        n_layers = len(names)
        out[seed] = episode(ep, spaces)
    return out, n_layers


# --- commands --------------------------------------------------------------------------------------------------------

def cmd_k(a) -> None:
    eps = episodes_in(_entries(Path(a.run)), "calibration")
    k = k_of(eps)
    Record(Path(a.run) / "record.jsonl").append("k", "code", step="step0_fg", k=k, episodes=len(eps))
    print(json.dumps({"k": k, "episodes": len(eps)}))


def cmd_screen(a) -> None:
    result = screen(_entries(Path(a.run)))
    Record(Path(a.run) / "record.jsonl").append("screen", "code", step="step0_fg", **result)
    print(json.dumps(result, indent=1))


def cmd_score(a) -> None:
    run = Path(a.run)
    eps, n_layers = load(run)
    result = score(eps, a.k, n_layers, label_fn=labels, within=WITHIN, variants=False)
    Record(run / "record.jsonl").append("score", "code", step="step0_fg", **result)
    print(json.dumps(result, indent=1, default=float))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("read", "k", "screen", "score"):
        s = sub.add_parser(name)
        s.add_argument("--run", required=True, help="an AWARE run folder holding Fault Garage's record.jsonl")
        if name == "read":
            s.add_argument("--model", required=True)
            s.add_argument("--revision", help="the model's revision, when the hub's commit is not recorded")
            s.add_argument("--device", default="auto")
            s.add_argument("--dtype", default="bfloat16")
            s.add_argument("--sets", default="material/paraphrases.jsonl")
            s.add_argument("--gpu-memory", type=float)
            s.add_argument("--draft", action="store_true")
        if name == "score":
            s.add_argument("--k", type=int, required=True, help="the k fixed in PLAN.md from the calibration pool")
    a = p.parse_args(argv)
    {"read": cmd_read, "k": cmd_k, "screen": cmd_screen, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
