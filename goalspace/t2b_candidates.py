"""T2b's marking pass: build the candidate list the owner marks (goalspace/t2b_wrong_instructions.md).

    python -m goalspace.t2b_candidates --corpus .../dara-session-transcripts

Writes `study/t2b_marking.md` (for the owner to fill in) and `study/t2b_candidates.json` (references, for the code)
inside the corpus, never in this repository: the list holds the owner's text. No reading is computed or attached.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SAMPLE, MIN_CHARS, MAX_CHARS, PREFIX, CONTEXT = 80, 40, 1000, 60, 300


def build(root: str, seed: int = 0) -> list[dict]:
    from product.labelled_logs import load_corpus

    R = json.load(open(Path(root) / "study" / "restorations_reading_v0_1.json"))
    L = json.load(open(Path(root) / "study" / "labels_confirmed.json"))
    sessions = load_corpus(root)
    out = []
    # 1. every owner or joint departure quote (the reader's who, or the owner's confirmed who)
    for d in R["departures"]:
        lab = L.get(d["id"], {})
        who = lab.get("who") or []
        if d["who"] != "owner" and not ({"owner", "joint"} & set(who)):
            continue
        prior = None
        if lab.get("real") == ["yes"] and "ascent" in (lab.get("kind") or []) and who == ["owner"]:
            prior = "ascent"
        out.append({"source": "departure", "departure": d["id"], "ord": d["ord"], "time": d["time"],
                    "ref": d["ref"], "text": d["quote"], "prior": prior,
                    "confirmed": {k: lab.get(k) for k in ("real", "kind", "who")} if lab else None})
    quotes = [c["text"][:PREFIX].strip() for c in out]
    # 2. a random sample of other owner instruction turns
    pool, seen = [], set()
    for ord_, s in sorted(sessions.items()):
        for i, c in enumerate(s["chunks"]):
            t = c["text"]
            if c["role"] != "human" or not (MIN_CHARS <= len(t) <= MAX_CHARS):
                continue
            if t in seen:
                seen.add(t)
                continue
            seen.add(t)
            if any(q and q in t for q in quotes):
                continue
            prev = next((s["chunks"][j]["text"] for j in range(i - 1, -1, -1)
                         if s["chunks"][j]["role"] == "agent_text"), "")
            pool.append({"source": "sample", "ord": ord_, "time": c["at"], "ref": c["ref"], "text": t,
                         "before": prev[:CONTEXT], "prior": None})
    out += random.Random(seed).sample(pool, min(SAMPLE, len(pool)))
    out.sort(key=lambda c: c["time"])
    for n, c in enumerate(out, 1):
        c["id"] = f"M{n:03d}"
    return out


def write(root: str, cands: list[dict]) -> None:
    study = Path(root) / "study"
    lines = [f"# T2b marking pass: {len(cands)} candidates",
             "",
             "For each candidate, one mark: **reversed** (you later reversed, undid, or would not give it again), "
             "**kept** (you stood by it), **ascent** (the aim moving up a level), or **unsure**. A line on why is "
             "optional. Candidates already confirmed as ascents are pre-marked; change the mark if it is wrong. No "
             "reading has been computed; nothing here comes from the test.",
             "",
             "Add any instruction you remember regretting at the end, under *Additions*, with its session and time.",
             ""]
    for c in cands:
        src = f"departure {c['departure']}" if c["source"] == "departure" else "random sample"
        lines += [f"## {c['id']}: {c['ord']}, {c['time'][:16]} ({src})", ""]
        if c.get("before"):
            lines += [f"**Agent just before (first {CONTEXT} characters):** {c['before'].strip()}", ""]
        lines += [f"**Instruction:** {c['text'].strip()}", "",
                  f"**Mark:** {c['prior'] or ''}", "",
                  "**Why:** ", ""]
    lines += ["## Additions", "", "(session, time, the instruction or its first words, and why)", ""]
    (study / "t2b_marking.md").write_text("\n".join(lines), encoding="utf-8")
    refs = [{k: c.get(k) for k in ("id", "source", "departure", "ord", "time", "ref", "prior")} for c in cands]
    (study / "t2b_candidates.json").write_text(json.dumps(refs, indent=1), encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True)
    a = p.parse_args(argv)
    cands = build(a.corpus)
    write(a.corpus, cands)
    n_dep = sum(c["source"] == "departure" for c in cands)
    print(f"{len(cands)} candidates: {n_dep} departure quotes ({sum(c['prior'] == 'ascent' for c in cands)} pre-marked "
          f"ascent), {len(cands) - n_dep} sampled; written to {Path(a.corpus) / 'study'}")


if __name__ == "__main__":
    main()
