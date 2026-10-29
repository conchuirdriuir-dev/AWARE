"""Scoring against the key, the owner's settled verdicts, and a first cut at diagnosing arm D's misses.

The key (JSONL, kept outside the repository) has one line per question:
  {"id": "q01", "accept": ["grey", "gray"]}               types 1, 2, 3, 5: any accepted phrase in the answer
  {"id": "q31", "items": [["the locket"], ["the map", "flood map"]]}   type 4: each item, any of its phrasings
  {"id": "q51"}                                            type 6: counted by the owner
A short answer with no accepted phrase is "check", not wrong: the owner settles it in the settled file,
  {"id": "q01", "arm": "B", "verdict": "correct" | "wrong"}  (type 4: "found": k, "wrong_items": m;
  type 6: "contradictions": k).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .store import norm


def read_jsonl(path: Path) -> list[dict]:
    if not path or not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def auto(answer: str, key: dict, qtype: int) -> dict:
    a = norm(answer)
    if qtype == 6:
        return {"verdict": "owner"}
    if qtype == 4:
        found = sum(any(norm(p) in a for p in item) for item in key.get("items", []))
        return {"verdict": "check", "found": found, "of": len(key.get("items", []))}
    if any(norm(p) in a for p in key.get("accept", [])):
        return {"verdict": "correct"}
    return {"verdict": "check"}


def in_store(conn, phrases: list[str]) -> bool:
    texts = []
    for table, cols in (("state", "attribute || ' ' || value"), ("knowledge", "proposition"), ("event", "summary"),
                        ("rule", "statement"), ("loop", "description"), ("entity", "name || ' ' || aliases")):
        texts += [norm(r[0]) for r in conn.execute(f"SELECT {cols} FROM {table}")]
    return any(norm(p) in t for p in phrases for t in texts)


def score(answers: list[dict], key: list[dict], settled: list[dict], views: dict, conn=None) -> list[dict]:
    keys = {k["id"]: k for k in key}
    owner = {(s["id"], s["arm"]): s for s in settled}
    rows = []
    for ans in answers:
        k = keys.get(ans["id"], {})
        row = {**{f: ans[f] for f in ("id", "arm", "type", "after", "run")}, **auto(ans["answer"], k, ans["type"])}
        row.update({f: v for f, v in owner.get((ans["id"], ans["arm"]), {}).items() if f not in ("id", "arm")})
        if ans["arm"] == "D" and row["verdict"] != "correct" and ans["type"] in (1, 2, 3):   # type 5's "no" matches anything
            phrases = k.get("accept", [])
            row["in_store"] = in_store(conn, phrases) if conn is not None else None
            row["in_view"] = any(norm(p) in norm(views.get((ans["id"], "D"), "")) for p in phrases)
        rows.append(row)
    return rows


def report(rows: list[dict]) -> str:
    table = defaultdict(lambda: {"correct": 0, "wrong": 0, "check": 0, "n": 0})
    for r in rows:
        if r["type"] in (4, 6):
            continue
        cell = table[(r["arm"], r["type"])]
        cell["n"] += 1
        cell[r["verdict"] if r["verdict"] in ("correct", "wrong") else "check"] += 1
    arms = sorted({r["arm"] for r in rows})
    out = ["# Rung 2 report", "", "## Short answers: correct / asked (unsettled in brackets)", "",
           "| Type | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for t in (1, 2, 3, 5):
        cells = []
        for a in arms:
            c = table.get((a, t))
            cells.append(f"{c['correct']}/{c['n']}" + (f" ({c['check']})" if c["check"] else "") if c else "-")
        out.append(f"| {t} | " + " | ".join(cells) + " |")
    loops = [r for r in rows if r["type"] == 4]
    if loops:
        out += ["", "## Open loops (type 4): items found / items in the key, summed", "",
                "| Arm | Found | Wrongly listed |", "|---|---|---|"]
        for a in arms:
            rs = [r for r in loops if r["arm"] == a]
            if rs:
                wrong = sum(r.get("wrong_items", 0) for r in rs)
                out.append(f"| {a} | {sum(r.get('found', 0) for r in rs)}/{sum(r.get('of', 0) for r in rs)} | "
                           f"{wrong if all('wrong_items' in r for r in rs) else 'owner to count'} |")
    cont = [r for r in rows if r["type"] == 6]
    if cont:
        out += ["", "## Continuations (type 6): contradictions counted by the owner", "", "| Arm | Contradictions | Counted |",
                "|---|---|---|"]
        for a in arms:
            rs = [r for r in cont if r["arm"] == a]
            if rs:
                done = [r for r in rs if "contradictions" in r]
                out.append(f"| {a} | {sum(r['contradictions'] for r in done)} | {len(done)}/{len(rs)} |")
    misses = [r for r in rows if r["arm"] == "D" and "in_store" in r]
    if misses:
        out += ["", "## Arm D's misses: where the answer was lost (a first cut, by phrase match)", "",
                "| Question | In the store | In the view | Reading |", "|---|---|---|---|"]
        for r in misses:
            reading = ("keeper" if not r["in_store"] else "builder" if not r["in_view"] else "model")
            out.append(f"| {r['id']} | {r['in_store']} | {r['in_view']} | {reading} |")
    return "\n".join(out) + "\n"
