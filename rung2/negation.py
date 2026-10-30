"""The negation test (SPEC.md, "The negation test"): does the keeper keep "not"?

DARA's spec keeps the ERG for negation scope (§12.3); AWARE lets the model read. This scores the keeper's store against
the probes of DARA's stage-1 pilot (`key_probes.py`, kept outside the repository with the chapter): for each probe, the
passage holding its sentence, the store's rows citing that passage, and the row that holds most of the probe's lemmas.
A probe marked `neg` asks one more thing of that row: does it carry the negation? A fact stored without it is stored
inverted, the costly error; a fact not stored at all is coverage, counted apart.

The probes are a key: only this scorer reads them, and nothing here reaches the model.
"""
from __future__ import annotations

import ast
import json
import re
import sqlite3
from pathlib import Path

from .keeper import parse
from .store import norm

NEGATION = {"not", "no", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor", "without", "cannot"}
UN = {"untold", "unseen", "unknown", "unsaid", "unheard", "unmade", "unkept"}
IRREGULAR = {"ran": "run", "left": "leave", "fell": "fall", "born": "bear", "told": "tell", "knew": "know",
             "came": "come", "meant": "mean", "slept": "sleep", "drew": "draw", "drawn": "draw", "hung": "hang",
             "stood": "stand", "kept": "keep", "gave": "give", "took": "take", "passed": "pass", "died": "die"}
WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*|n't")


def load_probes(path: Path) -> dict[str, list[dict]]:
    """PROBES from a key_probes.py, read as data: literals and dict(...) calls only, nothing executed."""
    def value(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
            return {k.arg: value(k.value) for k in node.keywords}
        if isinstance(node, ast.Dict):
            return {value(k): value(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, (ast.List, ast.Tuple)):
            return [value(x) for x in node.elts]
        return ast.literal_eval(node)

    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "PROBES" for t in node.targets):
            return value(node.value)
    raise ValueError(f"no PROBES in {path}")


def words(text: str) -> set[str]:
    ws = set(WORD.findall(norm(text)))
    return ws | {IRREGULAR[w] for w in ws if w in IRREGULAR}


def has(ws: set[str], alt: str) -> bool:
    """A lemma alternative among a row's words: the word itself, or a word it starts (tell: tells, telling)."""
    alt = alt.lower()
    if " " in alt:
        return all(has(ws, a) for a in alt.split())
    stems = ws | {w[2:] for w in ws if w.startswith("un")}           # unfastened: fasten, and its negation
    return alt in stems or (len(alt) >= 3 and any(w.startswith(alt) and len(w) - len(alt) <= 4 for w in stems))


def negated(ws: set[str]) -> bool:
    """A negation word, or an un- participle (unfastened, unlocked, untold)."""
    return bool(ws & NEGATION) or "n't" in ws or any(
        w.startswith("un") and (w.endswith(("ed", "en")) or w in UN) and w not in {"united", "under"} for w in ws)


def row_negated(row: dict) -> bool:
    """The row's own claim carries a negation: its value, proposition, summary or statement (never the entity names
    beside it), or an event stored as not having happened."""
    return negated(words(row["field"])) or row.get("happened") is False


def rows(conn: sqlite3.Connection, pid: str) -> list[dict]:
    """Every row citing the passage, as the text the store holds: entity names, values, propositions, summaries."""
    name = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM entity")}
    out = []
    for r in conn.execute("SELECT * FROM state WHERE passage_id = ?", (pid,)):
        out.append({"table": "state", "field": r["value"],
                    "text": f"{name.get(r['entity_id'], r['entity_id'])} {r['attribute']}: {r['value']}"})
    for r in conn.execute("SELECT * FROM knowledge WHERE passage_id = ?", (pid,)):
        out.append({"table": "knowledge", "field": r["proposition"],
                    "text": f"{name.get(r['character_id'], r['character_id'])} {r['stance']} that {r['proposition']}"})
    for r in conn.execute("SELECT * FROM event WHERE passage_id = ?", (pid,)):
        who = [name.get(x["entity_id"], x["entity_id"])
               for x in conn.execute("SELECT entity_id FROM event_entity WHERE event_id = ?", (r["id"],))]
        out.append({"table": "event", "field": r["summary"], "happened": bool(r["happened"]), "text": f"{r['summary']}" + (f" [{', '.join(who)}]" if who else "")
                    + ("" if r["happened"] else " (did not happen)")})
    for r in conn.execute("SELECT * FROM rule WHERE passage_id = ?", (pid,)):
        out.append({"table": "rule", "field": r["statement"], "text": r["statement"]})
    return out


FIELD = {"states": ("state", "value"), "knowledge": ("knowledge", "proposition"), "events": ("event", "summary"),
         "rules": ("rule", "statement")}


def quotes(log: Path) -> dict[tuple, str]:
    """The quote behind each stored row, from the keeper's replies in the record: (passage, table, field) -> quote."""
    out = {}
    if not Path(log).exists():
        return out
    for line in Path(log).read_text(encoding="utf-8").splitlines():
        e = json.loads(line)
        if e.get("kind") != "call" or e.get("role") != "keeper":
            continue
        try:
            reply = parse(e["reply"])
        except ValueError:
            continue
        for kind, (table, field) in FIELD.items():
            for rec in reply.get(kind, []):
                if isinstance(rec, dict) and rec.get(field) and rec.get("quote"):
                    out[(e["ref"], table, norm(rec[field]))] = rec["quote"]
    return out


def polarity(res: dict, q: dict[tuple, str]) -> None:
    """Would a polarity check in store.apply have caught this row: does its quote's negation differ from the row's?"""
    quote = q.get((res["passage"], res["row"]["table"], norm(res["row"]["field"])))
    res["quote"] = quote
    res["caught"] = quote is not None and negated(words(quote)) != res["neg"]


def score_probe(conn: sqlite3.Connection, probe: dict, q: dict[tuple, str] | None = None) -> dict:
    find = norm(probe["find"]).strip("\"' ")
    pids = [r["id"] for r in conn.execute("SELECT id, text FROM passage ORDER BY chapter, seq") if find in norm(r["text"])]
    if not pids:
        return {"status": "sentence not found", "missing": probe["lemmas"], "row": None}
    best = None
    for pid in pids:
        for row in rows(conn, pid):
            ws = words(row["text"])
            missing = [g for g in probe["lemmas"] if not any(has(ws, a) for a in g.split("|"))]
            entity_ok = not probe.get("entity") or probe["entity"].lower() in ws
            res = {"passage": pid, "row": row, "missing": missing, "entity_ok": entity_ok, "neg": row_negated(row)}
            key = (len(missing), not entity_ok, not res["neg"] if probe.get("neg") else 0)
            if best is None or key < best[0]:
                best = (key, res)
    if best is None:
        return {"status": "no rows", "missing": probe["lemmas"], "row": None, "passage": pids[0]}
    res = best[1]
    if res["missing"] or not res["entity_ok"]:
        res["status"] = "partial"
    elif probe.get("neg") and not res["neg"]:
        res["status"] = "inverted"          # every lemma in one row, without its negation
    elif not probe.get("neg") and res["neg"]:
        res["status"] = "negation added"    # a positive fact stored with a negation: read it, it may be the text's own
    else:
        res["status"] = "ok"
    if res["status"] in ("inverted", "negation added"):
        polarity(res, q or {})
    return res


def read_erg(report: Path) -> dict[tuple[str, str], dict]:
    """The pilot's own report (stage1_report.txt, Run A): its fixed-width probe rows, by item and label."""
    out = {}
    for line in Path(report).read_text(encoding="utf-8").splitlines():
        if len(line) < 50 or line.startswith(("item", " ")):
            continue
        status = line[44:64].strip()
        if status in ("ok", "partial", "unparsed", "sentence not found"):
            out[(line[:10].strip(), line[10:44].strip())] = {"status": status, "problems": line[80:].strip()}
    return out


def erg_keeps_negation(r: dict | None) -> bool | None:
    """The pilot's check is the sentence's: negation anywhere in it. None when the ERG produced nothing for it."""
    if r is None or r["status"] in ("unparsed", "sentence not found"):
        return None
    return "NEGATION LOST" not in r["problems"]


def reading(n: dict, erg: bool) -> str:
    """Which row of the pass criteria the counts point to (SPEC.md, "The negation test"), before the owner reads the
    rows: a reading, not a verdict."""
    if not erg:
        return "no reading without the ERG's report (--erg)"
    if n["comparable"] < 5:
        return f"no decision: {n['comparable']} negated facts comparable, fewer than 5; add negated facts to the key"
    if n["erg_lost"] > n["comparable"] / 2:
        return "the ERG loses most negated facts too: the chapter does not test negation; a harder key before deciding"
    if n["not stored"] > n["comparable"] / 2:
        return "most negated facts not stored by the keeper: coverage, not negation; the keeper's prompt first"
    if not n["inverted"] and not n["added"]:
        return "no inversions, no negation added: the ERG stays out"
    if n["uncaught_erg_kept"]:
        return (f"{n['uncaught_erg_kept']} inversion(s) the polarity check would miss, where the ERG keeps the "
                "negation: the ERG returns as the reader of negation scope")
    if n["uncaught"]:
        return "an inversion the polarity check would miss and the ERG loses too: read it before deciding"
    return "every inversion caught by a polarity check on the quotes: the check goes into store.apply, then rerun"


def report(conn: sqlite3.Connection, probes: dict[str, list[dict]], erg: dict | None = None,
           q: dict[tuple, str] | None = None) -> str:
    lines = ["The negation test: the keeper's store against the pilot's probes", "",
             f"{'item':<10}{'probe':<34}{'neg':<5}{'keeper':<20}" + (f"{'ERG':<20}" if erg is not None else "")
             + "the keeper's best row / what it lacks"]
    tally = {}
    n = {"kept": 0, "inverted": 0, "not stored": 0, "added": 0, "caught": 0, "uncaught": 0, "uncaught_erg_kept": 0,
         "comparable": 0, "erg_lost": 0, "erg_kept_keeper_inverted": 0, "erg_lost_keeper_kept": 0}
    for item, ps in probes.items():
        for p in ps:
            r = score_probe(conn, p, q)
            tally[r["status"]] = tally.get(r["status"], 0) + 1
            e = (erg or {}).get((item, p["label"][:33].strip()))
            ek = erg_keeps_negation(e) if p.get("neg") else None
            if r["status"] in ("inverted", "negation added"):
                n["caught" if r["caught"] else "uncaught"] += 1
                n["uncaught_erg_kept"] += r["status"] == "inverted" and not r["caught"] and bool(ek)
            n["added"] += r["status"] == "negation added"
            if p.get("neg"):
                n[{"ok": "kept", "inverted": "inverted"}.get(r["status"], "not stored")] += 1
                if ek is not None and r["status"] != "sentence not found":
                    n["comparable"] += 1
                    n["erg_lost"] += not ek
                n["erg_kept_keeper_inverted"] += bool(ek) and r["status"] == "inverted"
                n["erg_lost_keeper_kept"] += ek is False and r["status"] == "ok"
            what = []
            if r.get("row"):
                what.append(f"[{r['row']['table']}] {r['row']['text']}")
            if "caught" in r:
                what.append(f"quote: {r['quote']!r}, " + ("a polarity check catches it" if r["caught"]
                                                          else "a polarity check misses it"))
            if r.get("missing"):
                what.append("lacks " + ", ".join(r["missing"]))
            if r.get("entity_ok") is False:
                what.append(f"entity {p['entity']} not named")
            erg_col = "" if erg is None else f"{(e or {}).get('status', '-') + ('' if ek is not False else ', neg lost'):<20}"
            lines.append(f"{item:<10}{p['label'][:33]:<34}{'yes' if p.get('neg') else '':<5}{r['status']:<20}{erg_col}"
                         + " | ".join(what))
    total = sum(tally.values())
    lines += ["", "keeper totals: " + "   ".join(f"{k}: {v}/{total}" for k, v in sorted(tally.items())),
              f"negated facts ({n['kept'] + n['inverted'] + n['not stored']}): kept {n['kept']}, "
              f"stored inverted {n['inverted']}, not stored {n['not stored']}",
              f"negation added to a positive fact: {n['added']}",
              f"a polarity check on the quotes would catch {n['caught']} of {n['caught'] + n['uncaught']} of these"]
    if erg is not None:
        lines.append(f"against the ERG ({n['comparable']} negated facts comparable): the ERG kept the negation and the "
                     f"keeper inverted it on {n['erg_kept_keeper_inverted']}; the keeper kept it and the ERG lost it on "
                     f"{n['erg_lost_keeper_kept']}")
    lines += ["", "points to: " + reading(n, erg is not None), "",
              "Matching is by words, so read every 'inverted', 'negation added' and 'partial' row above before counting",
              "it: the owner settles what a row says, and the reading follows (SPEC.md, \"The negation test\")."]
    return "\n".join(lines) + "\n"
