"""Arm D's view as of chapter N, filled in a fixed order, each part stopping at its budget (SPEC.md, "The store")."""
from __future__ import annotations

import json
import re
from collections import Counter

from . import store
from .text import Count, fit, fit_end

BUDGET = {"map": 400, "rules": 150, "entities": 2500, "loops": 600, "cited": 1350, "recent": 900}   # +100 for headings


def named(conn, question: str) -> list[str]:
    """Entities whose name or an alias appears in the question, as whole words, longest names first."""
    q = question.lower()
    hits = []
    for r in conn.execute("SELECT id, name, aliases FROM entity").fetchall():
        names = [r["name"]] + json.loads(r["aliases"])
        best = max((n for n in names if n and re.search(r"\b" + re.escape(n.lower()) + r"\b", q)), key=len, default="")
        if best:
            hits.append((q.find(best.lower()), r["id"]))
    return [eid for _, eid in sorted(hits)]


def entity_lines(conn, eid: str, n: int, cited: Counter) -> list[str]:
    e = conn.execute("SELECT name, kind FROM entity WHERE id = ?", (eid,)).fetchone()
    lines = [f"{e['name']} ({e['kind']}):"]
    for s in conn.execute("SELECT * FROM state s WHERE entity_id = ? AND from_chapter <= ? AND id = "
                          "(SELECT MAX(id) FROM state WHERE entity_id = s.entity_id AND attribute = s.attribute "
                          "AND from_chapter <= ?) ORDER BY attribute", (eid, n, n)):
        lines.append(f"- {s['attribute']}: {s['value']} (from ch. {s['from_chapter']})")
        cited[s["passage_id"]] += 1
    for k in conn.execute("SELECT k.*, c.name AS who, (SELECT from_chapter FROM knowledge WHERE id = k.corrected_by) "
                          "AS corrected FROM knowledge k JOIN entity c ON c.id = k.character_id "
                          "WHERE (k.character_id = ? OR k.about_id = ?) AND k.from_chapter <= ? ORDER BY k.from_chapter",
                          (eid, eid, n)):
        note = f"; revised in ch. {k['corrected']}" if k["corrected"] and k["corrected"] <= n else ""
        false = "; false" if k["true_in_story"] == 0 else ""
        lines.append(f"- ch. {k['from_chapter']}: {k['who']} {k['stance']} that {k['proposition']}{false}{note}")
        cited[k["passage_id"]] += 1
    for ev in conn.execute("SELECT e.* FROM event e JOIN event_entity x ON x.event_id = e.id "
                           "WHERE x.entity_id = ? AND e.chapter <= ? ORDER BY e.chapter, e.id", (eid, n)):
        planned = "" if ev["happened"] else " (planned or cancelled; did not happen)"
        lines.append(f"- ch. {ev['chapter']}: {ev['summary']}{planned}")
        cited[ev["passage_id"]] += 1
    return lines


def open_loops(conn, n: int) -> list:
    """Loops open as of chapter N, least recently touched first."""
    return conn.execute("SELECT l.*, (SELECT MAX(chapter) FROM loop_touch WHERE loop_id = l.id AND chapter <= ?) AS touched "
                        "FROM loop l WHERE l.opened_chapter <= ? AND (l.closed_chapter IS NULL OR l.closed_chapter > ?) "
                        "ORDER BY touched, l.id", (n, n, n)).fetchall()


def build_view(conn, question: str, n: int, count: Count) -> str:
    cited: Counter = Counter()
    parts = []

    lines = [f"Ch. {r['chapter']}: {r['line']}"
             for r in conn.execute("SELECT * FROM chapter_line WHERE chapter <= ? ORDER BY chapter", (n,))]
    parts.append(("THE STORY SO FAR, CHAPTER BY CHAPTER", fit(lines, BUDGET["map"], count)))

    rules = conn.execute("SELECT * FROM rule WHERE chapter <= ? ORDER BY id", (n,)).fetchall()
    parts.append(("RULES OF THE WORLD", fit([f"- {r['statement']} (ch. {r['chapter']})" for r in rules],
                                            BUDGET["rules"], count)))

    lines = []
    for eid in named(conn, question):
        lines.extend(entity_lines(conn, eid, n, cited))
    kept = fit(lines, BUDGET["entities"], count)
    parts.append(("WHAT THE RECORD HOLDS ABOUT THOSE NAMED", kept))

    loops = open_loops(conn, n)
    parts.append(("OPEN THREADS (least recently touched first)",
                  fit([f"- opened ch. {r['opened_chapter']}, last touched ch. {r['touched']}: {r['description']}"
                       for r in loops], BUDGET["loops"], count)))
    for r in loops:
        cited[r["opened_passage"]] += 1

    recent = fit_end([r["text"] for r in store.passages(conn, n)], BUDGET["recent"], count)
    recent_ids = {r["id"] for r in store.passages(conn, n) if r["text"] in recent}
    by_id = {r["id"]: r for r in store.passages(conn, n)}
    cited_rows = [by_id[p] for p, _ in cited.most_common() if p in by_id and p not in recent_ids]
    texts = fit([f"[ch. {r['chapter']}] {r['text']}" for r in cited_rows], BUDGET["cited"], count)
    parts.append(("PASSAGES THE RECORD CITES", texts))
    parts.append((f"THE END OF CHAPTER {n}", recent))

    blocks = ("PASSAGES THE RECORD CITES", f"THE END OF CHAPTER {n}")
    return "\n\n".join(f"== {title} ==\n" + ("\n\n" if title in blocks else "\n").join(body)
                       for title, body in parts if body)
