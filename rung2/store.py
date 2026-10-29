"""The store for arm D (schema.sql), and what code does with each keeper reply (SPEC.md, "The keeper", steps 1-5).
Every row carries the passage it came from; the store belongs to one run, whose log records the model and seed.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

SCHEMA = Path(__file__).with_name("schema.sql")


def open_store(path: Path) -> sqlite3.Connection:
    new = not Path(path).exists()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    if new:
        conn.executescript(SCHEMA.read_text())
    return conn


def add_passages(conn: sqlite3.Connection, passages: list[dict]) -> None:
    conn.executemany("INSERT INTO passage (id, chapter, seq, text, tokens) VALUES (:id, :chapter, :seq, :text, :tokens)",
                     passages)
    conn.commit()


def passages(conn: sqlite3.Connection, upto: int | None = None) -> list[sqlite3.Row]:
    if upto is None:
        return conn.execute("SELECT * FROM passage ORDER BY chapter, seq").fetchall()
    return conn.execute("SELECT * FROM passage WHERE chapter <= ? ORDER BY chapter, seq", (upto,)).fetchall()


# -- what the keeper is shown ---------------------------------------------------------------------------------------

def entities_block(conn: sqlite3.Connection) -> str:
    rows = conn.execute("SELECT id, name, kind, aliases FROM entity ORDER BY rowid").fetchall()
    return "\n".join(f"{r['id']} | {r['name']} | {r['kind']} | {', '.join(json.loads(r['aliases']))}"
                     for r in rows) or "(none yet)"


def loops_block(conn: sqlite3.Connection) -> str:
    rows = conn.execute("SELECT id, description FROM loop WHERE closed_chapter IS NULL ORDER BY id").fetchall()
    return "\n".join(f"{r['id']} | {r['description']}" for r in rows) or "(none yet)"


# -- applying a reply -------------------------------------------------------------------------------------------------

QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "—": "-", "–": "-"})


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(QUOTES)).strip().lower()


def quoted(quote: str, passage: str) -> bool:
    q = norm(quote).strip("\"' ")
    return bool(q) and q in norm(passage)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "entity"


def resolve(conn: sqlite3.Connection, ref: str | None, new: dict[str, str]) -> str | None:
    """A known id; else a new entity's exact name from this reply; else a known name or alias. Never created here."""
    if not ref:
        return None
    ref = ref.strip()
    if conn.execute("SELECT 1 FROM entity WHERE id = ?", (ref,)).fetchone():
        return ref
    low = ref.lower()
    if low in new:
        return new[low]
    for r in conn.execute("SELECT id, name, aliases FROM entity").fetchall():
        if low in [r["name"].lower()] + [a.lower() for a in json.loads(r["aliases"])]:
            return r["id"]
    return None


def apply(conn: sqlite3.Connection, reply: dict, passage: dict) -> dict:
    """Write one keeper reply into the store. Returns the counts written and every record dropped, with why."""
    ch, pid, text = passage["chapter"], passage["id"], passage["text"]
    written = {k: 0 for k in ("entities", "states", "knowledge", "events", "rules", "loops_opened",
                              "loops_touched", "loops_closed")}
    dropped = []

    def keep(kind: str, rec: dict) -> bool:
        if quoted(rec.get("quote", ""), text):
            return True
        dropped.append({"kind": kind, "why": "quote not in passage", "record": rec})
        return False

    def ref(kind: str, rec: dict, value: str | None, required: bool = True) -> str | None:
        got = resolve(conn, value, new)
        if got is None and (required or value):
            dropped.append({"kind": kind, "why": f"unresolved entity {value!r}", "record": rec})
        return got

    new: dict[str, str] = {}
    for e in reply.get("new_entities", []):
        if not keep("new_entities", e):
            continue
        existing = resolve(conn, e["name"], new)
        if existing:                                   # the model called a known entity new: use the known one
            new[e["name"].strip().lower()] = existing
            continue
        eid, i = slug(e["name"]), 2
        while conn.execute("SELECT 1 FROM entity WHERE id = ?", (eid,)).fetchone():
            eid, i = f"{slug(e['name'])}-{i}", i + 1
        conn.execute("INSERT INTO entity (id, name, kind, aliases, first_passage) VALUES (?, ?, ?, ?, ?)",
                     (eid, e["name"].strip(), e["kind"], json.dumps(e.get("aliases", [])), pid))
        new[e["name"].strip().lower()] = eid
        written["entities"] += 1

    for s in reply.get("states", []):
        if not keep("states", s) or (eid := ref("states", s, s["entity"])) is None:
            continue
        attr = s["attribute"].strip().lower()
        cur = conn.execute("SELECT id, value FROM state WHERE entity_id = ? AND attribute = ? ORDER BY id DESC LIMIT 1",
                           (eid, attr)).fetchone()
        if cur and norm(cur["value"]) == norm(s["value"]):
            continue
        conn.execute("INSERT INTO state (entity_id, attribute, value, from_chapter, supersedes, passage_id) "
                     "VALUES (?, ?, ?, ?, ?, ?)", (eid, attr, s["value"].strip(), ch, cur["id"] if cur else None, pid))
        written["states"] += 1

    for k in reply.get("knowledge", []):
        if not keep("knowledge", k) or (cid := ref("knowledge", k, k["character"])) is None:
            continue
        about = ref("knowledge", k, k.get("about"), required=False)
        truth = None if k["true_in_story"] is None else int(k["true_in_story"])
        cur = conn.execute("INSERT INTO knowledge (character_id, proposition, about_id, stance, true_in_story, "
                           "from_chapter, passage_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (cid, k["proposition"].strip(), about, k["stance"], truth, ch, pid))
        if about is not None:
            conn.execute("UPDATE knowledge SET corrected_by = ? WHERE character_id = ? AND about_id = ? "
                         "AND corrected_by IS NULL AND id != ? AND (stance != ? OR true_in_story IS NOT ?)",
                         (cur.lastrowid, cid, about, cur.lastrowid, k["stance"], truth))
        written["knowledge"] += 1

    for ev in reply.get("events", []):
        if not keep("events", ev):
            continue
        place = ref("events", ev, ev.get("place"), required=False)
        cur = conn.execute("INSERT INTO event (chapter, summary, place_id, happened, passage_id) VALUES (?, ?, ?, ?, ?)",
                           (ch, ev["summary"].strip(), place, int(ev["happened"]), pid))
        for p in ev.get("participants", []):
            if (eid := ref("events", ev, p)) is not None:
                conn.execute("INSERT OR IGNORE INTO event_entity VALUES (?, ?)", (cur.lastrowid, eid))
        written["events"] += 1

    for r in reply.get("rules", []):
        if keep("rules", r):
            conn.execute("INSERT INTO rule (statement, chapter, passage_id) VALUES (?, ?, ?)",
                         (r["statement"].strip(), ch, pid))
            written["rules"] += 1

    for lp in reply.get("loops_opened", [])[:2]:
        if not keep("loops_opened", lp):
            continue
        cur = conn.execute("INSERT INTO loop (description, opened_chapter, opened_passage) VALUES (?, ?, ?)",
                           (lp["description"].strip(), ch, pid))
        conn.execute("INSERT INTO loop_touch VALUES (?, ?, ?)", (cur.lastrowid, ch, pid))
        for e in lp.get("entities", []):
            if (eid := ref("loops_opened", lp, e)) is not None:
                conn.execute("INSERT OR IGNORE INTO loop_entity VALUES (?, ?)", (cur.lastrowid, eid))
        written["loops_opened"] += 1

    def open_loop(kind: str, lid, rec) -> bool:
        if conn.execute("SELECT 1 FROM loop WHERE id = ? AND closed_chapter IS NULL", (lid,)).fetchone():
            return True
        dropped.append({"kind": kind, "why": f"no open loop {lid!r}", "record": rec})
        return False

    for lid in reply.get("loops_touched", []):
        if open_loop("loops_touched", lid, lid):
            conn.execute("INSERT OR IGNORE INTO loop_touch VALUES (?, ?, ?)", (lid, ch, pid))
            written["loops_touched"] += 1

    for c in reply.get("loops_closed", []):
        if keep("loops_closed", c) and open_loop("loops_closed", c["loop_id"], c):
            conn.execute("UPDATE loop SET closed_chapter = ?, closed_passage = ? WHERE id = ?", (ch, pid, c["loop_id"]))
            conn.execute("INSERT OR IGNORE INTO loop_touch VALUES (?, ?, ?)", (c["loop_id"], ch, pid))
            written["loops_closed"] += 1

    conn.commit()
    return {"written": written, "dropped": dropped}
