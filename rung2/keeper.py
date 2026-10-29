"""The keeper: one call per passage (keeper_prompt.txt, constrained by keeper_schema.json), and one line per chapter."""
from __future__ import annotations

import json
from pathlib import Path

from . import store
from .record import Recorder

HERE = Path(__file__).parent
PROMPT = (HERE / "keeper_prompt.txt").read_text()
SCHEMA = json.loads((HERE / "keeper_schema.json").read_text())
CHAPTER_LINE = ("In one sentence of 30 words or fewer, say what happens in this chapter that matters to the story. "
                "Use the characters' names. Reply with the sentence only.\n\nCHAPTER {n}:\n{text}")


def keep_passage(conn, rec: Recorder, passage: dict, previous: str) -> dict:
    prompt = PROMPT.format(chapter=passage["chapter"], passage_id=passage["id"], entities=store.entities_block(conn),
                           loops=store.loops_block(conn), previous_tail=" ".join(previous.split()[-150:]) or "(none)",
                           passage=passage["text"])
    reply = rec.chat("keeper", prompt, ref=passage["id"], schema=SCHEMA, max_tokens=1000)
    try:
        data = json.loads(reply)
    except json.JSONDecodeError as e:
        rec.event("keeper_unparsed", ref=passage["id"], error=str(e))
        return {"written": {}, "dropped": [{"kind": "reply", "why": "not JSON"}]}
    result = store.apply(conn, data, passage)
    rec.event("keeper_applied", ref=passage["id"], **result)
    return result


def keep_chapter_line(conn, rec: Recorder, n: int, text: str) -> str:
    line = rec.chat("chapter_line", CHAPTER_LINE.format(n=n, text=text), ref=f"c{n:02d}", max_tokens=80).strip()
    conn.execute("INSERT OR REPLACE INTO chapter_line VALUES (?, ?)", (n, line))
    conn.commit()
    return line
