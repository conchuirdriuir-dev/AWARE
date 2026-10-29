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


def parse(reply: str) -> dict:
    """The reply's JSON object, also when a model wraps it in a code fence or adds words around it."""
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        start, end = reply.find("{"), reply.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in the reply")
        try:
            data = json.loads(reply[start:end + 1])
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e}") from None
    if not isinstance(data, dict):
        raise ValueError("the reply is JSON but not an object")
    return data


def keep_passage(conn, rec: Recorder, passage: dict, previous: str) -> dict:
    prompt = PROMPT.format(chapter=passage["chapter"], passage_id=passage["id"], entities=store.entities_block(conn),
                           loops=store.loops_block(conn), previous_tail=" ".join(previous.split()[-150:]) or "(none)",
                           passage=passage["text"])
    reply = rec.chat("keeper", prompt, ref=passage["id"], schema=SCHEMA, max_tokens=1500)
    try:
        data = parse(reply)
    except ValueError as e:
        rec.event("keeper_unparsed", ref=passage["id"], error=str(e), finish=getattr(rec.model, "last_finish", None),
                  length=len(reply), head=reply[:300], tail=reply[-300:])
        return {"written": {}, "dropped": [{"kind": "reply", "why": "not JSON"}]}
    result = store.apply(conn, data, passage)
    rec.event("keeper_applied", ref=passage["id"], **result)
    return result


def keep_chapter_line(conn, rec: Recorder, n: int, text: str) -> str:
    line = rec.chat("chapter_line", CHAPTER_LINE.format(n=n, text=text), ref=f"c{n:02d}", max_tokens=80).strip()
    conn.execute("INSERT OR REPLACE INTO chapter_line VALUES (?, ?)", (n, line))
    conn.commit()
    return line
