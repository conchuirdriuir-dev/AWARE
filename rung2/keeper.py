"""The keeper: one call per passage (keeper_prompt.txt, constrained by keeper_schema.json); at each chapter's end, one
line (CHAPTER_LINE) and one loop pass (loops_prompt.txt, constrained by loops_schema.json) over the whole chapter."""
from __future__ import annotations

import json
from pathlib import Path

from . import store
from .record import Recorder

HERE = Path(__file__).parent
PROMPT = (HERE / "keeper_prompt.txt").read_text()
SCHEMA = json.loads((HERE / "keeper_schema.json").read_text())
LOOPS_PROMPT = (HERE / "loops_prompt.txt").read_text()
LOOPS_SCHEMA = json.loads((HERE / "loops_schema.json").read_text())
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


EMPTY_MIN_TOKENS = 150      # a passage this long with nothing in it is suspect (the Told Hives dry run: one was)
RETRY = (0.7, 1)            # temperature and seed of the one retry: at temperature 0 a new seed changes nothing


def empty_reply(data: dict) -> bool:
    return not any(data.get(k) for k in ("new_entities", "states", "knowledge", "events", "rules"))


def keep_passage(conn, rec: Recorder, passage: dict, previous: str) -> dict:
    prompt = PROMPT.format(chapter=passage["chapter"], passage_id=passage["id"], entities=store.entities_block(conn),
                           previous_tail=" ".join(previous.split()[-150:]) or "(none)",
                           passage=passage["text"])
    reply = rec.chat("keeper", prompt, ref=passage["id"], schema=SCHEMA, max_tokens=1500)
    try:
        data = parse(reply)
    except ValueError as e:
        rec.event("keeper_unparsed", ref=passage["id"], error=str(e), finish=getattr(rec.model, "last_finish", None),
                  length=len(reply), head=reply[:300], tail=reply[-300:])
        return {"written": {}, "dropped": [{"kind": "reply", "why": "not JSON"}]}
    if empty_reply(data) and passage["tokens"] >= EMPTY_MIN_TOKENS:
        # Both replies stay in the record; the store takes the retry's only if it holds something.
        temperature, seed = RETRY
        again = rec.chat("keeper", prompt, ref=passage["id"], schema=SCHEMA, max_tokens=1500,
                         temperature=temperature, seed=seed)
        try:
            retried = parse(again)
        except ValueError as e:
            retried, why = {}, str(e)
        else:
            why = "empty again" if empty_reply(retried) else None
        rec.event("keeper_empty", ref=passage["id"], said_by="code", tokens=passage["tokens"],
                  retry={"temperature": temperature, "seed": seed}, used_retry=why is None, why_not=why)
        if why is None:
            data = retried
    result = store.apply(conn, data, passage)
    rec.event("keeper_applied", ref=passage["id"], **result)
    return result


def keep_chapter_line(conn, rec: Recorder, n: int, text: str) -> str:
    line = rec.chat("chapter_line", CHAPTER_LINE.format(n=n, text=text), ref=f"c{n:02d}", max_tokens=80).strip()
    conn.execute("INSERT OR REPLACE INTO chapter_line VALUES (?, ?)", (n, line))
    conn.commit()
    return line


def keep_chapter_loops(conn, rec: Recorder, n: int, text: str, passages: list[dict]) -> dict:
    """The threads chapter n opens, advances and resolves, read from the whole chapter."""
    prompt = LOOPS_PROMPT.format(chapter=n, loops=store.loops_block(conn), text=text)
    reply = rec.chat("loops", prompt, ref=f"c{n:02d}", schema=LOOPS_SCHEMA, max_tokens=800)
    try:
        data = parse(reply)
    except ValueError as e:
        rec.event("loops_unparsed", ref=f"c{n:02d}", error=str(e), finish=getattr(rec.model, "last_finish", None),
                  length=len(reply), head=reply[:300], tail=reply[-300:])
        return {"written": {}, "dropped": [{"kind": "reply", "why": "not JSON"}]}
    result = store.apply_loops(conn, data, n, passages)
    rec.event("loops_applied", ref=f"c{n:02d}", **result)
    return result
