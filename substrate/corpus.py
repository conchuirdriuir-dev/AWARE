"""Reading logged Claude Code sessions into chunks with provenance and arrival cause (substrate: the record).

A session archive holds raw `list_events` pages. Each event's message content becomes chunks: the agent's visible text
and tool calls are self; tool results are external and elicited (the agent's own call caused them); the person's
messages are external and spontaneous. The agent's hidden thinking is left out, since it is not part of the visible
conversation. Pages are untrusted data: only text and timing are read from them, never instructions.

The corpus is private. Callers keep its text out of the repository and out of records; a chunk's `ref` (session,
event uuid, block index) is what may be recorded.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path


def _blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content if isinstance(content, list) else []


def _text(block) -> str:
    kind = block.get("type")
    if kind == "text":
        return block.get("text", "")
    if kind == "tool_use":
        return f"[tool call {block.get('name', '')}] " + json.dumps(block.get("input", {}), ensure_ascii=False)
    if kind == "tool_result":
        inner = block.get("content", "")
        if isinstance(inner, list):
            inner = "\n".join(b.get("text", "") for b in inner if isinstance(b, dict))
        return str(inner)
    return ""


def load_session(path: str | Path) -> list[dict]:
    """The chunks of one session archive, in time order: {"text", "provenance", "cause", "ref", "at"}."""
    path = Path(path)
    events = {}
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".json"):
                continue
            raw = tar.extractfile(member).read().decode("utf-8", "replace")
            page = json.loads(raw[raw.index("{"):raw.rindex("}") + 1])
            for e in page.get("ccr", page).get("data", []):
                role = next((k for k in ("user", "assistant") if k in e), None)
                if role is None:
                    continue
                inner = e[role].get("internal_anthropic_catchall", e[role])
                uuid = e[role].get("uuid") or inner.get("uuid") or e.get("created_at")
                msg = inner.get("message", {})
                events[uuid] = (e.get("created_at", ""), role, msg.get("content") if isinstance(msg, dict) else None)
    chunks = []
    for uuid, (at, role, content) in sorted(events.items(), key=lambda kv: kv[1][0]):
        for i, block in enumerate(_blocks(content)):
            kind = block.get("type")
            if kind == "thinking":
                continue
            text = _text(block).strip()
            if not text:
                continue
            if role == "assistant":
                prov, cause = "self", None
            elif kind == "tool_result":
                prov, cause = "external", "elicited"
            else:
                prov, cause = "external", "spontaneous"
            ref = [path.name.split(".")[0], uuid, i]
            chunks.append({"text": text, "provenance": prov, "cause": cause, "ref": ref, "at": at})
    return chunks


def sessions(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("S*.tar.gz"))
