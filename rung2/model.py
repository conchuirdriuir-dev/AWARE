"""The model behind one small client: llama-server's OpenAI-compatible API, or a fake that lets the pipeline run
without a GPU. The fake's replies are canned and mean nothing; a run with it tests the plumbing, never the result.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request


class Server:
    """llama-server (llama.cpp) for chat and tokens; a second server, or the same one, for embeddings."""

    def __init__(self, url: str, embed_url: str | None = None, timeout: float = 900):
        self.url = url.rstrip("/")
        self.embed_url = (embed_url or url).rstrip("/")
        self.timeout = timeout
        self.name = self._model_name()

    def _post(self, base: str, path: str, body: dict) -> dict:
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def _model_name(self) -> str:
        try:
            with urllib.request.urlopen(self.url + "/v1/models", timeout=30) as r:
                return json.loads(r.read())["data"][0]["id"]
        except Exception:
            return "unknown"

    def chat(self, prompt: str, schema: dict | None = None, max_tokens: int = 1024,
             temperature: float = 0.0, seed: int = 0) -> str:
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": temperature, "seed": seed}
        if schema is not None:
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}
        return self._post(self.url, "/v1/chat/completions", body)["choices"][0]["message"]["content"]

    def tokens(self, text: str) -> int:
        return len(self._post(self.url, "/tokenize", {"content": text})["tokens"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post(self.embed_url, "/v1/embeddings", {"input": texts})["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


# -- the fake --------------------------------------------------------------------------------------------------------

WORD = re.compile(r"[A-Za-z']+")
NAME = re.compile(r"\b[A-Z][a-z]{2,}\b")
COMMON = {"The", "And", "But", "She", "Her", "His", "They", "Then", "When", "What", "Where", "Who", "Why", "How",
          "There", "This", "That", "Not", "Yes", "Chapter", "Its", "Their", "You", "Your", "For", "With", "From"}


def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i)
    return text[i:j if j >= 0 else None]


def _first_sentence(text: str) -> str:
    m = re.search(r"[^.!?]*[.!?]", text.strip())
    return (m.group(0) if m else text).strip()


def _short(text: str, words: int = 12) -> str:
    return " ".join(text.split()[:words])


class Fake:
    """Deterministic replies shaped like the real ones: enough to exercise every code path."""

    name = "fake"

    def chat(self, prompt: str, schema: dict | None = None, max_tokens: int = 1024,
             temperature: float = 0.0, seed: int = 0) -> str:
        if schema is not None:
            return json.dumps(self._keeper(prompt))
        if prompt.startswith("In one sentence"):
            return _first_sentence(_between(prompt, ":\n", "\0"))
        if "SUMMARY SO FAR:" in prompt:
            old = _between(prompt, "SUMMARY SO FAR:\n", "\n\nCHAPTER").strip()
            return (old + " " + _first_sentence(_between(prompt, "TEXT:\n", "\0"))).strip()
        material = _between(prompt, "MATERIAL:\n", "\n\nQUESTION:").strip()
        return _first_sentence(material) if material else "Not stated in the material."

    def tokens(self, text: str) -> int:
        return max(1, math.ceil(len(text.split()) * 4 / 3))

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * 64
            for w in WORD.findall(t.lower()):
                v[int(hashlib.sha256(w.encode()).hexdigest(), 16) % 64] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    def _keeper(self, prompt: str) -> dict:
        passage = _between(prompt, "THIS PASSAGE:\n", "\n\nWhat to record:")
        known = {}
        for line in _between(prompt, "(refer to them by id):\n", "\n\n").splitlines():
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                known[parts[1]] = parts[0]
        names = []
        for n in NAME.findall(passage):
            if n not in COMMON and n not in names:
                names.append(n)
        new = [{"name": n, "kind": "character", "aliases": [], "quote": n} for n in names if n not in known]
        first = _first_sentence(passage)
        refs = [known.get(n, n) for n in names]
        reply = {"new_entities": new, "states": [], "knowledge": [], "rules": [],
                 "events": [{"summary": _short(first, 20), "place": None, "participants": refs[:3],
                             "happened": True, "quote": _short(first)}] if first else [],
                 "loops_opened": [], "loops_touched": [], "loops_closed": []}
        question = re.search(r"[^.!?]*\?", passage)
        if question:
            q = question.group(0).strip()
            reply["loops_opened"].append({"description": _short(q, 15), "entities": refs[:1], "quote": _short(q)})
        return reply
