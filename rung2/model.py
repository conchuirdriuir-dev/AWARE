"""The model behind one small client: llama-server's OpenAI-compatible API, or a fake that lets the pipeline run
without a GPU. The fake's replies are canned and mean nothing; a run with it tests the plumbing, never the result.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request


class ServerError(Exception):
    """A request llama-server refused, with the server's own explanation."""


class Server:
    """llama-server (llama.cpp) for chat and tokens; a second server, or the same one, for embeddings."""

    def __init__(self, url: str, embed_url: str | None = None, timeout: float = 900, model: str | None = None,
                 embed_model: str | None = None):
        self.url = url.rstrip("/")
        self.embed_url = (embed_url or url).rstrip("/")
        self.timeout = timeout
        self.schema_style = "response_format"     # falls back to llama.cpp's own "json_schema" field if refused
        self.models = self._models(self.url)
        if model is None and len(self.models) > 1:
            raise ServerError("the server offers several models; choose one with --model: " + ", ".join(self.models))
        self.name = model or (self.models[0] if self.models else "unknown")
        if embed_model is None and self.embed_url != self.url:
            found = self._models(self.embed_url)
            embed_model = found[0] if len(found) == 1 else None
        self.embed_model = embed_model or self.name

    def _post(self, base: str, path: str, body: dict) -> dict:
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:2000]
            raise ServerError(f"{path} returned HTTP {e.code}: {detail}") from None

    def _models(self, base: str) -> list[str]:
        try:
            with urllib.request.urlopen(base + "/v1/models", timeout=30) as r:
                return [m["id"] for m in json.loads(r.read())["data"]]
        except Exception:
            return []

    def chat(self, prompt: str, schema: dict | None = None, max_tokens: int = 1024,
             temperature: float = 0.0, seed: int = 0) -> str:
        body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "temperature": temperature, "seed": seed,
                "chat_template_kwargs": {"enable_thinking": False}}    # Qwen3 and the like: no thinking (needs --jinja)
        if schema is not None and self.schema_style == "response_format":
            try:
                return self._chat({**body, "response_format": {"type": "json_schema",
                                                               "json_schema": {"name": "reply", "schema": schema}}})
            except ServerError as first:
                try:
                    reply = self._chat({**body, "json_schema": schema})
                except ServerError:
                    raise first from None            # refused either way: the first refusal is the one to read
                self.schema_style = "json_schema"
                return reply
        if schema is not None:
            body["json_schema"] = schema
        return self._chat(body)

    def _chat(self, body: dict) -> str:
        body = {"model": self.name, **body}
        choice = self._post(self.url, "/v1/chat/completions", body)["choices"][0]
        self.last_finish = choice.get("finish_reason")      # "length": the reply was cut off at max_tokens
        return strip_thinking(choice["message"]["content"])

    def tokens(self, text: str) -> int:
        return len(self._post(self.url, "/tokenize", {"model": self.name, "content": text})["tokens"])

    def embed(self, texts: list[str]) -> list[list[float]]:
        data = self._post(self.embed_url, "/v1/embeddings", {"model": self.embed_model, "input": texts})["data"]
        return [d["embedding"] for d in sorted(data, key=lambda d: d["index"])]


def check(url: str, embed_url: str | None, schema: dict, model: str | None = None,
          embed_model: str | None = None) -> list[tuple[str, bool, str]]:
    """Each kind of request the harness makes, on its own: which one a server refuses, and what it says."""
    results = []

    def step(name, fn):
        try:
            results.append((name, True, str(fn())[:300]))
        except Exception as e:
            results.append((name, False, str(e)))

    try:
        s = Server(url, embed_url, model=model, embed_model=embed_model)
    except ServerError as e:
        return [("choosing the model", False, str(e))]
    step("models the server offers (GET /v1/models)", lambda: ", ".join(s.models) or "(none listed)")
    step("model used", lambda: s.name)
    step("token count (POST /tokenize)", lambda: s.tokens("Mara hid the silver locket under the board."))
    step("plain chat", lambda: s._chat({"messages": [{"role": "user", "content": "Say hello in three words."}],
                                        "max_tokens": 20, "temperature": 0, "seed": 0}))
    step("plain chat, thinking off", lambda: s.chat("Say hello in three words.", max_tokens=20))
    prompt = ("Record facts from this passage as JSON. Every record needs a quote from the passage.\n\n"
              "PASSAGE:\nMara hid the silver locket under the loose board by the stove.")
    step("schema: response_format", lambda: s._chat({"messages": [{"role": "user", "content": prompt}],
                                                     "max_tokens": 400, "temperature": 0, "seed": 0,
                                                     "response_format": {"type": "json_schema", "json_schema":
                                                                         {"name": "reply", "schema": schema}}}))
    step("schema: json_schema field", lambda: s._chat({"messages": [{"role": "user", "content": prompt}],
                                                       "max_tokens": 400, "temperature": 0, "seed": 0,
                                                       "json_schema": schema}))
    step("embeddings (arm B only; not needed for the dry run)", lambda: len(s.embed(["a test sentence"])[0]))
    return results


def strip_thinking(reply: str | None) -> str:
    """Drop a <think>...</think> block a reasoning model emits before its answer, if one still comes through."""
    return re.sub(r"^\s*<think>.*?</think>\s*", "", reply or "", flags=re.S)


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
    last_finish = "stop"

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
        passage = _between(prompt, "THIS PASSAGE:\n", "\n\nWhat to record")
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
