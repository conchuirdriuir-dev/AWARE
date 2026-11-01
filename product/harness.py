"""The harness, arm A (PLAN.md, step 6, first part): a plain agent loop on a local model, writing the record at the
source.

The person types; the model answers and may call three file tools confined to a workspace folder. Every chunk is
appended to the record when it is created, tagged with who made it, its provenance and (if external) its arrival
cause; the context of every forward pass is built from the record alone, by one rendering rule (ChatML, Qwen3 with
thinking off), and the pass is recorded with the chunks it read and a hash of its token ids. States captured while a
chunk is generated are a cache beside the record (`cache.npz`), never the record; `verify` rebuilds every pass from
the record and checks the cache against a replay.

    python -m product.harness --model Qwen/Qwen3-0.6B --run runs/armA-1 --workspace runs/armA-1/workspace
    python -m product.harness --model Qwen/Qwen3-0.6B --run runs/armA-1 --script turns.txt   # one turn per line
    python -m product.harness --model Qwen/Qwen3-0.6B --run runs/armA-1 --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from substrate.record import Record, read

RENDERING = "chatml-qwen3-nothink-v1"
GEN_PROMPT = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
STOP = "<|im_end|>"
SYSTEM = ("You are an agent working with a person in a shared workspace folder. Use the tools when they help.\n\n"
          "# Tools\n\nYou may call one or more functions. Function signatures are within <tools></tools>:\n<tools>\n"
          '{"type": "function", "function": {"name": "list_files", "description": "List files in a folder of the '
          'workspace.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}\n'
          '{"type": "function", "function": {"name": "read_file", "description": "Read a file in the workspace.", '
          '"parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}\n'
          '{"type": "function", "function": {"name": "write_file", "description": "Write a file in the workspace.", '
          '"parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, '
          '"required": ["path", "content"]}}}\n</tools>\n\n'
          "For each function call, return a json object with the function name and arguments within "
          '<tool_call></tool_call> tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n'
          "</tool_call>")
ROLE = {"system": "system", "person": "user", "model": "assistant", "tool": "user"}


# ---- tools ----------------------------------------------------------------------------------------------------------

def inside(workspace: Path, path: str) -> Path | None:
    p = (workspace / (path or ".")).resolve()
    return p if p == workspace.resolve() or p.is_relative_to(workspace.resolve()) else None


def call_tool(workspace: Path, name: str, args: dict) -> str:
    """A tool's result as text. A path outside the workspace, an unknown tool or bad arguments are refused in words."""
    if name not in ("list_files", "read_file", "write_file") or not isinstance(args, dict):
        return f"refused: unknown tool {name!r}"
    p = inside(workspace, str(args.get("path", ".")))
    if p is None:
        return f"refused: {args.get('path')!r} is outside the workspace"
    try:
        if name == "list_files":
            return "\n".join(sorted(str(q.relative_to(workspace.resolve())) + ("/" if q.is_dir() else "")
                                    for q in p.iterdir())) or "(empty)"
        if name == "read_file":
            return p.read_text(encoding="utf-8", errors="replace")[:20000]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(args.get("content", "")), encoding="utf-8")
        return f"wrote {len(str(args.get('content', '')))} characters to {p.relative_to(workspace.resolve())}"
    except OSError as e:
        return f"error: {e.strerror or e}"


def tool_calls(text: str) -> list[tuple[str, dict]]:
    out = []
    for body in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        try:
            obj = json.loads(body)
            out.append((str(obj.get("name")), obj.get("arguments") or {}))
        except (json.JSONDecodeError, AttributeError):
            out.append(("(unparsed)", {}))
    return out


# ---- rendering: the record's chunks to token ids --------------------------------------------------------------------

def segments(chunks: list[dict]) -> list[str]:
    """The rendering rule: each chunk its own ChatML turn; tool results inside <tool_response>; then the prompt for
    the model's next turn. Segments are tokenized one by one, so a pass's ids are a function of the record alone."""
    out = []
    for c in chunks:
        body = f"<tool_response>\n{c['text']}\n</tool_response>" if c["role"] == "tool" else c["text"]
        out += [f"<|im_start|>{ROLE[c['role']]}\n", body, f"{STOP}\n"]
    return out + [GEN_PROMPT]


def encode(tok, segs: list[str]) -> list[int]:
    ids = list(tok("", add_special_tokens=True)["input_ids"])
    for s in segs:
        ids += tok(s, add_special_tokens=False)["input_ids"]
    return ids


def digest(ids) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()


# ---- the loop -------------------------------------------------------------------------------------------------------

class Session:
    def __init__(self, model, tok, run: str | Path, workspace: str | Path | None = None, layers=None,
                 max_new_tokens: int = 512, temperature: float = 0.0, seed: int = 0, max_tool_rounds: int = 4,
                 revision: str | None = None, name: str | None = None):
        import torch

        self.model, self.tok, self.run = model, tok, Path(run)
        self.workspace = Path(workspace or self.run / "workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
        n = model.config.num_hidden_layers
        self.layers = layers or sorted({n // 4, n // 2, 3 * n // 4})
        self.max_new_tokens, self.temperature, self.seed = max_new_tokens, temperature, seed
        self.max_tool_rounds = max_tool_rounds
        self.record = Record(self.run / "record.jsonl")
        self.chunks: list[dict] = []
        self.cache: dict[str, np.ndarray] = {}
        self.stop_id = tok.convert_tokens_to_ids(STOP) if STOP in tok.get_vocab() else tok.eos_token_id
        torch.manual_seed(seed)
        self.record.append("session", "code", arm="A", model=name or getattr(model.config, "name_or_path", ""),
                           revision=revision, commit=getattr(model.config, "_commit_hash", None),
                           dtype=str(next(model.parameters()).dtype), rendering=RENDERING, layers=self.layers,
                           max_new_tokens=max_new_tokens, temperature=temperature, seed=seed,
                           tools=["list_files", "read_file", "write_file"], workspace=str(self.workspace))
        self._add(SYSTEM, "code", "external", "spontaneous", "system")

    def _add(self, text, by, provenance, cause, role, **fields):
        e = self.record.chunk(text, by, provenance, cause, role=role, **fields)
        self.chunks.append({"seq": e["seq"], "text": text, "role": role})
        return e

    def person(self, text: str) -> list[str]:
        """The person's turn, then the model's turns (and tool rounds) until it answers without a tool call."""
        self._add(text, "person", "external", "spontaneous", "person")
        replies = []
        for _ in range(self.max_tool_rounds + 1):
            reply = self._generate()
            replies.append(reply)
            calls = tool_calls(reply)
            if not calls:
                break
            for name, args in calls:
                self._add(call_tool(self.workspace, name, args), "code", "external", "elicited", "tool", tool=name)
        return replies

    def _generate(self) -> str:
        import torch

        ids = encode(self.tok, segments(self.chunks))
        reads = [c["seq"] for c in self.chunks]
        captured = {l: [] for l in self.layers}
        start = len(ids)
        layers = self.model.model.layers

        def hook(l):
            def fn(mod, args, out):
                h = out[0] if isinstance(out, tuple) else out
                captured[l].append(h[0].detach().float().cpu().numpy())
            return fn

        hs = [layers[l].register_forward_hook(hook(l)) for l in self.layers]
        try:
            with torch.no_grad():
                kw = dict(do_sample=self.temperature > 0, max_new_tokens=self.max_new_tokens,
                          eos_token_id=self.stop_id, pad_token_id=self.stop_id)
                if self.temperature > 0:
                    kw["temperature"] = self.temperature
                else:
                    kw.update(temperature=None, top_p=None, top_k=None)
                out = self.model.generate(torch.tensor([ids]), **kw)[0].tolist()
        finally:
            for h in hs:
                h.remove()
        gen = out[start:]
        content = gen[:-1] if gen and gen[-1] == self.stop_id else gen
        text = self.tok.decode(content)
        round_trip = self.tok(text, add_special_tokens=False)["input_ids"] == content
        e = self._add(text, "model", "self", None, "model", generated_tokens=len(content), round_trip=round_trip)
        # a sampled token's state is computed when it is fed back, one step later; the last token is not fed back
        # unless it is the stop, so the cache pools the content tokens that were fed back, and says how many
        fed = 0
        if content:
            for l in self.layers:
                rows = np.concatenate(captured[l])[start:start + len(content)]
                fed = len(rows)
                if fed:
                    self.cache[f"{e['seq']}:{l}"] = rows.mean(0)
            np.savez(self.run / "cache.npz", **self.cache)                    # derived, beside the record
        self.record.append("pass", "code", reads=reads, tokens=len(ids), ids_sha256=digest(ids), wrote=e["seq"],
                           stopped=bool(gen and gen[-1] == self.stop_id), cached_tokens=fed)
        return text


# ---- verification: PLAN step 6's tests 1-3 from the record ----------------------------------------------------------

def verify(run: str | Path, model, tok) -> dict:
    """Rebuild every pass from the record; check provenance, exact replay and the cache against a replayed state."""
    import torch

    run = Path(run)
    entries = read(run / "record.jsonl")
    session = next(e for e in entries if e["kind"] == "session")
    chunks = {e["seq"]: e for e in entries if e["kind"] == "chunk"}
    cache = dict(np.load(run / "cache.npz")) if (run / "cache.npz").exists() else {}
    want = {"person": ("external", "spontaneous"), "tool": ("external", "elicited"), "system": ("external", "spontaneous"),
            "model": ("self", None)}
    problems, passes, diffs, checked, round_trips = [], 0, [], 0, []
    for e in entries:
        if e["kind"] != "pass":
            continue
        passes += 1
        if any(s not in chunks or s > e["seq"] for s in e["reads"]):
            problems.append(f"pass {e['seq']} reads a chunk recorded after it")
        segs = segments([{"text": chunks[s]["text"], "role": chunks[s]["role"]} for s in e["reads"]])
        ids = encode(tok, segs)
        if digest(ids) != e["ids_sha256"] or len(ids) != e["tokens"]:
            problems.append(f"pass {e['seq']}: replayed ids differ from the ids it read")
        w = chunks[e["wrote"]]
        round_trips.append(bool(w.get("round_trip")))
        if not w.get("round_trip") or not w["text"]:
            continue
        gen = tok(w["text"], add_special_tokens=False)["input_ids"][:e.get("cached_tokens", 0)]
        if not gen:
            continue
        captured_rows = {}

        def hook(l):
            def fn(mod, args, out):
                h = out[0] if isinstance(out, tuple) else out
                captured_rows[l] = h[0, len(ids):len(ids) + len(gen)].float().mean(0).cpu().numpy()
            return fn

        hs = [model.model.layers[l].register_forward_hook(hook(l)) for l in session["layers"]]
        with torch.no_grad():
            model(torch.tensor([ids + gen]), use_cache=False)
        for h in hs:
            h.remove()
        for l in session["layers"]:
            key = f"{w['seq']}:{l}"
            if key in cache:
                a, b = cache[key], captured_rows[l]
                diffs.append(float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)))
                checked += 1
    for c in chunks.values():
        if (c["provenance"], c.get("cause")) != want.get(c.get("role"), (None, None)):
            problems.append(f"chunk {c['seq']} ({c.get('role')}) tagged {c['provenance']}/{c.get('cause')}")
    report = {"passes": passes, "chunks": len(chunks), "problems": problems,
              "round trip share": float(np.mean(round_trips)) if round_trips else float("nan"),
              "states checked": checked, "max relative difference": max(diffs) if diffs else float("nan"),
              "provenance ok": not any("tagged" in p or "after it" in p for p in problems),
              "replay exact": not any("replayed" in p for p in problems),
              "cache faithful": bool(diffs) and max(diffs) <= 1e-3}
    return report


def load(name: str, revision: str | None = None, dtype: str = "float32"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=getattr(torch, dtype)).eval()
    return model, tok


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--run", required=True)
    p.add_argument("--workspace")
    p.add_argument("--script", help="a file of the person's turns, one per line, instead of typing them")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--verify", action="store_true", help="check a recorded run against PLAN step 6's tests 1-3")
    a = p.parse_args(argv)
    model, tok = load(a.model, a.revision)
    if a.verify:
        print(json.dumps(verify(a.run, model, tok), indent=1))
        return
    s = Session(model, tok, a.run, a.workspace, max_new_tokens=a.max_new_tokens, temperature=a.temperature,
                seed=a.seed, revision=a.revision, name=a.model)
    turns = Path(a.script).read_text().splitlines() if a.script else None
    while True:
        text = turns.pop(0) if turns else (None if turns is not None else input("you> "))
        if not text:
            break
        if turns is not None:
            print(f"you> {text}")
        for r in s.person(text):
            print(f"model> {r}")


if __name__ == "__main__":
    main()
