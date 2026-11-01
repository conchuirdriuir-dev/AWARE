"""Labelled logs: goal attention and goal alignment on the owner-confirmed departures (product/labelled_logs.md).

    python -m product.labelled_logs --model Qwen/Qwen3-8B --corpus .../dara-session-transcripts --run runs/labels-8b \\
        --device auto --gpu-memory 6.5
    python -m product.labelled_logs --model Qwen/Qwen3-0.6B --corpus ... --run runs/labels-06 --limit 40 --context 4096

The corpus is private: no text from it is recorded, only references and numbers. Each anchor (a departure or a
control) is checkpointed, so an interrupted run resumes when the same command is run again.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import json
import os
import math
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from product.toybed import auc
from substrate.corpus import load_session, sessions as session_paths
from substrate.echo_floor import session_resumable
from substrate.record import Record

W, CONTROLS_PER_SESSION, CAP = 100, 5, 256
SYSTEM_PREFIXES = ("[SYSTEM", "<system-reminder>", "[Artifact comment")
KINDS = ("closure", "accommodation", "ascent", "undecidable")


# --- the corpus, as the text study read it ---------------------------------------------------------------------

def ts(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s.replace("Z", "")[:26])


def role(chunk: dict) -> str:
    if chunk["provenance"] == "self":
        return "agent_tool" if "tool" in chunk else "agent_text"
    if chunk["cause"] == "elicited":
        return "tool_result"
    return "system" if chunk["text"].startswith(SYSTEM_PREFIXES) else "human"


def load_corpus(root: str) -> dict:
    sessions = {}
    for path in session_paths(Path(root) / "sessions"):
        chunks = load_session(path)
        for c in chunks:
            c["role"] = role(c)
        sessions[path.name.split(".")[0]] = {"chunks": chunks, "times": [ts(c["at"]) for c in chunks]}
    return sessions


def locate(session: dict, when: str) -> int:
    i = bisect.bisect_left(session["times"], ts(when))
    return min(max(i, 0), len(session["times"]) - 1)


def anchors(root: str, sessions: dict, seed: int = 0) -> list[dict]:
    """The owner-confirmed real departures, grouped as the text study's run 3 grouped them, and its controls."""
    R = json.load(open(Path(root) / "study" / "restorations_reading_v0_1.json"))
    L = json.load(open(Path(root) / "study" / "labels_confirmed.json"))
    out = []
    for d in R["departures"]:
        lab = L.get(d["id"])
        if not lab or lab["real"] != ["yes"] or d["ord"] not in sessions:
            continue
        kinds = lab["kind"]
        group = next((k for k in KINDS[:3] if k in kinds), "undecidable")
        out.append({"id": d["id"], "session": d["ord"], "index": locate(sessions[d["ord"]], d["time"]),
                    "time": d["time"], "group": group, "who": (lab["who"] or ["?"])[0],
                    "departed_from": (d.get("departs_from_aims") or [None])[0]})
    rng = random.Random(seed)
    for k in sorted({a["session"] for a in out}):
        taken = [a["index"] for a in out if a["session"] == k]
        chunks = sessions[k]["chunks"]
        cands = [i for i, c in enumerate(chunks) if c["role"] == "agent_text" and len(c["text"]) >= 200
                 and all(abs(i - j) > W for j in taken)]
        rng.shuffle(cands)
        for i in cands[:CONTROLS_PER_SESSION]:
            out.append({"id": f"C-{k}-{i}", "session": k, "index": i, "time": chunks[i]["at"], "group": "control",
                        "who": "agent", "departed_from": None})
    return [a for a in out if a["index"] >= W]


def live_aim(R: dict, session: str, when: str) -> dict | None:
    aims = [a for a in R["aims"] if a["ord"] == session and ts(a["time"]) <= ts(when)]
    return max(aims, key=lambda a: ts(a["time"])) if aims else None


# --- the text study's four statistics (study/pilot.py, the owner's; reused unchanged on the same anchors) --------

_TOK = lambda s: re.findall(r"[a-z][a-z'\-]{2,}", s.lower())  # noqa: E731
_STOP = set("the and for that with this from have are was were will would could should not but you your they them "
            "then than what which when where who how can into about just like also been being has had does did doing "
            "very more most some such only over under out off its it's i'm we're don't can't let's".split())


def text_stats(chunks: list[dict], a: int, b: int, opening: str) -> dict:
    win = chunks[a:b]
    agent = " ".join(c["text"] for c in win if c["role"] == "agent_text")
    human = " ".join(c["text"] for c in win if c["role"] == "human")
    tools = " ".join(c["text"] for c in win if c["role"] == "tool_result")
    prev = " ".join(c["text"] for c in chunks[max(0, a - W):a] if c["role"] == "agent_text")
    aw, hw, pw = _TOK(agent), _TOK(human), _TOK(prev)
    tot = len(agent) + len(human) + len(tools)
    earlier = " ".join(c["text"] for c in chunks[:a] if c["role"] == "agent_text")
    e4 = {tuple(g) for g in zip(*[_TOK(earlier)[i:] for i in range(4)])}
    a4 = list(zip(*[aw[i:] for i in range(4)]))

    def bag(ws):
        return Counter(w for w in ws if w not in _STOP)

    def cos(c1, c2):
        num = sum(c1[w] * c2[w] for w in c1 if w in c2)
        d = math.sqrt(sum(v * v for v in c1.values())) * math.sqrt(sum(v * v for v in c2.values()))
        return num / d if d else float("nan")
    hc, pset = [w for w in hw if w not in _STOP], set(pw)
    return {"agent_frac_all": len(agent) / tot if tot else float("nan"),
            "agent_rep4": sum(g in e4 for g in a4) / len(a4) if a4 else float("nan"),
            "drift_from_open": 1 - cos(bag(aw + hw), bag(_TOK(opening))),
            "human_echo": sum(w in pset for w in hc) / len(hc) if hc else float("nan")}


# --- the readings ------------------------------------------------------------------------------------------------

def attention_indices(A: np.ndarray, spans: list[tuple[int, int]], kinds: list[str]) -> dict:
    """G and S: observed attention from self chunks' tokens onto the goal's tokens and onto the entity's own other
    tokens, over what other external tokens at the same distance draw (amended baseline: the set being measured must
    not feed its own baseline). A is (layers, T, T), head-averaged; the sequence's first token (the sink) and a token's
    attention to itself are left out; a query chunk's own tokens are left out of every set."""
    Lyr, T, _ = A.shape
    ind = {"goal": np.zeros(T), "self": np.zeros(T), "external": np.zeros(T)}
    owner = np.full(T, -1)
    for n, ((a, b), kind) in enumerate(zip(spans, kinds)):
        ind[kind][a:b] = 1.0
        owner[a:b] = n
    for v in ind.values():
        v[0] = 0.0
    queries = [n for n, k in enumerate(kinds) if k == "self"]
    tot = {s: [0.0, 0.0] for s in ("goal", "self")}
    for layer in range(Lyr):
        M = A[layer].copy()
        M[:, 0] = 0.0                                                   # the sink
        np.fill_diagonal(M, 0.0)                                        # a token's attention to itself
        e = np.zeros(T)
        ext = ind["external"]
        for d in range(1, T):
            vals = np.diagonal(M, -d)                                     # M[d + j, j], keys j = 0 .. T-d-1
            m = ext[:T - d]
            e[d] = (vals * m).sum() / m.sum() if m.sum() else vals[1:].mean() if len(vals) > 1 else 0.0
        for s in ("goal", "self"):
            obs = M @ ind[s]                                              # per query token
            exp = np.convolve(ind[s], e)[:T]                              # sum over keys k < q of e(q - k)
            for n in queries:
                a, b = spans[n]
                if s == "self" and not ind["self"][1:a].any():         # no earlier self tokens: nothing to read
                    continue
                o, x = obs[a:b].sum(), exp[a:b].sum()
                if s == "self":                                          # take the query's own chunk out
                    own = M[a:b, a:b]
                    o -= np.tril(own, -1).sum()
                    q = np.arange(a, b)
                    dd = q[:, None] - q[None, :]
                    x -= e[np.clip(dd, 0, None)][dd > 0].sum()
                tot[s][0] += o
                tot[s][1] += x
    return {k: tot[s][0] / tot[s][1] if tot[s][1] > 1e-9 else float("nan") for k, s in (("G", "goal"), ("S", "self"))}


def alignment(states: np.ndarray, goals: np.ndarray, live: int) -> float:
    """Mean over chunks and layers of the live goal's scaled rank by distance: 1 nearest of all goals, 0 farthest."""
    d = np.linalg.norm(states[:, None] - goals[None], axis=3)          # chunks x goals x layers
    rank = (d < d[:, live:live + 1]).sum(axis=1)                         # goals nearer than the live one
    return float(np.mean(1 - rank / max(1, goals.shape[0] - 1)))


def goal_turn(session: dict, aim: dict) -> int:
    """The chunk index of an aim's owner turn: the turn holding its quote, else the owner turn nearest its time."""
    chunks = session["chunks"]
    j = locate(session, aim["time"])
    quote = aim["quote"][:60].strip()
    near = [k for k in range(len(chunks)) if chunks[k]["role"] == "human" and quote in chunks[k]["text"]]
    if not near:
        near = [k for k in range(max(0, j - 5), min(len(chunks), j + 6)) if chunks[k]["role"] == "human"]
    return min(near, key=lambda k: abs(k - j)) if near else j


def ntok(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False)["input_ids"])


def trim(tok, texts: list[str], kinds: list[str], goal_tokens: int, context: int) -> list[tuple[str, str]]:
    """The window's latest chunks that fit beside a goal turn of `goal_tokens` in `context` tokens."""
    budget = context - goal_tokens - 8
    keep, used = [], 0
    for t, k in zip(reversed(texts), reversed(kinds)):
        n = ntok(tok, t)
        if used + n > budget:
            break
        keep.append((t, k))
        used += n
    keep.reverse()
    return keep


def capper(tok):
    def capped(t):
        return tok.decode(tok(t, add_special_tokens=False)["input_ids"][:CAP])
    return capped


def window(chunks, a, b, capped):
    win = chunks[a:b]
    return win, [capped(c["text"]) + "\n\n" for c in win], ["self" if c["provenance"] == "self" else "external" for c in win]


def read_anchor(reader, sessions, R, anchor, layers, aim_states, aim_ids, line_states, line_ids, context, after):
    s = sessions[anchor["session"]]
    chunks, i = s["chunks"], anchor["index"]
    aim = live_aim(R, anchor["session"], anchor["time"])
    out = {"text": text_stats(chunks, i - W, i, " ".join(c["text"] for c in chunks[:W])), "aim": aim and aim["id"]}
    tok = reader.tokenizer
    capped = capper(tok)
    windows = {"before": (i - W, i)} | ({"after": (i + 1, i + 1 + W)} if after and i + 1 + W <= len(chunks) else {})
    for name, (a, b) in windows.items():
        win, texts, kinds = window(chunks, a, b, capped)
        if aim is not None:
            goal_text = capped(chunks[goal_turn(s, aim)]["text"]) + "\n\n"
            keep = trim(tok, texts, kinds, ntok(tok, goal_text), context)
            A, spans = reader.attention_in_context([goal_text] + [t for t, _ in keep], layers)
            out[f"{name}: attention"] = attention_indices(A, spans, ["goal"] + [k for _, k in keep])
            del A
        Aall = reader.states([capped(c["text"]) for c in win], layers)       # every chunk alone, read once
        selfs = [n for n, c in enumerate(win) if c["provenance"] == "self"]
        if selfs and aim is not None:
            S = Aall[selfs]
            out[f"{name}: A"] = alignment(S, aim_states, aim_ids.index(aim["id"]))
            out[f"{name}: A'"] = alignment(S, line_states, line_ids.index(aim["line"])) if aim["line"] in line_ids else None
        news = [np.min(np.linalg.norm(Aall[:k] - Aall[k], axis=2), axis=0).mean()
                for k in range(1, len(win)) if win[k]["provenance"] == "external"]
        out[f"{name}: news"] = float(np.median(news)) if news else None
    return out


# --- the placebo check (amendment, 2026-11-01): G over the live aim's turn against G over a placebo aim's turn ---

def placebo_pool(sessions, R, tok, seed: int = 0) -> list[dict]:
    """Every aim's owner turn, capped, with its token length, in a fixed order shuffled by `seed` (ties go to the
    first in this order)."""
    capped = capper(tok)
    pool = []
    for a in R["aims"]:
        s = sessions.get(a["ord"])
        if not s or not s["chunks"]:
            continue
        text = capped(s["chunks"][goal_turn(s, a)]["text"]) + "\n\n"
        pool.append({"id": a["id"], "session": a["ord"], "line": a["line"], "text": text, "tokens": ntok(tok, text)})
    random.Random(seed).shuffle(pool)
    return pool


def pick_placebo(pool: list[dict], live: dict, live_tokens: int) -> dict | None:
    """The aim from another session and another line nearest the live turn in token length."""
    cands = [p for p in pool if p["session"] != live["ord"] and p["line"] != live["line"]]
    return min(cands, key=lambda p: abs(p["tokens"] - live_tokens)) if cands else None


def read_placebo(reader, sessions, R, anchor, layers, context, pool, raw_g):
    """The window before onset read twice over the same chunks: with the live aim's turn, and with the placebo's.
    The live pass is reused from the raw reading (`raw_g`) when the placebo turn is no longer, since the chunks are
    then the same."""
    s = sessions[anchor["session"]]
    chunks, i = s["chunks"], anchor["index"]
    aim = live_aim(R, anchor["session"], anchor["time"])
    if aim is None:
        return {}
    tok = reader.tokenizer
    capped = capper(tok)
    win, texts, kinds = window(chunks, i - W, i, capped)
    live_text = capped(chunks[goal_turn(s, aim)]["text"]) + "\n\n"
    lt = ntok(tok, live_text)
    raw_keep = trim(tok, texts, kinds, lt, context)
    out = {"goal tokens": lt, "window tokens": sum(ntok(tok, t) for t, _ in raw_keep)}
    p = pick_placebo(pool, aim, lt)
    if p is None:
        return out
    keep = trim(tok, texts, kinds, max(lt, p["tokens"]), context)
    kk = ["goal"] + [k for _, k in keep]

    def g(goal):
        A, spans = reader.attention_in_context([goal] + [t for t, _ in keep], layers)
        return attention_indices(A, spans, kk)["G"]
    live_g = raw_g if p["tokens"] <= lt and raw_g is not None else g(live_text)
    placebo_g = g(p["text"])
    out.update({"placebo": p["id"], "placebo tokens": p["tokens"], "G live (paired)": live_g, "G placebo": placebo_g,
                "G ratio": live_g / placebo_g if placebo_g and placebo_g == placebo_g else float("nan")})
    return out


def score(rows: dict, key: str, events: list[str], controls: list[str], direction: int, draws: int = 2000,
          seed: int = 0) -> dict:
    ev = [rows[k][key] for k in events if rows[k].get(key) is not None and rows[k][key] == rows[k][key]]
    ct = [rows[k][key] for k in controls if rows[k].get(key) is not None and rows[k][key] == rows[k][key]]
    if len(ev) < 3 or len(ct) < 3:
        return {"n": len(ev), "controls": len(ct), "AUC": float("nan")}
    ev, ct = np.array(ev) * direction, np.array(ct) * direction
    rng = np.random.default_rng(seed)
    boot = [auc(rng.choice(ev, len(ev)), rng.choice(ct, len(ct))) for _ in range(draws)]
    lo, hi = np.percentile(boot, [5, 95])
    return {"n": len(ev), "controls": len(ct), "AUC": auc(ev, ct), "90% interval": [float(lo), float(hi)]}


def flat(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if isinstance(v, dict):
            out.update({f"{k} {kk}" if k != "text" else f"text {kk}": vv for kk, vv in v.items()})
        else:
            out[k] = v
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--corpus", required=True, help="the dara-session-transcripts folder (with sessions/ and study/)")
    p.add_argument("--run", required=True)
    p.add_argument("--context", type=int, default=8192, help="tokens of each attention pass")
    p.add_argument("--limit", type=int, help="a rehearsal: at most this many anchors (closures first, then controls)")
    p.add_argument("--no-after", action="store_true", help="skip the windows after each anchor (reported only)")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"])
    p.add_argument("--gpu-memory", type=float)
    p.add_argument("--dtype", choices=["bfloat16", "float32"])
    a = p.parse_args(argv)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before CUDA starts: less fragmentation
    if not session_paths(Path(a.corpus) / "sessions"):
        raise SystemExit(f"no session archives in {a.corpus!r}/sessions")
    from substrate.states import Reader
    reader = Reader(a.model, device=a.device, dtype=a.dtype, gpu_memory=a.gpu_memory)
    n = reader.info["layers"]
    layers = list(range(math.ceil(2 * n / 3), n))
    record = Record(Path(a.run) / "record.jsonl")
    sessions = load_corpus(a.corpus)
    R = json.load(open(Path(a.corpus) / "study" / "restorations_reading_v0_1.json"))
    todo = anchors(a.corpus, sessions)
    if a.limit:
        order = {"closure": 0, "control": 1, "ascent": 2, "accommodation": 3, "undecidable": 4}
        todo = sorted(todo, key=lambda x: (order[x["group"]], x["id"]))
        per = {"closure": a.limit // 2, "control": a.limit - a.limit // 2}
        todo = [x for x in todo if x["group"] in per and sum(y["group"] == x["group"] for y in todo[:todo.index(x)]) < per[x["group"]]]
    record.append("run", "code", **reader.info, read_layers=layers, context=a.context, limit=a.limit,
                  anchors=[{k: v for k, v in x.items()} for x in todo])
    aim_ids = [x["id"] for x in R["aims"]]
    line_ids = [x["id"] for x in R["lines"]]

    def goals():
        return ({"aims": reader.states([x["quote"][:2000] for x in R["aims"]], layers),
                 "lines": reader.states([x["statement"] for x in R["lines"]], layers)}, {})
    g, _ = session_resumable(record, "goal-states", goals)
    rows = {}
    for x in todo:
        def compute(x=x):
            r = read_anchor(reader, sessions, R, x, layers, g["aims"], aim_ids, g["lines"], line_ids, a.context,
                            not a.no_after)
            return {"done": np.array([1])}, flat(r)
        _, row = session_resumable(record, "anchor-" + x["id"], compute)
        rows[x["id"]] = row
    pool = placebo_pool(sessions, R, reader.tokenizer)                      # the amendment: the placebo stage
    for x in todo:
        def compute_placebo(x=x):
            return {"done": np.array([1])}, read_placebo(reader, sessions, R, x, layers, a.context, pool,
                                                         rows[x["id"]].get("before: attention G"))
        _, extra = session_resumable(record, "placebo-" + x["id"], compute_placebo)
        rows[x["id"]].update(extra)
    groups = {grp: [x["id"] for x in todo if x["group"] == grp and (grp != "closure" or x["who"] == "agent")]
              for grp in ("closure", "accommodation", "ascent", "control")}
    groups["accommodation"] = [x["id"] for x in todo if x["group"] == "accommodation" and x["who"] == "agent"]
    groups["ascent"] = [x["id"] for x in todo if x["group"] == "ascent" and x["who"] == "owner"]
    primary = {"S": score(rows, "before: attention S", groups["closure"], groups["control"], +1),
               "G": score(rows, "before: attention G", groups["closure"], groups["control"], -1),
               "A": score(rows, "before: A", groups["closure"], groups["control"], -1)}
    text = {k: score(rows, f"text {k}", groups["closure"], groups["control"], +1)
            for k in ("agent_frac_all", "agent_rep4", "drift_from_open", "human_echo")}
    best_text = max((max(v["AUC"], 1 - v["AUC"]) for v in text.values() if v["AUC"] == v["AUC"]), default=float("nan"))
    ascent_S = score(rows, "before: attention S", groups["ascent"], groups["control"], +1)
    ratio = score(rows, "G ratio", groups["closure"], groups["control"], -1)
    lengths = {k: score(rows, k, groups["closure"], groups["control"], +1) for k in ("window tokens", "goal tokens")}

    def bar(v):
        return v["AUC"] == v["AUC"] and v["AUC"] >= 0.7 and v.get("90% interval", [0])[0] > 0.5 and v["AUC"] > best_text
    ratio_holds = ratio["AUC"] == ratio["AUC"] and ratio.get("90% interval", [0])[0] > 0.5
    supported = bar(primary["S"]) or bar(primary["A"]) or (bar(primary["G"]) and ratio_holds)
    g_note = ("G's pass is not distinguishable from position and length"
              if bar(primary["G"]) and not ratio_holds else None)
    passed = {"1 batch claim supported": supported,
              "2 ascents not read as closures": ascent_S["AUC"] <= 0.6 if ascent_S["AUC"] == ascent_S["AUC"] else None}
    reported = {grp: {k: score(rows, k, groups[grp], groups["control"], +1)
                      for k in ("before: attention S", "before: attention G", "before: A", "before: A'", "before: news")}
                for grp in ("accommodation", "ascent")}
    report = record.append("report", "code", primary=primary, G_ratio=ratio, lengths=lengths, G_note=g_note,
                           text=text, best_text=best_text, ascent_S=ascent_S, passed=passed, reported=reported,
                           groups={k: len(v) for k, v in groups.items()})
    print(f"{reader.info['model']}; anchors {report['groups']}")
    for k, v in primary.items():
        print(f"  closure, {k}: AUC {v['AUC']:.2f} interval {v.get('90% interval')} (n {v['n']}, controls {v['controls']})")
    print(f"  closure, G live / G placebo: AUC {ratio['AUC']:.2f} interval {ratio.get('90% interval')} (n {ratio['n']})")
    for k, v in lengths.items():
        print(f"  {k}, closures against controls: AUC {v['AUC']:.2f} (n {v['n']}, controls {v['controls']})")
    if g_note:
        print("  " + g_note)
    print(f"  best text statistic: {best_text:.2f}   ascents' S: {ascent_S['AUC']}")
    print("  passed:", passed)


if __name__ == "__main__":
    main()
