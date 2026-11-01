"""Attention to the owner as the agent runs on without hearing from them (a probe beside `attention_share_long.py`,
not a step; neither track's criterion).

The clearest result so far: when the agent writes, it leans hard toward the owner's most recent message within 512
tokens. Drift tends to happen in long stretches without new input from the owner. This asks how the agent's attention
to the owner's messages, and to the owner's first message (the origin, the nearest thing in the window to the goal),
changes with the number of tokens since the owner last spoke.

Written 2026-11-01, before its code.
- **Window:** as `attention_share_long.py`: the first W tokens of each session from its start (W = 16,384 on
  Qwen3-0.6B here), the same seeded sessions, chunks capped at 256 tokens, each its own ChatML turn with its true role.
- **Queries:** up to 32 evenly spaced tokens of each of the agent's prose chunks (not tool calls). Attention rows
  recomputed per layer, averaged over heads, as there.
- **The gap:** tokens from the end of the owner's most recent message before the query (owner messages are the
  spontaneous external chunks; tool results are not the owner). Bins: 0-512, 512-2,048, 2,048-4,096, 4,096-8,192,
  8,192 or more. Queries with no owner message before them are left out.
- **Measured:** over all content keys before the query outside its own chunk (prose, tool calls, tool results and
  owner messages; not the sink or the wrappers), the log-odds lift over the base rate of (a) the share on all owner
  messages and (b) the share on the origin, the owner's first message in the window. For (b) also among queries more
  than 4,096 tokens into the window only, so the origin is always far and the gap is not just position.
- **How it is read,** over the middle layers (6-26 of 28, scaled for other depths), for (a) and (b) separately:
  attention falls as the gap grows if the middle-layer mean lift in the widest gap bin with at least five sessions is
  below the 0-512 bin's at at least two thirds of those layers, and lower in at least two thirds of the sessions
  measured in both bins.
- **Not measured here:** drift. The confirmed departure labels (`study/labels_confirmed.json` on the transcripts
  repository's default branch) are for the next step, with windows ending at each departure.
- **Amended before any result** (the smoke test found archives that begin mid-session, so the owner's first message
  in the window is not the session's): only sessions whose window opens with an owner message are read.

    python probes/attention_gap.py --model Qwen/Qwen3-0.6B --corpus ../dara-session-transcripts/sessions \\
        --run runs/attn-gap-qwen06 --window 16384 --sessions 20
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from substrate.corpus import load_session, sessions as session_paths  # noqa: E402
from substrate.record import Record  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


long = _load("attention_share_long")
share = long.share
SELF, EXT, OTHER = share.SELF, share.EXT, share.OTHER
GAPS = ((0, 512), (512, 2048), (2048, 4096), (4096, 8192), (8192, 10 ** 9))
FAR_FROM_ORIGIN = 4096


def prose_queries(owner, cls, n_chunks):
    """Up to 32 evenly spaced token positions of each agent-prose chunk."""
    out = []
    for i in range(n_chunks):
        pos = np.flatnonzero(owner == i)
        if len(pos) and cls[pos[0]] == SELF:
            out.extend(pos[np.linspace(0, len(pos) - 1, min(long.PER_CHUNK, len(pos))).round().astype(int)].tolist())
    return np.unique(np.array(out, dtype=int))


def gaps(qpos, cls):
    """Tokens since the last owner token before each query; -1 where the owner has not spoken yet."""
    owner_pos = np.flatnonzero(cls == EXT)
    i = np.searchsorted(owner_pos, qpos) - 1
    return np.where(i >= 0, qpos - owner_pos[np.maximum(i, 0)], -1)


def reduce(R, qpos, cls, owner, origin):
    """Per query: attention mass and token counts on content, on owner messages and on the origin (other chunks)."""
    k = np.arange(len(cls))[None, :]
    q = qpos[:, None]
    m = (k <= q) & (owner[None, :] != owner[qpos][:, None]) & np.isin(cls, (SELF, EXT, OTHER))[None, :]
    own = m & (cls[None, :] == EXT)
    org = m & (owner[None, :] == origin)
    return {"content": (R * m).sum(1), "n content": m.sum(1), "owner": (R * own).sum(1), "n owner": own.sum(1),
            "origin": (R * org).sum(1), "n origin": org.sum(1)}


def lift(obs, total, n, n_total, eps=1e-6):
    ok = (n > 0) & (n < n_total) & (total > 0)
    p = np.clip(obs / np.maximum(total, eps), eps, 1 - eps)
    b = np.clip(n / np.maximum(n_total, 1), eps, 1 - eps)
    return np.where(ok, np.log(p / (1 - p)) - np.log(b / (1 - b)), np.nan)


def read_window(model, ids, qpos, cls, owner, origin):
    import torch

    layers = model.model.layers
    found = {}

    def hook(l):
        def pre(mod, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            cos, sin = kwargs["position_embeddings"]
            with torch.no_grad():
                parts = [reduce(R, qpos[sl], cls, owner, origin) for sl, R in long.blocks(mod, hidden, cos, sin, qpos)]
            found[l] = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        return pre

    hooks = [layer.self_attn.register_forward_pre_hook(hook(l), with_kwargs=True) for l, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            model(torch.from_numpy(ids)[None].to(model.device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    return [found[l] for l in range(len(layers))]


def run(model_name, corpus, run_dir, W, n_sessions, device, dtype, revision=None, seed=0, cap=256):
    model, tok = long.load(model_name, device, dtype, revision)
    L = len(model.model.layers)
    assert not getattr(model.config, "use_sliding_window", False), "sliding-window attention is not handled"
    record = Record(Path(run_dir) / "record.jsonl")
    record.append("attention gap, run", "code", model=model_name, revision=revision,
                  commit=getattr(model.config, "_commit_hash", None), layers=L, window=W, cap=cap, gaps=GAPS,
                  per_chunk=long.PER_CHUNK, device=device, dtype=dtype, seed=seed,
                  attention="recomputed rows, head mean", formatting="ChatML per chunk, true roles")
    rng = np.random.default_rng(seed)
    paths = session_paths(corpus)
    paths = [paths[i] for i in rng.permutation(len(paths))]
    rows = []                                         # per query: session, gap, position, owner lift, origin lift
    done = 0
    for path in paths:
        chunks = load_session(path)
        ids, cls, _, owner, kept = share.window(tok, chunks, W, cap, match="prose")
        if len(ids) < 0.9 * W:
            continue
        if not kept or share.kind(kept[0], "prose") != EXT:
            continue                                 # the archive begins mid-session: no origin in the window
        origin = 0
        qpos = prose_queries(owner, cls, len(kept))
        g = gaps(qpos, cls)
        red = read_window(model, ids, qpos, cls, owner, origin)
        own = np.stack([lift(r["owner"], r["content"], r["n owner"], r["n content"]) for r in red], 1)
        org = np.stack([lift(r["origin"], r["content"], r["n origin"], r["n content"]) for r in red], 1)
        name = path.name.split(".")[0]
        for j, p in enumerate(qpos):
            if g[j] >= 0:
                rows.append({"session": name, "gap": int(g[j]), "position": int(p), "chunk": int(owner[p]),
                             "owner": own[j], "origin": org[j]})
        refs = [{"ref": ch["ref"], "provenance": ch["provenance"], "cause": ch["cause"],
                 "kind": int(share.kind(ch, "prose")), "position": int(np.flatnonzero(owner == i)[0]),
                 "truncated": ch["truncated"]} for i, ch in enumerate(kept)]
        record.append("attention gap, window", "code", session=name, tokens=int(len(ids)), queries=int(len(qpos)),
                      origin=origin, chunks=refs)
        done += 1
        print(f"{name}: {len(kept)} chunks, {len(ids)} tokens, {len(qpos)} queries", flush=True)
        if done == n_sessions:
            break
    return summarise(rows, L, record, run_dir)


def table(rows, key, mid, L):
    """By gap bin: queries, sessions, lift by layer, middle mean, per-session middle means."""
    out = []
    for lo, hi in GAPS:
        sel = [r for r in rows if lo <= r["gap"] < hi]
        X = np.array([r[key] for r in sel]) if sel else np.full((0, L), np.nan)
        # chunk means first, then session means, so long chunks and busy sessions do not dominate
        per = {}
        for s in sorted({r["session"] for r in sel}):
            ch = {}
            for r in sel:
                if r["session"] == s:
                    ch.setdefault(r["chunk"], []).append(r[key])
            per[s] = np.nanmean([np.nanmean(v, 0) for v in ch.values()], 0)
        per = {s: v for s, v in per.items() if not np.isnan(v[mid]).all()}
        out.append({"gap": f"{lo}-{hi if hi < 10 ** 9 else 'end'}", "queries": len(sel), "sessions": len(per),
                    "lift by layer": [float(v) for v in (np.nanmean(list(per.values()), 0) if per else np.full(L, np.nan))],
                    "per session": {s: float(np.nanmean(v[mid])) for s, v in per.items()}})
        out[-1]["middle mean"] = float(np.nanmean(np.array(out[-1]["lift by layer"])[mid])) if per else float("nan")
    return out


def reading(tab, mid):
    far = max((b for b in range(len(tab)) if tab[b]["sessions"] >= 5), default=0)
    if far == 0:
        return {"falls": None, "note": "no gap bin beyond 0-512 with five sessions"}
    near, wide = tab[0], tab[far]
    lower = [l for l in mid if wide["lift by layer"][l] < near["lift by layer"][l]]
    both = [s for s in wide["per session"] if s in near["per session"]]
    down = [s for s in both if wide["per session"][s] < near["per session"][s]]
    return {"widest bin": wide["gap"], "layers lower": len(lower), "of layers": len(mid),
            "sessions lower": f"{len(down)}/{len(both)}",
            "falls": len(lower) >= 2 / 3 * len(mid) and len(both) > 0 and len(down) >= 2 / 3 * len(both)}


def summarise(rows, L, record, run_dir):
    mid = list(range(round(6 * L / 28), round(26 * L / 28) + 1))
    far_rows = [r for r in rows if r["position"] > FAR_FROM_ORIGIN]
    report = {"sessions": len({r["session"] for r in rows}), "queries": len(rows), "middle layers": [mid[0], mid[-1]],
              "owner": table(rows, "owner", mid, L), "origin": table(rows, "origin", mid, L),
              "origin, far from it": table(far_rows, "origin", mid, L)}
    report["reading"] = {k: reading(report[k], mid) for k in ("owner", "origin", "origin, far from it")}
    record.append("attention gap, report", "code", **report)
    np.savez(Path(run_dir) / "attn_gap.npz", owner=np.array([r["owner"] for r in rows]),         # derived
             origin=np.array([r["origin"] for r in rows]), gap=np.array([r["gap"] for r in rows]),
             position=np.array([r["position"] for r in rows]), session=np.array([r["session"] for r in rows]),
             chunk=np.array([r["chunk"] for r in rows]))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--corpus", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--window", type=int, default=16384)
    p.add_argument("--sessions", type=int, default=20)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = p.parse_args(argv)
    rep = run(a.model, a.corpus, a.run, a.window, a.sessions, a.device, a.dtype, a.revision)
    print(f"{rep['sessions']} sessions, {rep['queries']} agent-prose queries; middle layers {rep['middle layers']}")
    print("log-odds lift over the base rate among content keys, middle-layer mean, by tokens since the owner spoke")
    for key in ("owner", "origin", "origin, far from it"):
        print(f"  toward {key}:")
        for b in rep[key]:
            print(f"   {b['gap']:>11}: {b['middle mean']:+.2f}  ({b['queries']} queries, {b['sessions']} sessions)")
        print("   reading:", rep["reading"][key])


if __name__ == "__main__":
    main()
