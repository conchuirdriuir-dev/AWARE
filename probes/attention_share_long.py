"""Attention share at session length: does the agent's own far-back prose outcompete the owner's messages more the
further back they sit? (A probe beside `attention_share.py`, not a step; neither track's criterion.)

`attention_share.py --match prose` found, in 4,096-token windows, that the agent's prose leans toward the owner's
messages within 512 tokens and toward its own earlier prose further back. Real sessions run to tens of thousands of
tokens, and the owner's early messages set the goal. This reads the same statistic by how far back the key sits.

Written 2026-11-01, before its code.
- **Window:** the first W tokens of each session from its start, so the owner's first message is inside (W = 16,384
  on Qwen3-0.6B in this container; 32,768 with the 8B on a GPU). Chunks capped at 256 tokens, each its own ChatML turn
  with its true role, as `attention_share.py`. Sessions with at least W tokens of chunks, in a seeded order.
- **Classes:** the agent's prose (SELF) against the owner's messages (EXT); tool calls and results stay in the window
  but are left out of the share. Queries: up to 32 evenly spaced tokens of each prose chunk and each owner message.
- **Read:** the model runs with its ordinary attention; at each layer the attention rows of the chosen queries are
  recomputed from the layer's own projections and rotary embeddings, in blocks, averaged over heads. On a short
  window this must match the eager probabilities `attention_share.py` reads (checked before the run, `--check`).
- **Measured:** the log-odds lift of the agent-prose share over the base rate, as before, for keys in four distance
  bands: 0-512, 512-2,048, 2,048-8,192 and 8,192 or more tokens back; the query's own chunk is left out.
- **How it is read:** over the middle layers (6-26 of 28, scaled for other depths), the agent's prose outcompetes
  the owner's messages more with distance if the lift for prose queries in the furthest band measured exceeds the
  512-2,048 band's at at least two thirds of those layers, and the furthest band's lift is above zero in at least two
  thirds of sessions measured there. The owner's queries are reported the same way.
- **Not measured:** drift itself (no departure labels yet).

    python probes/attention_share_long.py --model Qwen/Qwen3-0.6B --corpus ../dara-session-transcripts/sessions \\
        --run runs/attn-long-qwen06 --window 16384
    python probes/attention_share_long.py --model Qwen/Qwen3-8B --corpus ../dara-session-transcripts/sessions \\
        --run runs/attn-long-qwen8b --window 32768 --device cuda --dtype bfloat16
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

_spec = importlib.util.spec_from_file_location("attention_share", HERE / "attention_share.py")
share = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(share)
SELF, EXT, OTHER = share.SELF, share.EXT, share.OTHER

BANDS = ((0, 512), (512, 2048), (2048, 8192), (8192, 10 ** 9))
PER_CHUNK = 32
BLOCK = 256


def queries(owner, cls, n_chunks):
    """Up to PER_CHUNK evenly spaced token positions of each prose chunk and owner message."""
    out = []
    for i in range(n_chunks):
        pos = np.flatnonzero(owner == i)
        if len(pos) and cls[pos[0]] in (SELF, EXT):
            out.extend(pos[np.linspace(0, len(pos) - 1, min(PER_CHUNK, len(pos))).round().astype(int)].tolist())
    return np.unique(np.array(out, dtype=int))


def rows(attn, hidden, cos, sin, qpos):
    """Head-mean attention rows (len(qpos), T) of one layer, recomputed from its projections."""
    return np.concatenate([R for _, R in blocks(attn, hidden, cos, sin, qpos)])


def blocks(attn, hidden, cos, sin, qpos):
    """The same rows, BLOCK queries at a time: (slice of qpos, rows)."""
    import torch

    rope = sys.modules[type(attn).__module__].apply_rotary_pos_emb
    T, hd = hidden.shape[1], attn.head_dim
    shape = (1, T, -1, hd)
    q = attn.q_proj(hidden).view(shape)
    k = attn.k_proj(hidden).view(shape)
    if hasattr(attn, "q_norm"):
        q, k = attn.q_norm(q), attn.k_norm(k)
    q, k = rope(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
    groups = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(groups, dim=1)                       # (1, H, T, hd)
    keys = torch.arange(T, device=hidden.device)
    for a in range(0, len(qpos), BLOCK):
        idx = torch.as_tensor(qpos[a:a + BLOCK], device=hidden.device)
        s = (q[:, :, idx] @ k.transpose(-1, -2)).float() * attn.scaling          # (1, H, b, T)
        s = s.masked_fill(keys[None, None, None, :] > idx[None, None, :, None], float("-inf"))
        yield slice(a, a + BLOCK), torch.softmax(s, -1).mean(1)[0].cpu().numpy()


def reduce(R, qpos, cls, owner):
    """Per chosen query: attention mass and token counts on other chunks' prose (SELF) and owner (EXT) content, by
    distance band."""
    T = len(cls)
    k = np.arange(T)[None, :]
    q = qpos[:, None]
    other = (owner[None, :] != owner[qpos][:, None])
    out = {}
    for b, (lo, hi) in enumerate(BANDS):
        band = (q - k >= lo) & (q - k < hi) & (k <= q) & other
        for name, c in (("self", SELF), ("ext", EXT)):
            m = band & (cls[None, :] == c)
            out[(b, name)] = (R * m).sum(1)
            out[(b, "n " + name)] = m.sum(1)
    return out


def load(model_name, device, dtype, revision, attn="sdpa"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, attn_implementation=attn,
                                                 dtype=getattr(torch, dtype), device_map=device)
    model.eval()
    return model, tok


def read_window(model, ids, qpos, cls, owner):
    """One forward pass; each layer's chosen attention rows reduced as the layer runs (nothing T x T is kept)."""
    import torch

    layers = model.model.layers
    found = {}

    def hook(l):
        def pre(mod, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            cos, sin = kwargs["position_embeddings"]
            with torch.no_grad():
                parts = [reduce(R, qpos[sl], cls, owner) for sl, R in blocks(mod, hidden, cos, sin, qpos)]
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


def check(model_name, corpus, W=1024):
    """The recomputed rows against the eager probabilities nnterp reads, on one short window."""
    import torch
    from nnterp import StandardizedTransformer

    ref = StandardizedTransformer(model_name, enable_attention_probs=True, dtype=torch.float32, device_map="cpu")
    model, tok = load(model_name, "cpu", "float32", None)
    path = next(p for p in session_paths(corpus) if len(load_session(p)) >= 40)
    ids, cls, _, owner, kept = share.window(tok, load_session(path), W, 256, match="prose")
    qpos = queries(owner, cls, len(kept))
    saved = []
    with torch.no_grad():
        with ref.trace(torch.from_numpy(ids)[None]):
            for l in range(ref.num_layers):
                saved.append(ref.attention_probabilities[l].mean(1)[0].save())
    got = {}
    layers = model.model.layers

    def hook(l):
        def pre(mod, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            got[l] = rows(mod, hidden, *kwargs["position_embeddings"], qpos)
        return pre

    hs = [layer.self_attn.register_forward_pre_hook(hook(l), with_kwargs=True) for l, layer in enumerate(layers)]
    with torch.no_grad():
        model(torch.from_numpy(ids)[None], use_cache=False)
    for h in hs:
        h.remove()
    err = max(float(np.abs(got[l] - saved[l].float().numpy()[qpos]).max()) for l in range(len(layers)))
    print(f"check: {len(qpos)} query rows x {len(ids)} keys x {len(layers)} layers; max abs difference {err:.2e}")
    return err


def run(model_name, corpus, run_dir, W, n_sessions, device, dtype, revision=None, seed=0, cap=256):
    model, tok = load(model_name, device, dtype, revision)
    L = len(model.model.layers)
    assert not getattr(model.config, "use_sliding_window", False), "sliding-window attention is not handled"
    record = Record(Path(run_dir) / "record.jsonl")
    record.append("attention share long, run", "code", model=model_name, revision=revision,
                  commit=getattr(model.config, "_commit_hash", None), layers=L, window=W, cap=cap, bands=BANDS,
                  per_chunk=PER_CHUNK, device=device, dtype=dtype, seed=seed, match="prose",
                  attention="recomputed rows, head mean", formatting="ChatML per chunk, true roles")
    rng = np.random.default_rng(seed)
    paths = session_paths(corpus)
    paths = [paths[i] for i in rng.permutation(len(paths))]
    out = []                                         # per chunk: session, kind, position, lift (bands x layers)
    done = 0
    for path in paths:
        chunks = load_session(path)
        ids, cls, _, owner, kept = share.window(tok, chunks, W, cap, match="prose")
        if len(ids) < 0.9 * W:
            continue
        qpos = queries(owner, cls, len(kept))
        red = read_window(model, ids, qpos, cls, owner)
        lifts = np.full((len(kept), len(BANDS), L), np.nan)
        for l, r in enumerate(red):
            for b in range(len(BANDS)):
                per_q = share.lift(r[(b, "self")], r[(b, "ext")], r[(b, "n self")], r[(b, "n ext")])
                for i in np.unique(owner[qpos]):
                    v = per_q[owner[qpos] == i]
                    v = v[~np.isnan(v)]
                    if len(v):
                        lifts[i, b, l] = v.mean()
        name = path.name.split(".")[0]
        refs = []
        for i, ch in enumerate(kept):
            c = share.kind(ch, "prose")
            pos = int(np.flatnonzero(owner == i)[0])
            refs.append({"ref": ch["ref"], "provenance": ch["provenance"], "cause": ch["cause"], "kind": int(c),
                         "position": pos, "truncated": ch["truncated"]})
            if c != OTHER:
                out.append({"session": name, "kind": "prose" if c == SELF else "owner", "position": pos,
                            "lift": lifts[i]})
        record.append("attention share long, window", "code", session=name, tokens=int(len(ids)),
                      queries=int(len(qpos)), chunks=refs)
        done += 1
        print(f"{name}: {len(kept)} chunks, {len(ids)} tokens, {len(qpos)} queries", flush=True)
        if done == n_sessions:
            break
    return summarise(out, L, record, run_dir)


def summarise(out, L, record, run_dir):
    mid = list(range(round(6 * L / 28), round(26 * L / 28) + 1))
    report = {"sessions": len({r["session"] for r in out}), "middle layers": [mid[0], mid[-1]], "kinds": {}}
    for kind in ("prose", "owner"):
        sel = [r for r in out if r["kind"] == kind]
        X = np.array([r["lift"] for r in sel])                              # chunks x bands x layers
        sess = np.array([r["session"] for r in sel])
        per = {s: np.nanmean(X[sess == s], 0) for s in np.unique(sess)}     # bands x layers
        bands = []
        for b, (lo, hi) in enumerate(BANDS):
            col = X[:, b, :]
            measured = [s for s in per if not np.isnan(per[s][b, mid]).all()]
            bands.append({"band": f"{lo}-{hi if hi < 10 ** 9 else 'end'}",
                          "chunks measured": int(np.sum(~np.isnan(col).all(1))),
                          "sessions measured": len(measured),
                          "lift by layer": [float(v) for v in np.nanmean(col, 0)],
                          "middle mean": float(np.nanmean(col[:, mid])),
                          "sessions above zero (middle mean)": int(sum(np.nanmean(per[s][b, mid]) > 0 for s in measured))})
        report["kinds"][kind] = {"chunks": len(sel), "bands": bands}
    pb = report["kinds"]["prose"]["bands"]
    far = max(b for b in range(len(BANDS)) if pb[b]["sessions measured"] >= min(3, report["sessions"]))
    grows = [l for l in mid if pb[far]["lift by layer"][l] > pb[1]["lift by layer"][l]]
    report["reading"] = {"furthest band measured": pb[far]["band"],
                         "prose: grows with distance": len(grows) >= 2 / 3 * len(mid), "layers growing": grows,
                         "prose: furthest band above zero in sessions": f"{pb[far]['sessions above zero (middle mean)']}/"
                                                                         f"{pb[far]['sessions measured']}"}
    record.append("attention share long, report", "code", **report)
    np.savez(Path(run_dir) / "attn_long.npz", lift=np.array([r["lift"] for r in out]),            # derived
             kind=np.array([r["kind"] for r in out]), session=np.array([r["session"] for r in out]),
             position=np.array([r["position"] for r in out]))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--corpus", required=True)
    p.add_argument("--run")
    p.add_argument("--window", type=int, default=16384)
    p.add_argument("--sessions", type=int, default=15)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--check", action="store_true", help="compare recomputed rows with eager probabilities, then stop")
    a = p.parse_args(argv)
    if a.check:
        check(a.model, a.corpus)
        return
    rep = run(a.model, a.corpus, a.run, a.window, a.sessions, a.device, a.dtype, a.revision)
    print(f"{rep['sessions']} sessions; middle layers {rep['middle layers']}")
    for kind, k in rep["kinds"].items():
        print(f"{kind} queries ({k['chunks']} chunks): lift toward agent prose by key distance, middle-layer mean")
        for b in k["bands"]:
            print(f"   {b['band']:>12}: {b['middle mean']:+.2f}  ({b['chunks measured']} chunks, "
                  f"{b['sessions above zero (middle mean)']}/{b['sessions measured']} sessions above zero)")
    print("reading:", rep["reading"])


if __name__ == "__main__":
    main()
