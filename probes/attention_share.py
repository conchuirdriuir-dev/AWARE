"""Attention share on self-generated tokens in logged sessions (a probe, not a step; neither track's criterion).

Question: when the model writes, does its own attention favour its own earlier output over what came from outside,
beyond what the token counts and recency alone would give? If so, capture ("focusing so completely on your own last
thought that nothing from outside gets in", Entity tab) is the model's attention working as built, and the system's
lever, what enters the window, acts on the right mechanism.

Written 2026-11-01, before its first run.
- **Sessions:** the owner's logged Claude Code sessions (`dara-session-transcripts`, private: no text enters this
  repository or the record, only references and numbers), at least 40 chunks each, in a seeded order; the first W
  tokens of chunks of each (W = 4,096 on Qwen3-0.6B in this container), chunks capped at 256 tokens.
- **Formatting:** each chunk wrapped as its own chat turn (ChatML: self as assistant, external as user), so the model
  sees roles as it would in use. The wrapper tokens are their own class.
- **Read:** the model's attention probabilities at every layer (through nnterp, eager attention), averaged over
  heads. Each key token is the sink (the sequence's first token), wrapper, self or external. For a query token, the
  keys in its own chunk are set apart: attending within the chunk being written is not attending to the past.
- **Measured:** for each query token in a self chunk, the self share of its attention over other chunks' content
  (self / (self + external)), against the base rate (self tokens / content tokens among those keys), as a log-odds
  lift; separately for near keys (within 512 tokens) and far keys (further back), so recency cannot make the lift.
  Averaged over each chunk's tokens, then over chunks; by session, by position in the window, and for self chunks
  that repeat earlier text (word 3-gram overlap, as the echo floor) against new ones. External chunks' queries are
  reported the same way, for comparison.
- **How it is read:** the model favours its own output if the far-key lift for self queries is above zero in at
  least two thirds of sessions at at least two thirds of layers. It rises as the session goes on if the last position
  bin's lift exceeds the first's at most layers. Repeats are compared with new only if there are at least 10.
- **Role controls** (written 2026-11-01 after the first run, before their code): `--roles swap` labels self chunks
  as user and external as assistant, text unchanged; `--roles random` labels each chunk by a seeded coin flip,
  independent of who wrote it, and measures the lift twice in one pass: by who wrote the text, and by label (the
  same statistic with assistant-labelled chunks in place of self). The swap keeps the partition, so it tests only
  whether the assistant label carries the self side's larger lift; random labels separate text from label. Read over
  layers 6-26 (where the first run's lift was positive in every session), against the first run's mean there: under
  the swap, the preference follows the text if the self lift keeps at least half of it, the label if under a
  quarter; under random labels, text and label each carry the preference if their lift is above zero in at least
  two thirds of sessions, and the larger mean says which carries more.
- **Content-matched** (written 2026-11-01 after the role controls, before its code): `--match prose`. The role
  controls showed the preference follows the text, not the label; this asks whether it is the agent's own text or
  just its kind (prose against tool output). Only natural language is compared: the agent's prose (its visible text,
  not tool calls) against the owner's messages; tool calls and tool results stay in the window but are left out of
  the share, like wrappers. Each session's window is the W-token stretch holding the most of both (the larger of the
  two counts' minimum; chosen by counts only), over every session with at least 40 chunks. Read over layers 6-26: the
  agent's prose favours its own prose if the far lift is above zero in at least two thirds of sessions measured at
  at least two thirds of those layers; content type alone carried the earlier result if the mean far lift there is
  within 0.2 of zero. The owner's messages are read the same way toward the agent's prose.
- **Not measured:** drift itself. The sessions carry no departure labels yet, so this says whether the mechanism is
  there, not whether it precedes drift.

    python probes/attention_share.py --model Qwen/Qwen3-0.6B --corpus ../dara-session-transcripts/sessions \\
        --run runs/attn-share-qwen06
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from substrate.corpus import load_session, sessions as session_paths  # noqa: E402
from substrate.echo_floor import classify  # noqa: E402
from substrate.record import Record  # noqa: E402

NEAR = 512
SINK, WRAP, SELF, EXT, OTHER = range(5)
MID = range(6, 27)


def label(ch, roles, rng):
    """The chat role a chunk is shown under: by provenance (true), the other way (swap), or a coin flip (random)."""
    own = ch["provenance"] == "self"
    if roles == "swap":
        own = not own
    elif roles == "random":
        own = bool(rng.random() < 0.5)
    return "assistant" if own else "user"


def kind(ch, match):
    """A chunk's class for the share: by provenance, or with --match prose the agent's prose (SELF) against the
    owner's messages (EXT), everything else OTHER."""
    if match != "prose":
        return SELF if ch["provenance"] == "self" else EXT
    if ch["provenance"] == "self":
        return OTHER if ch["text"].startswith("[tool call") else SELF
    return EXT if ch["cause"] == "spontaneous" else OTHER


def start(tok, chunks, W, cap, match):
    """Where the window begins: the first chunk, or with --match prose the start of the W-token stretch holding the
    most of both classes (chosen by counts only)."""
    if match != "prose":
        return 0
    lens = [min(len(tok(c["text"], add_special_tokens=False)["input_ids"]), cap) + 6 for c in chunks]
    kinds = [kind(c, match) for c in chunks]
    best, at, j, n, cs, ce = -1, 0, 0, 0, 0, 0
    for i in range(len(chunks)):
        n += lens[i]; cs += kinds[i] == SELF; ce += kinds[i] == EXT
        while n > W:
            n -= lens[j]; cs -= kinds[j] == SELF; ce -= kinds[j] == EXT; j += 1
        if min(cs, ce) > best:
            best, at = min(cs, ce), j
    return at


def window(tok, chunks: list[dict], W: int, cap: int, roles: str = "true", rng=None, match: str = "provenance"):
    """Token ids of the first chunks that fit in W, with each token's class by who wrote it, its class by the label
    it is shown under (SELF for assistant), and its chunk index (-1 outside content)."""
    ids, cls, lab, owner, kept = [], [], [], [], []
    first = tok("", add_special_tokens=True)["input_ids"]
    for t in first:
        ids.append(t); cls.append(WRAP); lab.append(WRAP); owner.append(-1)
    for ch in chunks:
        body = tok(ch["text"], add_special_tokens=False)["input_ids"][:cap]
        role = label(ch, roles, rng)
        head = tok(f"<|im_start|>{role}\n", add_special_tokens=False)["input_ids"]
        tail = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]
        if len(ids) + len(head) + len(body) + len(tail) > W:
            break
        k = len(kept)
        for t in head:
            ids.append(t); cls.append(WRAP); lab.append(WRAP); owner.append(-1)
        for t in body:
            ids.append(t); cls.append(kind(ch, match))
            lab.append(SELF if role == "assistant" else EXT); owner.append(k)
        for t in tail:
            ids.append(t); cls.append(WRAP); lab.append(WRAP); owner.append(-1)
        kept.append({**ch, "text": tok.decode(body), "truncated": len(body) == cap, "label": role})
    cls[0] = lab[0] = SINK                          # the sequence's first token: the attention sink
    return np.array(ids), np.array(cls), np.array(lab), np.array(owner), kept


def reduce_layer(P, cls, owner):
    """P: (T, T) head-mean attention. Per query: mass on other chunks' self and external content, near and far."""
    T = len(cls)
    q = np.arange(T)[:, None]
    k = np.arange(T)[None, :]
    other = (owner[None, :] != owner[:, None]) | (owner[:, None] < 0)
    near = (q - k) <= NEAR
    out = {}
    for name, c in (("self", SELF), ("ext", EXT)):
        m = (cls[None, :] == c) & other & (k <= q)
        out[f"{name} near"] = (P * (m & near)).sum(1)
        out[f"{name} far"] = (P * (m & ~near)).sum(1)
        out[f"n {name} near"] = (m & near).sum(1)
        out[f"n {name} far"] = (m & ~near).sum(1)
    return out


def lift(obs_s, obs_e, n_s, n_e, eps=1e-6):
    """Log-odds of the observed self share over the base rate; nan where either class is absent."""
    ok = (n_s > 0) & (n_e > 0) & (obs_s + obs_e > 0)
    p = np.clip(obs_s / np.maximum(obs_s + obs_e, eps), eps, 1 - eps)
    b = n_s / np.maximum(n_s + n_e, 1)
    return np.where(ok, np.log(p / (1 - p)) - np.log(np.clip(b, eps, 1 - eps) / (1 - np.clip(b, eps, 1 - eps))), np.nan)


def run(model_name: str, corpus: str, run_dir: str, W: int = 4096, n_sessions: int = 10, cap: int = 256,
        seed: int = 0, revision: str | None = None, roles: str = "true", match: str = "provenance") -> dict:
    import torch
    from nnterp import StandardizedTransformer

    model = StandardizedTransformer(model_name, enable_attention_probs=True, dtype=torch.float32,
                                    device_map="cpu", revision=revision)
    tok = model.tokenizer
    L = model.num_layers
    config = model._model.config
    record = Record(Path(run_dir) / "record.jsonl")
    record.append("attention share, run", "code", model=model_name, revision=revision,
                  commit=getattr(config, "_commit_hash", None), layers=L, window=W, cap=cap, near=NEAR, seed=seed,
                  attention="eager, head mean", formatting="ChatML per chunk", roles=roles,
                  match=match)
    rng = np.random.default_rng(seed)
    label_rng = np.random.default_rng(seed + 1)
    paths = session_paths(corpus)
    paths = [paths[i] for i in rng.permutation(len(paths))]
    rows = []                                       # one per chunk: session, index, provenance, class, position, lifts
    done = 0
    for path in paths:
        chunks = load_session(path)
        if len(chunks) < 40:
            continue
        at = start(tok, chunks, W, cap, match)
        ids, cls, lab, owner, kept = window(tok, chunks[at:], W, cap, roles, label_rng, match)
        kinds = classify([c["text"] for c in kept])
        groupings = {"": cls, "label ": lab}
        lifts = {g + band: np.full((len(kept), L), np.nan) for g in groupings for band in ("near", "far")}
        saved = []
        with torch.no_grad():
            with model.trace(torch.from_numpy(ids)[None]):
                for l in range(L):
                    saved.append(model.attention_probabilities[l].mean(1)[0].save())
        for l, P in enumerate(saved):
            P = P.float().cpu().numpy()
            for g, c in groupings.items():
                r = reduce_layer(P, c, owner)
                for band in ("near", "far"):
                    per_q = lift(r[f"self {band}"], r[f"ext {band}"], r[f"n self {band}"], r[f"n ext {band}"])
                    for i in range(len(kept)):
                        v = per_q[owner == i]
                        v = v[~np.isnan(v)]
                        if len(v):
                            lifts[g + band][i, l] = v.mean()
        del saved
        name = path.name.split(".")[0]
        refs = []
        for i, ch in enumerate(kept):
            pos = int(np.flatnonzero(owner == i)[0])
            refs.append({"ref": ch["ref"], "provenance": ch["provenance"], "label": ch["label"], "cause": ch["cause"],
                         "kind": int(kind(ch, match)), "position": pos, "truncated": ch["truncated"]})
            if kind(ch, match) == OTHER:
                continue
            rows.append({"session": name, "i": i, "label": ch["label"],
                         "provenance": "self" if kind(ch, match) == SELF else "external",
                         "class": kinds[i][0], "position": pos, **{k: v[i] for k, v in lifts.items()}})
        record.append("attention share, window", "code", session=name, start=at, tokens=int(len(ids)), chunks=refs)
        done += 1
        print(f"{name}: {len(kept)} chunks, {len(ids)} tokens", flush=True)
        if done == n_sessions:
            break
    report = summarise(rows, L, W, record, run_dir)
    if roles != "true":
        report["by label"] = by_label(rows, L)
        record.append("attention share, by label", "code", **report["by label"])
    return report


def by_label(rows, L):
    """The same far lift with chunks grouped by the label they were shown under: assistant in place of self."""
    a = [r for r in rows if r["label"] == "assistant"]
    sessions = sorted({r["session"] for r in a})
    X = np.array([r["label far"] for r in a])
    per = np.array([np.nanmean(np.array([r["label far"] for r in a if r["session"] == s]), 0) for s in sessions])
    return {"assistant chunks": len(a), "layers": [
        {"layer": l, "assistant far": float(np.nanmean(X[:, l])), "sessions with assistant far > 0":
         int(np.sum(per[:, l] > 0)), "sessions measured": int(np.sum(~np.isnan(per[:, l])))} for l in range(L)]}


def summarise(rows, L, W, record, run_dir):
    def mean(sel, band):
        X = np.array([r[band] for r in sel]) if sel else np.full((0, L), np.nan)
        return np.nanmean(X, axis=0) if len(X) else np.full(L, np.nan), X

    selfs = [r for r in rows if r["provenance"] == "self"]
    exts = [r for r in rows if r["provenance"] == "external"]
    sessions = sorted({r["session"] for r in rows})
    by_session = np.array([mean([r for r in selfs if r["session"] == s], "far")[0] for s in sessions])
    pos_bins = [(0, 1024), (1024, 2048), (2048, W)]
    report = {"sessions": len(sessions), "self chunks": len(selfs), "external chunks": len(exts),
              "classes": {k: sum(r["class"] == k for r in selfs) for k in ("exact repeat", "near repeat", "new", "between")},
              "layers": []}
    for l in range(L):
        row = {"layer": l,
               "self far": float(mean(selfs, "far")[0][l]), "self near": float(mean(selfs, "near")[0][l]),
               "ext far": float(mean(exts, "far")[0][l]), "ext near": float(mean(exts, "near")[0][l]),
               "sessions with self far > 0": int(np.sum(by_session[:, l] > 0)),
               "sessions measured": int(np.sum(~np.isnan(by_session[:, l])))}
        for lo, hi in pos_bins:
            row[f"self far {lo}-{hi}"] = float(mean([r for r in selfs if lo <= r["position"] < hi], "far")[0][l])
        reps = [r for r in selfs if r["class"] in ("exact repeat", "near repeat")]
        row["self far, repeats"] = float(mean(reps, "far")[0][l]) if reps else float("nan")
        row["self far, new"] = float(mean([r for r in selfs if r["class"] == "new"], "far")[0][l])
        report["layers"].append(row)
    lay = report["layers"]
    favours = [r["layer"] for r in lay if r["sessions measured"] and r["sessions with self far > 0"] >= 2 / 3 * r["sessions measured"]]
    first, last = f"self far {pos_bins[0][0]}-{pos_bins[0][1]}", f"self far {pos_bins[-1][0]}-{pos_bins[-1][1]}"
    rises = [r["layer"] for r in lay if r[last] > r[first]]
    n_rep = report["classes"]["exact repeat"] + report["classes"]["near repeat"]
    report["reading"] = {"favours own output (far)": len(favours) >= 2 / 3 * L, "layers favouring": favours,
                         "rises with position": len(rises) > L / 2, "layers rising": rises,
                         "repeats compared": n_rep >= 10, "repeats": n_rep}
    record.append("attention share, report", "code", **report)
    np.savez(Path(run_dir) / "attn_share.npz",                                  # derived
             far=np.array([r["far"] for r in rows]), near=np.array([r["near"] for r in rows]),
             label_far=np.array([r["label far"] for r in rows]), label=np.array([r["label"] for r in rows]),
             provenance=np.array([r["provenance"] for r in rows]), cls=np.array([r["class"] for r in rows]),
             position=np.array([r["position"] for r in rows]), session=np.array([r["session"] for r in rows]))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--corpus", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--window", type=int, default=4096)
    p.add_argument("--sessions", type=int, default=10)
    p.add_argument("--match", choices=["provenance", "prose"], default="provenance",
                   help="prose: the agent's prose against the owner's messages only (the content-matched comparison)")
    p.add_argument("--roles", choices=["true", "swap", "random"], default="true",
                   help="chat roles by provenance, swapped, or by coin flip (the role controls)")
    a = p.parse_args(argv)
    rep = run(a.model, a.corpus, a.run, W=a.window, n_sessions=a.sessions, revision=a.revision, roles=a.roles,
              match=a.match)
    print(f"{rep['sessions']} sessions; {rep['self chunks']} self chunks, {rep['external chunks']} external; "
          f"self classes {rep['classes']}")
    print("log-odds lift of self share over base rate (0 = no preference); far = more than 512 tokens back")
    print(f"{'layer':>5} {'self far':>9} {'self near':>9} {'ext far':>8} {'ext near':>8} {'sess>0':>7} "
          f"{'0-1k':>6} {'1-2k':>6} {'2k+':>6} {'repeat':>7} {'new':>6}")
    for r in rep["layers"]:
        b = [v for k, v in r.items() if k.startswith("self far ") and "-" in k]
        print(f"{r['layer']:>5} {r['self far']:>9.2f} {r['self near']:>9.2f} {r['ext far']:>8.2f} {r['ext near']:>8.2f} "
              f"{r['sessions with self far > 0']:>3}/{r['sessions measured']:<3} "
              + " ".join(f"{v:>6.2f}" for v in b) + f" {r['self far, repeats']:>7.2f} {r['self far, new']:>6.2f}")
    print("reading:", rep["reading"])
    if "by label" in rep:
        print(f"by label: {rep['by label']['assistant chunks']} assistant-labelled chunks; far lift toward "
              "assistant-labelled keys")
        for r in rep["by label"]["layers"]:
            print(f"{r['layer']:>5} {r['assistant far']:>9.2f} {r['sessions with assistant far > 0']:>3}/"
                  f"{r['sessions measured']}")


if __name__ == "__main__":
    main()
