"""The echo floor (PLAN.md, substrate): how far an arrival that is the entity's own content, read alone, sits from the
in-context state it repeats, against how far news sits. Measured beside D, per layer.

    python -m substrate.echo_floor --model Qwen/Qwen3-0.6B --run runs/echo-qwen06 --draft    # the dress rehearsal

Sessions are statements from the paraphrase sets, read in one pass. Each arrival (an exact echo of a session's
statement, a paraphrase of one, or a statement from another set) is read alone and measured against the nearest of the
session's in-context states. Distances are reported in units of D. The sets must be frozen unless --draft.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from substrate.measure_d import default_layers, load_sets, measure
from substrate.record import Record

SESSIONS, PER_SESSION = 10, 8


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, dtype=float)[:, None], np.asarray(neg, dtype=float)[None, :]
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def build_sessions(sets: list[dict], n: int = SESSIONS, k: int = PER_SESSION, seed: int = 0) -> list[dict]:
    """Each session: k statements from k different sets (their first paraphrase), the paraphrased echoes (each set's
    second paraphrase), and k news statements from sets the session does not use."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        order = rng.permutation(len(sets))
        used, other = order[:k], order[k:2 * k]
        out.append({"sets": [sets[i]["id"] for i in used],
                    "chunks": [sets[i]["texts"][0] for i in used],
                    "paraphrases": [sets[i]["texts"][1] for i in used],
                    "news": [sets[i]["texts"][0] for i in other]})
    return out


def nearest(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """For each row of X (arrivals x layers x width), its distance to the nearest of S (chunks x layers x width), per
    layer: arrivals x layers."""
    return np.min(np.linalg.norm(X[:, None] - S[None], axis=3), axis=1)


def echo_floor(reader, sets: list[dict], layers: list[int], record: Record, draft: bool = False,
               skip_first: bool = True) -> dict:
    d_report = measure(reader, sets, layers, record, draft=draft, skip_first=skip_first)
    D = np.array([row["D"] for row in d_report["layers"]])
    sessions = build_sessions(sets)
    record.append("echo sessions", "code", sessions=sessions, layers=layers, draft=draft)
    dist = {"exact echo": [], "paraphrased echo": [], "news": []}
    positions = []
    for sess in sessions:
        chunks = [c + "\n\n" for c in sess["chunks"]]
        S = reader.states_in_context(chunks, layers, skip_first)
        dist["exact echo"].append(nearest(reader.states(sess["chunks"], layers, skip_first), S))
        dist["paraphrased echo"].append(nearest(reader.states(sess["paraphrases"], layers, skip_first), S))
        dist["news"].append(nearest(reader.states(sess["news"], layers, skip_first), S))
        positions.extend(range(len(chunks)))
    dist = {k: np.concatenate(v) / D for k, v in dist.items()}          # arrivals x layers, in units of D
    rows = []
    for j, layer in enumerate(layers):
        ex, pa, nw = dist["exact echo"][:, j], dist["paraphrased echo"][:, j], dist["news"][:, j]
        rows.append({"layer": layer, "D": float(D[j]),
                     "exact echo, median": float(np.median(ex)), "exact echo, 95th": float(np.percentile(ex, 95)),
                     "paraphrased echo, median": float(np.median(pa)),
                     "news, median": float(np.median(nw)), "news, 5th": float(np.percentile(nw, 5)),
                     "AUC news over exact echo": auc(nw, ex), "AUC news over paraphrased echo": auc(nw, pa),
                     "exact echo against position, correlation": float(np.corrcoef(positions, ex)[0, 1])})
    report = record.append("echo floor", "code", layers=rows, draft=draft, unit="D per layer", skip_first=skip_first,
                           separable_at=[r["layer"] for r in rows if r["AUC news over exact echo"] >= 0.9])
    np.savez(record.path.parent / "echo.npz", **{k.replace(" ", "_"): v for k, v in dist.items()})   # derived
    return report


BINS = ((0, 8), (8, 24), (24, 56))


def build_long_sessions(sets: list[dict], n: int = 4, held_out: int = 6, seed: int = 0) -> list[dict]:
    """Each long session: the four paraphrases other than the second of all but `held_out` sets, shuffled (56
    statements from 20 sets); the held-out sets supply news, each session set's second paraphrase its paraphrased
    echo."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        order = rng.permutation(len(sets))
        inside, outside = order[held_out:], order[:held_out]
        chunks = [(sets[i]["id"], sets[i]["texts"][j]) for i in inside for j in (0, 2, 3, 4)]
        chunks = [chunks[k] for k in rng.permutation(len(chunks))]
        out.append({"chunks": [c for _, c in chunks], "sets": [i for i, _ in chunks],
                    "paraphrases": [sets[i]["texts"][1] for i in inside],
                    "news": [t for i in outside for t in sets[i]["texts"]]})
    return out


def echo_floor_long(reader, sets: list[dict], layers: list[int], record: Record, draft: bool = False,
                    skip_first: bool = True) -> dict:
    """Does the echo floor hold as a session grows? Exact echoes by their original's position, against news measured
    against the whole session."""
    d_report = measure(reader, sets, layers, record, draft=draft, skip_first=skip_first)
    D = np.array([row["D"] for row in d_report["layers"]])
    sessions = build_long_sessions(sets)
    record.append("echo sessions, long", "code", sessions=sessions, layers=layers, draft=draft)
    echo, news, pos = [], [], []
    for sess in sessions:
        S = reader.states_in_context([c + "\n\n" for c in sess["chunks"]], layers, skip_first)
        alone = reader.states(sess["chunks"], layers, skip_first)
        own = np.linalg.norm(alone - S, axis=2)                              # each echo against its own original
        echo.append(np.minimum(own, nearest(alone, S)))
        news.append(nearest(reader.states(sess["news"], layers, skip_first), S))
        pos.extend(range(len(sess["chunks"])))
    echo, news, pos = np.concatenate(echo) / D, np.concatenate(news) / D, np.array(pos)
    rows = []
    for j, layer in enumerate(layers):
        row = {"layer": layer, "D": float(D[j]), "news, median": float(np.median(news[:, j])),
               "news, 5th": float(np.percentile(news[:, j], 5))}
        for lo, hi in BINS:
            ex = echo[(pos >= lo) & (pos < hi), j]
            row[f"echo {lo + 1}-{hi}, median"] = float(np.median(ex))
            row[f"echo {lo + 1}-{hi}, 95th"] = float(np.percentile(ex, 95))
            row[f"AUC {lo + 1}-{hi}"] = auc(news[:, j], ex)
        rows.append(row)
    last = f"AUC {BINS[-1][0] + 1}-{BINS[-1][1]}"
    report = record.append("echo floor, long", "code", layers=rows, draft=draft, unit="D per layer",
                           skip_first=skip_first, holds_at=[r["layer"] for r in rows if r[last] >= 0.9])
    np.savez(record.path.parent / "echo_long.npz", echo=echo, news=news, position=pos)      # derived
    return report


def _bins(W: int) -> list[tuple[int, int]]:
    edges, e = [0, 512], 512
    while e * 2 < W:
        e *= 2
        edges.append(e)
    edges.append(W)
    return list(zip(edges[:-1], edges[1:]))


def echo_floor_corpus(reader, sets: list[dict], layers: list[int], record: Record, corpus: str, window: int = 4096,
                      n_sessions: int = 10, news_per: int = 50, draft: bool = False, skip_first: bool = True,
                      seed: int = 0, cap: int = 256) -> dict:
    """The echo floor on logged sessions: each window's self chunks read alone against the nearest in-context state,
    by token position, against self chunks from other sessions. No corpus text is recorded, only references."""
    from substrate.corpus import load_session, sessions as session_paths

    d_report = measure(reader, sets, layers, record, draft=draft, skip_first=skip_first)
    D = np.array([row["D"] for row in d_report["layers"]])
    tok = reader.tokenizer

    def capped(text):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return tok.decode(ids[:cap]), len(ids) > cap, min(len(ids), cap)

    rng = np.random.default_rng(seed)
    paths = [session_paths(corpus)[i] for i in rng.permutation(len(session_paths(corpus)))]
    loaded = {p.name.split(".")[0]: load_session(p) for p in paths}
    chosen = [k for k in loaded if len(loaded[k]) >= 40][:n_sessions]
    bins = _bins(window)
    echo, news, pos = [], [], []
    windows = []
    for name in chosen:
        texts, refs, offsets, selfs, n_tok = [], [], [], [], 0
        for ch in loaded[name]:
            text, cut, n = capped(ch["text"])
            if n_tok + n + 2 > window:
                break
            texts.append(text)
            refs.append({"ref": ch["ref"], "provenance": ch["provenance"], "cause": ch["cause"], "truncated": cut,
                         "position": n_tok})
            offsets.append(n_tok)
            if ch["provenance"] == "self":
                selfs.append(len(texts) - 1)
            n_tok += n + 2
        S = reader.states_in_context([t + "\n\n" for t in texts], layers, skip_first)
        alone = reader.states([texts[i] for i in selfs], layers, skip_first)
        echo.append(nearest(alone, S))
        pos.extend(offsets[i] for i in selfs)
        others = [ch for k in chosen if k != name for ch in loaded[k] if ch["provenance"] == "self"]
        pick = [others[i] for i in rng.choice(len(others), size=min(news_per, len(others)), replace=False)]
        news.append(nearest(reader.states([capped(ch["text"])[0] for ch in pick], layers, skip_first), S))
        windows.append({"session": name, "tokens": n_tok, "chunks": refs, "news refs": [ch["ref"] for ch in pick]})
    record.append("echo corpus windows", "code", windows=windows, window=window, cap=cap, draft=draft)
    echo, news, pos = np.concatenate(echo) / D, np.concatenate(news) / D, np.array(pos)
    rows = []
    for j, layer in enumerate(layers):
        nmed = float(np.median(news[:, j]))
        row = {"layer": layer, "D": float(D[j]), "news, median": nmed, "news, 5th": float(np.percentile(news[:, j], 5))}
        for lo, hi in bins:
            ex = echo[(pos >= lo) & (pos < hi), j]
            if len(ex):
                row[f"echo {lo}-{hi}, median"] = float(np.median(ex))
                row[f"echo {lo}-{hi}, 95th"] = float(np.percentile(ex, 95))
                row[f"echo {lo}-{hi}, of news median"] = float(np.median(ex) / nmed)
                row[f"AUC {lo}-{hi}"] = auc(news[:, j], ex)
        rows.append(row)
    last = f"AUC {bins[-1][0]}-{bins[-1][1]}"
    report = record.append("echo floor, corpus", "code", layers=rows, bins=bins, window=window, draft=draft,
                           skip_first=skip_first, holds_at=[r["layer"] for r in rows if r.get(last, 0) >= 0.9],
                           echoes=int(len(pos)), news=int(len(news)))
    np.savez(record.path.parent / "echo_corpus.npz", echo=echo, news=news, position=pos)      # derived
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--run", required=True)
    p.add_argument("--sets", default="material/paraphrases.jsonl")
    p.add_argument("--layers", type=int, nargs="+")
    p.add_argument("--draft", action="store_true", help="allow unfrozen sets; recorded as a draft run")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"], help="auto: split the model across GPU and RAM")
    p.add_argument("--gpu-memory", type=float, help="with --device auto: GiB of GPU memory to use (e.g. 6.5 on 8 GB)")
    p.add_argument("--dtype", choices=["bfloat16", "float32"])
    p.add_argument("--quantize", choices=["4bit", "8bit"], help="a quick check only: quantisation adds noise to D")
    p.add_argument("--long", action="store_true", help="long sessions: does the floor hold as a session grows?")
    p.add_argument("--corpus", help="a directory of logged session archives (S*.tar.gz): the floor at real lengths")
    p.add_argument("--window", type=int, default=4096, help="with --corpus: tokens of each session read in one pass")
    p.add_argument("--keep-first", action="store_true",
                   help="keep each sequence's first token; by default it is left out (an attention sink)")
    a = p.parse_args(argv)
    from substrate.states import Reader
    reader = Reader(a.model, revision=a.revision, device=a.device, dtype=a.dtype, gpu_memory=a.gpu_memory,
                    quantize=a.quantize)
    layers = a.layers or default_layers(reader.info["layers"])
    if a.corpus:
        rep = echo_floor_corpus(reader, load_sets(a.sets), layers, Record(Path(a.run) / "record.jsonl"), a.corpus,
                                window=a.window, draft=a.draft, skip_first=not a.keep_first)
        print(f"{reader.info['model']} ({reader.info['dtype']}){'  DRAFT' if a.draft else ''}; corpus windows of "
              f"{a.window} tokens; {rep['echoes']} echoes, {rep['news']} news; units of D (of news's median)")
        for r in rep["layers"]:
            cells = [f"{lo}-{hi}: {r[f'echo {lo}-{hi}, median']:.2f} ({r[f'echo {lo}-{hi}, of news median']:.2f}) "
                     f"AUC {r[f'AUC {lo}-{hi}']:.2f}" for lo, hi in rep["bins"] if f"AUC {lo}-{hi}" in r]
            print(f"layer {r['layer']:>3}  D {r['D']:.3f}  news {r['news, median']:.2f} (5th {r['news, 5th']:.2f})  "
                  + " | ".join(cells))
        print("holds in the last position bin (AUC >= 0.9) at layers:", rep["holds_at"])
        return
    if a.long:
        rep = echo_floor_long(reader, load_sets(a.sets), layers, Record(Path(a.run) / "record.jsonl"), draft=a.draft,
                              skip_first=not a.keep_first)
        print(f"{reader.info['model']} ({reader.info['dtype']}){'  DRAFT' if a.draft else ''}; long sessions, units of D")
        print(f"{'layer':>5} {'D':>8} {'news med':>9} {'news 5':>7} " + " ".join(
            f"{'echo ' + str(lo + 1) + '-' + str(hi):>12} {'AUC':>5}" for lo, hi in BINS))
        for r in rep["layers"]:
            print(f"{r['layer']:>5} {r['D']:>8.3f} {r['news, median']:>9.2f} {r['news, 5th']:>7.2f} " + " ".join(
                f"{r[f'echo {lo + 1}-{hi}, median']:>6.2f}/{r[f'echo {lo + 1}-{hi}, 95th']:<5.2f} {r[f'AUC {lo + 1}-{hi}']:>5.2f}"
                for lo, hi in BINS))
        print("holds in the last bin (AUC >= 0.9) at layers:", rep["holds_at"])
        return
    report = echo_floor(reader, load_sets(a.sets), layers, Record(Path(a.run) / "record.jsonl"), draft=a.draft,
                        skip_first=not a.keep_first)
    print(f"{reader.info['model']} ({reader.info['dtype']}){'  DRAFT' if a.draft else ''}; distances in units of D")
    print(f"{'layer':>5} {'D':>8} {'echo med':>9} {'echo 95':>8} {'para med':>9} {'news med':>9} {'news 5':>7} "
          f"{'AUC n>e':>8} {'AUC n>p':>8} {'pos r':>6}")
    for r in report["layers"]:
        print(f"{r['layer']:>5} {r['D']:>8.3f} {r['exact echo, median']:>9.2f} {r['exact echo, 95th']:>8.2f} "
              f"{r['paraphrased echo, median']:>9.2f} {r['news, median']:>9.2f} {r['news, 5th']:>7.2f} "
              f"{r['AUC news over exact echo']:>8.2f} {r['AUC news over paraphrased echo']:>8.2f} "
              f"{r['exact echo against position, correlation']:>6.2f}")
    print("separable (news over exact echo at AUC >= 0.9) at layers:", report["separable_at"])


if __name__ == "__main__":
    main()
