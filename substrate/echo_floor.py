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
               skip_first: bool = False) -> dict:
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--run", required=True)
    p.add_argument("--sets", default="material/paraphrases.jsonl")
    p.add_argument("--layers", type=int, nargs="+")
    p.add_argument("--draft", action="store_true", help="allow unfrozen sets; recorded as a draft run")
    p.add_argument("--skip-first", action="store_true", help="leave each sequence's first token (an attention sink) out")
    a = p.parse_args(argv)
    from substrate.states import Reader
    reader = Reader(a.model, revision=a.revision)
    layers = a.layers or default_layers(reader.info["layers"])
    report = echo_floor(reader, load_sets(a.sets), layers, Record(Path(a.run) / "record.jsonl"), draft=a.draft,
                        skip_first=a.skip_first)
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
