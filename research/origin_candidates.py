"""How far apart the candidates for Gate 1's origin sit (PLAN.md, research track). Input to the owner's choice of c, not
a test: nothing is judged. No corpus text enters the record, only references and distances.

    python -m research.origin_candidates --run runs/origin-qwen8b --sink-run runs/sink-check-qwen8b \
        --corpus ~/dara-session-transcripts/sessions
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from research.d_jump import LAYERS, MODEL
from research.sink_check import N_SESSIONS
from substrate.record import Record

REPORT = [8, 12, 24, 28, 32, 35]
CAP = 256


def mirror_shifts(dc: np.ndarray, means: np.ndarray) -> np.ndarray:
    """2|dc.u| for u each direction between two set means: how far moving c by dc moves a mirror's prediction."""
    out = []
    for i, j in combinations(range(len(means)), 2):
        u = means[i] - means[j]
        out.append(2 * abs(dc @ (u / np.linalg.norm(u))))
    return np.array(out)


def earliest_externals(reader, paths, layers) -> tuple[np.ndarray, np.ndarray, list]:
    """Each session's first 5 and first 10 external chunks, read alone and averaged: (sessions, layers, width) twice, with
    the chunk references used. References only: no chunk text leaves here."""
    from substrate.corpus import load_session

    tok = reader.tokenizer
    rng = np.random.default_rng(0)
    order = [paths[i] for i in rng.permutation(len(paths))]
    five, ten, refs = [], [], []
    for path in order:
        chunks = load_session(path)[:300]
        if len(chunks) < 60:
            continue
        external = [c for c in chunks if c["provenance"] == "external"]
        texts, used = [], []
        for c in external:
            ids = tok(c["text"], add_special_tokens=False)["input_ids"][:CAP]
            if len(ids) >= 3:
                texts.append(tok.decode(ids))
                used.append(c["ref"])
            if len(texts) == 10:
                break
        A = reader.states(texts, layers).astype(float)
        five.append(A[:5].mean(0))
        ten.append(A.mean(0))
        refs.append(used)
        if len(five) == N_SESSIONS:
            break
    return np.stack(five), np.stack(ten), refs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--sink-run", required=True)
    p.add_argument("--corpus", required=True)
    a = p.parse_args(argv)
    from substrate.echo_floor import corpus_paths
    from substrate.states import Reader

    paths = corpus_paths(a.corpus)
    record = Record(Path(a.run) / "record.jsonl")
    reader = Reader(MODEL, device="auto", gpu_memory=6.5)
    record.append("run", "code", **reader.info, read_layers=REPORT, sink_run=a.sink_run, draft=True)

    S = np.load(Path(a.sink_run) / "states.npy").astype(float)     # sets, texts, LAYERS, width
    reports = [json.loads(line) for line in (Path(a.sink_run) / "record.jsonl").open()]
    D_rows = [r for r in reports if r["kind"] == "report"][-1]["layers"]
    D = {r["layer"]: r["D"] for r in D_rows}

    five, ten, refs = earliest_externals(reader, paths, REPORT)
    record.append("earliest externals", "code", refs=refs)

    print(f"{'layer':>5} {'D':>7} {'radius':>7} {'zero':>7} {'all5':>7} {'sess med':>9} {'range':>15} {'s-s med':>8} "
          f"{'5v10':>6} {'mir med':>8} {'mir 95':>7}   (all in D)")
    rows = []
    for j, x in enumerate(REPORT):
        states = S[:, :, LAYERS.index(x)]
        d = D[x]
        par = states.reshape(-1, states.shape[-1]).mean(0)
        means = states.mean(1)
        radius = np.sqrt(((states - par) ** 2).sum(-1).mean())
        to_session = np.linalg.norm(five[:, j] - par, axis=1)
        pairs = [np.linalg.norm(five[a, j] - five[b, j]) for a, b in combinations(range(len(five)), 2)]
        within = np.linalg.norm(five[:, j] - ten[:, j], axis=1)
        shifts = np.concatenate([mirror_shifts(five[i, j] - par, means) for i in range(len(five))])
        row = {"layer": x, "D": d, "radius": radius / d, "zero": np.linalg.norm(par) / d,
               "sessions_mean": np.linalg.norm(five[:, j].mean(0) - par) / d,
               "session_median": np.median(to_session) / d, "session_min": to_session.min() / d,
               "session_max": to_session.max() / d, "session_session_median": np.median(pairs) / d,
               "five_vs_ten_median": np.median(within) / d, "mirror_median": np.median(shifts) / d,
               "mirror_95": np.percentile(shifts, 95) / d}
        rows.append({k: float(v) for k, v in row.items()})
        r = rows[-1]
        print(f"{x:>5} {d:>7.1f} {r['radius']:>7.2f} {r['zero']:>7.2f} {r['sessions_mean']:>7.2f} "
              f"{r['session_median']:>9.2f} {r['session_min']:>7.2f}-{r['session_max']:<7.2f} "
              f"{r['session_session_median']:>8.2f} {r['five_vs_ten_median']:>6.2f} {r['mirror_median']:>8.2f} "
              f"{r['mirror_95']:>7.2f}")
    record.append("origin candidates", "code", layers=rows, sessions=len(five), draft=True)


if __name__ == "__main__":
    main()
