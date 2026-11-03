"""The sink by norm, checked (PLAN.md, research track): on the 100 draft texts against the position rule, then on real
chunks the rule has not seen. No corpus text enters the record, only references, positions and norms.

    python -m research.sink_check --run runs/sink-check-qwen8b --corpus ~/dara-session-transcripts/sessions
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from research.d_jump import HIGH, LAYERS, LOW, MODEL, SAVED, jump
from substrate.measure_d import load_sets, measure
from substrate.record import Record
from substrate.states import sink_tokens

MATCH = 0.01            # the largest distance from the position rule, as a share of the state's norm
HOLDS = 0.99            # the share of real chunks with exactly one token left out
PER_SESSION = 30
N_SESSIONS = 10
CAP = 256


def flagged(reader, ids) -> tuple[np.ndarray, np.ndarray]:
    """The token norms at the sink layer, and the positions the rule leaves out."""
    saved = reader._read(ids, [])
    norms = saved[reader.info["sink_layer"]][0].float().norm(dim=-1).cpu().numpy()
    return norms, np.flatnonzero(sink_tokens(norms))


def largest_unflagged_ratio(norms: np.ndarray, where: np.ndarray) -> float:
    rest = np.delete(norms, where)
    return float(rest.max() / np.median(np.sort(norms)[:-1])) if len(rest) else 0.0


def read_texts(positions: list[list[int]], starts_find: list[bool]) -> bool:
    """One token left out of each text: token 1 where the text begins "Find", token 0 elsewhere."""
    return all(p == [1 if find else 0] for p, find in zip(positions, starts_find))


def read_corpus(counts: list[int]) -> tuple[float, bool]:
    share = float(np.mean([c == 1 for c in counts]))
    return share, share >= HOLDS


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--sets", default="material/paraphrases.jsonl")
    a = p.parse_args(argv)
    import torch
    from substrate.corpus import load_session
    from substrate.echo_floor import corpus_paths
    from substrate.states import Reader

    paths = corpus_paths(a.corpus)
    sets = load_sets(a.sets)
    record = Record(Path(a.run) / "record.jsonl")
    reader = Reader(MODEL, device="auto", gpu_memory=6.5)
    tok = reader.tokenizer

    with torch.no_grad():
        texts = [t for s in sets for t in s["texts"]]
        found = [flagged(reader, tok(t, return_tensors="pt")["input_ids"]) for t in texts]
    positions = [w.tolist() for _, w in found]
    starts_find = [t.split()[0] == "Find" for t in texts]
    one = read_texts(positions, starts_find)

    report = measure(reader, sets, LAYERS, record, draft=True)
    rows = {r["layer"]: r for r in report["layers"]}
    D = np.array([rows[x]["D"] for x in LAYERS])
    J = jump(D, LAYERS)

    S = np.load(Path(a.run) / "states.npy")
    saved = np.load(Path(SAVED) / "states.npy")
    saved_layers = json.loads((Path(SAVED) / "record.jsonl").open().readline())["read_layers"]
    now = S[:, :, [LAYERS.index(x) for x in saved_layers]]
    distance = np.linalg.norm(now - saved, axis=3) / np.linalg.norm(saved, axis=3)
    mask = ~np.array(starts_find).reshape(distance.shape[:2])
    worst = float(distance[mask].max())

    verdict = {"one token, where expected": one, "J at most 3": J <= 3, "matches position rule": worst <= MATCH,
               "read twice": report["repeat_max_abs_diff"] == 0}
    record.append("sink check, texts", "code", positions=positions, J=J, worst_share=worst, verdict=verdict,
                  largest_unflagged=max(largest_unflagged_ratio(n, w) for n, w in found), draft=True)
    print(f"texts: {verdict}; J({HIGH}/{LOW}) {J:.2f}; worst distance from the position rule {worst:.4%}")
    print(f"largest unflagged ratio {max(largest_unflagged_ratio(n, w) for n, w in found):.2f}")
    print(f"position counts: {Counter(tuple(x) for x in positions).most_common(5)}")
    for x in LAYERS:
        r = rows[x]
        print(f"  layer {x:>2}: D {r['D']:>8.2f}  between {r['between']:>8.2f}  ratio {r['ratio']:.2f}")

    rng = np.random.default_rng(0)
    order = [paths[i] for i in rng.permutation(len(paths))]
    counts, where_all, ratios, refs, done = [], [], [], [], 0
    with torch.no_grad():
        for path in order:
            chunks = load_session(path)[:300]
            if len(chunks) < 60:
                continue
            done += 1
            for i in sorted(rng.choice(len(chunks), PER_SESSION, replace=False).tolist()):
                ids = tok(chunks[i]["text"], add_special_tokens=False)["input_ids"][:CAP]
                if len(ids) < 3:
                    continue
                norms, w = flagged(reader, torch.tensor([ids]))
                counts.append(len(w))
                where_all.append(w.tolist())
                ratios.append(largest_unflagged_ratio(norms, w))
                refs.append({"ref": chunks[i]["ref"], "tokens": len(ids), "left_out": w.tolist(),
                             "norms_left_out": norms[w].tolist(), "median": float(np.median(norms)),
                             "provenance": chunks[i]["provenance"]})
            if done == N_SESSIONS:
                break
    share, holds = read_corpus(counts)
    where = Counter(x for w in where_all for x in w)
    record.append("sink check, corpus", "code", chunks=refs, share_exactly_one=share, holds=holds)
    print(f"corpus: {len(counts)} chunks of {done} sessions; exactly one left out {share:.2%} (holds: {holds})")
    print(f"  counts left out: {sorted(Counter(counts).items())}")
    print(f"  positions left out: {where.most_common(10)}")
    print(f"  largest unflagged ratio: max {max(ratios):.2f}, 99th {np.percentile(ratios, 99):.2f}, "
          f"median {np.median(ratios):.2f}")


if __name__ == "__main__":
    main()
