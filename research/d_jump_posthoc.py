"""Post hoc, after the D jump's verdicts (PLAN.md): which texts carry it, and how it relates to the first-token sink.
Descriptive only: no threshold was fixed in advance, and nothing here is a verdict.

    python -m research.d_jump_posthoc --run runs/d-jump-qwen8b
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from research.d_jump import MODEL
from substrate.measure_d import load_sets
from substrate.record import Record

LAYER = 6
COORDINATES = [2276, 233, 4081]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    a = p.parse_args(argv)
    import torch
    from substrate.states import Reader

    reader = Reader(MODEL, device="auto", gpu_memory=6.5, sink="first")
    rows = []
    with torch.no_grad():
        for s in load_sets("material/paraphrases.jsonl"):
            for t, text in enumerate(s["texts"]):
                ids = reader.tokenizer(text, return_tensors="pt")["input_ids"]
                with reader.model.trace(ids):
                    saved = reader.model.layers_output[LAYER].save()
                T = saved[0].float().cpu().numpy()
                norms = np.linalg.norm(T, axis=1)
                rows.append({"set": s["id"], "text": t, "first": reader.tokenizer.decode(ids[0, 0]),
                             "second": reader.tokenizer.decode(ids[0, 1]), "norm_0": float(norms[0]),
                             "norm_1": float(norms[1]), "median": float(np.median(norms[2:])),
                             "coords_0": np.argsort(-T[0] ** 2)[:3].tolist(),
                             "coords_1": np.argsort(-T[1] ** 2)[:3].tolist(),
                             "share_0": float((T[0, COORDINATES] ** 2).sum() / (T[0] ** 2).sum()),
                             "share_1": float((T[1, COORDINATES] ** 2).sum() / (T[1] ** 2).sum())})
    Record(Path(a.run) / "record.jsonl").append("post hoc", "code", layer=LAYER, coordinates=COORDINATES, rows=rows,
                                                about="the first two tokens of every text")
    groups = defaultdict(list)
    for r in rows:
        groups[r["first"]].append(r)
    print(f"layer {LAYER}, first word: texts, median norm of tokens 0 and 1, their share in {COORDINATES}")
    for word, g in sorted(groups.items(), key=lambda kv: -max(r["norm_1"] for r in kv[1])):
        print(f"{word!r:>14} n={len(g):>2}  norm_0 {np.median([r['norm_0'] for r in g]):>8.0f}  "
              f"norm_1 {np.median([r['norm_1'] for r in g]):>8.0f}  max_1 {max(r['norm_1'] for r in g):>8.0f}  "
              f"share_0 {np.median([r['share_0'] for r in g]):.2f}  share_1 {np.median([r['share_1'] for r in g]):.2f}  "
              f"coords_0 {g[0]['coords_0']}")


if __name__ == "__main__":
    main()
