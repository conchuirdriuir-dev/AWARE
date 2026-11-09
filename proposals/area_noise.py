"""The paraphrase spread that sets the noise of area's toy power check (proposals/area-sense.md, section 4).

    python -m proposals.area_noise --run runs/area-noise-8b --device auto --gpu-memory 5

Reads the frozen paraphrase sets (`material/paraphrases.jsonl`, 20 sets of 5) with the 8B, as `stepm1 read` reads a
chunk: each text on its own, mean over its tokens, the first token left out (sink "first"), bf16, every layer of the
last third. Per layer, fixed before the run:

- **within:** the spread of a paraphrase about its set's mean, as a root-mean-square distance, corrected for the mean
  being taken from 5 members (x sqrt(5/4));
- **between:** the median distance between two sets' means, corrected for the means' own noise (median of the squared
  distances less 2 within^2 / 5, then the root);
- **ratio = within / between.** The toy places four contents at distance 1 and adds a visit's noise of norm `ratio`.

Writes the per-layer numbers, stamped with the model, revision, precision and this repository's commit, to
`<run>/noise.json` and copies it to `material/area_noise_8b.json`; the states go to `<run>/states.npy` (a cache,
rebuilt from the texts). It reads no habitat text and decides no sense.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np


def spread(S: np.ndarray, groups: np.ndarray) -> dict:
    """S: texts x width at one layer; groups: the set of each text. Returns within, between, ratio."""
    ids = np.unique(groups)
    means = np.stack([S[groups == g].mean(0) for g in ids])
    k = np.array([(groups == g).sum() for g in ids])
    res = np.concatenate([S[groups == g] - means[i] for i, g in enumerate(ids)])
    within = float(np.sqrt((res ** 2).sum(1).mean() * k.mean() / (k.mean() - 1)))
    iu = np.triu_indices(len(ids), 1)
    d2 = ((means[:, None] - means[None]) ** 2).sum(-1)[iu] - 2 * within ** 2 / k.mean()
    between = float(np.sqrt(max(np.median(d2), 1e-12)))
    return {"within": within, "between": between, "ratio": within / between}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--revision")
    p.add_argument("--run", required=True)
    p.add_argument("--sets", default="material/paraphrases.jsonl")
    p.add_argument("--device", default="auto")
    p.add_argument("--gpu-memory", type=float)
    p.add_argument("--dtype", default="bfloat16")
    a = p.parse_args(argv)
    from substrate.measure_d import load_sets
    from substrate.states import Reader

    sets = load_sets(a.sets)
    texts = [t for s in sets for t in s["texts"]]
    groups = np.array([i for i, s in enumerate(sets) for _ in s["texts"]])
    reader = Reader(a.model, revision=a.revision, device=a.device, dtype=a.dtype, gpu_memory=a.gpu_memory,
                    sink="first")
    n = reader.info["layers"]
    layers = list(range(math.ceil(2 * n / 3), n))
    S = np.concatenate([reader.states(texts[b:b + 8], layers) for b in range(0, len(texts), 8)])  # texts x layers x w
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    np.save(run / "states.npy", S.astype(np.float32))
    per = {str(L): spread(S[:, j].astype(np.float64), groups) for j, L in enumerate(layers)}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"layers": per, "mean_ratio": float(np.mean([v["ratio"] for v in per.values()])),
           "max_ratio": float(np.max([v["ratio"] for v in per.values()])),
           "sets": a.sets, "n_sets": len(sets), "n_texts": len(texts), "groups": groups.tolist(),
           "reader": {k: reader.info[k] for k in ("model", "revision", "dtype", "load", "sink")}, "commit": commit}
    (run / "noise.json").write_text(json.dumps(out, indent=1, default=str))
    Path("material/area_noise_8b.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("mean_ratio", "max_ratio")}))
    for L, v in per.items():
        print(f"layer {L}: within {v['within']:.3f}  between {v['between']:.3f}  ratio {v['ratio']:.3f}")


if __name__ == "__main__":
    main()
