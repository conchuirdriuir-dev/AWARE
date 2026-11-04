"""The echo floor as a file for the habitat's `read` (PR #28): per layer, in units of D, {"24": 0.31, ...}.

    python -m body.floor_file --model Qwen/Qwen3-8B --corpus .../dara-session-transcripts --run runs/echo-floor-8b-first \\
        --device auto --gpu-memory 5 --draft

Measured, not carried over. The 8B alone-against-alone run of 2026-10-31 read only layers 24, 28, 32 and 35 of the last
third, and its D came from the paraphrase sets before their revision of 2026-11-02; `read` multiplies each floor by D
measured on the current sets. So this re-runs substrate's alone-against-alone measurement (`echo_floor_alone`, the same
10 sessions, seed and caps) with `read`'s own settings: Qwen3-8B, the sink rule "first", the current sets, every layer
of the last third. The floor per layer is the one registered rule for a single number (PLAN.md, Substrate, "Retried
calls"): the midpoint between the 95th percentile of repeats (near and exact) and the 5th percentile of new content,
each chunk's distance to its nearest earlier chunk, in units of D. The corpus is private: only references, classes and
distances are recorded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from substrate.echo_floor import echo_floor_alone
from substrate.measure_d import load_sets
from substrate.record import Record


def floors(dist: np.ndarray, kinds: np.ndarray, layers: list[int]) -> dict:
    """The midpoint rule per layer; dist is chunks x layers in units of D."""
    rep = np.isin(kinds, ["near repeat", "exact repeat"])
    new = kinds == "new"
    return {str(L): float(0.5 * (np.percentile(dist[rep, j], 95) + np.percentile(dist[new, j], 5)))
            for j, L in enumerate(layers)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--corpus", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--sets", default="material/paraphrases.jsonl")
    p.add_argument("--draft", action="store_true")
    p.add_argument("--device", choices=["cuda", "cpu", "auto"])
    p.add_argument("--gpu-memory", type=float)
    p.add_argument("--dtype", choices=["bfloat16", "float32"])
    a = p.parse_args(argv)
    from substrate.states import Reader

    reader = Reader(a.model, device=a.device, dtype=a.dtype, gpu_memory=a.gpu_memory, sink="first")
    n = reader.info["layers"]
    layers = list(range(math.ceil(2 * n / 3), n))                    # the last third, as `read` takes it
    run = Path(a.run)
    record = Record(run / "record.jsonl")
    rep = echo_floor_alone(reader, load_sets(a.sets), layers, record, a.corpus, draft=a.draft, skip_first=True)
    z = np.load(run / "echo_alone.npz")
    out = floors(z["dist"], z["kinds"], layers)
    record.append("floor file", "code", rule="midpoint of repeats' 95th and new's 5th, units of D", floors=out,
                  counts=rep["counts"], sessions=rep["sessions"], sink="first", read_layers=layers)
    (run / "floor.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"counts": rep["counts"], "floors": out}, indent=1))
    for r in rep["layers"]:
        print(f"layer {r['layer']}: D {r['D']:.2f}  near 95th {r['near repeat, 95th']:.3f}  new 5th {r['new, 5th']:.3f}"
              f"  AUC {r['AUC new over near repeat']:.3f}")


if __name__ == "__main__":
    main()
