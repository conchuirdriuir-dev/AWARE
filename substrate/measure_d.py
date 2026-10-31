"""Measuring D on the real model (PLAN.md, step 1): the spread of states within paraphrase sets, per layer, beside the
spread between sets. The layer proposed is the one with the largest between-to-within ratio; the owner chooses.

    python -m substrate.measure_d --model meta-llama/Llama-3.1-8B-Instruct --run runs/d-llama
    python -m substrate.measure_d --model Qwen/Qwen3-8B --run runs/d-qwen --layers 8 16 24
    python -m substrate.measure_d --model Qwen/Qwen3-0.6B --run runs/d-plumbing --draft      # before the sets are frozen

The sets must be frozen by the owner; --draft runs on unfrozen ones and says so in the record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from substrate.geometry import between, spread
from substrate.record import Record


def load_sets(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def default_layers(n: int) -> list[int]:
    return sorted(set(range(0, n, max(1, n // 8))) | {n - 1})


def measure(reader, sets: list[dict], layers: list[int], record: Record, draft: bool = False,
            skip_first: bool = True) -> dict:
    if not draft and not all(s.get("frozen") for s in sets):
        raise SystemExit("the paraphrase sets are not frozen; the owner freezes them, or run with --draft")
    blob = json.dumps(sets, sort_keys=True).encode()
    record.append("run", "code", **reader.info, read_layers=layers, sets_sha256=hashlib.sha256(blob).hexdigest(),
                  draft=draft, pooling="mean over non-special tokens" + (", first token left out" if skip_first else ""))
    for s in sets:
        record.append("set", s.get("by", "person"), id=s["id"], texts=s["texts"], frozen=bool(s.get("frozen")))
    per_set = [reader.states(s["texts"], layers, skip_first) for s in sets]          # each: texts x layers x width
    again = reader.states(sets[0]["texts"][:1], layers, skip_first)
    repeat = float(np.abs(again[0] - per_set[0][0]).max())
    rows = []
    for j, layer in enumerate(layers):
        groups = [S[:, j, :].astype(float) for S in per_set]
        D, B = spread(groups), between(groups)
        rows.append({"layer": layer, "D": D, "between": B, "ratio": B / D if D > 0 else float("inf")})
    best = max(rows, key=lambda r: r["ratio"])["layer"]
    report = record.append("report", "code", layers=rows, proposed_layer=best, repeat_max_abs_diff=repeat)
    np.save(record.path.parent / "states.npy", np.stack([S for S in per_set]))   # derived: rebuilt by rerunning
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision")
    p.add_argument("--run", required=True, help="the run directory; its record.jsonl is appended to")
    p.add_argument("--sets", default="material/paraphrases.jsonl")
    p.add_argument("--layers", type=int, nargs="+")
    p.add_argument("--draft", action="store_true", help="allow unfrozen sets; recorded as a draft run")
    a = p.parse_args(argv)
    from substrate.states import Reader
    reader = Reader(a.model, revision=a.revision)
    layers = a.layers or default_layers(reader.info["layers"])
    report = measure(reader, load_sets(a.sets), layers, Record(Path(a.run) / "record.jsonl"), draft=a.draft)
    print(f"{reader.info['model']}  ({reader.info['dtype']}){'  DRAFT sets' if a.draft else ''}")
    print(f"{'layer':>6} {'D':>10} {'between':>10} {'ratio':>7}")
    for r in report["layers"]:
        print(f"{r['layer']:>6} {r['D']:>10.4f} {r['between']:>10.4f} {r['ratio']:>7.2f}")
    print(f"proposed layer: {report['proposed_layer']}   same text read twice, max difference: "
          f"{report['repeat_max_abs_diff']:.2e}")


if __name__ == "__main__":
    main()
