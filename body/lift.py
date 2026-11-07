"""Lift v1: a trained lift for shape (PLAN.md, "Shape, a trained lift: Lift v1"; drafted with the owner 2026-11-07).

A fixed linear map per read layer, from the model's states to K whitened dimensions, learned once from a domain-free
corpus (MASC, never the habitat) and frozen before any reading of a deciding pool. Shape is then coded in it, by
source, as before (`body/stepm1.shape_by_source`).

    python -m body.lift build --model Qwen/Qwen3-0.6B --corpus DIR --run runs/lift-06 --draft
    python -m body.lift build --model Qwen/Qwen3-8B   --corpus DIR --run runs/lift-8b

`build` appends everything it does to `<run>/record.jsonl` and leaves the bases in `<run>/lift.npz`, a cache derived
from the record (the corpus's file hashes, the chunking seed, the model and revision, the code's commit).

- **Chunks:** the corpus's texts in sorted path order, each read as one token stream (Qwen3's tokenizer), cut into
  chunks whose lengths are drawn from the habitat's chunk-length distribution (`material/lift_chunk_lengths.json`,
  token counts only), each cut snapped to a sentence or line end within 25% of its drawn length if there is one.
- **Splits by document:** train 80%, held-out A 10% (chooses K), held-out B 10% (checks the lift), by a seeded
  permutation of the documents; then at most N_TRAIN, N_HELD and N_HELD chunks drawn from each.
- **The fit, per read layer:** `goalspace.formula.Fit` on the train chunks (PPCA, the noise floored at D^2/d, D from
  the paraphrase sets in the same run). **K** = the rank that codes held-out A in the fewest bits (0 to KMAX_LIFT).
- **The check, per layer:** held-out B coded at K in fewer bits than at k_v0, the rank Lift v0's own rule
  (`Fit.choose_k`, the formula's KMAX) picks on the same train chunks. A layer that fails is recorded; `ok` is True
  only if every read layer passes, and a lift that is not ok is not used.
- **The map:** z = diag(lam_K)^(-1/2) V_K (x - mu), whitened. The lift's own floor, for shape coded in it, is
  D_lift^2 / K, D_lift the paraphrase sets' spread in the lifted space: the floor's definition, in the new space.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from body.machine import record_machine
from goalspace.formula import KMAX, Fit
from substrate.record import Record

SEED = 20261107
N_TRAIN, N_HELD = 10_000, 1_000
SPLIT = (0.8, 0.1, 0.1)
KMAX_LIFT = 512
SNAP = 0.25
LENGTHS = Path(__file__).resolve().parent.parent / "material" / "lift_chunk_lengths.json"
_END = re.compile(r"[.!?]['\")\]]*$|\n")


def corpus_files(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*.txt") if not p.name.startswith("."))


def file_hashes(files: list[Path], root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def length_sampler(seed: int):
    counts = json.loads(LENGTHS.read_text())["counts"]
    ks = np.array([int(k) for k in counts]); ps = np.array(list(counts.values()), dtype=float)
    ps /= ps.sum()
    rng = np.random.default_rng(seed)
    return lambda: int(rng.choice(ks, p=ps))


def cut(text: str, tok, draw) -> list[str]:
    """One document's token stream cut into chunks of drawn lengths, each snapped to a sentence or line end within
    SNAP of its length if there is one."""
    ids = tok(text, add_special_tokens=False)["input_ids"]
    pieces = [tok.decode([i]) for i in ids]
    ends = [j + 1 for j, p in enumerate(pieces) if _END.search(p)]          # sorted
    out, i = [], 0
    while i < len(ids):
        L = max(1, draw())
        lo, hi = i + max(1, int(L * (1 - SNAP))), i + int(math.ceil(L * (1 + SNAP)))
        near = ends[bisect.bisect_left(ends, lo):bisect.bisect_right(ends, hi)]
        j = min(near, key=lambda e: abs(e - (i + L))) if near else min(i + L, len(ids))
        chunk = tok.decode(ids[i:j]).strip()
        if chunk:
            out.append(chunk)
        i = j
    return out


def chunks(root: Path, tok, seed: int = SEED) -> dict:
    """{"train": [...], "held_a": [...], "held_b": [...]}: chunks by split, documents split before cutting."""
    files = corpus_files(root)
    order = np.random.default_rng(seed).permutation(len(files))
    n = len(files)
    a, b = int(round(SPLIT[0] * n)), int(round((SPLIT[0] + SPLIT[1]) * n))
    split = {"train": [files[i] for i in order[:a]], "held_a": [files[i] for i in order[a:b]],
             "held_b": [files[i] for i in order[b:]]}
    draw = length_sampler(seed + 1)
    out = {}
    for name, fs in split.items():
        cs = [c for f in fs for c in cut(f.read_text(encoding="utf-8", errors="replace"), tok, draw)]
        want = N_TRAIN if name == "train" else N_HELD
        rng = np.random.default_rng([seed, len(name)])
        keep = sorted(rng.choice(len(cs), size=min(want, len(cs)), replace=False))
        out[name] = [cs[i] for i in keep]
    out["documents"] = {k: [str(p.relative_to(root)) for p in v] for k, v in split.items()}
    return out


def held_bits(fit: Fit, H: np.ndarray, kmax: int) -> np.ndarray:
    """Total bits of the rows H under the fit, for every k from 0 to kmax (vectorised `Fit.bits`)."""
    R = np.asarray(H, dtype=np.float64) - fit.mu
    P = R @ fit.V[:kmax].T
    rr = np.sum(R * R, 1)
    out = np.empty(kmax + 1)
    for k in range(kmax + 1):
        kk, lam_k, sigma2 = fit.params(k)
        p = P[:, :kk]
        pp = np.sum(p * p, 1)
        maha = np.sum(p * p / lam_k, 1) + (rr - pp) / sigma2
        logdet = float(np.log(lam_k).sum()) + (fit.d - kk) * math.log(sigma2)
        out[k] = float(np.sum(0.5 * (fit.d * math.log(2 * math.pi) + logdet + maha) / math.log(2)))
    return out


def fit_layer(train: np.ndarray, held_a: np.ndarray, held_b: np.ndarray, floor_var: float) -> dict:
    fit = Fit(train, floor_var)
    top = max(0, min(KMAX_LIFT, fit.n - 2, len(fit.lam)))
    a = held_bits(fit, held_a, top)
    K = int(np.argmin(a))
    k0 = fit.choose_k(KMAX)
    b = held_bits(fit, held_b, max(K, k0))
    kk, lam_k, _ = fit.params(K)
    return {"K": kk, "k_v0": int(k0), "held_a_bits_per_chunk": float(a[K] / len(held_a)),
            "held_b_bits_per_chunk": {"K": float(b[K] / len(held_b)), "k_v0": float(b[k0] / len(held_b))},
            "ok": bool(kk > 0 and b[K] < b[k0]),
            "mu": fit.mu, "V": fit.V[:kk], "scale": 1.0 / np.sqrt(lam_k)}


def apply(lift: dict, layer_index: int, X: np.ndarray) -> np.ndarray:
    """Rows of one read layer's states mapped into the lift: whitened, K dimensions."""
    mu, V, sc = lift["mu"][layer_index], lift["V"][layer_index], lift["scale"][layer_index]
    return ((np.asarray(X, dtype=np.float64) - mu) @ V.T) * sc


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    n = int(z["layers"].shape[0])
    return {"layers": [int(x) for x in z["layers"]], "ok": bool(z["ok"]),
            "mu": [z[f"mu{j}"] for j in range(n)], "V": [z[f"V{j}"] for j in range(n)],
            "scale": [z[f"scale{j}"] for j in range(n)], "floor": [float(x) for x in z["floor"]]}


def cmd_build(a) -> None:
    from substrate.echo_floor import d_resumable
    from substrate.geometry import spread
    from substrate.measure_d import load_sets
    from substrate.states import Reader

    run, root = Path(a.run), Path(a.corpus)
    record = Record(run / "record.jsonl")
    record_machine(record, "lift build")
    started = time.time()
    files = corpus_files(root)
    hashes = file_hashes(files, root)
    reader = Reader(a.model, device=a.device, dtype=a.dtype, gpu_memory=a.gpu_memory, sink="first")
    tok = reader.tokenizer
    parts = chunks(root, tok)
    record.append("corpus", "code", root=str(root), files=len(files), sha256=hashes,
                  lengths_file=hashlib.sha256(LENGTHS.read_bytes()).hexdigest(), seed=SEED,
                  documents=parts["documents"], chunks={k: len(parts[k]) for k in ("train", "held_a", "held_b")})
    n = reader.info["layers"]
    layers = list(range(math.ceil(2 * n / 3), n))
    sets = load_sets(a.sets)
    D = d_resumable(reader, sets, layers, record, a.draft, True)
    states = {k: reader.states(parts[k], layers) for k in ("train", "held_a", "held_b")}
    set_states = [reader.states(s["texts"], layers) for s in sets]
    out, report = {"layers": np.array(layers)}, []
    floors = []
    for j, layer in enumerate(layers):
        fv = float(D[j]) ** 2 / states["train"].shape[2]
        r = fit_layer(states["train"][:, j], states["held_a"][:, j], states["held_b"][:, j], fv)
        lift_j = {"mu": [r["mu"]], "V": [r["V"]], "scale": [r["scale"]]}
        d_lift = spread([apply(lift_j, 0, S[:, j]) for S in set_states]) if r["K"] else float("nan")
        floors.append(d_lift ** 2 / max(r["K"], 1))
        out[f"mu{j}"], out[f"V{j}"], out[f"scale{j}"] = r["mu"], r["V"], r["scale"]
        report.append({"layer": layer, **{k: v for k, v in r.items() if k not in ("mu", "V", "scale")},
                       "D": float(D[j]), "D_lift": d_lift})
    ok = all(x["ok"] for x in report)
    out["floor"], out["ok"] = np.array(floors), np.array(ok)
    np.savez(run / "lift.npz", **out)
    record.append("lift", "code", **reader.info, read_layers=layers, layers=report, ok=ok, draft=a.draft,
                  seconds=round(time.time() - started, 1),
                  lift_sha256=hashlib.sha256((run / "lift.npz").read_bytes()).hexdigest())
    print(json.dumps({"ok": ok, "layers": [{k: x[k] for k in ("layer", "K", "k_v0", "held_b_bits_per_chunk", "ok")}
                                           for x in report]}, indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("build")
    s.add_argument("--run", required=True)
    s.add_argument("--corpus", required=True, help="the unpacked MASC texts (masc_500k_texts)")
    s.add_argument("--model", required=True)
    s.add_argument("--device", default="auto")
    s.add_argument("--dtype", default="bfloat16")
    s.add_argument("--gpu-memory", type=float)
    s.add_argument("--sets", default="material/paraphrases.jsonl")
    s.add_argument("--draft", action="store_true")
    a = p.parse_args(argv)
    {"build": cmd_build}[a.cmd](a)


if __name__ == "__main__":
    main()
