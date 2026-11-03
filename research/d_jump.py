"""The D jump on Qwen3-8B (PLAN.md, research track): what D measures where it leaps between layers 4 and 8.

    python -m research.d_jump --run runs/d-jump-qwen8b --device auto --gpu-memory 6.5

Reads tokens one by one and leaves things out of the pooled state: a few tokens, a few coordinates. The model is only
read; substrate/ does not change. Thresholds are the plan's, fixed before this ran. Draft sets, labelled draft.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from substrate.geometry import between, spread
from substrate.measure_d import load_sets
from substrate.record import Record

MODEL = "Qwen/Qwen3-8B"
LAYERS = [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 28, 32, 35]
SAVED = "runs/echo-alone-qwen8b"
LOW, HIGH = 4, 8
EXPLAINED = 3.0         # J' at or below this holds
CONTROL = 0.8           # the control must keep J' at or above this share of J
MATCH = 0.01            # D must match the saved run within this share, or nothing is read
TOKEN_CONDITIONS = ("all", "top1", "top3", "random", "second", "last")


def dropped(T: np.ndarray, condition: str, pick: int) -> np.ndarray:
    """Indices left out of one text's tokens at one layer. `T` is (tokens, width); at least one token always stays.
    `pick` is the text's random token, fixed across layers."""
    n = len(T)
    if n <= 1 or condition == "all":
        return np.array([], dtype=int)
    if condition in ("top1", "top3"):
        k = min(int(condition[3:]), n - 1)
        return np.argsort(-np.linalg.norm(T, axis=1))[:k]
    return np.array([{"random": pick % n, "second": 0, "last": n - 1}[condition]])


def pooled(tokens: list[list[np.ndarray]], condition: str, seed: int = 0) -> np.ndarray:
    """(sets, texts, layers, width) from per-token states, each text (tokens, layers, width), under a condition."""
    rng = np.random.default_rng(seed)
    out = []
    for texts in tokens:
        row = []
        for T in texts:
            pick = int(rng.integers(1 << 30))
            means = []
            for j in range(T.shape[1]):
                keep = np.ones(len(T), dtype=bool)
                keep[dropped(T[:, j], condition, pick)] = False
                means.append(T[keep, j].astype(np.float64).mean(0))
            row.append(means)
        out.append(row)
    return np.array(out)


def D_per_layer(S: np.ndarray) -> np.ndarray:
    return np.array([spread(list(S[:, :, j].astype(float))) for j in range(S.shape[2])])


def between_per_layer(S: np.ndarray) -> np.ndarray:
    return np.array([between(list(S[:, :, j].astype(float))) for j in range(S.shape[2])])


def within_variance(S: np.ndarray) -> np.ndarray:
    """Per coordinate, the within-set variance. S is (sets, texts, width)."""
    return ((S - S.mean(1, keepdims=True)) ** 2).mean((0, 1))


def D_without_coordinates(S: np.ndarray, how: str, k: int = 5, seed: int = 0) -> np.ndarray:
    """D per layer with k coordinates removed: chosen per layer on one half of the sets and measured on the other,
    both ways, averaged. `how`: "top", most within-set variance on the choosing half; "random", the control."""
    half = S.shape[0] // 2
    folds = [(np.arange(half), np.arange(half, S.shape[0])), (np.arange(half, S.shape[0]), np.arange(half))]
    rng = np.random.default_rng(seed)
    out = np.zeros(S.shape[2])
    for choose, measure in folds:
        for j in range(S.shape[2]):
            if how == "top":
                coords = np.argsort(-within_variance(S[choose, :, j]))[:k]
            else:
                coords = rng.choice(S.shape[3], k, replace=False)
            kept = np.delete(S[measure, :, j].astype(float), coords, axis=2)
            out[j] += spread(list(kept)) / len(folds)
    return out


def jump(D: np.ndarray, layers: list[int]) -> float:
    return float(D[layers.index(HIGH)] / D[layers.index(LOW)])


def largest_step(D: np.ndarray, layers: list[int]) -> dict:
    """The single step of LOW to HIGH holding the most of log J."""
    steps = [(a, b) for a, b in zip(layers, layers[1:]) if LOW <= a and b <= HIGH]
    logs = {s: float(np.log(D[layers.index(s[1])] / D[layers.index(s[0])])) for s in steps}
    best = max(logs, key=logs.get)
    total = sum(logs.values())
    return {"from": best[0], "to": best[1], "ratio": float(np.exp(logs[best])),
            "share_of_log_J": logs[best] / total if total else 0.0}


def median_norms(tokens: list[list[np.ndarray]]) -> np.ndarray:
    return np.median(np.concatenate([np.linalg.norm(T, axis=2) for texts in tokens for T in texts], axis=0), axis=0)


def read(J: float, J_tokens: dict, J_coords: dict, scale: float) -> dict:
    """The plan's three causes, each on its own; more than one may hold."""
    kept = lambda control: control >= CONTROL * J  # noqa: E731
    return {"top 1 token": J_tokens["top1"] <= EXPLAINED and kept(J_tokens["random"]),
            "top 3 tokens": J_tokens["top3"] <= EXPLAINED and kept(J_tokens["random"]),
            "5 coordinates": J_coords["top"] <= EXPLAINED and kept(J_coords["random"]),
            "overall scale": scale >= J / EXPLAINED}


def token_states(reader, texts: list[str], layers: list[int]) -> tuple[list[np.ndarray], list[list[tuple[str, int]]]]:
    """Per text, (tokens, layers, width) over exactly the tokens `Reader.states` pools with skip_first, and each
    token's string and position in the sequence."""
    import torch

    states, labels = [], []
    with torch.no_grad():
        for text in texts:
            ids = reader.tokenizer(text, return_tensors="pt")["input_ids"]
            keep = ~torch.isin(ids[0], reader.special)
            if keep[1:].any():
                keep[0] = False
            saved = []
            with reader.model.trace(ids):
                for layer in layers:
                    saved.append(reader.model.layers_output[layer].save())
            states.append(np.stack([s[0, keep.to(s.device)].float().cpu().numpy() for s in saved], axis=1))
            where = keep.nonzero()[:, 0].tolist()
            labels.append([(reader.tokenizer.decode(ids[0, i]), i) for i in where])
    return states, labels


def largest_tokens(tokens, labels, layer_index: int) -> dict:
    """Which token is largest in each text at one layer, and how concentrated it is in its top 5 coordinates."""
    strings, positions, excess, share = Counter(), Counter(), [], []
    for texts, names in zip(tokens, labels):
        for T, name in zip(texts, names):
            norms = np.linalg.norm(T[:, layer_index], axis=1)
            i = int(np.argmax(norms))
            strings[name[i][0]] += 1
            positions[name[i][1]] += 1
            excess.append(float(norms[i] / np.median(norms)))
            squared = T[i, layer_index].astype(float) ** 2
            share.append(float(np.sort(squared)[-5:].sum() / squared.sum()))
    return {"strings": strings.most_common(10), "positions": positions.most_common(10),
            "largest_over_median_norm": float(np.median(excess)), "top5_share_of_squared_norm": float(np.median(share))}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True)
    p.add_argument("--sets", default="material/paraphrases.jsonl")
    p.add_argument("--device", default="auto")
    p.add_argument("--gpu-memory", type=float, default=6.5)
    a = p.parse_args(argv)
    from substrate.states import Reader

    sets = load_sets(a.sets)
    record = Record(Path(a.run) / "record.jsonl")
    reader = Reader(MODEL, device=a.device, gpu_memory=a.gpu_memory, sink="first")
    record.append("run", "code", **reader.info, read_layers=LAYERS, draft=True, sets=a.sets, saved=SAVED)
    tokens, labels = [], []
    for s in sets:
        states, names = token_states(reader, s["texts"], LAYERS)
        tokens.append(states)
        labels.append(names)

    pools = {c: pooled(tokens, c) for c in TOKEN_CONDITIONS}
    D = {c: D_per_layer(pools[c]) for c in TOKEN_CONDITIONS}
    B = {c: between_per_layer(pools[c]) for c in TOKEN_CONDITIONS}

    saved = np.load(Path(SAVED) / "states.npy")
    first = json.loads((Path(SAVED) / "record.jsonl").open().readline())
    common = [x for x in first["read_layers"] if x in LAYERS]
    D_saved = D_per_layer(saved)
    checks = {x: {"saved": float(D_saved[first["read_layers"].index(x)]), "now": float(D["all"][LAYERS.index(x)])}
              for x in common}
    worst = max(abs(c["now"] - c["saved"]) / c["saved"] for c in checks.values())
    record.append("check", "code", against=SAVED, layers=checks, worst_share=worst, passes=worst <= MATCH)
    print(f"check against the saved run: worst {worst:.4%} (must be within {MATCH:.0%})")
    if worst > MATCH:
        raise SystemExit("D does not match the saved run; nothing below is read")

    D_coords = {h: D_without_coordinates(pools["all"], h) for h in ("top", "random")}
    norms = median_norms(tokens)
    J = jump(D["all"], LAYERS)
    J_tokens = {c: jump(D[c], LAYERS) for c in TOKEN_CONDITIONS}
    J_coords = {h: jump(D_coords[h], LAYERS) for h in D_coords}
    scale = float(norms[LAYERS.index(HIGH)] / norms[LAYERS.index(LOW)])
    verdicts = read(J, J_tokens, J_coords, scale)
    step = largest_step(D["all"], LAYERS)
    biggest = {x: largest_tokens(tokens, labels, LAYERS.index(x)) for x in (4, 6, 8, 12, 24, 35)}
    rows = [{"layer": x, "median_token_norm": float(norms[j]),
             "D": {c: float(D[c][j]) for c in TOKEN_CONDITIONS},
             "between_over_D": {c: float(B[c][j] / D[c][j]) for c in TOKEN_CONDITIONS},
             "D_without_coordinates": {h: float(D_coords[h][j]) for h in D_coords}} for j, x in enumerate(LAYERS)]
    record.append("report", "code", J=J, J_tokens=J_tokens, J_coordinates=J_coords, scale=scale, verdicts=verdicts,
                  largest_step=step, largest_tokens=biggest, layers=rows, draft=True)

    print(f"J = D({HIGH}) / D({LOW}) = {J:.2f}; largest step {step['from']}->{step['to']}: x{step['ratio']:.2f}, "
          f"{step['share_of_log_J']:.0%} of log J")
    print(f"J': " + "  ".join(f"{c} {v:.2f}" for c, v in J_tokens.items()))
    print(f"J' coordinates: top5 {J_coords['top']:.2f}  random5 {J_coords['random']:.2f}; scale {scale:.2f}")
    print("verdicts:", verdicts)
    print(f"{'layer':>5} {'norm':>8} {'D':>8} {'top1':>8} {'top3':>8} {'random':>8} {'second':>8} {'last':>8} "
          f"{'-top5d':>8} {'-rnd5d':>8} {'B/D':>6} {'top1B/D':>7}")
    for r in rows:
        d = r["D"]
        print(f"{r['layer']:>5} {r['median_token_norm']:>8.1f} {d['all']:>8.2f} {d['top1']:>8.2f} {d['top3']:>8.2f} "
              f"{d['random']:>8.2f} {d['second']:>8.2f} {d['last']:>8.2f} {r['D_without_coordinates']['top']:>8.2f} "
              f"{r['D_without_coordinates']['random']:>8.2f} {r['between_over_D']['all']:>6.2f} "
              f"{r['between_over_D']['top1']:>7.2f}")
    for x, b in biggest.items():
        print(f"layer {x}: largest/median {b['largest_over_median_norm']:.1f}, top-5 coord share "
              f"{b['top5_share_of_squared_norm']:.2f}, positions {b['positions'][:5]}, tokens {b['strings'][:6]}")


if __name__ == "__main__":
    main()
