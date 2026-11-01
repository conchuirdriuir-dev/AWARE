"""The null check behind proposals/labelled-logs-goal-attention.md: what the goal attention index G reads when
attention depends on distance and content only, never on whether a key is the goal.

    python -m proposals.goal_attention_null

Attention here is synthetic: decaying with distance, scaled by a random salience per key, and (unlike the unit test
on the branch) normalised so each row sums to 1 with half of it on the sink, as softmax attention does.
"""
from __future__ import annotations

import numpy as np

from product import labelled_logs as L


def window(T, goal, tau, seed, boost=1.0, goal_seed=None, normalise=True, sink=0.5):
    rng = np.random.default_rng(seed)
    q, k = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    sal = rng.lognormal(0, 1.0, T)
    if goal_seed is not None:                                   # another text in the goal's place
        sal[:goal] = np.random.default_rng(goal_seed).lognormal(0, 1.0, goal)
    M = np.where(k <= q, (np.exp(-(q - k) / tau) + 0.02) * sal[None, :], 0.0)
    M[:, 1:goal] *= boost
    if normalise:
        M[:, 0] = 0
        M = (1 - sink) * M / np.maximum(M.sum(1, keepdims=True), 1e-12)
        M[:, 0] = sink
    spans, kinds, a, i = [(0, goal)], ["goal"], goal, 0
    while a < T:
        n = min(T - a, int(rng.integers(16, 96)))
        spans.append((a, a + n))
        kinds.append("self" if i % 2 == 0 else "external")
        a, i = a + n, i + 1
    return L.attention_indices(M[None], spans, kinds)


def main(seeds: int = 20):
    print("null G, rows not normalised (as the branch's unit test):",
          round(float(np.median([window(1024, 64, 128, s, normalise=False)["G"] for s in range(seeds)])), 2))
    print("null G, rows sum to 1:")
    for tau in (16, 128):
        for T in (512, 1024, 2048):
            for goal in (32, 128):
                g = np.median([window(T, goal, tau, s)["G"] for s in range(seeds)])
                print(f"  locality {tau:>3}, window {T:>4} tokens, goal {goal:>3} tokens: {g:.2f}")
    print("more attention on the goal, raw G against G over a placebo goal in the same slot:")
    for tau, T, goal in [(16, 512, 32), (128, 1024, 32), (128, 2048, 128)]:
        raw, ratio = [], []
        for s in range(seeds):
            live = window(T, goal, tau, s, boost=2.0, goal_seed=s + 100)["G"]
            placebo = window(T, goal, tau, s, goal_seed=s + 200)["G"]
            raw.append(live)
            ratio.append(live / placebo)
        print(f"  locality {tau:>3}, window {T:>4}, goal {goal:>3}: raw G {np.median(raw):.2f}, "
              f"live over placebo {np.median(ratio):.2f}")


if __name__ == "__main__":
    main()
