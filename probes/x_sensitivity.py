"""Can X see a planted dependence? A synthetic check, deciding nothing (PLAN.md, "Rate's calibration: the result":
the agent's suggested first step toward a repair, 2026-11-07).

Rate's calibration failed: X did not tell a player whose next move follows the world's last reply from one replaying
the same moves blind, and neither did the text baseline. This asks whether X, as built (Step 0's formula and fit,
`body.step0_habitat.codes`, unchanged), can see such a dependence at all, on states made here where the answer is
known. No model, no record: plain numpy on synthetic states, on the CPU.

Each episode is laid out as the calibration's: an opening, then 100 turns of an action (self) and a reply (partner),
201 chunks; W = 10 turns, burn-in 10. States live in d dimensions, replies and actions around two far-apart means (as
a room description and "go north" sit apart in the 8B), with isotropic noise. Two kinds of dependence:

- **category** (the calibration's case): each reply carries one of 4 fixed direction vectors; the contingent action
  carries the vector of the reply before it; the blind action, one drawn at random. The categories recur all episode.
- **content**: each reply carries a fresh random vector, new every turn; the contingent action carries the vector of the
  reply before it; the blind action, a fresh vector of its own (as replaying moves from another world would).
- **topic** (what Step 0's X was built for: the partner's recent chunks help code the agent's next one): the world's
  replies carry a topic, a fresh vector every 10 turns; the contingent action carries the topic of the reply before it;
  the blind action, a topic of its own, fresh every 10 turns on a schedule offset by 5.

Both arms hear replies from the same law, unlike the calibration (where the blind arm heard "You can't go that way."
a third of the time). The signal's strength is its norm against the noise's. Per kind and strength: the AUC of the
contingent episodes' mean X over the blind ones', 20 pairs as the calibration, with the same interval.

    python -m probes.x_sensitivity                       # the full grid; a few minutes on a CPU
    python -m probes.x_sensitivity --out x_sensitivity.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

KINDS = ("category", "content", "topic")
TOPIC_TURNS = 10
STRENGTHS = (0.25, 0.5, 1.0, 2.0)
D, TURNS, WORLDS, CATEGORIES, SEPARATION = 4096, 100, 40, 4, 3.0


def episode(kind: str, contingent: bool, strength: float, rng, d: int = D, turns: int = TURNS,
            means: tuple | None = None) -> tuple[np.ndarray, list[str]]:
    """One synthetic episode: states (chunks x d) and sources. Noise has unit variance per dimension, so its norm is
    about sqrt(d); a signal of `strength` has norm strength * sqrt(d), as do the means' separation (SEPARATION)."""
    scale = np.sqrt(d)
    unit = lambda v: v / np.linalg.norm(v)
    m_reply, m_action = means if means is not None else (unit(rng.standard_normal(d)) * SEPARATION * scale,
                                                         unit(rng.standard_normal(d)) * SEPARATION * scale)
    cats = [unit(rng.standard_normal(d)) * strength * scale for _ in range(CATEGORIES)]
    noise = lambda: rng.standard_normal(d)
    rows, srcs = [m_reply + noise()], ["partner"]           # the opening
    fresh = lambda: unit(rng.standard_normal(d)) * strength * scale
    carried = None                                          # what the last reply carried
    world_topic, own_topic = fresh(), fresh()
    for t in range(turns):
        if kind == "topic" and t % TOPIC_TURNS == 0:
            world_topic = fresh()
        if kind == "topic" and t % TOPIC_TURNS == TOPIC_TURNS // 2:
            own_topic = fresh()
        if contingent and carried is not None:
            mine = carried
        elif kind == "category":
            mine = cats[rng.integers(CATEGORIES)]
        elif kind == "content":
            mine = fresh()
        else:
            mine = own_topic
        rows.append(m_action + mine + noise())
        srcs.append("self")
        carried = (cats[rng.integers(CATEGORIES)] if kind == "category" else fresh() if kind == "content"
                   else world_topic)
        rows.append(m_reply + carried + noise())
        srcs.append("partner")
    return np.array(rows), srcs


def mean_x(X: np.ndarray, srcs: list[str], floor_var: float = 1.0) -> float:
    """The episode's mean X over its scored turns, by Step 0's codes, unchanged (no W = 5 or 20 variants)."""
    from body.step0_habitat import codes

    x = codes(X, srcs, floor_var, variants=False)["X"]
    return float(np.nanmean(x))


def run(kinds=KINDS, strengths=STRENGTHS, worlds: int = WORLDS, d: int = D, turns: int = TURNS, draws: int = 2000,
        seed: int = 0) -> dict:
    from body.rate_calibration import compare

    out = {}
    for kind in kinds:
        for s in strengths:
            rng = np.random.default_rng([seed, KINDS.index(kind), int(s * 1000)])
            by_world = {}
            for w in range(worlds):
                means = None
                vals = {}
                for arm, cont in (("contingent", True), ("blind", False)):
                    X, srcs = episode(kind, cont, s, rng, d=d, turns=turns, means=means)
                    vals[arm] = mean_x(X, srcs)
                by_world[w] = vals
            r = compare(by_world, draws)
            r["median X (bits)"] = {arm: float(np.median([v[arm] for v in by_world.values()]))
                                    for arm in ("contingent", "blind")}
            out[f"{kind}, strength {s:g}"] = r
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worlds", type=int, default=WORLDS)
    p.add_argument("--out", help="also write the result as JSON here")
    a = p.parse_args(argv)
    result = run(worlds=a.worlds)
    for name, r in result.items():
        print(f"{name:24s} AUC {r['AUC']:.3f}  90% {r['90% interval'][0]:.3f}-{r['90% interval'][1]:.3f}  "
              f"median X contingent {r['median X (bits)']['contingent']:8.2f}  blind {r['median X (bits)']['blind']:8.2f}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
