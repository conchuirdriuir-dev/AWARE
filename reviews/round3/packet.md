# Review packet: toy bed round 3

You are reviewing one experiment's implementation and the inference that can be drawn from it. Everything you need is
in this file: the design in brief, the pre-registration as it was frozen before the code, the code, and the raw
results. If you have the repository, the same code is at the paths named below; do not read PLAN.md, DECISIONS.md,
`product/rounds/`, `reviews/` or the git history, which hold the builder's own interpretation.

## What to do

Answer three questions:
1. **Fidelity.** Does the code implement the pre-registration? List every divergence, however small, with file and
   line, and say whether it could change a result.
2. **Diagnosis.** Every pre-registered bar failed. Why? Give the explanations you think are live, ranked, each with the
   evidence for it and a check that would rule it out. Say whether any of them is a bug.
3. **Next.** What would you measure or change next, and what result would tell you that you were wrong?

**The rule for findings.** Each finding is stated as a concrete change in what is measured or computed (the quantity,
the code path, the effect you expect), or it goes in a final section headed "Not actionable". Commentary that cannot be
turned into a change in a measurement does not count as a finding.

**Format.** A one-paragraph summary, then findings ranked most serious first, each with: severity (blocks the
inference / weakens it / minor), evidence, the proposed change, and what would show the finding wrong. Then "Not
actionable". Plain prose and tables; no preamble.

## The design in brief

The entity is an agent's trajectory through a frozen language model's activation space: one state per chunk the agent
produces. External chunks (tool results, the person's messages) arrive between them. An observer fits, on a window of
steps and in a low-dimensional code (the window's top principal axes), the joint model s(t+1) ≈ Â·s(t) + B̂·E(t).

**Coupling** is the unique contribution of the externals: the fall in held-out one-step prediction error when the
current external is added to a fit already using the entity's last p states. An external that merely echoes the
entity's own earlier state adds nothing beyond its past, so coupling should read zero for it whatever its provenance
tag says. The product's target is a **closed loop**: either the entity stops listening (dullness), or the externals
arrive already shaped by the entity (proliferation, a person inside the loop echoing it). Coupling should fall to zero
in both and stay open in healthy work.

**At zero** is judged against the session's own reference: δ is half the coupling measured in the first window after
an opening intake. Open, at zero or undecided by one-sided bounds; undecided is when to **probe** (the entity asks; the
partner's responses arrive), which should restore coupling in healthy work and not in a loop.

The toy bed tests the detectors on a linear generator where the planted structure is known:
s(t+1) = A·s(t) + B·E(t) + noise, with B = I − A so the externals set the equilibrium.

## Pre-registration (frozen before the code)

## Step 3, round 3: the step, at zero, and the probe (pre-registered 2026-10-31, before its code)

From the v10 tab's draft section "The step, and coupling at zero" as revised after review, and the owner's decisions
of the same day (DECISIONS.md). Two numbers are the agent's proposals, left unchanged by the owner: δ is half the
reference coupling, and readings use one-sided 90% bounds. Round 2's code stays as its record; round 3 is new code
(`python -m product.round3`).

**Generator.** Round 2's, with 32 dimensions, D = 1 and 500 steps, plus:
- **Sparse externals.** Each step is one self chunk. An opening intake (steps 0–39) delivers an external every step at
  twice the usual noise (files read, a briefing). After it, an external arrives on a step with probability 0.4. The
  entity is driven by the most recent external, which stays in context; the observer sees a new external only on the
  step it arrives, with a marker saying whether one did.
- **Onset at step 200**, so the reference window (below) closes before it. Ramps run 200–260.
- **Readings of an external.** The detectors use the external read alone (the raw external). For the framing
  comparison only, an in-context reading mixes in the entity's state: 0.5 × external + 0.5 × the entity's current state.
- **3b, proliferation:** each arriving external is the entity's state 1, 3 or 10 steps back plus noise 0.3 D.
- **7, valid inference:** the externals carry a trend; none arrive for steps 200–239 while the entity extrapolates on
  trend; then they resume on trend.
- **8, healthy and declining (new):** from onset the externals' novelty falls, both their walk and their noise ramping
  to 0.15 of normal over 60 steps, while the entity keeps listening (B unchanged).
- **The probe:** the entity asks; for the next 20 steps an external arrives every step as the partner's response. In
  healthy structures (progress, mastery, 8) the responses are fresh externals at full novelty. In 3a they are fresh but
  the entity no longer listens (B = 0). In 3b they echo the entity at the structure's lag. A probe is issued at step
  460 in every run whose late reading is not open.

**Coupling** (windows of 160 steps, 3 axes, p = 12, ridge 1.0): as round 2, plus the arrival marker in both fits, so
only the external's content is credited. Scored by contiguous 5-fold with a 12-step gap either side of each test fold,
held-out errors pooled. Its uncertainty comes from 200 bootstrap resamples of held-out steps in blocks of 12.

**At zero.** The reference is coupling in the first window after the intake (steps 40–200); δ is half of it.
- Open: the lower 90% bound is above δ.
- At zero: the upper 90% bound is below δ.
- Undecided: neither.
- A run whose reference coupling is not above zero has no δ, and counts against every bar it enters.

Windows: late, steps 300–460; after the probe, steps 340–500; valid inference after resume, windows starting at 240,
260, 280 and 300.

**Point contrast, admission, spectral descriptor:** as round 2, moved to the new onset (before 140–200, after 200–230,
later stretch 260–320). Their thresholds are the 99th percentiles over 500 progress runs with a separate seed, and the
admission floor is the median there, as in round 2.

**Shrinkage** (the review's version of "self-dynamics fitted once"): the past model is fitted per window with ridge
toward a prior instead of toward zero, λ = 20. The prior has the lag-1 block equal to â × identity and the other lags
and the intercept zero. â is the median mean diagonal of per-window lag-1 fits over 50 progress runs with a separate
seed. A scalar persistence is used because it is the same in any basis, and the code changes from window to window.

**Pass.**
1. **Separation.** Rows 1, 2, 5 and 6 each at 90% or better, read as in round 2, with "coupling open" replaced by
   "the late reading is not at zero". The closed loops (3a, and 3b at each lag) are called a dynamics shift in at most
   10% of runs.
2. **Coupling AUC, mastery over 3a and over 3b at each lag:** 0.8 or better, late window.
3. **Coupling AUC, progress over 3b at each lag:** 0.8 or better, late window.
4. **Valid inference:** at least 90% of the windows after resume not read at zero.
5. **At zero, late window, before any probe:** progress, mastery, topic shift and dynamics shift each read at zero in
   at most 10% of runs.
6. **After the probe:** 3a and 3b at each lag read at zero in at least 80% of runs; structure 8 reads not at zero in
   at least 90%. The undecided share is reported for every structure, before and after.
7. **Shrinkage is kept** only if it narrows the spread of progress's late coupling and does not raise 3b's
   false-open rate (late reading open) by more than 0.02 at any lag. This bar decides shrinkage, not round 3: rows 1–6
   are scored without shrinkage, and shrinkage's own readings are reported beside them.
8. **Framing** (reported, decides only the claim): reading alone is shown to help if, for progress over 3b, its AUC
   beats the in-context reading's by 0.05 or more at every lag. If not, the toy has not shown it; a linear framing term
   may only rescale coupling, which δ and AUCs ignore, and logged sessions decide.

Before the GPU, the same 200 trials a structure and seed 0 as rounds 1 and 2.

## Raw results (200 trials a structure, seed 0)

```json
{
 "trials": 200,
 "calibration": 500,
 "seed": 0,
 "thresholds": {
  "floor": 2.0767408656093957,
  "point contrast": 1.7834025678155523,
  "descriptor": 0.26870387168644233,
  "a_prior": 0.9299109262592538
 },
 "separation": {
  "topic shift": 0.995,
  "dynamics shift": 0.12,
  "noise spike": 0.995,
  "progress": 0.995
 },
 "loops called dynamics shift": {
  "closed loop (dullness)": 0.075,
  "proliferation, lag 1": 0.0,
  "proliferation, lag 3": 0.0,
  "proliferation, lag 10": 0.0
 },
 "coupling AUC, mastery over loop": {
  "closed loop (dullness)": 0.2181,
  "proliferation, lag 1": 0.3911,
  "proliferation, lag 3": 0.388775,
  "proliferation, lag 10": 0.133975
 },
 "coupling AUC, progress over echo": {
  "proliferation, lag 1": 0.550875,
  "proliferation, lag 3": 0.54725,
  "proliferation, lag 10": 0.2018
 },
 "valid inference not at zero, after resume": 0.0525,
 "at zero before probe": {
  "progress": 0.875,
  "mastery": 0.92,
  "topic shift": 0.905,
  "dynamics shift": 0.845
 },
 "loops at zero after probe": {
  "closed loop (dullness)": 0.255,
  "proliferation, lag 1": 0.28,
  "proliferation, lag 3": 0.31,
  "proliferation, lag 10": 0.04
 },
 "healthy declining not at zero after probe": 0.255,
 "shrinkage: progress spread": {
  "unshrunk": 0.019577628718933097,
  "shrunk": 0.0042514233301367615
 },
 "shrinkage: echo false-open": {
  "proliferation, lag 1": {
   "unshrunk": 0.0,
   "shrunk": 0.03
  },
  "proliferation, lag 3": {
   "unshrunk": 0.005,
   "shrunk": 0.015
  },
  "proliferation, lag 10": {
   "unshrunk": 0.13,
   "shrunk": 0.41
  }
 },
 "framing: AUC progress over echo": {
  "proliferation, lag 1": {
   "alone": 0.550875,
   "in context": 0.503725
  },
  "proliferation, lag 3": {
   "alone": 0.54725,
   "in context": 0.482325
  },
  "proliferation, lag 10": {
   "alone": 0.2018,
   "in context": 0.3544
  }
 },
 "passed": {
  "1 separation": false,
  "2 mastery over loops": false,
  "3 progress over echoes": false,
  "4 valid inference": false,
  "5 healthy not at zero": false,
  "6 after the probe": false
 },
 "decided": {
  "7 shrinkage kept": false,
  "8 reading alone shown to help": false
 },
 "readings": {
  "topic shift": {
   "before": {
    "open": 0.02,
    "undecided": 0.075,
    "at zero": 0.27,
    "no reference": 0.635
   },
   "after": {
    "open": 0.03,
    "undecided": 0.08,
    "at zero": 0.255,
    "no reference": 0.635
   },
   "coupling, median": -0.012837212278276527
  },
  "dynamics shift": {
   "before": {
    "open": 0.02,
    "undecided": 0.135,
    "at zero": 0.23,
    "no reference": 0.615
   },
   "after": {
    "open": 0.04,
    "undecided": 0.16,
    "at zero": 0.185,
    "no reference": 0.615
   },
   "coupling, median": -0.008201337668076542
  },
  "closed loop (dullness)": {
   "before": {
    "open": 0.02,
    "undecided": 0.055,
    "at zero": 0.25,
    "no reference": 0.675
   },
   "after": {
    "open": 0.02,
    "undecided": 0.05,
    "at zero": 0.255,
    "no reference": 0.675
   },
   "coupling, median": -0.004659594221487595
  },
  "proliferation, lag 1": {
   "before": {
    "open": 0.0,
    "undecided": 0.08,
    "at zero": 0.26,
    "no reference": 0.66
   },
   "after": {
    "open": 0.0,
    "undecided": 0.06,
    "at zero": 0.28,
    "no reference": 0.66
   },
   "coupling, median": -0.013614869438028698
  },
  "proliferation, lag 3": {
   "before": {
    "open": 0.005,
    "undecided": 0.075,
    "at zero": 0.305,
    "no reference": 0.615
   },
   "after": {
    "open": 0.01,
    "undecided": 0.065,
    "at zero": 0.31,
    "no reference": 0.615
   },
   "coupling, median": -0.014107638441298531
  },
  "proliferation, lag 10": {
   "before": {
    "open": 0.13,
    "undecided": 0.145,
    "at zero": 0.04,
    "no reference": 0.685
   },
   "after": {
    "open": 0.16,
    "undecided": 0.115,
    "at zero": 0.04,
    "no reference": 0.685
   },
   "coupling, median": 0.009134864581207935
  },
  "mastery": {
   "before": {
    "open": 0.015,
    "undecided": 0.065,
    "at zero": 0.345,
    "no reference": 0.575
   },
   "after": {
    "open": 0.015,
    "undecided": 0.05,
    "at zero": 0.36,
    "no reference": 0.575
   },
   "coupling, median": -0.023535125044086974
  },
  "noise spike": {
   "before": {
    "open": 0.02,
    "undecided": 0.125,
    "at zero": 0.215,
    "no reference": 0.64
   },
   "after": {
    "open": 0.035,
    "undecided": 0.095,
    "at zero": 0.23,
    "no reference": 0.64
   },
   "coupling, median": -0.009204643712697441
  },
  "progress": {
   "before": {
    "open": 0.01,
    "undecided": 0.115,
    "at zero": 0.26,
    "no reference": 0.615
   },
   "after": {
    "open": 0.02,
    "undecided": 0.13,
    "at zero": 0.235,
    "no reference": 0.615
   },
   "coupling, median": -0.0126360585903314
  },
  "valid inference": {
   "before": {
    "open": 0.005,
    "undecided": 0.025,
    "at zero": 0.155,
    "no reference": 0.815
   },
   "after": {
    "open": 0.005,
    "undecided": 0.03,
    "at zero": 0.15,
    "no reference": 0.815
   },
   "coupling, median": -0.015709745947886933
  },
  "healthy, declining": {
   "before": {
    "open": 0.005,
    "undecided": 0.07,
    "at zero": 0.295,
    "no reference": 0.63
   },
   "after": {
    "open": 0.095,
    "undecided": 0.16,
    "at zero": 0.115,
    "no reference": 0.63
   },
   "coupling, median": 7.348530043033152e-05
  }
 }
}
```

## Code

### `product/round3.py`

```python
"""Toy bed round 3: the step, coupling at zero against the post-intake reference, and the probe (PLAN.md, product
step 3, round 3; the v10 tab's draft section "The step, and coupling at zero").

    python -m product.round3                         # 200 trials a structure, seed 0
    python -m product.round3 --trials 50 --out runs/round3.json

Its own generator: each step is one self chunk, externals arrive sparsely after an opening intake, and the entity is
driven by the latest external, which stays in context. Rounds 1 and 2 keep theirs, so their records still reproduce.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from product import coupling as cp
from product.toybed import auc, orthonormal, rotation
from substrate.geometry import rms

D_DIM, T, T0, RAMP, INTAKE, ARRIVAL = 32, 500, 200, 60, 40, 0.4
PROBE, PROBE_LEN, PAUSE = 460, 20, 40
REFERENCE, LATE, AFTER_PROBE = (40, 200), (300, 460), (340, 500)
AFTER_RESUME = (240, 260, 280, 300)
WINDOW, AXES, LAGS, FOLDS, GAP, BOOT, RIDGE, LAMBDA = 160, 3, 12, 5, 12, 200, 1.0, 20.0
STRUCTURES = {                                  # name: (kind, echo lag)
    "topic shift": ("topic shift", 0), "dynamics shift": ("dynamics shift", 0),
    "closed loop (dullness)": ("dullness", 0),
    "proliferation, lag 1": ("proliferation", 1), "proliferation, lag 3": ("proliferation", 3),
    "proliferation, lag 10": ("proliferation", 10),
    "mastery": ("mastery", 0), "noise spike": ("noise spike", 0), "progress": ("progress", 0),
    "valid inference": ("valid inference", 0), "healthy, declining": ("declining", 0)}
LOOPS = ("closed loop (dullness)", "proliferation, lag 1", "proliferation, lag 3", "proliferation, lag 10")
ECHOES = LOOPS[1:]
HEALTHY_PROBE = ("progress", "mastery", "declining")


def generate(kind: str, rng, lag: int = 0, D: float = 1.0) -> dict:
    """One run with sparse externals and a probe at step 460. Returns the entity's states s, the externals as the
    observer sees them (E, zero where none arrived), the arrival marker m, the external read in context (E_in), the
    latest external in context at each step (H), and the origin's centroid c."""
    d = D_DIM
    c0 = rng.normal(scale=10.0, size=d)
    t = np.arange(T)
    novelty = np.ones(T)
    if kind == "declining":
        novelty = 1 - 0.85 * np.clip((t - T0) / RAMP, 0, 1)
    novelty[PROBE:PROBE + PROBE_LEN] = 1.0                       # the partner's response is fresh
    walk = np.cumsum(rng.normal(scale=0.3 * D / np.sqrt(d), size=(T, d)) * novelty[:, None], axis=0)
    if kind == "topic shift":
        walk[T0:] += 6 * D * orthonormal(rng, 1, d)[0]
    trend = np.zeros(d)
    if kind == "valid inference":
        trend = 0.3 * D * orthonormal(rng, 1, d)[0]
        walk += t[:, None] * trend
    noise_scale = np.where(t < INTAKE, 2 * D, D * novelty)
    fresh = c0 + walk + rng.normal(size=(T, d)) * (noise_scale / np.sqrt(d))[:, None]
    m = (rng.random(T) < ARRIVAL) | (t < INTAKE)
    if kind == "valid inference":
        m[T0:T0 + PAUSE] = False
    m[PROBE:PROBE + PROBE_LEN] = True
    I = np.eye(d)
    rot = 0.5 * I
    if kind == "dynamics shift":
        Q = orthonormal(rng, 4, d)
        R = np.zeros((4, 4))
        R[:2, :2], R[2:, 2:] = rotation(0.6, 0.9), rotation(0.6, 0.9)
        rot = 0.5 * I + Q.T @ (R - 0.5 * np.eye(4)) @ Q
    u = orthonormal(rng, 1, d)[0]
    s, E, H = np.zeros((T, d)), np.zeros((T, d)), np.zeros((T, d))
    s[0] = held = fresh[0]
    for k in range(T):
        if m[k]:
            if kind == "proliferation" and k >= T0:
                E[k] = s[k - lag] + rng.normal(scale=0.3 * D / np.sqrt(d), size=d)
            else:
                E[k] = fresh[k]
            held = E[k]
        H[k] = held
        if k == T - 1:
            break
        f = min(max((k - T0) / RAMP, 0.0), 1.0)
        A, beta, sigma, drift = 0.5 * I, 1.0, D, np.zeros(d)
        if kind == "dynamics shift" and k >= T0:
            A = rot
        elif kind == "dullness":
            A, beta, drift = (0.5 + 0.48 * f) * I, 1.0 - f, f * 0.3 * D * u
        elif kind == "mastery":
            A, sigma = (0.5 - 0.3 * f) * I, D * (1 - 0.7 * f)
        elif kind == "noise spike" and T0 <= k < T0 + 3:
            sigma = 20 * D
        drive = held - c0
        if kind == "valid inference" and T0 <= k < T0 + PAUSE:
            drive = walk[T0 - 1] + trend * (k - T0 + 1)           # the entity's own on-trend extrapolation
        s[k + 1] = c0 + A @ (s[k] - c0) + beta * (I - A) @ drive + drift + rng.normal(scale=sigma / np.sqrt(d), size=d)
    E_in = np.where(m[:, None], 0.5 * E + 0.5 * s, 0.0)
    return {"s": s, "E": E, "m": m, "E_in": E_in, "H": H, "c": fresh[:10].mean(0), "t0": T0, "kind": kind}


def make(name: str, rng, D: float = 1.0) -> dict:
    kind, lag = STRUCTURES[name]
    return generate(kind, rng, lag, D)


def _fit(X, Y, prior=None, lam=RIDGE):
    target = X.T @ Y + (lam * prior if prior is not None else 0.0)
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), target)


def coupling(run: dict, start: int, external: str = "E", a_prior: float | None = None, rng=None) -> dict:
    """Coupling on the window [start, start + 160): held-out errors by contiguous 5-fold with a 12-step gap, the
    arrival marker in both fits, block-bootstrap 90% bounds. With `a_prior`, the past fit is shrunk toward a scalar
    persistence instead of toward zero."""
    sl = slice(start, start + WINDOW)
    S, X, M, c = run["s"][sl], run[external][sl], run["m"][sl].astype(float), run["c"]
    U = cp.basis(S, c, AXES)
    z, e = (S - c) @ U.T, ((X - c) @ U.T) * M[:, None]
    rows = np.arange(LAGS - 1, WINDOW - 1)
    past = np.array([np.concatenate([z[t - j] for j in range(LAGS)] + [[1.0, M[t]]]) for t in rows])
    full = np.hstack([past, e[rows]])
    y = z[rows + 1]
    n = len(y)
    fold = np.arange(n) * FOLDS // n
    sq = {"past": np.zeros(n), "full": np.zeros(n)}
    for f in range(FOLDS):
        test = fold == f
        lo, hi = np.flatnonzero(test)[[0, -1]]
        train = ~test & ((np.arange(n) < lo - GAP) | (np.arange(n) > hi + GAP))
        for name, Xd in (("past", past), ("full", full)):
            prior, lam = None, RIDGE
            if a_prior is not None:
                prior = np.zeros((Xd.shape[1], AXES))
                prior[:AXES] = a_prior * np.eye(AXES)
                lam = LAMBDA
            b = _fit(Xd[train], y[train], prior, lam)
            sq[name][test] = np.sum((y[test] - Xd[test] @ b) ** 2, axis=1)
    value = float(np.sqrt(sq["past"].mean()) - np.sqrt(sq["full"].mean()))
    rng = rng if rng is not None else np.random.default_rng(start)
    blocks = [np.arange(i, min(i + GAP, n)) for i in range(0, n, GAP)]
    boot = []
    for _ in range(BOOT):
        idx = np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))])
        boot.append(np.sqrt(sq["past"][idx].mean()) - np.sqrt(sq["full"][idx].mean()))
    lo90, hi90 = np.percentile(boot, [10, 90])
    return {"coupling": value, "low": float(lo90), "high": float(hi90), "residual": float(np.sqrt(sq["full"].mean()))}


def reading(w: dict, delta: float | None) -> str:
    if delta is None:
        return "no reference"
    if w["low"] > delta:
        return "open"
    if w["high"] < delta:
        return "at zero"
    return "undecided"


def reference(run: dict, **kw) -> float | None:
    ref = coupling(run, REFERENCE[0], **kw)["coupling"]
    return 0.5 * ref if ref > 0 else None


def _held(run):
    """The point contrast and descriptor read the external in context at each step: the latest one to arrive."""
    return {"s": run["s"], "E": run["H"], "c": run["c"], "t0": run["t0"]}


def contrast(run: dict, floor: float | None = None) -> dict:
    """Round 2's point contrast and admission at the new onset (before 140-200, after 200-230)."""
    return cp.contrast(_held(run), floor)


def descriptor(run: dict) -> float:
    return cp.descriptor(_held(run), 260, 320)


def calibrate(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed + 10_000)
    runs = [make("progress", rng) for _ in range(n)]
    con = [contrast(r) for r in runs]
    prior_rng = np.random.default_rng(seed + 30_000)
    diag = []
    for _ in range(50):
        r = make("progress", prior_rng)
        S, c = r["s"][LATE[0]:LATE[0] + WINDOW], r["c"]
        U = cp.basis(S, c, AXES)
        z = (S - c) @ U.T
        b = _fit(np.hstack([z[:-1], np.ones((len(z) - 1, 1))]), z[1:])
        diag.append(np.mean(np.diag(b[:AXES])))
    return {"floor": float(np.median([x["empty residual"] for x in con])),
            "point contrast": float(np.percentile([x["statistic"] for x in con], 99)),
            "descriptor": float(np.percentile([descriptor(r) for r in runs], 99)),
            "a_prior": float(np.median(diag))}


def read(run: dict, th: dict, name: str) -> dict:
    delta = reference(run)
    late = coupling(run, LATE[0])
    before = reading(late, delta)
    after = before if before == "open" else reading(coupling(run, AFTER_PROBE[0]), delta)
    con = contrast(run, th["floor"])
    point, desc = con["statistic"] > th["point contrast"], descriptor(run) > th["descriptor"]
    if point:
        label = "topic shift" if con["admitted"] else "noise spike"
    elif desc:
        label = "no reference" if delta is None else ("dynamics shift" if before != "at zero" else "closed loop")
    else:
        label = "progress"
    out = {"label": label, "delta": delta, "late": late["coupling"], "before": before, "after": after}
    if name == "progress" or name in ECHOES:
        sd = reference(run, a_prior=th["a_prior"])
        shrunk = coupling(run, LATE[0], a_prior=th["a_prior"])
        out |= {"late shrunk": shrunk["coupling"], "before shrunk": reading(shrunk, sd),
                "late in context": coupling(run, LATE[0], external="E_in")["coupling"]}
    if name == "valid inference":
        out["after resume"] = [reading(coupling(run, a), delta) for a in AFTER_RESUME]
    return out


def evaluate(trials: int = 200, calibration: int = 500, seed: int = 0) -> dict:
    th = calibrate(calibration, seed)
    rng = np.random.default_rng(seed)
    res = {name: [read(make(name, rng), th, name) for _ in range(trials)] for name in STRUCTURES}

    def share(name, key, value):
        return float(np.mean([r[key] == value for r in res[name]]))

    def values(name, key):
        return [r[key] for r in res[name]]

    separation = {n: share(n, "label", n) for n in ("topic shift", "dynamics shift", "noise spike", "progress")}
    loops_as_dynamics = {n: share(n, "label", "dynamics shift") for n in LOOPS}
    mastery = {n: auc(values("mastery", "late"), values(n, "late")) for n in LOOPS}
    progress = {n: auc(values("progress", "late"), values(n, "late")) for n in ECHOES}
    vi = [x for r in res["valid inference"] for x in r["after resume"]]
    valid = float(np.mean([x not in ("at zero", "no reference") for x in vi]))
    zero_before = {n: float(np.mean([r["before"] in ("at zero", "no reference") for r in res[n]]))
                   for n in ("progress", "mastery", "topic shift", "dynamics shift")}
    loops_after = {n: share(n, "after", "at zero") for n in LOOPS}
    declining_after = float(np.mean([r["after"] not in ("at zero", "no reference") for r in res["healthy, declining"]]))
    sd = {"unshrunk": float(np.std(values("progress", "late"))), "shrunk": float(np.std(values("progress", "late shrunk")))}
    false_open = {n: {"unshrunk": share(n, "before", "open"), "shrunk": share(n, "before shrunk", "open")} for n in ECHOES}
    framing = {n: {"alone": auc(values("progress", "late"), values(n, "late")),
                   "in context": auc(values("progress", "late in context"), values(n, "late in context"))} for n in ECHOES}
    passed = {
        "1 separation": all(v >= 0.9 for v in separation.values()) and all(v <= 0.1 for v in loops_as_dynamics.values()),
        "2 mastery over loops": all(v >= 0.8 for v in mastery.values()),
        "3 progress over echoes": all(v >= 0.8 for v in progress.values()),
        "4 valid inference": valid >= 0.9,
        "5 healthy not at zero": all(v <= 0.1 for v in zero_before.values()),
        "6 after the probe": all(v >= 0.8 for v in loops_after.values()) and declining_after >= 0.9}
    decided = {
        "7 shrinkage kept": sd["shrunk"] < sd["unshrunk"]
        and all(v["shrunk"] - v["unshrunk"] <= 0.02 for v in false_open.values()),
        "8 reading alone shown to help": all(v["alone"] - v["in context"] >= 0.05 for v in framing.values())}
    readings = {n: {"before": {k: share(n, "before", k) for k in ("open", "undecided", "at zero", "no reference")},
                    "after": {k: share(n, "after", k) for k in ("open", "undecided", "at zero", "no reference")},
                    "coupling, median": float(np.median(values(n, "late")))} for n in STRUCTURES}
    return {"trials": trials, "calibration": calibration, "seed": seed, "thresholds": th,
            "separation": separation, "loops called dynamics shift": loops_as_dynamics,
            "coupling AUC, mastery over loop": mastery, "coupling AUC, progress over echo": progress,
            "valid inference not at zero, after resume": valid, "at zero before probe": zero_before,
            "loops at zero after probe": loops_after, "healthy declining not at zero after probe": declining_after,
            "shrinkage: progress spread": sd, "shrinkage: echo false-open": false_open,
            "framing: AUC progress over echo": framing, "passed": passed, "decided": decided, "readings": readings}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--calibration", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the full report as JSON")
    a = p.parse_args(argv)
    rep = evaluate(a.trials, a.calibration, a.seed)

    def show(v):
        if isinstance(v, dict):
            return {k: show(x) for k, x in v.items()}
        return round(v, 3) if isinstance(v, float) else v
    for key, v in rep.items():
        if key not in ("trials", "calibration", "seed"):
            print(f"{key}: {show(v)}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
```

### `product/coupling.py`

```python
"""v10's detectors (the Drift detection (v10) tab, "Operators"; PLAN.md, product step 3, round 2).

The observer is a joint fit in the code, s(t+1) ~ A s(t) + B E(t). Coupling is the unique contribution of E: the fall
in held-out one-step error when E(t) is added to a fit already using the last p states, so an echo of an earlier
state reads zero whatever its provenance. Dynamic residual is that fit's held-out error. Point contrast is taken on
the residual, its candidate tested by admission; the spectral descriptor reads A with B E removed.
"""
from __future__ import annotations

import numpy as np

from substrate.axes import Code
from substrate.geometry import rms

RIDGE = 1.0
COUPLING_WINDOW, COUPLING_AXES, LAGS = 160, 3, 12
CONTRAST_AXES, BEFORE, AFTER = 4, 60, 30


def basis(S: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    """The code for a window: its top k principal axes about c."""
    return np.linalg.svd(S - c, full_matrices=False)[2][:k]


def ridge(X: np.ndarray, Y: np.ndarray, alpha: float = RIDGE) -> np.ndarray:
    return np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ Y)


def coupling(S: np.ndarray, X: np.ndarray, c: np.ndarray, k: int = COUPLING_AXES, p: int = LAGS,
             train: float = 0.7) -> dict:
    """Coupling and dynamic residual on one window of states S with externals X."""
    U = basis(S, c, k)
    z, e = (S - c) @ U.T, (X - c) @ U.T
    rows = range(p - 1, len(S) - 1)
    past = np.array([np.concatenate([z[t - j] for j in range(p)] + [[1.0]]) for t in rows])
    full = np.hstack([past, e[p - 1:len(S) - 1]])
    y = z[p:]
    n = int(train * len(y))
    err = {}
    for name, M in (("past", past), ("full", full)):
        beta = ridge(M[:n], y[:n])
        err[name] = rms(y[n:] - M[n:] @ beta)
    return {"coupling": err["past"] - err["full"], "residual": err["full"]}


class Joint:
    """A joint fit with p = 1 in a window's own code; it predicts the full state, the part off the code as its mean."""

    def __init__(self, S: np.ndarray, X: np.ndarray, c: np.ndarray, k: int = CONTRAST_AXES):
        self.c, self.U = c, basis(S, c, k)
        z, e = (S - c) @ self.U.T, (X - c) @ self.U.T
        self.beta = ridge(self._design(z[:-1], e[:-1]), z[1:])
        Z = S - c
        self.off_mean = (Z - (Z @ self.U.T) @ self.U).mean(0)
        self.k = k

    @staticmethod
    def _design(z, e):
        return np.hstack([z, e, np.ones((len(z), 1))])

    def residuals(self, S: np.ndarray, X: np.ndarray) -> np.ndarray:
        """s(t+1) minus its prediction from s(t) and E(t), for each step in S after the first."""
        z, e = (S - self.c) @ self.U.T, (X - self.c) @ self.U.T
        pred = self.c + (self._design(z[:-1], e[:-1]) @ self.beta) @ self.U + self.off_mean
        return S[1:] - pred

    def modes(self) -> np.ndarray:
        """Sorted eigenvalue magnitudes of A: the dynamics with B E taken out."""
        return np.sort(np.abs(np.linalg.eigvals(self.beta[:self.k].T)))[::-1]


def contrast(run: dict, floor: float | None = None) -> dict:
    """Point contrast on the residual at onset, and, given the floor, whether admission keeps its candidate."""
    s, E, c, t0 = run["s"], run["E"], run["c"], run["t0"]
    model = Joint(s[t0 - BEFORE:t0], E[t0 - BEFORE:t0], c)
    r_before = model.residuals(s[t0 - BEFORE:t0], E[t0 - BEFORE:t0])
    r_after = model.residuals(s[t0 - 1:t0 + AFTER], E[t0 - 1:t0 + AFTER])
    candidate = r_after.mean(0) - r_before.mean(0)
    half = AFTER // 2
    fit, held = r_after[:half], r_after[half:]

    def residual(axes):
        if not len(axes):
            return rms(held)
        level = fit.mean(0) @ axes.T
        return rms(held - level @ axes)

    out = {"statistic": float(np.linalg.norm(candidate)), "empty residual": residual(np.zeros((0, len(c))))}
    if floor is not None:
        axis = Code(c, 1.0, "contrast", unit=floor).consider(candidate, residual=residual, source="point contrast",
                                                             at=t0)
        out["admitted"] = axis.kind == "predictive"
    return out


def descriptor(run: dict, start: int = 180, stop: int = 240) -> float:
    """How far the dynamics moved: modes before onset against a later stretch."""
    s, E, c, t0 = run["s"], run["E"], run["c"], run["t0"]
    before = Joint(s[t0 - BEFORE:t0], E[t0 - BEFORE:t0], c).modes()
    later = Joint(s[start:stop], E[start:stop], c).modes()
    return float(np.linalg.norm(before - later))
```

### `product/toybed.py`

```python
"""The toy bed (v9, "Toy bed specification"; PLAN.md, product step 3): six planted structures, each changing only A, B
or E in s(t+1) = A s(t) + B E(t) + noise at D, and the detectors asked to separate them.

    python -m product.toybed                       # 200 trials a structure, thresholds from 500 progress runs
    python -m product.toybed --trials 50 --out runs/toybed.json

Synthetic for now; v9 wants it in the real model's activation space so D is real. It tests the detectors, never the
generator: the closed loop planted here is a conjecture that the labeled logs check.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from product import detectors
from product.compass import cell_v9 as cell

KINDS = ("topic shift", "dynamics shift", "closed loop (dullness)", "closed loop (proliferation)", "mastery",
         "noise spike", "progress")
EXTRA = ("valid inference",)            # v10, round 2


def orthonormal(rng, k: int, d: int) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return q.T


def rotation(angle: float, scale: float) -> np.ndarray:
    return scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def generate(kind: str, rng, d: int = 32, D: float = 1.0, T: int = 240, t0: int = 120, ramp: int = 60,
             n_origin: int = 10, lag: int = 0, pause: int = 40) -> dict:
    """One run: the entity's states s, the external states E (both T x d), the origin's centroid c, the onset t0.
    In proliferation the externals echo the entity's state `lag` steps back (round 1: 0). In valid inference the
    externals carry a trend and pause for `pause` steps from onset while the entity extrapolates on trend."""
    if kind not in KINDS + EXTRA:
        raise ValueError(kind)
    c0 = rng.normal(scale=10.0, size=d)
    walk = np.cumsum(rng.normal(scale=0.3 * D / np.sqrt(d), size=(T, d)), axis=0)
    if kind == "topic shift":
        walk[t0:] += 6 * D * orthonormal(rng, 1, d)[0]
    trend = np.zeros(d)
    if kind == "valid inference":
        trend = 0.3 * D * orthonormal(rng, 1, d)[0]
        walk += np.arange(T)[:, None] * trend
    E = c0 + walk + rng.normal(scale=D / np.sqrt(d), size=(T, d))
    if kind == "valid inference":
        E[t0:t0 + pause] = E[t0 - 1]                    # nothing new arrives: the last external, repeated
    I = np.eye(d)
    rot = I * 0.5
    if kind == "dynamics shift":
        Q = orthonormal(rng, 4, d)
        R = np.zeros((4, 4))
        R[:2, :2], R[2:, 2:] = rotation(0.6, 0.9), rotation(0.6, 0.9)
        rot = 0.5 * I + Q.T @ (R - 0.5 * np.eye(4)) @ Q
    u = orthonormal(rng, 1, d)[0]
    s = np.zeros((T, d))
    s[0] = E[0]
    for t in range(T - 1):
        f = min(max((t - t0) / ramp, 0.0), 1.0)
        A, beta, sigma, drift = 0.5 * I, 1.0, D, np.zeros(d)
        if kind == "dynamics shift" and t >= t0:
            A = rot
        elif kind == "closed loop (dullness)":
            A, beta, drift = (0.5 + 0.48 * f) * I, 1.0 - f, f * 0.3 * D * u
        elif kind == "mastery":
            A, sigma = (0.5 - 0.3 * f) * I, D * (1 - 0.7 * f)
        elif kind == "noise spike" and t0 <= t < t0 + 3:
            sigma = 20 * D
        if kind == "closed loop (proliferation)" and t >= t0:
            E[t] = s[t - lag] + rng.normal(scale=0.3 * D / np.sqrt(d), size=d)
        x, e = s[t] - c0, E[t] - c0
        if kind == "valid inference" and t0 <= t < t0 + pause:
            e = walk[t0 - 1] + trend * (t - t0 + 1)     # the entity's own on-trend extrapolation drives it
        s[t + 1] = c0 + A @ x + beta * (I - A) @ e + drift + rng.normal(scale=sigma / np.sqrt(d), size=d)
    return {"s": s, "E": E, "c": E[:n_origin].mean(0), "t0": t0, "kind": kind}


def auc(pos, neg) -> float:
    """The chance a positive ranks above a negative, ties counting half."""
    pos, neg = np.asarray(pos, dtype=float)[:, None], np.asarray(neg, dtype=float)[None, :]
    return float(np.mean((pos > neg) + 0.5 * (pos == neg)))


def evaluate(trials: int = 200, calibration: int = 500, seed: int = 0, D: float = 1.0) -> dict:
    cal_rng = np.random.default_rng(seed + 10_000)
    null = [detectors.contrasts(generate("progress", cal_rng, D=D), D) for _ in range(calibration)]
    thresholds = {k: float(np.percentile([n[k] for n in null], 99)) for k in null[0]}
    rng = np.random.default_rng(seed)
    results = {}
    for kind in KINDS:
        runs = []
        for _ in range(trials):
            run = generate(kind, rng, D=D)
            stats = detectors.contrasts(run, D)
            fires = {k: stats[k] > thresholds[k] for k in stats}
            blocks = detectors.observe(run["s"], run["E"], run["c"], D)
            runs.append({"fires": fires, "label": detectors.classify(fires), "blocks": blocks})
        results[kind] = runs

    def late(kind, key, start):
        return [b[key] for r in results[kind] for b in r["blocks"] if b["start"] >= start]

    def growth(kind):
        return [sum(b["growth"] for b in r["blocks"] if b["start"] >= 120) for r in results[kind]]

    separation = {k: float(np.mean([r["label"] == k for r in results[k]]))
                  for k in ("topic shift", "dynamics shift", "noise spike", "progress")}
    margin = {v: auc(late(v, "margin", 180), late("mastery", "margin", 180))
              for v in ("closed loop (dullness)", "closed loop (proliferation)")}
    frame = auc(growth("progress"), growth("closed loop (proliferation)"))
    described = {}
    for kind in KINDS:
        rs = results[kind]
        cells = [cell(b["self"], b["frame"]) for r in rs for b in r["blocks"] if b["start"] >= 180]
        described[kind] = {
            **{f"fires {k}": float(np.mean([r["fires"][k] for r in rs])) for k in thresholds},
            "self-residual, 180-240": float(np.median(late(kind, "self", 180))),
            "frame-residual, 180-240": float(np.median(late(kind, "frame", 180))),
            "margin, 180-240": float(np.median(late(kind, "margin", 180))),
            "frame growth, 120-240": float(np.mean(growth(kind))),
            "distance to origin, 180-240": float(np.median(late(kind, "distance", 180))),
            "commonest cell, 180-240": max(set(cells), key=cells.count),
        }
    passed = {"separation": all(v >= 0.9 for v in separation.values()),
              "margin": all(v >= 0.8 for v in margin.values()), "frame growth": frame >= 0.8}
    return {"trials": trials, "calibration": calibration, "seed": seed, "thresholds": thresholds,
            "separation": separation, "margin AUC vs mastery": margin, "frame growth AUC, progress over 3b": frame,
            "passed": passed, "described": described}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--calibration", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write the full report as JSON")
    a = p.parse_args(argv)
    rep = evaluate(a.trials, a.calibration, a.seed)
    print("separation (pass >= 0.90 each):", {k: round(v, 2) for k, v in rep["separation"].items()})
    print("margin AUC vs mastery (pass >= 0.80 each):", {k: round(v, 2) for k, v in rep["margin AUC vs mastery"].items()})
    print(f"frame growth AUC, progress over proliferation (pass >= 0.80): {rep['frame growth AUC, progress over 3b']:.2f}")
    print("passed:", rep["passed"])
    for kind, row in rep["described"].items():
        print(f"  {kind}: " + ", ".join(f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
```

### `substrate/geometry.py`

```python
"""The geometry core, in numpy. States are rows; every quantity is taken about the origin's centroid `c`.

Shared by both tracks. Fitting and admission of axes are in `axes.py`; the hypotheses about negation are the research
track's (`research/gate1.py`).
"""
from __future__ import annotations

import numpy as np


def centroid(X: np.ndarray) -> np.ndarray:
    return np.asarray(X).mean(axis=0)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def reflect(X: np.ndarray, c: np.ndarray, u: np.ndarray) -> np.ndarray:
    """X mirrored in the hyperplane through c normal to u: c + H(x - c), H = I - 2uu^T."""
    u = unit(u)
    return X - 2.0 * ((X - c) @ u)[:, None] * u


def spread(sets: list[np.ndarray]) -> float:
    """D: the root-mean-square distance of a state from its set's mean, averaged over sets."""
    return float(np.mean([np.sqrt(np.mean(np.sum((S - S.mean(0)) ** 2, axis=1))) for S in sets]))


def between(sets: list[np.ndarray]) -> float:
    """The root-mean-square distance of the set means from their overall mean."""
    M = np.stack([S.mean(0) for S in sets])
    return float(np.sqrt(np.mean(np.sum((M - M.mean(0)) ** 2, axis=1))))


def off(X: np.ndarray, c: np.ndarray, axes: np.ndarray) -> np.ndarray:
    """X about c, with the part in the span of `axes` (orthonormal rows) removed."""
    Z = np.asarray(X, dtype=float) - c
    if len(axes):
        Z = Z - (Z @ axes.T) @ axes
    return Z


def rms(Z: np.ndarray) -> float:
    """Root-mean-square length of the rows: the unit of D."""
    return float(np.sqrt(np.mean(np.sum(Z ** 2, axis=1))))


def top_direction(Z: np.ndarray) -> np.ndarray:
    """The direction of largest second moment of the rows (not centred: a shift counts)."""
    _, _, vt = np.linalg.svd(Z, full_matrices=False)
    return vt[0]


def alignment(A: np.ndarray, B: np.ndarray) -> float:
    """How far two sets of orthonormal axes span the same space: the mean squared cosine of their principal angles,
    1 for the same space, 0 for orthogonal ones."""
    if len(A) == 0 or len(B) == 0:
        return 0.0
    return float(np.sum((A @ B.T) ** 2) / min(len(A), len(B)))


def rmse(Y: np.ndarray, Yhat: np.ndarray) -> float:
    return rms(Y - Yhat)
```

### `substrate/axes.py`

```python
"""Retained axes and their admission (the Entity tab, substrate decisions 2 and 3).

The code is retained, not consumed: every axis considered is kept as an object after it has served. Admission has two
criteria. An axis that lowers held-out residual by more than D enters the predictive code (the product criterion). One
that does not is retained as structural, tagged untested and excluded from residual, until the research track states a
structural test. Nothing is discarded.

By default residual is static: the root-mean-square distance from the code's affine subspace through the origin, in
the units of D. A caller may supply its own residual (v10: dynamic, the one-step error of a fit) and its own unit (v10:
the measured residual floor, since D as noise scale and D as threshold proved not to be the same unit).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from substrate.geometry import off, rms, unit


@dataclass
class Axis:
    vector: np.ndarray
    kind: str                 # "predictive" or "structural"
    gain: float               # held-out residual removed, in the units of D (negative: it made things worse)
    source: str               # where the candidate came from, e.g. "self" or "frame"
    at: int                   # the step it was considered at
    tags: list[str] = field(default_factory=list)


class Code:
    """Axes through the origin's centroid `c`. Only predictive axes take part in residual."""

    def __init__(self, c: np.ndarray, D: float, name: str = "code", unit: float | None = None):
        self.c, self.D, self.name = np.asarray(c, dtype=float), float(D), name
        self.unit = float(unit) if unit is not None else self.D       # what a gain must exceed to admit
        self.axes: list[Axis] = []

    def predictive(self) -> np.ndarray:
        rows = [a.vector for a in self.axes if a.kind == "predictive"]
        return np.stack(rows) if rows else np.zeros((0, len(self.c)))

    def structural(self) -> list[Axis]:
        return [a for a in self.axes if a.kind == "structural"]

    def residual(self, X: np.ndarray) -> float:
        return rms(off(X, self.c, self.predictive()))

    def consider(self, candidate: np.ndarray, held_out: np.ndarray | None = None, source: str = "", at: int = 0,
                 residual=None) -> Axis:
        """Test a candidate and retain it either way. `residual(axes)` gives held-out residual under a set of axes
        (orthonormal rows); without it, static residual on `held_out`."""
        P = self.predictive()
        v = unit(candidate - (P.T @ (P @ candidate)) if len(P) else candidate)
        if residual is None:
            residual = lambda axes: rms(off(held_out, self.c, axes))  # noqa: E731
        gain = (residual(P) - residual(np.vstack([P, v[None]]))) / self.unit
        if gain > 1.0:
            axis = Axis(v, "predictive", gain, source, at)
        else:
            axis = Axis(v, "structural", gain, source, at, tags=["untested"])
        self.axes.append(axis)
        return axis

    def save(self, path: str | Path) -> None:
        path = Path(path)
        np.save(path.with_suffix(".npy"), np.stack([a.vector for a in self.axes]) if self.axes else np.zeros((0, 0)))
        meta = {"name": self.name, "c": self.c.tolist(), "D": self.D, "unit": self.unit,
                "axes": [{k: v for k, v in asdict(a).items() if k != "vector"} for a in self.axes]}
        path.with_suffix(".json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: str | Path) -> "Code":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        code = cls(np.array(meta["c"]), meta["D"], meta["name"], meta.get("unit"))
        vectors = np.load(path.with_suffix(".npy"))
        code.axes = [Axis(vector=vectors[i], **a) for i, a in enumerate(meta["axes"])]
        return code
```
