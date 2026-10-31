# Plan

The design is the Big Picture doc, v9 ([Big Picture — v9](https://claude.ai/artifact/ByrrhB2Ue4NS8NzmyZXoCw), a
Claude Doc). Its **Entity** tab is the parent: what the entity is, the substrate decisions, the capacities, and the two
tracks. The **Drift detection (v10)** tab is the product track's full treatment (v9's is kept as the record);
**Tech stack** sets the build order. Gate 0 ends nothing: it is refined until the product passes it, and the doc's
"can end the project" line is ignored (owner, DECISIONS.md).
Where this file and the doc differ, the doc wins and this file is corrected.

**Two tracks, one substrate, kept apart.** The product track (drift detection, then self-correction) has the demo and
fast feedback; the research track (reasoning over the entity's own axes) has no demo yet and is the reason the project
exists. They never share a pass criterion: the product's faster feedback would win every local decision. The
substrate changes only by the owner's decision.

A step's pass criterion is written here before its code, and its thresholds are fixed before it runs. A step is done
when its test passes, not when its code exists.

## Substrate (`substrate/`)

The Entity tab's decisions, as code: the base model frozen (states read through nnterp, `states.py`); the code
retained, not consumed (`axes.py`: every axis considered is kept); admission by two criteria (predictive: lowers
held-out residual by more than D; otherwise structural, tagged untested, until the research track states a test);
provenance logged at the source (`record.py`: self or external on every chunk).

| Step | Test | Status |
|---|---|---|
| 1. Model access and D | states on the real model; D per layer on frozen paraphrase sets | **code built; plumbing passes on real Qwen3-0.6B weights on CPU (2026-10-31); awaits the card and the frozen sets** |
| 2. Logging | per-chunk activation capture, in context and alone; provenance tagged at creation and carried when re-read | **reader and record built; the harness that writes them is product step 4** |

**State** (provisional): the residual stream after a layer, mean-pooled over a chunk's tokens (special tokens left
out), read alone (`states`) or inside the whole sequence (`states_in_context`). **D**: the root-mean-square distance
of a state from its paraphrase set's mean, averaged over sets, per layer; the layer proposed is the one with the
largest between-to-within ratio, and the owner chooses. **Material:** 20 sets of 5 paraphrases
(`material/paraphrases.jsonl`), drafted by the model, not yet frozen. **Pass:** states for every text on both models
at every layer asked for, and the same text read twice gives the same state.

## Product track (`product/`): drift detection, then self-correction

| Step | Doc | Test | Status |
|---|---|---|---|
| 0. Text-only floor | v9 step 0 | the text panel on the owner's 45 sessions | **done by the owner, twice:** text sees the owner leave an aim (AUC 0.65–0.71), not the agent |
| 3. Toy bed | v9, v10 and the v10 draft | below | **round 2 (v10): coupling passes its three bars, separation fails on the "at zero" rule; round 3 (sparse externals) fails every bar: coupling vanishes when externals are sparse (below)** |
| 4. Labeled logs | Tests, in order | detectors on the 65 departures, agent-authored first | open: owes the owner's confirmation pass (drift or re-aiming) and logging on the card |
| 5. Reference scorer, coherence score | Tests, in order | intact versus shuffled work | open: needs the setting and the owner's reference work |
| 6. Compass and harness, arm A; Gate 0 | Tests, in order | the compass predicts drift on held-out stretches above a pre-registered level, and beats self-residual alone | open. Refined until passed; ends nothing |
| 7. Arms S and C; main demo | Main demo | C closes half the A–B gap on both scores and beats S by more than run-to-run noise | open |

### Step 3: the toy bed (pre-registered 2026-10-31, before its code)

**Generator** (v9): s(t+1) = A·s(t) + B·E(t) + noise at scale D, about the origin's centroid c (the mean of the first
10 external states). Synthetic for now: 32 dimensions, D = 1, 240 steps, the structure planted at step 120. Baseline:
A = 0.5·I, B = I − A (so E is the equilibrium), E a slow random walk (0.3 D a step) observed with noise D. The six
structures change only A, B or E:

| Structure | Planted at step 120 |
|---|---|
| 1. Topic shift | E jumps 6 D in a random direction |
| 2. Dynamics shift | A becomes two rotations (0.9, 0.6 rad) in a random 4-dimensional subspace; B = I − A, so the equilibrium stays |
| 3a. Closed loop, dullness | over 60 steps B → 0, A → 0.98·I, and a drift of 0.3 D a step along one direction grows in |
| 3b. Closed loop, proliferation | E arrives as s plus noise 0.3 D: provenance external, content self |
| 4. Mastery | over 60 steps noise → 0.3 D and A → 0.2·I |
| 5. Noise spike | three steps of noise 20 D |
| 6. Progress (control) | nothing: the baseline |

**Detectors.** Windows of 40 steps.
- Point contrast: the distance between the mean states of two windows, in D.
- Trajectory contrast: linear dynamics fitted in each window (each window's own mean removed, in the top 4 principal
  axes of the two together); the distance between the sorted magnitudes of the two fits' eigenvalues.
- Each at onset (the windows either side of step 120) and for persistence (the window before onset against steps
  160–200). A detector fires when its statistic exceeds the 99th percentile of the same statistic over 500 progress
  runs drawn with a separate seed.
- The observer, in blocks of 20 steps from the start: a self code (candidates from the block's own and external
  states) and a frame (candidates from external states only), both through c, each candidate the block's first half's
  top residual direction, admitted by `axes.py` on the block's second half of the entity's states. Per block:
  self-residual and frame-residual (static, in D) before updating, the margin (frame minus self), frame growth (1 if
  the frame admitted an axis) and distance to the origin.

**Pass** (v9's bars):
- Rows 1, 2, 5, 6 separated at 90% or better, each over 200 trials, by these signatures: topic shift, point contrast
  persists and trajectory contrast does not; dynamics shift, the reverse; noise spike, something fires at onset and
  nothing persists; progress, nothing fires.
- Row 3 against row 4: the margin over steps 180–240 ranks closed-loop blocks above mastery blocks at AUC 0.8 or
  better, each variant against mastery.
- Row 3b against row 6: frame growth over steps 120–240 is lower in 3b than in progress, at AUC 0.8 or better.

The band's edges and the compass cells are reported, not passed: self-residual in band between 0.75 and 2 D, frame
residual high above 2 D (provisional). The generator's closed loop is a conjecture (v9, "Limitation"); if the
detectors pass here and fail on labeled logs, the generator is at fault.

**First run** (2026-10-31, `python -m product.toybed`, 200 trials a structure, seed 0). Not passed.

| Bar | Result | Pass |
|---|---|---|
| Separation: topic shift / dynamics shift / noise spike / progress | 0.99 / **0.25** / **0.74** / 0.97 | no |
| Margin AUC, dullness over mastery / proliferation over mastery | 0.86 / **0.57** | no |
| Frame growth AUC, progress over proliferation | **0.00** | no |

What each miss says, each mapped to an item already open in v9's tension log:
- **Dynamics shift (0.25).** The fitted eigenvalues do rise (about 0.6 to 0.9), but the externals' slow walk puts a
  near-1 mode in every window, and 40-step fits are too noisy to clear a 99th-percentile threshold set on progress
  runs. A post-hoc check, not a retune. Bears on "What contrast yields".
- **Noise spike (0.74).** Only point contrast sees it: a spike decays at the entity's own rate, so its dynamics do
  not change, and trajectory contrast is right to stay quiet. The signature asks one detector to catch it.
- **Proliferation.** Frame growth is higher, not lower, when externals are generated from the entity (5.8 admissions
  against 2.1 in progress), and the margin cannot tell it from mastery. This is v9's decisive row 3b failing as the
  doc warned: the frame measures provenance, not new directions, and the person-inside-the-loop case is invisible to
  it. Bears on "What the closed-loop statistic is" (coupling B̂ rather than frame growth).
- **The band.** Static self-residual is about 2 D in every structure (it carries the externals' walk), above the band
  in the closed loop (5.1) rather than below it, and mastery does not fall below the band; the commonest cell in
  progress is "lost". The tension log predicted the lower edge would not survive the generator's noise. Bears on
  "Which residual" (dynamic, not static).
- **What passed.** Point contrast separates topic shifts, and the margin catches the dullness closed loop (0.86).

Any change to the detectors in answer to this is a new pre-registration and a new run, not an edit to this one.

### Step 3, round 2: v10's detectors (pre-registered 2026-10-31, before its code)

v10 replaced frame growth with coupling, static residual with dynamic, and removed the band. Same generator as round 1,
with 360 steps (onset at 120) so the late windows clear the 60-step ramps, and these changes from v10:
- **3b, proliferation,** at three echo lags: from onset the externals are the entity's state 1, 3 or 10 steps back plus
  noise 0.3 D. Scored at each lag separately, never pooled.
- **7, valid inference:** the externals carry a trend (0.3 D a step along one direction); at onset they pause for 40
  steps (the last external is repeated, stale); meanwhile the entity is driven by its own on-trend extrapolation; then
  the externals resume on trend.

**Detectors.** All in the code: the top principal axes of the window's own states about c. Fits are least squares
with a small ridge (1.0), fitted on the first 70% of a window and judged on the rest.
- **Coupling** (windows of 160 steps, 3 axes): the fall in held-out one-step error when the current external state is
  added to a fit already using the last p = 12 states (above the longest echo lag, 10). A Granger-style reduction,
  not the size of B̂.
- **Dynamic residual:** the held-out one-step error of that full fit.
- **Point contrast on the residual** (4 axes, p = 1): a joint fit on the 60 steps before onset; the candidate is the
  mean residual vector over the 30 steps after onset minus that over the 60 before; the statistic is its length.
- **Admission of that candidate** (`substrate/axes.py`, residual supplied by the product): the candidate's level is
  re-estimated on onset's first 15 steps and judged on the next 15; admitted if it lowers the held-out residual by more
  than the floor, the median of the same residual over progress runs with no candidate.
- **Spectral descriptor** (4 axes, p = 1): the sorted eigenvalue magnitudes of Â from the joint fit, so B̂·E is
  removed; the distance between the 60 steps before onset and steps 180–240.
- Thresholds from 500 progress runs with a separate seed: point contrast and the descriptor fire above their 99th
  percentile; coupling is "at zero" below its 5th percentile and the residual "at floor" below its 95th. A window is
  flagged closed loop when both hold.

**Pass** (v10's bars):
- Rows 1, 2, 5, 6 at 90% or better, each over 200 trials, read in order: point contrast fires and the candidate is
  admitted, topic shift; fires and rejected, noise spike; otherwise the descriptor fires and coupling is open (late
  window 200–360), dynamics shift; fires with coupling at zero, closed loop; nothing fires, progress. Row 2 also needs
  3a and 3b (every lag) called dynamics shift in no more than 10% of their runs.
- Coupling over steps 200–360 ranks mastery above 3a, and above 3b at each lag, at AUC 0.8 or better.
- Coupling over steps 200–360 ranks progress above 3b at AUC 0.8 or better, at every lag separately. A fail means
  coupling is also measuring provenance (v10: then the in-context/out-of-context divergence is needed).
- Row 7: of the windows starting after the externals resume (160, 180 and 200), at least 90% not flagged closed loop.
  v10 does not say which windows; the pause itself is predicted to dip, so this reading counts only the windows after
  it, and the whole run is reported beside it.

**Round 2 result** (2026-10-31, `python -m product.round2`, 200 trials a structure, seed 0). Three of four bars pass.

| Bar | Result | Pass |
|---|---|---|
| Separation: topic / dynamics / noise spike / progress | 1.00 / **0.82** / 1.00 / 0.98 | no |
| Closed loop called a dynamics shift: dullness / lag 1 / lag 3 / lag 10 (at most 0.10) | **0.55** / 0.00 / 0.02 / 0.02 | no |
| Coupling AUC, mastery over: dullness / lag 1 / lag 3 / lag 10 | 1.00 / 1.00 / 1.00 / 1.00 | yes |
| Coupling AUC, progress over proliferation: lag 1 / lag 3 / lag 10 | 0.93 / 0.93 / 0.94 | yes |
| Valid inference not flagged, after resume (whole run 0.94) | 0.96 | yes |

- **Coupling does what v10 says.** It reads zero for externals that echo the entity at every lag, though their
  provenance says external, and it recovers when valid inference resumes. Admission now separates topic shifts from
  noise spikes on its own (1.00 each).
- **The separation fail is the threshold for "at zero", not coupling.** Rule as written: below the 5th percentile of
  progress coupling. Progress coupling is wide (median 0.09, standard deviation 0.09), so that percentile is −0.02,
  and the dullness loop, centred on zero (median −0.016), clears it in 55% of runs. Coupling itself ranks loop below
  mastery at AUC 1.00. A post-hoc check, not a retune. Bears on v10's "Coupling under collinearity" item, and on how
  "at zero" is defined at all; a fix is a round 3 with its own pre-registration.
- **Not bars, worth knowing.** Mastery's descriptor fires (its A really does change, 0.5 to 0.2). Valid inference's
  pause makes point contrast fire and admission reject it, so it reads as a noise spike at onset, then recovers.
  Every structure's commonest late cell is "on track", for the same reason as the separation fail.

### Step 3, round 3: the step, at zero, and the probe (pre-registered 2026-10-31, before its code)

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

**Round 3 result** (2026-10-31, `python -m product.round3`, 200 trials a structure, seed 0; report in
`runs/round3.json`, not committed). **No bar passes.**

| Bar | Result | Pass |
|---|---|---|
| 1 Separation: topic / dynamics / spike / progress; loops called dynamics | 0.995 / **0.12** / 0.995 / 0.995; ≤ 0.075 | no |
| 2 Coupling AUC, mastery over dullness / lag 1 / 3 / 10 | **0.22 / 0.39 / 0.39 / 0.13** | no |
| 3 Coupling AUC, progress over lag 1 / 3 / 10 | **0.55 / 0.55 / 0.20** | no |
| 4 Valid inference not at zero after resume | **0.05** | no |
| 5 At zero before the probe: progress / mastery / topic / dynamics (at most 0.10) | **0.88 / 0.92 / 0.91 / 0.85** | no |
| 6 After the probe: loops at zero (0.26 / 0.28 / 0.31 / 0.04); structure 8 not at zero | **0.26** | no |
| 7 Shrinkage: progress spread 0.020 → 0.004; lag-10 false-open 0.13 → **0.41** | not kept | — |
| 8 Framing: alone over in context, by lag | 0.55 vs 0.50 / 0.55 vs 0.48 / 0.20 vs 0.35 | not shown |

- **Coupling vanishes with sparse externals.** Its median is at or below zero in every structure, progress (−0.013)
  and mastery (−0.024) included, so 58–82% of runs have no reference and every bar that reads δ fails with them.
- **Not a bug** (post-hoc check): the same code with an external every step gives round 2's pattern back (progress
  +0.10, mastery +0.27, dullness −0.015). At 40% arrivals, an external's unique effect is confined to the one step
  after it arrives, and that signal is smaller than the cost of fitting about 40 parameters on about 90 training
  rows. Under a heavily regularised past model, an echo at lag 10 even reads as more coupled than progress.
- **The review's warning about shrinkage, confirmed.** A scalar-persistence prior cut progress's spread fivefold and
  raised the lag-10 echo's false-open rate from 0.13 to 0.41: E absorbs what the prior gets wrong.
- **What held.** Point contrast with admission separated topic shifts, noise spikes and progress at 0.995.
- **What it says for the design.** With externals as sparse as real sessions, coupling must be read where the
  information is: around arrivals (event-locked), with a past model small enough for the window. A round 4 is its
  own pre-registration.

## Research track (`research/`): reasoning over the entity's own axes

No demo and no pass bar yet, by design. What it has:

| Item | Test | Status |
|---|---|---|
| Gate 1: negation | mirror, additive shift or contraction toward the origin; the winner beats both others by more than D | `research/gate1.py` built and tested on planted pairs; runs on the accumulated axes after the demo's arms exist (v9 build order) |
| Structural axes | retained, tagged untested (`substrate/axes.py`) | waiting for a structural test from this track |
| Relations, composition, order | angles, nesting, symmetry; predicted position under combined axes; non-commuting moves | no probe yet |

## Retired

Freeform chat (rung 1), the rung-2 memory harness, the toy world and the port of DARA's kernel, at the tag
`aware-rungs-final` (DECISIONS.md, 2026-10-30).
