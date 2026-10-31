# Toy bed, round 3

The implementation record for one round of the product track's toy bed (PLAN.md, product step 3): its
pre-registration, frozen by the commit before its code, then its result and any post-hoc checks. The design it tests
is the Big Picture doc (v10's drift tab, draft section "The step, and coupling at zero"). Run with `python -m product.round3`.

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

**Correction after review** (2026-10-31; `reviews/round3/`). The diagnosis above is wrong. Three independent reviews
found that the observer is misspecified for externals held in context: the fit sees an external only on arrival and
cannot switch its lag coefficients when one arrives. It is not a power problem (coupling stays negative on windows of
2,600–3,000 steps). With arrival-gated lags added to both fits, 30 runs a structure: reference above zero in 97–100%,
every ranking AUC 1.00. The bootstrap bounds were also too narrow, and the shrinkage test aimed at the wrong
coefficient. Round 3's bars stand as failed; the triage is `reviews/round3/triage.md`.
