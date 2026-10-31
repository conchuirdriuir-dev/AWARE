# Toy bed, round 2

The implementation record for one round of the product track's toy bed (PLAN.md, product step 3): its
pre-registration, frozen by the commit before its code, then its result and any post-hoc checks. The design it tests
is the Big Picture doc (v10's drift tab, Tests). Run with `python -m product.round2`.

## Step 3, round 2: v10's detectors (pre-registered 2026-10-31, before its code)

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
