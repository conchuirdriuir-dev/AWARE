Reviewer: fresh Claude agent, system-identification lens; given the round 3 packet only.

## Summary

The round 3 failures come from one modelling error, not from the detectors' thresholds. The observer's model structure
does not match how the plant uses its input. The generator applies each external as a zero-order hold: once it arrives
it drives every later step until the next one does (`round3.py:455-461, 474, 477`). The observer instead sees a sparse,
masked impulse `m(t)·e(t)` with an additive marker, against a past model that is linear in the lags and the same on
arrival and non-arrival steps (`round3.py:499-502`). Such a model cannot switch regimes when an external arrives. On
non-arrival steps the lags have to rebuild the held input. On arrival steps they have to stop doing that and hand over
to `e(t)`. A shared linear lag block cannot do both. As a result, coupling has no positive expectation even in healthy
work: the progress median is −0.013 in the raw results, and it is still −0.0135 on a 3000-step window. With δ coming
from a coupling that is zero or negative, 57–82% of runs have no reference, and that alone fails bars 1, 4, 5 and 6. The
same flaw explains the inverted lag-10 AUC (0.13 and 0.20). The masked echo `m(t)·z(t−10)` is a gated function of the
past, which the linear past model cannot represent, so it gets credited as coupling.

I made one change: arrival-gated lags `m(t)·z(t−j)` in both fits, with everything else as coded. In 30 trials a
structure, the no-reference share went from 40–77% to 0–7%. The mastery-over-loop and progress-over-echo AUCs went to
0.97–1.00 at every lag. Loops read at zero in 93–97% of runs before the probe and 90–97% after. Valid inference went
from 0.06 to 0.83.

What still fails after that change has sysid causes:
- **The probe reading is diluted.** It is scored over the whole 148-row window, of which the probe is 20 rows. The
  contiguous folds and the gap also guarantee that B̂ is never trained on probe rows when it is tested on them.
- **The code wastes an axis.** Its first axis is the window's mean offset, and only about 10% of the external's change
  lands in the code.
- **The bounds ignore two sources of variance.** They leave out parameter-estimation variance and δ's own sampling
  variance.

The code implements the pre-registration closely. The divergences I found are minor, except that shrinkage is applied
to the input block too. None of them is the cause of the failures. The cause is a specification error in the observer.

## Findings

### 1. The observer's model cannot represent arrival-switched dynamics, so coupling has no positive expectation (blocks the inference)

**Evidence.**
- **Plant.** `s(t+1) = A s(t) + (I−A)(H(t)−c0)`, where `H` is the latest arrival (`round3.py:461, 474, 477`).
- **Observer.** The past model is `[z(t..t−11), 1, m(t)]` and the full model adds `m(t)·e(t)` (`round3.py:499-502`).
  The correct predictor of `H(t)` is `E(t)` on arrival steps and the lag-rebuilt `H(t−1)` otherwise. That needs lag
  coefficients that depend on `m(t)`, and an additive marker only shifts the intercept.
- **The raw results show this.**
  - Every structure's median late coupling is negative or about 0 (progress −0.0126, mastery −0.0235).
  - 57–82% of runs have no reference.
  - The bootstrap bounds are wide next to δ. In my 30-trial rerun of the code, the median 80% CI width was 0.024
    against a median δ of 0.006.
- **It is structural, not a lack of power.** On long stationary progress runs (window 3000, 8 runs), the coded
  estimator gave mean −0.0135 (sd 0.009). With gated lags it gave +0.0315 (sd 0.003).
  Command: `python /tmp/review-sysid/long.py`.
- **The lag-10 inversion has the same cause.** `m(t)·(z(t−10)+noise)` adds information only because the past model
  cannot gate its lag-10 term.
- **Rerun of the readings with gated lags.** Command: `python /tmp/review-sysid/gated_round.py {coded|gated} 30`,
  30 trials a structure, seed 7, same windows, same δ rule, same bootstrap.

| quantity | coded | gated past |
|---|---|---|
| no reference, progress / mastery / dynamics shift | 0.40 / 0.67 / 0.77 | 0.00 / 0.00 / 0.00 |
| AUC mastery over 3a, 3b lag 1/3/10 | 0.25, 0.43, 0.37, 0.12 | 1.00, 1.00, 1.00, 1.00 |
| AUC progress over 3b lag 1/3/10 | 0.64, 0.56, 0.20 | 0.98, 0.97, 0.98 |
| loops at zero before probe (3a, 3b ×3) | 0.10–0.27 | 0.93–0.97 |
| loops at zero after probe | 0.07–0.27 | 0.90–0.97 |
| progress / topic / dynamics / mastery at zero before probe (incl. no reference) | 0.83 / 0.93 / 0.90 / 0.97 | 0.13 / 0.07 / 0.10 / 0.00 |
| valid inference not at zero after resume | 0.06 | 0.83 |

- **Two simpler alternatives are worse.** Feeding the held input `H(t)` (which the observer can build from `E` and `m`)
  restores progress coupling to 0.10. But lag-10 echoes then read open, with progress-over-echo AUC 0.34, because a
  held echo is up to about 20 steps old. Centring the basis alone raises the share of progress runs with a positive
  reference to only 0.67. Command: `python /tmp/review-sysid/variants.py 30` and `python /tmp/review-sysid/v2.py`.

**Bug or not.** This is not a bug against the pre-registration: the code does what was frozen ("the arrival marker in
both fits"). It is an error in the specification of the observer's model structure.

**Proposed change.** In `coupling()`, add `m(t)·z(t−j)` for j = 0…p−1 to the past design, so it is also in the full
design. Pre-register this as a new round with the same bars, and run it at 200 trials, seed 0.

**What would show it wrong.** On the 200-trial, seed-0 run with gated lags, either of these:
- progress has no reference in more than 5% of runs;
- any mastery-over-loop or progress-over-echo AUC is below 0.9.

Either result would mean the gain I measured at 30 trials is noise, or that something else is limiting.

### 2. The probe is read over the whole window and B̂ is never trained on probe rows (blocks bar 6 for structure 8)

**Evidence.**
- **The probe is a small part of the window.** In the after-probe window [340, 500), the probe takes rows 109–128 of
  148.
- **The probe rows never reach training.** With contiguous 5-fold and a 12-step gap (`round3.py:505-510`), the folds
  that test probe rows (3 and 4) exclude every probe row from their training set.
- **The training input carries little excitation.** In structure 8 the training rows' input innovation power is
  0.15² ≈ 2% of normal, so with ridge 1.0 B̂ is shrunk hard.
- **Coupling measures input innovation times gain.** That is why structure 8 reads at zero before the probe by
  construction (gated: 93%), and why it reads at zero in 77% after.
- **Gated past, 20 trials.** Command: `python /tmp/review-sysid/probe.py`.

| structure | window coupling (as scored) | probe rows only, contiguous folds | probe rows only, interleaved folds |
|---|---|---|---|
| progress | 0.041 | 0.071 | 0.063 |
| healthy, declining | 0.007 | 0.024 | 0.047 |
| 3a | −0.007 | −0.005 | 0.001 |
| 3b lag 1 | 0.002 | 0.007 | −0.001 |
| 3b lag 10 | 0.003 | 0.006 | 0.001 |

  Median δ is about 0.02. Cutting the ridge to 1e-3 overfits and makes every structure negative.

**Proposed change.** Read the probe as an event-locked estimate:
- Score the held-out error difference only on rows whose target follows a probe response.
- Use folds that put probe rows in both training and test, for example interleaved folds over the window, or
  leave-one-probe-block-out.
- Compare the result with δ.

The pre-probe reading stays as it is.

**What would show it wrong.** At 200 trials, structure 8's event-locked probe coupling has its lower 90% bound above δ
in fewer than 90% of runs. That would mean 20 responses are too few to identify B at this noise level, and the next
step would be a longer probe rather than a different readout.

### 3. The per-window code spends an axis on the mean offset and is not chosen for the input's observability (weakens)

**Evidence.**
- **The first axis is the mean offset.** `cp.basis` takes the SVD of `S − c`, uncentred (`coupling.py:697-699`). Its
  first axis is the window's mean offset (median |cos| = 1.000 over 20 progress late windows), which the intercept
  already models.
- **The code is chosen by state variance.** It captures 95% of the state power about `c`. But the external's change
  from one arrival to the next puts only 0.101 of its power in the 3-axis code, against 3/32 = 0.094 if it were
  isotropic.
- Command: `python /tmp/review-sysid/basis.py`.
- **Effect under the gated estimator** (`python /tmp/review-sysid/axes.py`, 20 trials).
  - Centring the basis raised the progress late median from 0.029 to 0.040 with 2 axes, and from 0.050 to 0.056 with 3.
  - The AUCs were already saturated (0.96–1.00).

**Proposed change.** Centre the basis on the window mean. Then test one alternative code against PCA: axes chosen by
reduced-rank regression of the past-model residual on the arrival innovation. Compare the two on the ratio of progress
coupling to its bootstrap width.

**What would show it wrong.** The centred or input-chosen code does not raise that ratio by at least 20% on 50 progress
runs.

### 4. The bounds leave out estimation variance and δ's variance (weakens bar 5)

**Evidence.**
- **What the bootstrap holds fixed.** It resamples pooled held-out errors with the fitted models fixed
  (`round3.py:521-526`). So it leaves out the variance of the 41-parameter fits and the variance of δ, which is a
  single noisy reading (`round3.py:541-542`).
- **Gated estimator (finding 1's run).** Progress still reads at zero in 13% of runs, against a bar of 10%. Its median
  late coupling is 0.038 against a median δ of 0.026.

**Proposed change.** Read at zero on the ratio late / reference, with a joint bootstrap. Resample blocks from both
windows' held-out errors in each replicate. Then:
- at zero: the upper 90% bound of the ratio is below 0.5;
- open: the lower 90% bound is above 0.5.

Optionally, refit inside each replicate by resampling residual blocks.

**What would show it wrong.** With the gated estimator, progress's at-zero share stays above 10% under the ratio rule.
That would put the cause in genuine window-to-window variation in input innovation energy, not in how the bounds are
built.

### 5. The lag horizon p sets which echoes coupling can see, and p = 12 costs prediction (weakens; extend the test)

**Evidence.** Gated estimator, 20 trials, command `python /tmp/review-sysid/order.py`.

| p | progress-over-echo AUC, lag 1 | progress-over-echo AUC, lag 10 | progress held-out residual |
|---|---|---|---|
| 1 | 0.96 | 0.04 | 0.415 |
| 2 | 1.00 | 0.05 | 0.414 |
| 4 | 1.00 | 0.05 | 0.401 |
| 12 | 0.99 | 0.99 | 0.483 |

An echo reads as redundant only if p is greater than its lag. Meanwhile p = 12 raises the held-out residual by about
20%, because the gated design has 74 parameters against about 94 training rows. So p is a detection horizon, not a
model order to be picked by order selection.

**Proposed change.**
- Add proliferation at lags 15 and 20 as structures. They are expected to read open with p = 12.
- State the echo horizon as a property of the detector.
- Consider sparse lag sets, such as {0, 1, 2, 4, 8, 16}, so the horizon grows without the parameter cost.

**What would show it wrong.** Lag-20 echoes read at zero with p = 12. That would mean the redundancy comes through the
held state rather than through the lags.

### 6. Fidelity divergences (minor unless noted)

| # | where | divergence | could it change a result |
|---|---|---|---|
| a | `round3.py:512-517` | Shrinkage puts λ = 20 on every column of both fits: the input block `e`, the intercept and the marker. The pre-registration says only "the past model ... with ridge toward a prior". | Yes, for bar 7. Shrinking B̂ toward zero shrinks coupling, so the "spread narrows" test is met by scale. In 25 progress runs, sd went from 0.022 to 0.0032 but the median went from −0.0054 to 0.0021, and the correlation between shrunk and unshrunk is only 0.69 (`python /tmp/review-sysid/shrink.py`). Put λ = 20 on the lag blocks only, and judge the spread as sd / median or relative to δ. |
| b | `round3.py:588, 614` | A loop run with no reference is labelled "no reference", so it never counts toward "loops called dynamics shift". The rule says a run with no δ counts against every bar it enters. | It understates the loop-as-dynamics-shift rate. Bar 1 fails anyway. |
| c | `round3.py:545-547` | Point contrast and descriptor read the held external `H`, while coupling reads the sparse `E`. So two detectors in one round use different input models. | Small for their results. It is relevant to finding 1, because `H` is the representation the plant uses. |
| d | `round3.py:565-571` | â comes from a lag-1 fit with no input term, on window 300–460. It absorbs the held input's persistence: â = 0.93 against a true A = 0.5. The pre-registration does not name the window. | The prior pulls toward the wrong self-dynamics. Bar 7 only. |
| e | `round3.py:456` | Proliferation switches on abruptly at step 200, although "Ramps run 200–260". | No. The proliferation readings come from the late (300–460) and after-probe (340–500) windows, which start after the ramp would have ended; the reference window ends at 200. |
| f | `round3.py:430, 443, 582` | The probe is generated in every run, and only the after-reading is gated on "not open". | No. The late window's last regressor is step 458. |
| g | `round3.py:478` | The in-context reading `m·(0.5e + 0.5z(t))` hands the past model exactly the gated lag-0 term it is missing (finding 1). | Bar 8 is confounded. Rerun it with the gated past before drawing the framing claim. |

Divergences checked and found faithful:
- The window bounds, the rows, the 12-step gap (exactly 12 rows excluded on each side), the pooled held-out errors, and
  the 10th and 90th percentiles giving one-sided 90% bounds.
- The intake: steps 0–39, twice the noise, every step.
- Arrival probability 0.4.
- Structure 8's novelty ramp to 0.15.
- Valid inference: the pause and on-trend extrapolation, 200–239.
- Separate calibration seeds.
- Bars 4, 5 and 6 counting "no reference" against the run.

## Not actionable

- The design defines coupling as unique contribution: input innovation times gain. This makes a healthy partner who
  supplies little new information indistinguishable from a closed loop without a probe. That is a property of the
  definition, not of the estimator. Finding 2 only makes the probe able to show it.
- The toy's isotropic external noise is probably a worse case for observability in a 3-axis code than real sessions,
  where tool results may be concentrated in a few directions. Whether that holds is for logged sessions to decide.
