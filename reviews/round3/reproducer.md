Reviewer: fresh Claude agent, adversarial-reproducer lens; given the round 3 packet only.

**Reproduction.** The reported numbers reproduce exactly. `python -m product.round3 --trials 200 --out /tmp/review-reproducer/round3.json` (81 s) gives a JSON identical, key for key, to the packet's raw results. `product/round3.py` in the repository is byte-identical to the packet's copy. The test suite passes (27 tests).

## Summary

I traced the code by hand and it implements the windows, timing, folds, gap, readings and pass arithmetic as pre-registered. I checked these points specifically:
- An external arriving at step k drives s(k+1), and the fit row t pairs e(t) with z(t+1).
- The late window's last target is s(459), so the probe at step 460 cannot leak into it, and generating the probe in every run is harmless.
- The gap excludes exactly 12 rows on each side of each test fold.
- A missing reference counts against every bar it enters.

Every coupling-based bar fails for one root cause, and I did not find a single-line bug behind it. The pre-registered encoding of the external has no measurable signal under sparse arrivals: the raw external, zero-filled when none arrives and entered additively beside the arrival marker. The entity is driven by the held external, which persists, while the linear fit only sees it on the step it arrives. As a result:
- Coupling in healthy progress is centred slightly below zero, not above it.
- Its sign stays negative with 2,600-step windows, so this is not a power problem.
- It recovers at once if the fit is given the held external, which the observer can rebuild from what it has seen, or the innovation at arrival.

From this, δ is a positive draw of noise, selected by "reference above zero". "No reference" is 57–82% of every structure. Among runs that have a reference, healthy and closed-loop runs read "at zero" at about the same rate.

Three further problems are secondary:
- The block-bootstrap bounds cover about half as often as they claim.
- The point contrast and descriptor read the held external rather than the raw external the pre-registration names. This divergence is load-bearing for the descriptor.
- Shrinkage's narrower spread comes mostly from shrinking the external's own coefficients.

## Findings

### 1. Coupling as encoded carries no signal for fresh sparse externals (blocks the inference)

**Evidence.** The encoding is at `product/round3.py:117`: `e = ((X - c) @ U.T) * M`, beside a marker column. The generator drives the entity by `held` (lines 78, 92), which persists between arrivals. The table shows medians of late-window coupling (start 300) over 40–80 runs each. All columns use the code's own `coupling()` except the long-window column, which uses T = 3,000 and a 2,600-step window.

| structure | as coded | arrivals every step | long window (2,600 steps) | held external H, observer-rebuilt | innovation M·(E − last seen) |
|---|---|---|---|---|---|
| progress | −0.010 to −0.016 | +0.094 | −0.008 (0/10 runs > 0) | +0.093 | +0.022 |
| mastery | −0.018 to −0.024 | +0.265 | −0.014 (0/10 > 0) | +0.206 | +0.110 |
| 3a dullness | −0.005 | −0.016 | | −0.035 | −0.002 |
| 3b lag 1 | −0.016 | +0.014 | | +0.036 | +0.007 |
| 3b lag 10 | +0.007 | +0.006 | | +0.135 | +0.052 |

Other checks on the coded quantity:
- **Folds.** Shuffled (non-contiguous) folds give progress −0.0001 and mastery +0.006, so it is not extrapolation across contiguous folds.
- **Lag order.** p = 1 or p = 3 does not help (progress −0.010 and −0.007), so it is not mainly the 41-parameter fit.
- **Reference against late.** In progress the reference is above zero in 43% of runs, and its correlation with the same run's late coupling is −0.12.

Late-window AUCs over 80 runs a structure:

| encoding | mastery over 3a / lag 1 / lag 3 / lag 10 | progress over lag 1 / lag 3 / lag 10 |
|---|---|---|
| as coded | 0.23 / 0.43 / 0.45 / 0.15 | 0.59 / 0.61 / 0.25 |
| held external H | 1.00 / 1.00 / 1.00 / 0.88 | 0.90 / 0.88 / 0.23 |
| innovation | 1.00 / 1.00 / 1.00 / 0.95 | 0.91 / 0.86 / 0.10 |

**Mechanism.** On an arrival step, the entity's drive switches from the old held value to the new one. The only part of this the past cannot predict is the new external's fresh noise: about 0.5·√(3/32) of its scale on 40% of steps. Using it needs the past coefficients to change with the marker, which is a product term an additive linear fit cannot form. The repository's own test `test_round_three_coupling_recovers_when_an_external_arrives_every_step` passes only because it sets ARRIVAL = 1.0. No test checks the sparse case the pre-registration runs.

**Change.** Pre-register a coupling variant whose external column is the held external H (the last arrival, rebuilt from the observer's own record) or the innovation M·(E − last seen E), with the marker kept. Rerun bars 2 and 3 at 200 trials.

**What would show this wrong.** Progress reference coupling under the H encoding is still not above zero in more than 20% of 200 runs, or its correlation with late coupling stays below 0.3.

### 2. δ is selected noise, so bars 1, 4, 5 and 6 are scored mostly on "no reference" (blocks the inference)

**Evidence.**
- "No reference" (reference ≤ 0, line 160) runs from 0.575 for mastery to 0.815 for valid inference.
- Among runs with a reference, the "at zero" rate before the probe is similar for healthy and closed-loop structures:

| structure | at zero, given a reference |
|---|---|
| progress | 0.26 / 0.385 = 0.68 |
| mastery | 0.81 |
| 3a | 0.77 |
| 3b lag 1 | 0.76 |
| 3b lag 3 | 0.79 |
| 3b lag 10 | 0.13 |

- Bar 5's "at zero" shares (0.845–0.92) are about 70% "no reference" (line 237 counts both).
- Dynamics-shift separation of 0.12 is the descriptor firing (≈0.80, finding 4) times a coupling gate that is passed in only 0.155 of runs.

Conditioning on a positive point estimate of a near-zero quantity puts δ in the upper tail of noise, and the late window is an independent draw. This follows from finding 1, but it would persist in a weaker form after the fix.

**Change.**
- Report every coupling bar both unconditionally and conditional on a reference existing.
- Require the reference's own lower 90% bound to be above zero before a δ is formed, and report the share of runs that have one.

**What would show this wrong.** After finding 1's fix, progress runs with a reference read "at zero" late in no more than 10%, while 3a and 3b at lags 1 and 3 read "at zero" in more than 50%.

### 3. The block-bootstrap bounds undercover by about 2× (weakens the inference)

**Evidence.**
- In 150 progress runs, the across-run sd of late coupling is 0.0185, but the median distance from the point estimate to each bound is 0.012. A normal 80% interval would need about 0.024.
- Taking the across-run median as the window's true value, P(low > truth) = 0.31 and P(high < truth) = 0.17, against 0.10 nominal for each.

**Cause.** Lines 138–144 resample held-out squared errors in 13 blocks without refitting, so the fold models' estimation variance is left out. This variance dominates with about 100 training rows for 41 coefficients. Undercoverage inflates both "open" and "at zero" and shrinks "undecided", which is what the probe rule keys on.

**Change.** Refit inside the bootstrap by resampling row blocks, refitting the fold models and recomputing the held-out errors. Report the empirical coverage of both bounds on progress at 200 runs.

**What would show this wrong.** A refitting bootstrap still gives P(low > truth) > 0.2.

### 4. Fidelity: the point contrast and descriptor read the held external, not the raw one (weakens the inference)

**Evidence.**
- The divergence is at `product/round3.py:163–165`: `_held` passes `run["H"]` as E. The pre-registration says: "The detectors use the external read alone (the raw external)".
- The thresholds are recalibrated on 300 progress runs each way. "Raw" means the observer's own zero-filled external, set to c where none arrived.

| structure | descriptor fires, held external H | descriptor fires, raw external |
|---|---|---|
| dynamics shift | 0.80 | 0.13 |
| 3a | 1.00 | 0.88 |
| 3b lag 1 | 0.04 | 0.18 |
| progress | 0.01 | 0.01 |

Rows 1, 2 and 5 of separation, and "loops called a dynamics shift", would change materially if the code followed the pre-registration.

**Change.** Either amend the pre-registration to say that every detector reads the observer-rebuilt held external, which also legitimises finding 1's fix, or rerun with the raw external. Add the descriptor-fire rate per structure to the report, so that gate failures and descriptor failures are separable.

**What would show this wrong.** Separation and loop-as-dynamics shares unchanged by more than 0.03 when the code is switched to raw externals.

### 5. Lag-10 echoes read more coupling than progress under every encoding (weakens bar 3)

**Evidence.**
- Progress-over-lag-10 AUC is 0.20 as coded, 0.23 under H and 0.10 under the innovation.
- Lag-10 median coupling is +0.009 against progress −0.013.

The held echo s(k′−10) persists after its arrival at k′. For k − k′ ≥ 2 (probability 0.6² = 0.36 a step), its source lies before t − 11, outside the p = 12 past. To the past fit it is genuinely new information, whatever its provenance. On the arrival step itself the echo is inside the lags, but only multiplied by the marker (finding 1).

**Change.** Report lag-10 coupling as a function of p (12, 16, 24), using 2 axes or a longer window to keep the fit determined. Alternatively, add the observer's state at the external's best-matching lag to the past block.

**What would show this wrong.** Lag-10 coupling stays above progress's at p = 24.

### 6. Shrinkage's narrowed spread comes mostly from shrinking the external's coefficients (minor; bar 7's decision is unchanged)

**Evidence.** Lines 131–135 apply λ = 20 to every coefficient of both fits, including the external block and the marker, toward zero. The pre-registration says "the past model is fitted per window with ridge toward a prior". Progress late-coupling sd over 100 runs:

| fit | sd |
|---|---|
| unshrunk | 0.0222 |
| as coded | 0.0044 |
| λ = 20 toward zero, no persistence prior | 0.0072 |
| λ = 20 on the past block only, ridge 1 on the external | 0.0147 |

Most of the narrowing is coupling being forced toward zero, not steadier self-dynamics.

**Change.** Apply λ and the prior to the past block only, then re-score bar 7, including 3b's false-open rate.

**What would show this wrong.** Past-only shrinkage gives an sd within 0.002 of the coded version.

### 7. Every run's bootstrap uses the same resampling indices (minor)

**Evidence.** Line 138 uses `np.random.default_rng(start)` when no rng is passed, and `read()` never passes one. All 2,200 runs share the same block draws at a given window start, so the bootstrap errors in the shares are correlated across runs rather than averaging out.

**Change.** Pass a per-run rng, derived from the trial index.

**What would show this wrong.** Re-scored shares move by less than 0.01.

### 8. â's estimation window is unstated in the pre-registration (minor)

**Evidence.** `calibrate` fits lag-1 plus intercept with ridge 1 on the late window, 300–460 (line 185). The pre-registration names no window. â = 0.93 reflects the persistence of the held external and the walk, not A = 0.5.

**Change.** State the window in the pre-registration. Report â fitted on the reference window as well.

**What would show this wrong.** â on 40–200 differs from 0.93 by less than 0.02.

## Not actionable

- Things I checked and found to match the pre-registration:
  - **Timing:** E(k) drives s(k+1) (lines 75–95), and the fit rows pair e(t) with z(t+1).
  - **Windows:** the reference closes at s(199), before the ramps reach s(201). The late window closes at s(459), before the probe.
  - **Folds and gap:** five contiguous folds with 12 rows excluded on each side (line 128).
  - **Bounds:** one-sided 90% bounds are taken as the 10th and 90th percentiles.
  - **Echo noise:** the 0.3 D echo noise has rms 0.3 D.
  - **Declining structure:** its novelty ramp, and its reset during the probe.
  - **In-context reading:** 0.5 × external + 0.5 × s(k).
  - **Pass arithmetic:** every bar's formula, and "no reference" counted against.
- Generating the probe unconditionally, where the pre-registration probes only runs not read "open", has no effect on any reading. "After" is only consulted when "before" is not "open", and the probe cannot reach the late window.
