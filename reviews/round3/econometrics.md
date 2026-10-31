Reviewer: fresh Claude agent, econometrics lens; given the round 3 packet only.

**Summary.** The main reason round 3 failed is a specification error in the coupling regression. It is not lack of power, and it is not that the closed loops fail to close. Externals are observed only on the step they arrive, but the entity is driven by the held external. So the correct one-step model for s(t+1) needs the arrival marker *interacted* with the lagged states: on an arrival step the stale held value, which the past lags encode, has to be switched off and the new external switched on. The fit in `product/round3.py:501-502` includes the marker only as an intercept shift. That has two effects. (i) The true held-out gain from the external under the linear specification is tiny: pooled oracle coupling is +0.004 to +0.008 against a per-run estimator SD of about 0.018. Add the cost of 9 extra parameters on about 94 training rows and the median coupling comes out negative in every healthy structure. That is why 58–69% of runs have "no reference", in a reference window that is pre-onset and identical for every structure. (ii) An arrival-gated echo M(t)·s(t−L) is *not* in the span of the linear past, so it does not read zero. The lag-10 echo has oracle coupling +0.013, more than progress at +0.004, which is why the lag-10 AUC is 0.20. I added M(t)·z(t−j) for j < 12 to both fits and changed nothing else (ridge 1, 5-fold, gap 12, same bootstrap). The reference is then positive in 100% of runs. Mastery-over-loop and progress-over-echo AUCs are all 1.00 (40 runs per arm), and dullness and all three echoes read at zero in 28–29 of 30 runs. Three problems remain with the gated model. The block bootstrap gives bounds about 1.7× too narrow. Progress still reads at zero in 17% of runs. Structure 8 (declining) cannot be told apart from a loop by an absolute-RMSE coupling, and the 340–500 after-probe window dilutes the probe's 20 steps among 148 rows. The shrinkage arm has a separate problem: its prior is set from a quantity (AR(1) persistence, 0.93) that is not the coefficient it shrinks (the lag-1 block of a 12-lag fit, about 0.61). This biases the comparison toward false openings.

All checks are in `/tmp/review-econometrics/` (`flex.py` reproduces `round3.coupling` exactly with `p=12, interact=0`; checked with `np.allclose` against `round3.coupling`).

## Findings, most serious first

### 1. Coupling regression omits the arrival × lag interaction; the stated null ("an echo adds nothing beyond its past") does not hold under this specification. Blocks the inference. A bug relative to the design's own claim.

*Evidence.* `round3.py:501` builds the past design as `[z(t)…z(t−11), 1, M(t)]`, and `:499` sets e(t) = M(t)·(X−c)U. The data-generating process is s(t+1) = 0.5 s(t) + 0.5 h(t) + noise, where h is the held external. Between arrivals the past lags identify h. On an arrival step h jumps, and an optimal linear predictor has to change its lag weights. A model that is linear in lags, with M only in the intercept, cannot do that. For the full model the effect is that most of the information in e(t) cannot be used. For an echo, e(t) = M(t)·s(t−L) is an interaction term, so it is outside the information set of the past model, and Granger non-causality fails even though the echo carries no new information. Pooled oracle coupling, from 100 runs with run fixed effects (A is scalar, so coefficients do not depend on the basis): `python c2.py <structure> 300`

| structure (late window) | round-3 spec, p=12 | gated spec, p=12 |
|---|---|---|
| progress | +0.0038 | +0.0335 |
| mastery | +0.0075 | +0.1351 |
| proliferation lag 1 | +0.0003 | +0.0014 |
| proliferation lag 10 | **+0.0126** | +0.0016 |

Finite sample, same CV, gap, ridge and bootstrap as round 3, 30–40 runs per arm (`python c4.py`, `python c5.py`):

| quantity | round 3 spec | gated spec |
|---|---|---|
| reference coupling > 0 (progress) | 0.33 | 1.00 |
| AUC mastery over dullness / L1 / L3 / L10 | 0.24 / 0.43 / 0.40 / 0.15 | 1.00 / 1.00 / 1.00 / 1.00 |
| AUC progress over L1 / L3 / L10 | 0.63 / 0.62 / 0.28 | 1.00 / 1.00 / 1.00 |
| late reading at zero: dullness, L1, L3, L10 | 8, 8, 6, 5 of 30 (rest mostly no reference) | 28, 29, 28, 29 of 30 |
| late at zero or no ref: progress, mastery, topic, dynamics | 27, 25, 24, 28 of 30 | 5, 0, 0, 0 of 30 |
| valid inference, window 240, at zero or no ref | 29 of 30 | 2 of 30 |

The interaction matters only together with long enough lags. With p=4, gated, lag 10 is not spanned, coupling reads +0.10, and the AUC of progress over L10 is 0.05.

*Proposed change.* In both fits, add M(t)·z(t−j) for j = 0…p−1 to the past design. Keep e(t) only in the full fit. Re-run all eight bars at 200 trials and seed 0 as a pre-registered amendment, since the pre-registration says "as round 2, plus the arrival marker". Expected result: bars 2 and 3 pass, bar 6 passes for the loops, and the "no reference" share falls to 0–3%.

*What would show it wrong.* At 200 trials, the gated reference is still ≤ 0 in more than 10% of progress runs. Or the AUC of progress over the lag-10 echo stays below 0.8. Either would mean the gain in my 30–40-run checks was a small-sample accident.

### 2. Regularisation and over-parameterisation bias the coupling toward zero or below. Weakens the inference.

*Evidence.* Under the round-3 spec the past design has 38 columns and the full design 41, per output, fitted on 94 training rows for interior folds (148 rows − 30 test − 2×12 gap). The out-of-sample estimate is below the in-sample one by about 0.015: the in-sample OLS median is +0.013, the CV median −0.002 (`python c1.py`). That is three times the oracle signal, so the round-3 estimator's sign is mostly noise plus penalty. Ridge interacts with collinearity in a way that matters directly. On an arrival step e(t) ≈ h(t), which is collinear with the lag block, so penalising e shifts its explanatory power onto the lags. That is omitted-variable bias induced by shrinkage. Under the gated spec, raising the ridge from 1 to 30 drops progress coupling from 0.034 to 0.006 and mastery from 0.118 to 0.010, and at ridge 300 the AUC of progress over L10 is 0.35 (`python c4.py`). Ridge is applied to unstandardised columns and to the intercept and marker (`round3.py:489`). Its effective strength therefore depends on the scale of z, which differs by structure (mastery σ = 0.3, dullness drifts to about 15 D along u), so AUCs across structures compare different amounts of shrinkage.

*Proposed change.* Standardise columns within each training fold. Do not penalise the intercept, marker or e block, and penalise only the nuisance lag and interaction block. Choose that penalty by inner blocked CV on the reference window only. Report the coefficient on e and its fold-wise variation next to coupling.

*What would show it wrong.* If coupling medians and AUCs under the gated spec stay within ±0.005 and ±0.02 when the penalty structure changes like this, the scale and penalty issue is immaterial.

### 3. The block bootstrap under-states uncertainty by about 1.7×. Weakens every "open / at zero" reading.

*Evidence.* `round3.py:521-526` resamples held-out squared errors in fixed blocks of 12 and conditions on the 5 fitted coefficient vectors. Each fold's 30 test rows share one fitted b, so its estimation error is a common component across more than two blocks, and the bootstrap does not see it. Check: in progress, late minus reference has expected difference near zero (both windows are healthy). Standardised by the two bootstrap SDs, it should have SD about 1. It has SD **1.63** under the round-3 spec and **1.77** under the gated spec (30 runs, `python c6.py`). This is an upper bound on the miscalibration, since window geometry differs, but per-run bootstrap normal-equivalent SDs (0.010–0.011) are less than half the across-run SDs (0.020–0.025; `python c5.py`). Separately, `round3.py:520` seeds the bootstrap with `default_rng(start)`, so every run uses the identical resample sequence. Errors in the bounds are then correlated across trials, and the shares in bars 4–6 have more than binomial variance. The final block has 4 rows (148 = 12×12 + 4), so resample lengths vary.

*Proposed change.* Replace this with a bootstrap that refits. Either (a) a sieve or residual bootstrap: resimulate the window from the fitted past model plus block-resampled residuals, then rerun the whole CV; or (b) resample folds and the training data together, a moving-block bootstrap of rows with refit inside each replicate. Seed per run from the trial RNG. Expected result: intervals about 1.6× wider, which moves some readings into "undecided".

*What would show it wrong.* A coverage experiment fixes the window geometry (same s path, fresh noise draws only) and finds the 10–90% bootstrap band covers about 80% of replicate couplings.

### 4. δ is a noisy random threshold treated as fixed. Weakens bars 4–6.

*Evidence.* `round3.py:540-542` takes δ = ½ × the reference point estimate and ignores the reference's own sampling error, which is about 0.018 under round 3 and about 0.025 across runs under gated. Under the gated spec, progress still reads at zero in 5/30 runs (17%, bar 5 needs ≤ 10%), with a median late value of 0.040 against a median reference of 0.039. These misses come from runs where the reference landed high by chance.

*Proposed change.* Read the paired statistic Δ = late − ½·reference, with bounds from a joint bootstrap. With the refit bootstrap of finding 3, resample both windows in the same replicate. Call open when the lower 90% bound of Δ > 0 and at zero when the upper bound < 0. Expected result: progress at zero falls below 10% under the gated spec.

*What would show it wrong.* Progress still reads at zero in more than 10% of runs at 200 trials after this change. In that case the variance of the reference itself, not the way it is used, is the limit, and the reference window would need to be longer.

### 5. The shrinkage prior targets the wrong coefficient, and the full model is also shrunk at λ = 20. Weakens bar 7; also a fidelity divergence.

*Evidence.* â comes from a lag-1-only fit on the late window (`round3.py:570`) and soaks up the persistence of the external's random walk: â = 0.930. The coefficient it is a prior for, the lag-1 block of the 12-lag fit, has a median mean diagonal of **0.61** in the past model and 0.60 in the full model (20 progress runs, ridge 1, `python c6.py`). The true A is 0.5. So shrinkage at λ = 20 pulls toward a value about 0.32 too high. The full model can offset the bias through e(t) and the past model cannot, which inflates coupling for exactly the structures where e is collinear with lags. That fits the reported jump in the lag-10 false-open rate from 0.13 to 0.41. The pre-registration says "the past model is fitted … toward a prior". `round3.py:511-516` applies the prior and λ = 20 to the full model as well, which shrinks the e block 20× harder than in the unshrunk arm. The narrower "spread" (0.0043 against 0.0196) follows automatically when both models are pulled hard toward the same point. It is not evidence of better estimation.

*Proposed change.* Set â to the median lag-1 diagonal of the *same* 12-lag design (about 0.6), or put the prior on the sum of lag coefficients. Leave the e block at ridge 1 (or unpenalised, per finding 2). Judge shrinkage by the mean squared error of coupling against the pooled oracle in `c2.py`, not by spread.

*What would show it wrong.* With â matched to the 12-lag fit, the lag-10 false-open rate under shrinkage is still more than 0.02 above unshrunk. Then the mismatch is not what drives it.

### 6. Absolute-RMSE coupling cannot tell "healthy but low-novelty" from "closed", and the after-probe window dilutes the probe. Blocks the structure-8 half of bar 6.

*Evidence.* Coupling is the RMSE gain, which scales with B × the novelty variance of the external. In structure 8, B is unchanged but novelty falls to 0.15. Even under the gated spec, declining reads at zero in 29/30 runs before the probe and 26/30 after (`python c5.py`, `python c7.py`). The 340–500 window holds 20 probe rows among 148 (`round3.py:405`). A probe-only out-of-sample statistic does better: fit on 300–460, score one-step errors only on the 20 probe rows. Its median is +0.027 for declining against −0.012 (dullness) and +0.004 to +0.007 (echoes). The AUC of declining over dullness / L1 / L3 / L10 is 0.88 / 0.78 / 0.79 / 0.80 (30 runs, `python c7.py`).

*Proposed change.* Score the probe as a pure forecast evaluation: models fitted on the pre-probe window, errors only on probe steps, which is in effect a Diebold–Mariano-style test over the 20 rows. Put a scale-free Granger statistic beside it (Wald or F on the e block, with HAC variance) so that a low-novelty but listening entity is not read as closed. Lengthening PROBE_LEN would raise power roughly as √n. Expected result: declining is not at zero in ≥ 90% of runs only if the probe is lengthened (40 or more steps by my √n estimate; I did not run this).

*What would show it wrong.* At 200 trials and 40 probe steps, the probe-only statistic still leaves declining not at zero in < 90% of runs while the loops read at zero in ≥ 80%. Then the probe cannot carry structure 8 at this signal level.

### 7. Point contrast and descriptor read the held external at every step. Minor fidelity divergence.

*Evidence.* `round3.py:545-547` passes H (latest external, every step) to `cp.contrast` and `cp.descriptor`. The pre-registration says the observer sees an external only on the step it arrives, with a marker, and that the detectors use the raw external. The observer could rebuild H from E and m, so this is not cheating, but it is a specification choice the pre-registration does not state. It could change separation for rows 1, 2, 5 and 6.

*Proposed change.* State it in the pre-registration as "the observer carries forward the last arrived external", or feed E and m with the same gating as finding 1.

*What would show it wrong.* Separation numbers are unchanged when H is replaced by a forward fill built from (E, m). They should be identical, which would make this finding pure documentation.

### 8. Smaller divergences that I do not expect to change a result.

The intercept and marker are penalised (`round3.py:489`). The 3b echoes switch on abruptly at step 200 with no ramp (`round3.py:456`); the late window is unaffected. The probe is generated in every run (`round3.py:443`) rather than only when the late reading is not open; this is equivalent, because `:582` copies "open" forward. The CV gap of 12 equals p and is enough to stop target leakage: the last test target z(hi+1) does not enter any training regressor, which I checked by index. It does not handle the dependence induced by the near-unit-root walk, but that affects past and full equally and mostly cancels in the difference.

## Not actionable

- Statistical power of the AUC bars at 200 runs per arm (SE about 0.02–0.03) is not the binding constraint. The round-3 failures are point-estimate failures (AUCs of 0.13–0.55 against 0.8), not borderline ones.
- Whether the arrival-gated observer is legitimate for logged sessions, where "arrival" may be less crisp than a marker, is a design question for the doc. It needs a decision before finding 1's amendment is adopted.
