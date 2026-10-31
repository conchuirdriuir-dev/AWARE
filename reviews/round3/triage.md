# Round 3 reviews: triage

Three reviews, each by a fresh Claude agent given the packet only (`packet.md`): econometrics, system identification,
and an adversarial reproducer. All three are one model family playing roles; reviews from other families or a person
would use the same packet. Triage by the builder, 2026-10-31; what is accepted becomes round 4's pre-registration or a
proposal for the doc, by the owner's decision.

**The numbers reproduce** (reproducer: a 200-trial rerun matched the reported JSON key for key; 27 tests pass).

**The builder's diagnosis was wrong.** Round 3's record said coupling lacks the power with sparse externals. All three
reviewers found instead that the observer cannot represent how the generator uses its input: the entity is driven by
the latest external, held in context between arrivals, while the fit sees an external only on its arrival step and
uses the same lag coefficients whether or not one arrived. More data does not help (negative at 2,600 and 3,000
steps). The builder's own check after the reviews, 30 runs a structure with arrival-gated lags added to both fits:
reference above zero in 97–100% of runs, progress late coupling +0.034, loops near zero, every ranking AUC 1.00.

| # | Finding | Raised by | Kind | Triage | Goes to |
|---|---|---|---|---|---|
| 1 | The observer is misspecified for held externals; fix by arrival-gated lags (m(t)·z(t−j)) or by giving the fit the last external received | all three | error in the pre-registered model | **accepted**; verified | round 4: choose the representation (event-locked discussion) |
| 2 | Bootstrap bounds too narrow (about 1.7–2×): resamples held-out errors without refitting; the same draws in every run | all three | error in the build | **accepted** | round 4: refit per resample, a seed per run |
| 3 | δ treated as exact; read late against reference jointly (a bootstrapped ratio against 0.5) | econometrics, system identification | gap in the pre-registration | **accepted** | round 4 |
| 4 | The probe is read over a whole window (20 of 148 rows); score the probe steps alone, with a model fitted before it | econometrics, system identification | design | **accepted in principle**; this is the event-locked estimate | event-locked discussion |
| 5 | An echo older than the p past steps is invisible to coupling; at p = 4 the progress-over-echo AUC falls to 0.05 | reproducer, system identification | design | **accepted**; test echo lags of 15 and 20 | round 4; the doc's "Echo lag" item |
| 6 | Ridge and over-parameterisation pull coupling toward zero (41 columns, about 94 rows); standardise, leave the external unpenalised | econometrics | error in the build | **accepted** | round 4 |
| 7 | Shrinkage aimed at the wrong coefficient (prior 0.93 from a one-lag fit; the 12-lag lag-1 block is about 0.61; truth 0.5) and also shrank the external's coefficients, which the pre-registration did not say | all three | divergence and error | **accepted**; round 3's shrinkage verdict is void | round 4, if shrinkage is retested |
| 8 | The 3-axis code spends one axis on the window's mean offset, and little of the input lands in it (10.1% against 9.4% for an even spread) | system identification | design | **accepted for testing**: centre on the window mean; try axes chosen for the input (reduced-rank regression) | round 4 option; the code is a doc-level object |
| 9 | Point contrast and the descriptor read the held external, while the pre-registration says the raw one; it moves dynamics-shift detection from 0.13 to 0.80 | all three | divergence in the build | **accepted**; it was the builder's choice at build time and should have been pre-registered | round 4 states it |
| 10 | Minor: loop runs with no reference are not counted against "loops called dynamics shift"; the window for â is unspecified; proliferation has no ramp; the framing comparison is confounded by #1 | system identification, reproducer | minor | **accepted** | round 4 |

**Not taken up:** none. Every finding was stated as a change in a measurement, and none conflicts with another; the
two fixes for #1 (gated lags, held external) differ in representation, which is a design choice for the owner.
