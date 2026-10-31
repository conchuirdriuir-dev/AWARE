# Toy bed, round 5

The implementation record for one round of the product track's toy bed (PLAN.md, product step 3): its
pre-registration, frozen by the commit before its code, then its result and any post-hoc checks. The design it tests
is the observer as an event record, accepted 2026-10-31 (the v10 tab, "The observer as an event record"; DECISIONS.md).
Run with `python -m product.round5`.

## Step 3, round 5: the event record (pre-registered 2026-10-31, before its code)

**What changes from round 4.** Round 4's three misses, each answered from the reframe rather than patched:
- **Separation is read from events.** The window-based point contrast, admission and spectral descriptor retire.
- **The probe is read by its own responses**, not by whatever arrives after it.
- **The response profile is measured as a curve**, and the detection horizon is chosen by a rule. A single "K at 90%
  of the plateau" is dropped.
Plus the two toy rules: a blind structure, and the freeze rule.

### Generator

Round 4's (`product/round4.py`), unchanged: 32 dimensions, D = 1, 500 steps, an intake at 0–39, externals on 40% of
steps with a content jump of 3 D each, onset at 200, ramps over 60 steps, the probe's responses at 460–479. The same 13
structures: topic shift, dynamics shift, dullness, proliferation at echo lags 1, 3, 10, 15 and 20, mastery, noise spike,
progress, valid inference, healthy and declining. Plus the blind structure (below).

### Readings

As round 4 unless said otherwise.
- **Innovation and news:** as round 4. The external is read alone, against the nearest of the entity's last H = 21
  states and the external held before it. The news floor is D.
- **Gain profile:** gain at each lead k from 1 to 20, Σ⟨s(t+k) − s(t), u⟩ / Σ‖u‖².
- **The detection horizon K:** the k from 1 to 20 that maximises the median, over 200 progress calibration runs, of
  the z-score of the gain averaged over leads 1 to k (reference-window arrivals, round 4's cluster-robust standard
  error). The gain reading is the gain averaged over leads 1 to K, with round 4's interval.
- **Unexplained surprise** at step t: the length of s(t+1) − s(t) − Σᵢ (g(t − tᵢ + 1) − g(t − tᵢ)) uᵢ, summed over
  arrivals i with 0 ≤ t − tᵢ < 20. Here g is the session's reference gain profile, with g(0) = 0. In words: the step
  the entity took, minus what the innovations it received account for.

### Separation, from events

Read in the onset window (steps 200–230) and a later stretch (260–440), against thresholds from 500 progress
calibration runs (separate seed), taken in order:
1. **Topic shift:** an arrival in the onset window whose news is above the 99th percentile of progress's news per
   arrival in the same window, while the gain reading at 460 is not at zero.
2. **Noise spike:** otherwise, a step in the onset window whose surprise is above the 99th percentile of progress's
   maximum surprise in that window, with no arrival above the news threshold, and the later stretch's median surprise
   within progress's 95th percentile.
3. **Dynamics shift:** otherwise, the gain profile over leads 1–8 from arrivals in the later stretch, against the same
   from reference-window arrivals: a distance above the 99th percentile of progress's, with the reading at 460 not
   closed loop. With the reading closed loop, it is labelled closed loop.
4. **Progress:** otherwise.

### Reading at a point, and the probe

News at zero, the gain budget, the closed-loop rule and the references are round 4's. The pooled gain uses the
detection horizon K. **The probe is read by its own responses:** if the reading at 460 is undecided, the re-read at
480 + K uses only the probe's arrivals (460–479), for both news (their mean, with round 4's upper bound) and gain
(pooled over them, against the session's reference and δ). The budget is not required for this re-read; the bounds
decide, and a reading neither open nor at zero stays undecided.

### Tuning (calibration runs only, separate seeds)

N and M by round 4's rules; K by the rule above; separation thresholds as above.

### The blind structure

Written before this code by a fresh reviewer given only the design and the generator's interface:
- `product/blind/round5_structure.py`, sha256 `700867f78af1df956fcbe78360d701f113cc9172c5761c939f510df3aa3f206c`;
- `product/blind/round5_expected.json`, sha256 `41f85b6479bdb638465dfa4901e70d23f9931d4431296001340d11b898031c6b`.

The builder has not opened either file. The round's code imports the module to generate runs. It reads the expected
file only in scoring: it compares the final reading with the file's "final reading". The file's other fields are
printed only after the result is recorded.

### Pass

1. **Separation:** rows 1, 2, 5 and 6 each at 90% or better. Each loop (dullness, every echo lag) is called a dynamics
   shift in at most 10% of runs, counting runs without a reference against it.
2. **Gain ranks mastery above dullness:** AUC 0.8 or better (gain reading at 460).
3. **News ranks progress above proliferation:** AUC 0.8 or better at each echo lag (mean news over the last N arrivals
   at 460).
4. **Valid inference:** read points 260, 280, 300 and 320 not closed loop in at least 90%.
5. **Healthy structures** (progress, mastery, topic shift, dynamics shift, healthy and declining) read closed loop at
   460, before any probe, in at most 10% of runs each.
6. **Final reading:** every loop closed loop in at least 80% of runs; healthy and declining not closed loop in at least
   90%.
7. **The response profile is recovered.** In this generator, an arrival's external drives the entity until the next
   arrival replaces it (probability 0.4 a step). So the expected response at lead k is
   Σ over i from 0 to k − 1 of 0.5^(k − 1 − i) × 0.5 × 0.6^i, which peaks at lead 2. The measured mean profile over the
   200 calibration runs must peak at lead 2, and correlate with the expected one over leads 1–8 at 0.95 or better.
8. **The intervals stay honest:** as round 4, on the reference gain at K.
9. **The blind structure:** its final reading matches the expected file's "final reading" in at least 80% of 200 runs.

Reported, deciding nothing: K and the profile; N and M's sweeps; the undecided and probe rates; surprise by structure;
lead-1 gain; news and gain read in context against alone; the blind structure's reading at 460.

### The freeze rule

If every bar passes, the generator, readings and thresholds of this round are frozen under the tag `toybed-frozen`.
After that they change only for a finding on real data, recorded with the finding. If a bar fails, nothing is frozen,
and round 6 meets a new blind structure.

### Amendment, 2026-10-31, before any test or blind result

The first build of this round could not finish: the refitting bootstrap returned a standard error of zero. The
cause is the rule for K above. At a horizon of 8 or more, every reference-window arrival falls in one overlap block,
so the cluster-robust error collapses toward zero, z grows without bound, and the rule picks a degenerate horizon (12).
Only calibration numbers had been computed; no test structure's or the blind structure's result had been computed or
seen. Amended rule: **K is the horizon that maximises the median z among the horizons whose reference-window arrivals
form at least 10 overlap blocks (median over the 200 calibration runs)**, so that the error is defined. Nothing else
changes.
