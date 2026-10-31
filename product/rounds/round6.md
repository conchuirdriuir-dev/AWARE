# Toy bed, round 6

The implementation record for one round of the product track's toy bed (PLAN.md, product step 3): its
pre-registration, frozen by the commit before its code, then its result and any post-hoc checks. The design is the
observer as an event record (the v10 tab; DECISIONS.md), with naming a change in the entity's dynamics moved to the
discovery capacity (DECISIONS.md, 2026-10-31). Run with `python -m product.round6`.

## Step 3, round 6 (pre-registered 2026-10-31, before its code)

**What changes from round 5.**
- **The news-spike rule uses the per-run maximum**, as the surprise rule beside it always did. Round 5's per-arrival
  threshold, applied to any of about 12 arrivals, gave progress false topic shifts in about 12% of runs.
- **The product's separation bar narrows.** A dynamics shift must not be called a closed loop or a topic shift; naming
  it is no longer the product's job. The gain-profile change rule retires.
- **Discovery gets its own test**, scored apart from the product: a residual descriptor for changes in the entity's
  own dynamics.
- **A new blind structure.** Everything else is round 5's, with its amended K rule.

### Generator

Round 4's, unchanged, and the same 13 structures. Plus the new blind structure.

### Product readings

As round 5: innovations against the nearest of the last 21 entity states and the held external; news against the
floor D; gain over leads 1 to K, with K chosen by round 5's amended rule; the gain budget, N and M by round 4's rules;
the probe read by its own responses. Separation, from 500 progress calibration runs with a separate seed, taken in
order:
1. **Topic shift:** the largest news among the arrivals in the onset window (steps 200–230) is above the 99th
   percentile of progress's *largest* onset news per run, while the gain reading at 460 is not at zero.
2. **Noise spike:** otherwise, as round 5: onset surprise above progress's 99th percentile of per-run maxima, no news
   spike, and the later stretch's median surprise within progress's 95th percentile.
3. **Progress:** otherwise.

### Discovery reading: the residual descriptor (its own test)

The unexplained surprise vector at each step: s(t+1) − s(t) minus what the innovations account for, using the
session's reference gain profile (round 5's surprise, kept as a vector). Take the vectors in the reference window
(steps 40–199) and in the later stretch (260–439). Remove each window's own mean, and find the top 4 principal axes of
the two windows together. In that basis, fit each window's step-to-step map (ridge 1.0). The descriptor is the
distance between the sorted eigenvalue magnitudes of the two maps. It **fires** above the 99th percentile of
progress's descriptor over the 500 calibration runs.

### The blind structure

Written before this code by a fresh reviewer given only the design and the generator's interface; the builder has not
opened either file:
- `product/blind/round6_structure.py`, sha256 `53d971f8adccb38c37edae68726a3914f06fe24ceb971b729bcdb4f94d00c712`;
- `product/blind/round6_expected.json`, sha256 `c5a0f17d7a53a0fb8ec3a592cbfeace8ff87f3720744ba831305192d4a523c85`.

As in round 5, the code imports the module to generate runs and reads the expected file only in scoring. The file's
other fields are printed after the result is recorded.

### Product pass

1. **Separation:** topic shift, noise spike and progress each labelled correctly in at least 90% of runs. A dynamics
   shift is neither read closed loop at 460 nor labelled a topic shift in at least 90%.
2. **Gain ranks mastery above dullness:** AUC 0.8 or better.
3. **News ranks progress above proliferation:** AUC 0.8 or better at each echo lag (1, 3, 10, 15, 20).
4. **Valid inference** not closed loop at 260, 280, 300 and 320 in at least 90%.
5. **Healthy structures** read closed loop at 460 in at most 10% each.
6. **Final reading:** every loop closed loop in at least 80%; healthy and declining not closed loop in at least 90%.
7. **Response profile:** the measured mean profile peaks at lead 2 and correlates with this generator's expected
   profile at 0.95 or better over leads 1–8.
8. **Intervals honest:** as round 4.
9. **Blind structure:** its final reading matches the expected file's in at least 80% of 200 runs.

### Discovery pass (scored apart; does not gate the product or the freeze)

D. The residual descriptor fires on the dynamics shift in at least 90% of runs, on the topic shift in at most 10%
(modes of working, not topics: v9's decisive question), and on healthy and declining work in at most 10% (it must not
track how much news there is, as round 5's profile change did). Reported beside it: mastery, whose dynamics do change,
and every other structure.

### The freeze rule

If product bars 1–9 all pass, the toy's generator, product readings and thresholds are frozen under the tag
`toybed-frozen`, changed after that only for a finding on real data. The discovery test is not frozen with them: it
belongs to a capacity shared by both tracks, and its own bar decides it. If a product bar fails, nothing is frozen,
and round 7 meets a new blind structure.

Reported, deciding nothing: K, the profile, N and M's sweeps, undecided and probe rates, surprise by structure,
framing, and the blind structure's reading at 460.
