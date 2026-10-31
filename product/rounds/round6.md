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

## Round 6 result

2026-10-31, `python -m product.round6`, 200 trials a structure, seed 0; report in `runs/round6.json`, not committed.
**Product: seven of nine bars. Discovery: fails. Nothing is frozen;** round 7 meets a new blind structure.
Calibration: K = 1, N = 4, M = 10, profile peak at lead 2 (correlation 0.993).

| Bar | Result | Pass |
|---|---|---|
| 1 Separation: topic / noise spike / progress; dynamics not closed loop or topic | 1.00 / **0.88** / 0.99; 0.955 | no |
| 2 Gain ranks mastery above dullness | 1.00 | yes |
| 3 News ranks progress above each echo, lags 1 to 20 | 1.00 at every lag | yes |
| 4 Valid inference not closed loop | 1.00 | yes |
| 5 Healthy read closed loop at 460 (at most 0.10) | 0.00 / 0.00 / 0.00 / 0.005 / 0.02 | yes |
| 6 Final: every loop closed; healthy and declining not closed | 1.00 for every loop; 0.98 | yes |
| 7 Response profile | lead 2; 0.993 | yes |
| 8 Intervals honest | misses 0.09 / 0.055; SE ratio 1.006 | yes |
| 9 Blind structure matches its expected final reading | **0.00** (it read open in every run) | no |
| D Residual descriptor fires: dynamics ≥ 0.9 / topic ≤ 0.1 / healthy and declining ≤ 0.1 | 1.00 / 0.005 / **0.88** | no |

- **The news-spike fix worked:** progress is now labelled correctly in 0.99 of runs (0.875 in round 5).
- **The noise-spike miss is a threshold choice in this pre-registration** (post-hoc, 100 runs). The spike's surprise
  cleared its threshold in every run. But the rule also requires the later stretch's surprise to be within progress's
  95th percentile, which rejects about 5% of real spikes by construction (4 of 100), and chance news spikes at onset
  take about 3 more. A rule that discards 5% by design leaves no margin under a 90% bar.
- **The residual descriptor sees dynamics, not topics,** as v9 asked (1.00 against 0.005). But it fires on healthy
  and declining work (0.88), on proliferation (0.21–1.00 by lag) and barely on mastery (0.035, whose dynamics do
  change). So it reads changes in where the entity's movement comes from, not only in its own dynamics: once
  externals carry less news, or echo the entity, the movement left unexplained changes character.
- **The blind structure read open in every run,** against an expected reading recorded before the code. What it is,
  and why, is below; the file was opened only after this result was written.
