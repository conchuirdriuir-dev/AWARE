# Toy bed, round 4

The implementation record for one round of the product track's toy bed (PLAN.md, product step 3): its
pre-registration, frozen by the commit before its code, then its result and any post-hoc checks. The design it tests
is event-locked coupling, decided 2026-10-31 (DECISIONS.md: horizons, pooling into news and gain, latency), which
answers round 3's review (`reviews/round3/triage.md`). Run with `python -m product.round4`.

## Step 3, round 4: event-locked coupling, news and gain (pre-registered 2026-10-31, before its code)

**What changes from round 3.** Coupling is no longer a window's error reduction. Each arrival contributes an
*innovation*: what it carries that the entity's recent history and the external before it did not. Arrivals are pooled
into two readings: **news**, how large the innovations are, and **gain**, how much of an innovation shows up in the
entity afterwards. Closed loop is gain at zero while news is present (dullness), or news at zero (proliferation). Low
news with the gain intact is not a loop; it waits, then probes.

### Generator

Round 3's (`product/round3.py`): 32 dimensions, D = 1, 500 steps, an opening intake at steps 0–39, externals arriving
on 40% of steps after it, onset at 200, ramps over 60 steps, the entity driven by the external held in context. Three
changes:
- **Externals carry news above the reading noise.** Each arriving external adds a fresh content jump of size 3 D in a
  random direction to the slow walk, beside its reading noise of D. In round 3 an external's only novelty was noise at
  D, which is what D marks as no news. Structure 8's jump ramps to 0.15 of that (0.45 D); the probe's responses carry
  full news.
- **Echo lags 1, 3, 10, 15 and 20** for proliferation (the reviews: an echo older than the history is invisible).
- **The probe fires by rule**, not at a fixed step: at the read point (step 460), if the gain's news budget is unmet
  within M arrivals and news is not at zero, the next 20 steps each bring the partner's response (fresh for progress,
  mastery and structure 8; fresh but unheard in dullness; an echo at the structure's lag in proliferation), and the
  readings are taken again at step 495.

### Readings

All in the full state space, so no code is fitted for coupling.
- **Innovation** of an arrival at step t: the external read alone, minus whichever of its candidates it is nearest:
  the entity's last H = 21 states (above the longest echo lag, 20) and the external held before it. Its length is the
  arrival's **news**.
- **Gain at lead k** over a set of arrivals: Σ⟨s(t+k) − s(t), u⟩ / Σ‖u‖², a scalar. s(t) is the entity's state when
  the external arrives, before it can respond. Later arrivals' responses fall in the residual. They are not modelled,
  because innovations are unpredictable from what came before and so uncorrelated with an earlier arrival's.
- **The gain reading** is the mean of gain over leads 1 to K. Lead 1 is reported beside it.
- **Its interval:** a cluster-robust sandwich over blocks of arrivals less than K steps apart. One-sided 90% bounds.
- **K** is measured, not chosen: the lead at which the mean gain profile over 200 progress calibration runs first
  reaches 90% of its plateau (the mean over leads 15–20).

### References and readings at a point

- **References**, from the arrivals in steps 40–200: news_ref, the mean news; g_ref, the gain reading. δ_gain is half
  of g_ref. A run whose g_ref has a lower bound not above zero has no reference and counts against every bar it
  enters.
- **News at zero:** the upper 90% bound of the mean news over the last N arrivals is below the floor, D.
- **The gain budget:** pool arrivals backwards from the read point (each with t + K at or before it) until the
  interval's half-width is at most 0.25 × g_ref, or M arrivals are used. Met: open if the lower bound is above δ_gain,
  at zero if the upper bound is below it, undecided otherwise. Unmet: undecided.
- **At a read point:** closed loop if news is at zero, or the gain reads at zero. Otherwise open if the gain reads
  open, undecided if not. Undecided at step 460 fires the probe.

### Tuning (on calibration runs only, separate seeds)

- **N:** the smallest of 4, 8, 16, 32 for which progress reads news at zero in at most 10% of 200 calibration runs.
- **M:** the smallest of 10, 20, 40, 80 for which progress and mastery each meet the gain budget at step 460, without a
  probe, in at least 90% of 200 calibration runs. The whole sweep is reported.
- The point contrast and descriptor thresholds, and the admission floor, as round 3, on 500 progress runs.

### Unchanged from round 3, now stated

Point contrast, admission and the spectral descriptor read the external held in context at each step, as round 3's
code did; round 3's pre-registration said the raw external, and the reviews caught the divergence. Windows as round 3
(before onset 140–200, after 200–230, later stretch 260–320).

### Pass

1. **Separation:** rows 1, 2, 5, 6 each at 90% or better, read as round 3, with "the late reading is not at zero"
   replaced by "the reading at 460 is not closed loop". The loops (dullness and every echo lag) are called a dynamics
   shift in at most 10% of runs, counting runs without a reference against them.
2. **Gain ranks mastery above dullness** at AUC 0.8 or better (gain reading at 460, before any probe).
3. **News ranks progress above proliferation** at AUC 0.8 or better at each echo lag separately (mean news over the
   last N arrivals at 460).
4. **Valid inference:** read points 260, 280, 300 and 320 not closed loop in at least 90%.
5. **Healthy structures** (progress, mastery, topic shift, dynamics shift, structure 8) read closed loop at 460,
   before any probe, in at most 10% of runs each.
6. **After the probe** (reading at 495 where a probe fired, at 460 otherwise): dullness reads closed loop in at least
   80%, each echo lag in at least 80%, structure 8 not closed loop in at least 90%.
7. **K is recovered:** the measured K is within one lead of the generator's own, which is 4: under A = 0.5·I the
   response to a held external reaches 1 − 0.5^k of its plateau, first above 90% at k = 4.
8. **The intervals are honest:** over the 200 progress runs, the analytic lower 90% bound is above, and the upper below,
   the mean gain over those runs each in at most 15% of runs; and the analytic standard error is within 25% of a
   bootstrap that resamples arrival blocks and refits, in median.

Reported, deciding nothing: the undecided share and probe rate per structure; N and M's sweeps; lead-1 gain; news and
gain with the in-context reading of externals (0.5 × external + 0.5 × the entity's state) beside the alone reading.

### What this cannot settle

The innovation measures an external against the entity's own states in one space. On the real model an external is
read alone and the entity's states in context, so even an exact echo of the entity's words will sit some distance
from its state, possibly above D. The floor for news on the real model needs its own measurement (a known echo read
alone against the in-context state it repeats), beside D, before news at zero means anything there.
