# Toy bed, round 7a

The implementation record for one round of the product track's toy bed (PLAN.md, product step 3): its
pre-registration, frozen by the commit before its code, then its result and any post-hoc checks. The design: the
observer as an event record, with two changes from the biologist review decided by the owner (DECISIONS.md,
2026-10-31): each external records its arrival cause, and news is measured short and long. Round 7 is split; 7b adds
contingency, the entity-captured reading and the reactive toy. Run with `python -m product.round7a`.

## Step 3, round 7a: provenance and news (pre-registered 2026-10-31, before its code)

**What changes from round 6.**
- **Arrival cause.** Every external is spontaneous or elicited. An elicited arrival with little news confirms the
  entity's action; it is never evidence of capture.
- **Short and long news.** Short news is round 6's (against the last 21 states and the held external). Long news is
  against the session's whole record.
- **The partner-side loop is read on long news over spontaneous arrivals**, so content that left recent history and
  comes back is not news to the session.
- **The noise-spike rule's later-surprise condition moves from the 95th to the 99th percentile**, so it no longer
  discards about 5% of real spikes by design (round 6's miss).
- **Two known structures join the toy**, and a new blind structure.

### Generator

Round 4's, plus an arrival-cause marker: every arrival is spontaneous except the probe's responses (460–479), which
are elicited. The 13 structures as before, plus two:
- **Replay** (round 6's blind structure, now known): from step 200 every arrival is a fresh reading (noise D) of an
  external the session received at steps 60–199, replayed in order on a cycle. Arrivals are spontaneous; the entity
  keeps listening.
- **Confirmations:** from step 200 each arrival is, with probability 0.6, an elicited confirmation: the entity's own
  current state plus noise 0.3 D (a tool result that returns what the entity expected). The rest are ordinary
  spontaneous externals.

### Readings

As round 6 unless said otherwise.
- **Long news** of an arrival at step t: the external minus whichever is nearest of every earlier entity state (steps
  0 to t) and every earlier external. Short news, innovations and gain are round 6's, over all arrivals.
- **News at zero** (closed loop on the partner's side): the upper 90% bound of the mean *long* news over the last N
  *spontaneous* arrivals is below D. **Re-presentation**, reported: over the same arrivals, mean short news is above D
  while long news is at zero.
- **Confirmations**, reported: elicited arrivals whose short news is below D.
- **The probe's re-read**, as round 6, on its own responses: news at zero if their mean long news has its upper bound
  below D; gain as round 6.
- **Separation** as round 6. The noise spike's later-stretch condition is now within progress's 99th percentile.
- **The residual descriptor** is computed and reported only. Its fix (taking the self-drive term out of surprise) is
  7b's.
- K, N and M by round 5's and round 4's rules. N now counts spontaneous arrivals.

### The blind structure

Written before this code by a fresh reviewer given only the design and the generator's interface (now with the
arrival-cause marker); the builder has not opened either file:
- `product/blind/round7a_structure.py`, sha256 `f72436a046299bdcf99a954d091dd4011499809579be047b6d413e58bf800b9b`;
- `product/blind/round7a_expected.json`, sha256 `663c36540aa01df86fd8ea2acedca3693ad65c19e40809d7cdfe6e127b2340b1`.

As before, the code imports the module to generate runs and reads the expected file only in scoring.

### Pass

1. **Separation:** topic shift, noise spike and progress each labelled correctly in at least 90%. A dynamics shift is
   neither read closed loop at 460 nor labelled a topic shift in at least 90%.
2. **Gain ranks mastery above dullness:** AUC 0.8 or better.
3. **News ranks progress above proliferation:** AUC 0.8 or better at each echo lag. The statistic is mean long news over
   the last N spontaneous arrivals at 460.
4. **Valid inference** not closed loop at 260, 280, 300 and 320 in at least 90%.
5. **Healthy structures** (progress, mastery, topic shift, dynamics shift, healthy and declining) read closed loop at
   460 in at most 10% each.
6. **Final reading:** dullness, every echo lag and **replay** closed loop in at least 80% each; healthy and declining
   not closed loop in at least 90%.
7. **Response profile:** peak at lead 2, correlation with the expected profile at least 0.95 over leads 1–8.
8. **Intervals honest:** as round 4.
9. **Blind structure:** final reading matches the expected file's in at least 80%.
10. **Confirmations are not capture:** the confirmations structure is not closed loop at 460, nor finally, in at least
    90% of runs. Reported beside it: the same reading with the arrival cause ignored (every arrival treated as
    spontaneous), to show what the field does.

Reported, deciding nothing: the re-presentation reading by structure; confirmations counted; short against long news
on healthy structures (the biologist's falsifier: if they agree everywhere, long news adds nothing); the residual
descriptor; K, N, M; undecided and probe rates.

### No freeze at 7a

Round 7b changes the toy (a reactive environment), so the freeze rule is evaluated after 7b, not here. A 7a that passes
every bar is recorded as such and carried into 7b unchanged.

## Round 7a result

2026-10-31, `python -m product.round7a`, 200 trials a structure, seed 0; report in `runs/round7a.json`, not committed.
**Seven of ten bars.** Calibration: K = 1, N = 4, M = 10, profile peak at lead 2 (correlation 0.993).

| Bar | Result | Pass |
|---|---|---|
| 1 Separation: topic / noise spike / progress; dynamics not closed loop or topic | 1.00 / 0.975 / 0.975; 0.955 | yes |
| 2 Gain ranks mastery above dullness | 1.00 | yes |
| 3 Long news ranks progress above each echo, lags 1 to 20 | 1.00 at every lag | yes |
| 4 Valid inference not closed loop | 1.00 | yes |
| 5 Healthy read closed loop at 460 (at most 0.10) | 0.00 / 0.00 / 0.00 / 0.01 / **1.00** (healthy and declining) | no |
| 6 Final: dullness and every echo closed; **replay** closed; declining not closed | 1.00 each; **0.00**; **0.00** | no |
| 7 Response profile | lead 2; 0.993 | yes |
| 8 Intervals honest | misses 0.08 / 0.08; SE ratio 1.005 | yes |
| 9 Blind structure matches its expected final reading | **0.00** (it read closed loop in every run) | no |
| 10 Confirmations not capture | 0.985 | yes |

The noise-spike fix worked (0.975, against 0.88 in round 6); separation passes for the first time.

**What the misses are** (post-hoc checks, 60 runs each):
- **Replay is a units error in this pre-registration.** A replayed arrival's long news has a median of 1.35 D, close
  to √2 D = 1.41 D: two independent readings of one content sit √2 D apart, because D is one reading's spread about
  its centre. A floor of D for a comparison between two readings is the wrong unit. It is v10's "D is three units"
  item again; the floor for news should be the echo floor measured on the real model, or √2 D in the toy.
- **Healthy and declining work is a gap between the pre-registrations and the design.** With long news its low news
  falls under the floor, so it reads closed loop, and the rule fires the probe only on an undecided reading, so a
  closed loop is never probed. But v10's compass ("closed loop: probe; if coupling does not respond, prune") and the
  accepted event-record table ("news at zero: probe; flag the capture") both probe a partner-side closed loop first.
  Had the probe fired, declining would have read open in all 60 runs checked (the probe's responses carry long news
  of 2.93 D). Rounds 4 to 6 carried the same rule; it did not matter while short news kept declining above the floor.
- **The arrival-cause field was not exercised.** Confirmations read the same with the field and without it (0.985
  both): the 40% of arrivals that are ordinary keep the mean of the last four arrivals' news high, so mixing never
  pushes it under the floor. The toy did not show what the field does; 7b's reactive tool should.
- **Short and long news agree on healthy work** (median correlation 0.86–0.89; 0.46 for healthy and declining). Long
  news earns its place only where content returns, which is replay's job and, on this result, the blind structure's.

**The blind structure, opened after the result was recorded.** "blind-7a": from step 200 the partner goes quiet,
its rare spontaneous messages content-free acknowledgements re-presenting one point, while the entity drives its own
work through tool calls whose elicited results each carry content new to the session, and takes them up. The reviewer
expected open: news is present, only it arrives because the entity asked, and gain is open. It read closed loop in
every run (re-presentation in 82%). **The rule treated every elicited arrival as a confirmation.** Only an elicited
arrival with little news confirms an action; one with news is the world answering the entity's own sensing (the
biologist review's active sensing, finding 1). The partner-side loop should exclude confirmations, not every elicited
arrival.
