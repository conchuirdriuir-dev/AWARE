# Toy bed, round 7b

The implementation record for one round of the product track's toy bed (PLAN.md, product step 3): its
pre-registration, frozen by the commit before its code, then its result and any post-hoc checks. The design: the
observer as an event record, with the biologist review's changes decided by the owner (DECISIONS.md, 2026-10-31).
This is the second half of round 7, and folds in round 7a's three fixes (owner, 2026-10-31). Run with
`python -m product.round7b`.

## Step 3, round 7b: contingency and the reactive toy (pre-registered 2026-10-31, before its code)

### Round 7a's fixes, folded in

- **The news floor is F = √2·D,** for short and long news alike. A news comparison is between two readings, which sit
  √2·D apart when they hold the same content.
- **A confirmation is an elicited arrival whose short news is below F.** Only confirmations are set aside from the
  partner-side test. An elicited arrival with news counts: it is the world answering the entity's own sensing.
- **A partner-side closed loop is probed before it is final,** as v10's compass and the event-record table say. The
  probe fires at 460 on an undecided reading or on news at zero. An entity-side closed loop (gain at zero, or the loop
  below) is final at 460.

### New in 7b

- **The probe as a contingency test** (biologist, 1e). At step 460 the entity asks: its state at 461 is displaced by
  3·D along a random unit direction p. The partner's responses arrive at 460–479, elicited. **Contingency:** the mean
  projection onto p of the responses, minus the mean projection onto p of the last 10 non-confirmation arrivals before
  460, divided by 3·D. A mirror partner's responses follow the probe; an independent or stale one's do not.
- **Reading after the probe.** If the responses' long news (confirmations aside) is at zero, the reading is closed
  loop on the partner's side. It is a **mirror** (capture) when contingency is above progress's 99th percentile, and
  **stale** otherwise. If the responses carry news, gain is read on them as in round 6.
- **Entity captured** (biologist, 3). At 460, the gain's lower bound is above 1.5 × the reference gain, and the
  entity's mean distance from the origin's centroid over steps 400–460 is above progress's 99th percentile. Final at
  460.
- **Persistence and recurrence** (biologist, 5). *Persistent gain:* gain averaged over leads 12–20, from arrivals in
  steps 260–440. *Recurrence:* the share of steps 300–459 whose state is within F of some state at least 22 steps
  earlier. An entity-side closed loop also reads when persistent gain is below progress's 1st percentile and
  recurrence is above progress's 99th. Thresholds come from progress calibration runs.

### Generator

Round 7a's (with its arrival cause, replay and confirmations), plus the probe displacement for every structure and
six reactive or nonlinear conditions (biologist, 10):
- **Confirmations, heavy:** from 200, 90% of arrivals are elicited confirmations (the entity's own state plus noise
  0.3·D); the rest are ordinary.
- **Active sensing:** from 200 the partner's spontaneous arrivals are content-free acknowledgements re-presenting one
  fixed point, while elicited tool results (60% of arrivals) carry fresh content, which the entity takes up.
- **Half echo:** from 200, half the arrivals echo the entity at lag 3, half are ordinary.
- **Saturating:** healthy work whose response to an external saturates: the drive is the held external's offset
  passed through r·tanh(|x|/r), with r = 3·D, along its own direction.
- **Attractor loop:** from 200 the entity cycles through its states at 194–199, but each arrival pulls it 0.5 of
  the way toward the external for one step, after which it returns to the cycle.
- **Captured entity:** from 200 the partner's content drifts away from the origin (a trend of 0.3·D a step), and the
  entity copies the held external (A → 0, B = I).

Probe responses: fresh externals for healthy structures, dullness, confirmations heavy, active sensing and
saturating. Echo partners echo at their lag, the half-echo partner as before. Replay continues its replay. The
captured entity's partner continues its drift. The attractor loop's responses are fresh.

### Deferred: the self-drive term

The biologist's self-drive term (movement explained by the entity's own read-back outputs, finding 6) has no
correlate in this toy. An entity driven by its own read-back content moves toward itself, so there is no movement to
explain. It is deferred to logged sessions. The residual descriptor is reported only.

### The blind structure

Written before this code by a fresh reviewer given only the design and the generator's interface (now with the
probe's direction); the builder has not opened either file:
- `product/blind/round7b_structure.py`, sha256 `4ae894df8fbdf324e0a1ce553cec16c591031ac90044351b0ec50a50dbaacebe`;
- `product/blind/round7b_expected.json`, sha256 `37c71da5b65f61e02334e44962a537d5493e91ded5d5bc97af03ae5334ecca5c`.

### Pass

1. **Separation:** as 7a (topic, noise spike, progress each at least 90%; dynamics shift not closed loop or topic in
   at least 90%).
2. **Gain ranks mastery above dullness:** AUC 0.8 or better.
3. **Long news ranks progress above each echo lag:** AUC 0.8 or better.
4. **Valid inference** not closed loop at 260, 280, 300 and 320 in at least 90%.
5. **Healthy, final:** progress, mastery, topic shift, dynamics shift, healthy and declining, saturating, active
   sensing and confirmations heavy, each not closed loop and not entity captured in their final reading, in at least
   90%.
6. **Loops, final:** dullness, every echo lag, replay and the attractor loop each read closed loop finally in at least
   80%.
7. **Response profile:** peak at lead 2, correlation at least 0.95 over leads 1–8.
8. **Intervals honest:** as round 4.
9. **Blind structure:** final reading matches the expected file's in at least 80%.
10. **The arrival cause does work:** confirmations heavy reads not closed loop finally in at least 90% with the field.
    With the field ignored (every arrival treated as spontaneous), it reads closed loop in at least 50%. The second
    clause is there because 7a's toy never exercised the field.
11. **Capture is contingency, not low news:** contingency ranks every echo lag above replay at AUC 0.8 or better. Each
    echo lag's partner-side loops are called mirror in at least 80%; replay's are called stale in at least 80%.
12. **Entity captured:** the captured entity reads entity captured in at least 80%; mastery in at most 10%.

Reported, deciding nothing: half echo's readings and contingency; recurrence and persistent gain by structure; the
residual descriptor; K, N, M; probe rates.

### The freeze rule

7b is the last planned change to the toy. If product bars 1–12 all pass, the generator, readings and thresholds are
frozen under the tag `toybed-frozen`, changed after that only for a finding on real data. If any fails, nothing is
frozen, and the next round meets a new blind structure.

## Round 7b result

2026-10-31, `python -m product.round7b`, 200 trials a structure, seed 0; report in `runs/round7b.json`, not committed.
**Seven of twelve product bars. Nothing is frozen.** Calibration: K = 1, N = 4, M = 10, profile peak at lead 2
(0.993); thresholds include recurrence 0.0 (progress's 99th percentile) and contingency 0.234.

| Bar | Result | Pass |
|---|---|---|
| 1 Separation: topic / noise spike / progress; dynamics | 1.00 / 0.97 / 0.98; 0.95 | yes |
| 2 Gain ranks mastery above dullness | 1.00 | yes |
| 3 Long news ranks progress above each echo | 1.00 at every lag | yes |
| 4 Valid inference | 1.00 | yes |
| 5 Healthy not closed or captured, final (at least 0.90 each) | 1.00 / 0.995 / 1.00 / 0.99 / 0.975 / **0.80** (saturating) / 0.93 / 0.985 | no |
| 6 Loops closed, final (at least 0.80 each) | dullness 1.00; echoes 0.92–1.00; **replay 0.57**; attractor 0.98 | no |
| 7 Response profile | lead 2; 0.993 | yes |
| 8 Intervals honest | misses 0.11 / 0.115; SE ratio 1.005 | yes |
| 9 Blind structure matches its expected final reading | **0.115** | no |
| 10 Arrival cause does work: open with the field / closed without it | 0.985 / **0.015** | no |
| 11 Capture is contingency: AUC echo over replay; echoes called mirror; replay called stale | 1.00 / 0.98 / 0.81 / 0.81 / **0.53**; **0.995 / 0.86 / 0 / 0 / 0**; 0.95 | no |
| 12 Entity captured: the copying entity / mastery | 0.825 / 0.005 | yes |

**What worked.** Entity captured fires on an entity copying a partner off the task and not on mastery. Probing
partner-side loops first rescues healthy and declining work (0.975, against 0.00 in 7a). Contingency tells a
short-lag mirror from a stale partner (lag 1 called mirror in 0.995, replay called stale in 0.95). The attractor loop
is caught (0.98) by persistence and recurrence.

**What the misses are** (post-hoc checks, 60 runs each):
- **Contingency can only identify a mirror whose lag is short against the probe window.** Long-lag echoes are probed and
  read closed loop, as they should; only the label fails. A lag-10 partner's responses in the 20-step window mostly
  echo states from before the probe moved the entity, so the probe's direction is diluted below the threshold; at lag
  20 none of them carry it. A design limit, not a bug: a probe can only test what returns within its window.
- **Saturating** reads closed loop in some runs through a degenerate threshold in this pre-registration: progress's
  99th percentile of recurrence is exactly 0, so "recurrence above it" fires on any recurrence at all, and saturating's
  lower persistent gain completes the condition. The saturation also compresses gain on large innovations, leaving
  about a fifth of runs undecided after the probe.
- **Replay** reads closed in 0.57. With the √2·D floor its long news (median about 1.35 D) sits on the floor, so the
  reading is close to a coin toss; a floor between two readings of the same content cannot separate content that is
  exactly that, which is what replay is. The real model's measured echo floor, with its own spread, has to decide it.
- **The arrival-cause field's work is absorbed by the probe.** With the field ignored, confirmations heavy reads news
  at zero, the probe fires, and the partner's fresh responses read open: 0.015 closed. Probing first makes the field
  redundant for this structure in the toy. It still matters where a probe is costly or unanswered, which the toy does
  not model.

**The blind structure, opened after the result was recorded.** "blind-7b": from step 200 the entity runs a fixed
retry loop, repeating nine actions in order on a 23-step cycle, each drawing back the same result again (elicited, a
fixed vector plus reading noise). Each result sits about 3 D from the recent window, so it is not a confirmation and
short news looks present; the entity takes it up, so short gain looks intact, as healthy active sensing does. But
nothing new to the session arrives, and no innovation lasts past a cycle. The reviewer expected a closed loop on the
entity's side (an agent stuck re-issuing the same calls), after the probe too. It read open in 89% of runs. The cause
is replay's: a repeat re-read with noise lands on the √2·D floor, so long news cannot reliably call it nothing new.
Persistence did not complete the loop reading either.

**Across 7a and 7b, one limit recurs.** Detecting that content *repeats* (replay, the retry loop) depends on a floor
for "the same content, read again" that the toy cannot supply: in the toy it is √2·D by construction, exactly where
repeats sit. On the real model it is the echo floor, measured with its own spread (DECISIONS.md, the event record),
and only that measurement can say whether repeats are separable from news.
