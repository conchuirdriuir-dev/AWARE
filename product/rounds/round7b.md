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
