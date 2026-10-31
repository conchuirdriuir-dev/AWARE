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
