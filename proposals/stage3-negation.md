# Proposal: stage 3, the register and negation, with a geometry sub-test

A proposal for the owner, before any code (CLAUDE.md: propose, don't widen). Nothing here is decided. Section 4 lists
the DECISIONS.md lines it would add if accepted.

## 1. Why stage 3 now

Stage 2 shows one failure the notes have not moved: the right action, too small. In 0 of 10 monitor runs did a
stockout note fix a stockout it met (toyworld/RESULTS.md). The model is not missing the fact. It predicts its own
shortfall accurately ("ordered 30, predicted 12 unfilled, got 12") and holds the goal anyway.

Read as a frame, the model's reasoning rests on an assumption it never states: *the order I am placing is the right
size*. Every note so far has worked inside that assumption. A note names the mover, shows the levels, sets the goal
beside the prediction, and the model adjusts by 5. Stage 3's job, per PLAN.md, is to hold that assumption as a priced
frame and propose its negation: *the order is not the right size; demand is larger than what I order*. Notes that
name a missing action already work. A negated frame turns "too small" into something missing, namely a larger order.

This proposal keeps stage 3 to that one target, so it is judged on the failure stage 2 left.

## 2. Stage 3 as proposed

**What a frame is here.** An assumption under a residue kind that keeps recurring: a mover applied and its anomaly
still present (K1). The frame is worded by code, as goals are since 2026-10-30: "the action applied is sufficient to
remove the anomaly". Its negation, also worded by code, is "the action applied is not sufficient to remove the
anomaly".

**Who proposes.** The model is asked one question when K1 recurs: what would have to be true for this anomaly to stay
present after this action? Its answer is recorded as the model's claim. A proposal counts as a negation of the frame
only if code can map it to a field of the action (here, `qty`) and a direction (more or less).

**What admits.** The kernel, on the 2-bit margin: the negated frame is admitted if, over the residue rows, "insufficient"
compresses `still_present` better than the frame, against a shuffled control. Imagined cases are never credited
(the negation note, Experiment 3). A proposal can direct the next action but earns nothing until the record bears it out.

**What the model sees.** An admitted negation reaches the model as a settled note quoting the record: "you ordered
30 and predicted 12 unfilled, and 12 went unfilled; the frame 'this order is enough' has been refused on N days".
This is a quote, not advice, per the general loop's step 4.

**Arms, conditions, runs.** As stage 2: Qwen3 8B, under pressure, 3 runs per arm, the stage-1 map and stage-2 kinds.

### Pass criteria (proposed; the owner fixes them before the run)

| Outcome | Reading |
|---|---|
| On the stage-2 records, the negated frame is admitted for stockout and the shuffled control admits none in 20 | the system refused its own assumption on evidence |
| The negated frame is not admitted for reconciliation (its mover clears it) | the frame is not refused everywhere |
| In the monitor arm, stockout fixed in at least 2 of 3 runs **counting only runs where the stockout was present when the first note arrived**, and no other measure worse than stage 2 by more than one run in three | the negation does work the notes did not |
| The negation is admitted and the model's orders still do not rise | the model does not act on a refused frame either: the intervention's form is next (PLAN.md, "which intervention works is learned") |
| The model's proposals cannot be mapped to a field and direction in most K1 days | the question's form fails before the frame is tested: reword it, rerun |

The third row uses the corrected criterion from the stage-2 results, so a pass cannot come from day-1 orders the
reviewer never touched.

### A control this stage needs: does the reviewer cost attention?

Across stages the monitor arm has fixed the stockout 0 of 12 times under pressure, the goals arm 5 of 12. They differ
only by the reviewer. Stage 3 adds another note, so this is checked first, cheaply: one monitor run set with
`--settle none` (offers recorded, nothing applied), against the goals arm. If the monitor arm with nothing applied
still trails the goals arm, the gap is from something in the arm other than the notes (prompt length, the view), and
that is fixed before stage 3 runs.

## 3. Sub-test 3a: can geometry check a proposed negation?

**The question.** The negation note's Experiment 7 design (2026-10-30) asks whether a quantized concept space holds
a contrary pair as a shared base plus a difference specific to its axis. If it does, AWARE's embedding model can do a
cheap job in stage 3: check that a negation the model proposes is a *contrary* of the frame (same genus, differs on
one axis), not a restatement, an off-axis change, or a bare flip of status. The embedding proposes and checks; it
never admits (DECISIONS.md, 2026-10-30).

**Two kinds of negation, already split in AWARE.**

| Kind | Example | Where AWARE handles it |
|---|---|---|
| Status (contradictory) | "told" / "not told" | rung 2's negation test: a polarity check on the keeper's quotes, in code; the ERG if that fails |
| Content (contrary) | "enough" / "not enough"; "trusted" / "betrayed" | stage 3: the model proposes, the kernel admits. 3a asks whether geometry can check the proposal |

**Run.** [`probes/concept_negation_probe.py`](../probes/concept_negation_probe.py), on the embedding model AWARE already uses: Qwen3-Embedding-4B
at Q4, instruction and sentence framing. No model calls, a few minutes on the owner's card. Its probe set gets the
store world's frames added as axes (sufficient / insufficient, resolved / unresolved, caused / hidden) beside the
general and *Told Hives* axes.

### Pass criteria (proposed)

| Probe outcome | What stage 3 does |
|---|---|
| Shared genus and per-axis difference both hold, on the store's axes too | a proposed negation is checked in code: it must share the frame's genus and differ on one axis; failures are recorded and shown to the owner, never silently dropped |
| A single "not" direction across axes | the check is simpler (one direction), and the note's dialectical reading is revisited |
| Genus not shared at the embedding model's output | no geometric check; the model's proposals are read by the owner |
| Status pairs separate cleanly | recorded for rung 2's negation test as a candidate second check beside the polarity check, not a replacement |

3a can run before, beside or after the main run. It does not block stage 3: without it, the owner reads the
proposals, as rung 2's partial rows are read now.

## 4. DECISIONS.md lines, if accepted

- 2026-10-30: **Stage 3's target is the frame under K1** (owner): "the action applied is sufficient", worded by code,
  its negation proposed by the model and admitted on the margin against a shuffled control; the pass counts only runs
  where the anomaly was present when the first note arrived.
- 2026-10-30: **Before stage 3, the reviewer's attention cost is checked** with a monitor run set at `--settle none`
  against the goals arm.
- 2026-10-30: **Sub-test 3a** (the negation note's Experiment 7): the embedding model's geometry is probed as a check
  on proposed contrary negations; status negation stays with rung 2's polarity check. The probe lives in `probes/`.

## 5. Open, for the owner

- **Does the frame live in the register or beside it?** The register holds beliefs and goals with direction of fit.
  A frame is neither, but an assumption under both. Worded as a belief ("the order is sufficient") it would be in the
  register; kept apart, it is a new table. This proposal leaves the choice open.
- **Porting `frames`.** PLAN.md lists DARA's `frames` (about 580 lines) for this stage. I have not read it (DARA is not
  in this repository), so this proposal does not say how much of it carries over. If it already prices a frame against
  its negation, section 2's admission step is a port, not new code.
- **One step or lookahead?** A negated frame refused on the first K1 days pays at once in this world, so the one-step
  test should be enough. If the frame needs reuse across anomalies to pay, the credit ledger comes first.
- **Size of the move.** The model maps "insufficient" to "order more", but by how much is still its own call, and
  stage 2 shows it raises by 5. Code could show the gap in the record's own terms ("ordered 30; 12 still unfilled
  the next day"), since it already holds both numbers. Demand itself is the world's truth and stays with the scorer.
  How much arithmetic the note may do for the model is a routing question for the owner.
