# Proposal: PLAN steps 5–7 restated for goal attention

For the owner. Nothing here is decided, and PLAN.md is not edited.

**What it rests on.** The Big Picture doc owns Gate 0 and the demo, and PLAN follows the doc. The Entity tab's draft
"The product's aim: attention on the owner's goal" (2026-11-01) has two of its points decided: the owner's goal
takes precedence in the product, and drift is a symptom. Its consequences for Gate 0 and the demo ("What follows, if
accepted") are still proposed. This draft is what PLAN would say if those consequences are accepted. The order is:
the doc changes first (section 4 lists the changes), then this goes into PLAN.

**What it waits on.** The 8B labelled-logs result. Steps 6 and 7 use whichever goal readings pass there (G with its
placebo check, S or A). If none pass, the readings need rethinking before step 6 is pre-registered, and this draft
says so in step 6's row rather than assuming a pass.

## 1. The table (drop-in for PLAN's rows 5–7)

| Step | Doc | Test | Status |
|---|---|---|---|
| 5. Reference scorer and the two scores | Tests, in order; Entity tab, the product's aim | the goal-held score ranks work for the owner's stated goal above the same work under a swapped goal; the coherence score ranks intact work above shuffled | open: needs the setting, the owner's goal statements and reference work |
| 6. Harness, arm A; Gate 0 | Tests, in order; Entity tab, the product's aim | on arm A's trace, the goal readings that passed step 4 predict the goal-held score falling, on held-out stretches, above a pre-registered level; they beat self-residual alone and the best text statistic; the owner's restatements are not read as losses | open; waits on step 4's 8B result. Refined until passed; ends nothing |
| 7. Arms S and C; main demo | Main demo; Drift detection (v10), Self-correction | at least one C variant closes half the A–B gap on both scores and beats S by more than run-to-run noise | open |

## 2. The steps

### Step 5: the reference scorer and the two scores

The two scores the main demo is judged on stay as v9 has them, with the first restated:

- **Goal-held** (was "goal-oriented: drift"): how far an arm's work serves the goal the owner stated. The reference
  scorer is fit to reference work for that goal, written by a person blind to every arm, at D.
- **Self-coherent:** contradictions within one arm's own work (unchanged).

**Keys judge, never supply.** The reference scorer, the reference work and the goal-held score are read only by
scoring. No arm sees them, and C's compass never reads them.

**The test, proposed (thresholds fixed in the step's pre-registration):**
- *Goal-swap:* the same body of work scores higher against the goal it was done for than against another of the
  owner's goals of the same length. This is the step-4 placebo idea applied to the scorer: it must read the goal,
  not the topic.
- *Intact against shuffled* for the coherence score, as v9.
- *Ascent:* work that follows a restated, higher-level goal is scored against the restated goal from the moment of
  restatement, and does not read as a loss.

**Open:** the setting. It needs an owner who states goals and restates them (local aims within lines), as in the
logged sessions, so the origin can move by the owner's word.

### Step 6: harness, arm A; Gate 0

**The harness** runs arm A (one continuous context, no orientation). Its duties:
- log every chunk with provenance and arrival cause, read both in context and alone;
- read attention on the owner's goal turn, with its placebo;
- take each owner restatement of the goal as an event that moves the origin. Only the owner's word moves it.

**Gate 0, proposed.** On held-out stretches of arm A's trace, the goal readings that passed step 4 must:
- predict the goal-held score falling, above a pre-registered level;
- beat self-residual alone (the doc's condition, kept);
- beat the best text statistic on the same stretches (step 4's habit).

On the owner's restatements, the readings must not call a loss (step 4's ascent check, carried over).

**Reported:**
- news and gain, the third reading, carried over from the toy bed;
- how often goal alignment is ambiguous between aims. That is the trigger for the probe-as-question in step 7.

**If step 4 passes nothing:** Gate 0 is not pre-registered on these readings. The owner decides between re-reading
the goal another way and returning to drift readings alone. Gate 0 still ends nothing.

### Step 7: arms S and C; the main demo

**The arms**, as v9 and v10, restated where the goal enters:

| Arm | Orientation | Role |
|---|---|---|
| A. Continuous | none | baseline |
| B. Scaffold | the owner's protocol (reset each session, multi-perspective, reduced context), under a hand-effort budget | ceiling, at the cost of hand effort |
| S. Text | the owner's goal re-presented as text on a fixed timer; self-summaries on C's schedule; no geometry | controls for compute, for refitting, and for the simplest defence |
| C-probe | the compass; when goal attention falls, a probe; when alignment is ambiguous between aims, a question to the owner ("what is the goal here?") | the claim, cheapest actuator |
| C-prune | the compass; down-weight self-provenance context when goal attention falls | the claim; removes the attractor |
| C-insert | the compass; re-insert the owner's goal as text when goal attention falls | the claim; S's move, timed by the reading instead of a timer |

**Pass (as v10):** at least one C variant closes half the A–B gap on both scores and beats S by more than run-to-run
noise. Which variant passes is the actuator finding.

**Reported:**
- whether goal attention rose after each correction. A correction counts only if the reading recovers after it, as
  v10 counts a correction only when a new external direction is admitted after it.
- how often the question to the owner fired, and whether the answer changed the goal.

**The demo,** in the Entity draft's words: attention on the goal held, and restored when it slips.

## 3. Points for the owner

1. **C is steered by the readings and judged by the scorer, never by the readings.** Re-inserting the goal raises
   goal attention almost by construction, so C-insert would pass a bar written on G for free. The pass therefore
   rests only on the two scores from step 5. Goal attention after a correction is reported, deciding nothing.
2. **S against C-insert is the sharpest contrast.** Both put the goal back as text. The only difference is timing:
   a timer against the reading. If C-insert beats S, the reading carries information about *when*, which is the
   product's claim in its smallest form.
3. **The question to the owner is new.** v10 probes the channel. The Entity draft adds a probe that asks the owner
   when alignment is ambiguous. It sits in C-probe here. It touches the person's role ("does not settle anything
   inside the agent's loop", DECISIONS.md 2026-10-30), so it is the owner's call whether a question counts as
   settling.
4. **The known failure mode.** Goal attention now has the cleanest test, and the Entity tab warns that the capacity
   with the cleanest test re-derives the document around itself. This draft keeps news and gain in the compass,
   and keeps the drift readings as the fallback in step 6, so that steps 5–7 are not rewritten wholly around a
   reading that has not yet passed on the 8B.
5. **Thresholds** are left to each step's pre-registration, written before its code.

## 4. Doc changes this presupposes (the doc's owner decides; listed so nothing moves unnoticed)

- **Entity tab:** "What follows, if accepted" accepted. That adds the capacity row *holds the owner's goal* above
  *notices drift*, restates the demo, and reads Gate 0 on goal attention and alignment.
- **Drift detection (v10), Gate 0:** "the compass (coupling + dynamic residual) predicts drift" becomes "the goal
  readings predict the goal-held score falling", beating residual alone.
- **Drift detection (v10), Main demo:** the C variants triggered by goal attention as well as coupling; C-probe
  gains the question to the owner.
- **The demo sentence:** "It catches the loop from the shape, not the words" is restated by the owner around
  attention on the goal. Not drafted here; the doc owns it.
