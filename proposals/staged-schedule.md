# Proposal: a staged schedule for every planned capacity

For the owner. Nothing here is decided until accepted. Drafted by the agent, 2026-11-05, from the owner's draft "The
staged schedule: who plays, then what is learned" (itself from a design discussion with the owner) and a chat review of
it. It is read against PLAN.md as it stands on PR #31 (branch `body-step0-rehearsal`, commit fb1dad4): "Two stages: the
player and the perceiver apart", fixed by the owner on 2026-11-05, and Stage 1's registered scripted play, done.
Where this differs from the North star, the North star wins; where it differs from PLAN, PLAN stands until the owner
amends it.

## 1. What the draft gets right, and keeps

Calibrate the instrument before the subject, changing one thing per stage, so a failure says where the fault is:

| Stage | Who plays | Who reads | Asks | Status |
|---|---|---|---|---|
| 1 | The script | Qwen3-8B | Instrument validity (Step −1) | Fixed (owner, 2026-11-05); play done |
| 2 | A weaker Qwen | Qwen3-8B | Predictive validity on another's play (Step 0) | Fixed (owner, 2026-11-05) |
| 3 | Qwen3-8B | Qwen3-8B | The same on its own play; the self/world split | **Open (Q1)** |

Then Babble, Act, Dream, in that order. What a failure means: Stage 1 fails, the gauge is wrong; Stage 2 fails after
Stage 1 passes, the senses read but do not anticipate; Stage 3 fails after Stage 2 passes, perceiving another works and
perceiving itself does not, which babbling's forward model is meant to address.

## 2. Where the draft conflicts with what was fixed on 2026-11-05

1. **Stage 3.** PLAN leaves open whether the 8B playing and reading its own play returns, or the babbling gate covers
   it. The draft adds it, and maps "Step −1 as registered (8B play)" onto it; but registered Step −1 is now the scripted
   player. The draft's "Step −1S" is largely Stage 1 as fixed.
2. **Harder worlds.** Retired by the two-stages section (criteria and C1–C3 kept as the record). The draft needs them
   back for Stage 3 (the 8B's episodes must reach turn 35). Only if Q1 is yes.
3. **The script.** The fixed script alternates explore (8–15 turns) and loop (5–12 turns) phases from
   `default_rng(seed + 700000)` and never issues the quest's final command. The draft's recipes (walkthrough segments,
   loops injected at chosen turns) are a second script; any test that needs them is a new form on a new seed pool.
4. **Stage 2's screen.** The fixed screen takes the smallest of Qwen3-1.7B and Qwen3-4B projected to give at least 150
   closures and 75 ascents in Step 0's 400 worlds and winning at least 10% of episodes. The draft replaces this with
   the most episodes progressing N turns before a closure. The agent recommends keeping the fixed screen: it is tied to
   Step 0's power.

## 3. Problems with the draft's new tests, stated before any result

1. **Bar 1's restated part, moved after a rehearsal.** Rehearsal 3 (0.6B) gave restated self-talk 12% against the 5%
   bar; the draft moves that part to the tickle test after babbling. The reason is sound (a history predictor cannot
   tell restated self from world; that is the forward model's job), but moving a bar after seeing it fail is a retune,
   and the draft's own rule says results never retune a bar. It is allowed only as an owner amendment, before the 8B
   run, recorded as made after rehearsal 3. Otherwise bar 1 stands and Stage 1 may fail on it. **(Q2)**
2. **Step 0S (rate on scripted play) is circular as drafted.** PLAN's reason for not scripting Step 0 applies: X is how
   much the replies help code the player's next action, so a script's rate is its own rules. And the draft's
   contrast fails by construction: random actions and a fixed walkthrough are both blind to the replies, so world rate
   should sit near zero in both. **Proposed instead (for Stage 1, optional):** two scripts differing only in
   contingency, one choosing its next action from the reply (an exit the reply names), one blind to it on the same
   worlds. Pass: world rate higher under the contingent script, AUC **[owner fixes]**. This uses the circularity as a
   known answer; it tests the gauge's calibration, never lead. **(Q3)**
3. **Control 5 (informative vs distractor) tests a property v0's boundary does not claim.** Boundary is news against
   the episode's history, not "improves prediction of the next step"; rehearsal 3 noted that any short unseen sentence
   clears the floor. A matched-novelty distractor should fire as hard as the rule change, so the control fails as built.
   The question is rate's, not boundary's. Recommended: reported only in Stage 1; restated as a rate test later.
4. **Control 4 (self-caused vs spontaneous novelty)** needs an efference copy, and in Stage 1 the actions are not the
   perceiver's. Recommended: moved to the tickle test after babbling, not run in Stage 1.
5. **Control 6 (habituation)** falls across repeats by definition (the earlier identical reading is in the history).
   Only the recovery at a one-detail change is informative; state the bar as that alone.
6. **Lift invariance is undefined.** Lift v0 is a prequential code length; "code-space location over matched windows"
   has no definition for it yet. That definition comes before any AUC bar. Lift is not the current step's test.

## 4. Scope

CLAUDE.md asks for the simplest thing that can pass the current step's test, and warns of a clean test re-deriving
everything around itself. The draft takes Stage 1 from three bars to seven plus lift, Step 0S and chords. Proposed
split:

**Now, beside Stages 1–2 as fixed:**
- *Reported, deciding nothing:* Stage 1's plants against first-seen replies of the same length (draft control 7; the
  gap rehearsal 3 noted).
- *Optional, if the owner accepts Q3:* the contingent-vs-blind rate check.
- **Shape (Stage 1, from the draft's Step 0S; dropped from this proposal's first version by mistake).** On scripted
  play, shape (the code length of the current fit) rises at the planted rule change and falls back as the code re-fits
  within **[owner fixes]** turns; the rule change's turn is known, so this is a calibration check with a known answer.
  Unlike rate it is not circular: it reads the fit to the text, not how much the replies help code the next action.
  It needs no new play, only scoring the plants already recorded. **(Q6)**

**Proposals, no bars yet, each written into PLAN before its code when its step comes:**
- **Stage 3 and the self/world split**, if Q1 is yes. Learnable structure planted in the world's replies or in the
  agent's own notes; the rise in rate should land on the planted source's part. **If Q1 is no**, the split moves to
  babbling (Step 2), where the 8B's own actions return; otherwise nothing tests rate's self part.
- **At babbling (Step 2):** the active-vs-yoked-passive arm (a second graft on the same episodes replayed with no acting,
  matched in data and steps; if it does no worse, the sensory-substitution framing is doing no work) and the tickle test
  (the forward model frozen as its own graft, decision 1; restated self-talk and draft control 4 rerun there). The
  agent rates these the draft's strongest additions.
- **Chords**, after Step 0W has data: the closure chord against the best single sense, on the AUC; its form (fixed rule,
  or fitted on a calibration split only) fixed first. Its pure-reasoning control needs a definition of world rate where
  there is no world.
- **Dream (Step 4)**, one line until babbling exists: dream vs replay-only at matched compute on held-out worlds'
  task success (functional); rollback catching a degraded graft (safeguard); the sleep trigger firing on an injected
  plateau (instrument). Three measures, kept apart. Two more slots, from the North star's Dreaming table and ascent
  cycle:
  - *The library of codes (abstract):* a code admitted to the library codes held-out worlds it was not fitted on
    shorter than codes fitted per world. An instrument measure of the abstraction, apart from task success.
  - *Rivals and ascent (dream):* after a closure, the dreamed route yields a code shorter than both rivals on both
    episodes' held-out turns. The North star's ascent cycle (compress, go lateral, hold two codes, ascend) has a label
    in Step 0 but no test that the organism itself ascends; this is that test.
- **Act (Step 3), one more slot: the perturbation test** (North star, "Design changes", goal code as setpoint). The
  route to the goal is blocked mid-episode (an exit closed, an object moved); an agent holding the goal reaches it by
  another route, one running a habit repeats the old one. Functional (task success after the perturbation), apart
  from Felt vs Tell. Needs the goal code (decision 5).
- **The dyad, after Dream: a placeholder only.** The North star puts it at the next scale up: three-class provenance
  (agent, collaborator, world), recoupling as the drift response, split tracking, cone metrics. Paused with the
  agent–user work (owner, 2026-11-04); the slot marks where it returns, with no content until then.

**Naming** *(draft's choice, kept)*: new forms take a suffix (0W, and any new Stage 1 form); registered steps keep
their names, so records, amendments and run folders line up.

## 5. Coverage: every capacity has a step

The draft's purpose (the owner's): every part of the planned system gets a development step, or it will not be built.
A slot is a named step with its question; its bars are written into PLAN before its code, when its turn comes. The
capacities are the North star's five steps, its senses table and its Dreaming table. An empty cell is a gap.

| Capacity | Step | Kind | State |
|---|---|---|---|
| Lift (trajectory to code-space) | Re-skin invariance, before or beside Stage 1 | instrument | Slot; the distance on Lift v0 is not yet defined (§3.6) |
| Shape (code length) | Stage 1, shape at the rule change (§4) | instrument | Decided; window and bar from the 0.6B record, fixed before the 8B reads |
| Rate, world part | Step 0, Stage 2 | instrument | Fixed |
| Rate, world part, calibration | Stage 1, contingent vs blind (§3.2) | instrument | Decided; reported only |
| Rate, self part (the self/world split) | Babbling (Step 2) | instrument | Slot (Q1: no Stage 3) |
| Boundary, world | Step −1, Stage 1 | instrument | Fixed; play done |
| Boundary, efference copy | Tickle test at babbling | instrument | Slot; bar 1's restated part gates here (Q2) |
| Adaptation | Habituation's recovery (Stage 1, reported); sleep trigger (Dream) | instrument | Partial: reported only before Dream |
| Chords (senses together) | After Step 0W | instrument | Slot |
| Sense tokens, never words | Step 2, babbling gate | channel | Registered, not written |
| Babble: forward model | Step 2; active vs yoked passive | channel | Gate registered; yoked arm a slot |
| Act: acting on the senses | Step 3, Felt vs Tell and Felt-shuffled | functional | Registered, not written |
| Act: goal code as setpoint | Step 3, perturbation test | functional | Slot |
| Dream: replay and write the graft | Dream vs replay-only | functional | Slot |
| Dream: morning check and rollback | Rollback of a degraded graft | safeguard | Slot |
| Dream: library of codes | Held-out worlds' code length | instrument | Slot |
| Dream: rivals and ascent | Ascent beats both rivals | instrument | Slot |
| The dyad | After Dream | — | Placeholder; paused (owner, 2026-11-04) |

Every capacity now has a slot. Adaptation is scored only by the sleep trigger; habituation before Dream is reported,
never gating.

## 6. Questions for the owner (answered 2026-11-05)

*The owner answered all seven on 2026-11-05, each with the agent's recommended option; recorded in DECISIONS.md.*
Q1: no Stage 3, babbling covers it. Q2: amend, labelled as after rehearsal 3. Q3: add, reported only. Q4: keep the
fixed screen. Q5: Stage 1 gates, Stage 2 informs. Q6: shape's bar set from the 0.6B record, fixed before the 8B reads.
Q7: slots count; the table moves into PLAN.

**To carry into PLAN.md once PR #31 is merged** (not edited here, to stay clear of that PR): Step −1 amendment 4 (bar
1's restated part, Q2); the two-stages section's open Stage 3 question closed (Q1); Stage 1's shape test and rate
calibration (Q3, Q6); gating (Q5); the coverage table into "Current state and direction" (Q7).

The questions as asked:

- **Q1.** Does Stage 3 (the 8B playing and reading its own play) return after Stages 1–2, or does the babbling gate
  cover it? Q2's bar, the harder worlds and the self/world split depend on the answer.
- **Q2.** Bar 1's restated part: stands as registered, or amended before the 8B run (recorded as after rehearsal 3)
  and moved to the tickle test?
- **Q3.** Add the contingent-vs-blind rate check to Stage 1, with its bar fixed first, or leave rate to Stage 2?
- **Q4.** Stage 2's screen: keep the fixed one (the agent's recommendation) or the draft's?
- **Q5.** Gating: Stage 1 gates Stages 2 and 3; Stage 2 informs and does not gate (the draft's choice)?
- **Q6.** Shape at the rule change in Stage 1: added, with its re-fit window and bar fixed before scoring, or reported
  only?
- **Q7.** Coverage (§5): is a slot (a named step with its question, bars later) enough to count a capacity as
  scheduled, and should the table move into PLAN's "Current state and direction" once accepted?

## 7. Limits

- Stage 1 tests the reading, not the prediction; a pass there says nothing about early warning.
- The script's loops are cleaner than real ones; a gauge that reads them may still miss real stalls.
- Stage 2's player is another organism to the reader; a result there transfers to the 8B's own play only by assumption
  until something like Stage 3 runs.
- Re-skinned worlds keep sentence templates, so lift's re-skin pairs share more surface than names; restructured pairs
  should keep templates too.
