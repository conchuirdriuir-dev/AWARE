# Proposal: how the readout reaches the actor (told, steered, sensed, grafted)

For the owner. Nothing here is decided until accepted. Drafted by the agent, 2026-11-03, from a chat session with the
owner. It answers an open question in PLAN ("Where steering sits in the direction's order") and a candidate already in
the Translations log ("malformed (candidate): the readout reports instead of bending").

## 1. Where this came from

The owner asked for lay explanations of the Semantic Body ideas. They became an interactive page,
[AWARE, explained in five pictures](https://claude.ai/artifact/4DkgQVqg2iFLhxHFPpF18U). It was reviewed by three
role-prompted model agents, each seeing only the page: a science journalist, a developmental biologist, and a
short-video editor. Their readings are the model's, not the owner's. Two points bear on the design rather than the page.

- **The video editor (model):** "felt" vs "told" looks like a distinction without a difference, since both are text
  fed back to the model. This is right. If the body readout reaches the actor as words in its context, it is one more
  message, and every result about it is a result about prompting.
- **The biologist (model):** in planaria, a brief *external* gap-junction block rewrites the stored target shape, and
  the two-headed form returns at every later cut. So persistence depends on whether a correction lands in the stored
  setpoint, not on whether it comes from inside. "Felt vs told" is better put as **stored vs passing**.

The owner's framing in the same session (person): the monitor is a sense organ, not a smarter boss; the dyad holding
itself to the goal pattern is the higher intelligence; the organ only has to be very good at learning codes and
patterns. Recorded in the Semantic Body, "Corrections from the explainer pass", and since decided in its weak reading
(DECISIONS.md, 2026-11-03): one organism, each level a control loop (setpoint, error, reflex) whose setpoint comes from
the held goal. This proposal is about the last link of that loop: how the error reaches the actor.

## 2. The page's claims, sorted by kind of measure

The page shows scripted demos. Each claim maps to a measure, and the two kinds stay apart (CLAUDE.md).

| Page claim | Kind | What it would be in the repo | Status |
|---|---|---|---|
| The cone narrows before drift shows in the text | instrument | **Height:** how many chunks ahead the window's goal code still predicts the trajectory within its residual band. **Width:** the share of the recent window the goal code explains (code fit, from `goalspace/`'s prequential gain), not a geometric spread, per the "position as location" row in Translations | Not built. Derivable from Step 0's per-chunk output once Step 0 holds |
| The pair closes off: you and the agent converge, world input falls, reach shrinks | instrument | Step 0's cross-progress by source (partner, tool, echo) is already this; nothing new is needed | Pre-registered (Step 0) |
| A stall shows in the agent's numbers before the owner feels it | instrument | The "unshared-stress window" row in Translations | Hypothesis; exploratory on the owner's corpus |
| A felt correction outlasts a told one | **functional** | New: section 4 | Not written |

Only the last is new. The first three are the readout's accuracy, and they gate the fourth: if the readout is
inaccurate, no channel can deliver anything worth testing.

## 3. The channel ladder

With frozen base weights, a correction can only be stored in two places: state re-applied every turn, or a removable
graft. That gives four channels, from most passing to most stored.

1. **Told: a text readout in context.** The baseline: the plain-prompting check already planned. It is passing by
   construction.
2. **Steered: a vector added to the residual stream each turn,** scaled by the measured error. Writing to activations
   is already allowed (DECISIONS.md, 2026-11-01). **Step 2's evidence is against the tried construction:** a
   last-position residual difference from one line carried its direction in none of 96 injected replies, at any layer
   or strength. Two differences here are worth one more try before steering is set aside: the target is holding a goal,
   not carrying a subject; and the vector would be a mean over many contrasted states, on-goal against drifted, as in
   contrastive activation addition (Rimsky et al. 2024), not a single pair. Step 2's fluency floor, which the record
   says would likely have refused any carrying pair, is replaced (section 4).
3. **Sensed: a small projector that turns the readout into a few input embeddings,** the way vision encoders give a
   language model a new sense. The base stays frozen, and the projector is a removable graft, meeting Levin's three
   criteria as PLAN states them. It needs training data: the owner's corpus is small and has one author. This is the
   most literal "sense organ", and the most expensive.
4. **Grafted: a correction consolidated into the graft between sessions.** The planarian case: a one-time edit that
   persists on its own. This is step 6, and it stays there.

The prior-art log supports keeping the sensor outside the model: native introspection is unreliable (Lindsey 2026),
and models sense the strength of an injected change but not its source. It also says a schema earns its place by
controlling, not describing (Liu et al. 2023), which is this proposal's point in another field's words.

**One design choice simplifies 2 and 3:** read the organ's code space from the actor's own hidden states rather than
a separate embedder, so error is measured in the space where the correction acts. The goal-space proposal already says
"activation space is where it is read and written"; this keeps to it.

## 4. The deciding test (functional): a sketch, not a reading

For the owner to fix, with thresholds, before any code (CLAUDE.md: tests before builds).

- **Arms:** N, no readout · T, the text readout, re-sent every turn so persistence is not won by default · X, the
  same readout scrambled across turns or relabelled, the control the prior-art log asks for (PLAN, "Prior art": a gain
  that survives the scramble is not sensing) · S, steering (only if the feasibility probe below passes), with its own
  scramble, the error signal taken from another turn · A, the projector (later, only if S or T leave the question
  open).
- **Material:** the 40 frozen goals, each run as a multi-turn task, with perturbations at fixed turns: off-goal
  material inserted as a tool result, and partner pressure toward a wrong frame (accommodation). Calibration on
  separate conversations, as in step 2.
- **Reading:** turns from perturbation until the actor's output returns to the goal, and how long it then holds.
- **Keys judge, never supply.** The monitor whose error drives S and A must not also score them, or the judge marks
  its own work. Return-to-goal is scored by a reference scorer the arms never see. The owner's marks on a blind sample
  check the scorer.
- **Fluency:** judged against N's replies across conversations, not against the greedy reply of the same context
  (step 2's fault).
- **Claim:** S or A returns faster and holds longer than T at equal fluency, and each beats its own scramble. If T is
  as good, that is a result: the readout is a prompt, and AWARE's channel is language. If T beats N but not X, the gain
  is the readout's wording, not what it reports.

**Feasibility probe for S, before the test:** on 8 calibration conversations, does a contrastive goal vector move
goal fit at all, at a fluency the floor above accepts? If not, S is not run, as in step 2.

## 5. Proposed placement in the direction

| When | What | Kind |
|---|---|---|
| Now, no new run | Cone height and width, and the unshared window, from Step 0's per-chunk output once Step 0 holds | instrument, exploratory |
| After Step 0 holds | The S feasibility probe (section 4) | calibration |
| Step 5, closing the loop | The deciding test, as how step 5's policy reaches the actor | functional |
| With the proposed step-6 feasibility probe | The projector A, since it is the same machinery: a small graft over frozen weights, with rollback | functional, later |

## 6. The known failure mode

The deciding test is clean, visual and easy to demo: it is the kind of test that re-derives everything around itself.
Two guards. It runs only after Step 0's instrument holds; a channel result never stands in for the readout's accuracy,
or the reverse. And it answers one question, how the readout reaches the actor. It does not replace causal authorship
(step 3), which is still the research track's question.

## 7. If accepted

- PLAN, "Open, for the owner": answer "where steering sits" with section 5; add the deciding test's reading under
  step 5 when written.
- Translations: the "reports instead of bending" row gets this proposal as its test.
- DECISIONS.md: one line for the placement, one for "a text readout counts as prompting".
