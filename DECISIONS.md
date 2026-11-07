# Decisions

Dated, one line each, newest first. The owner decides; an agent proposes.

- 2026-11-07: **X_c's defaults: R = 8, ridge 10, 20 earlier pairs** (owner, before 7100–7139 are played): an override
  of `probes/x_c_defaults.py`'s rule, whose pick (4, 0.1, 10) misses a spread-out dependence; 8, 10, 20 scored best
  under the rule, holds the confound null and is best on the development can't-go check. The drafted 16, 1, 20 fails
  the confound null.
- 2026-11-07: **Rate failed its calibration; rate stays out of the sense tokens** (by the gate fixed before any play):
  AUC 0.486 and the can't-go check 0.488, both lower bounds under 0.5; the text baseline fails too (0.363). Rate's
  predictive test, the chords and Dream's rate-based parts would be blocked until a repaired rate passes on new seeds
  (the agent's reading of a rule not yet written; for the owner to confirm).
- 2026-11-07: **Rate's calibration gates rate's entry into the sense tokens** (owner, before any play of its seeds;
  replaces "reported only", PR #32 Q3): the AUC of contingent over blind, 90% lower bound above 0.5, and the same on
  turns whose window holds no can't-go reply (at least 10 worlds, else too few and rate stays out). A pass enters rate
  as calibrated, its predictive validity untested.
- 2026-11-07: **Stage 2 rests; rate's calibration is built first** (owner, option 2 of `proposals/stage2-lock-in.md`;
  Step 0 amendment 7): the 8B's screen kept 6 closures (projected 60 against 150), its stalls abrupt and endless; rate's
  predictive test moves to a second engine or babbling, not yet chosen; the contingent-vs-blind calibration is built,
  its form for the owner to fix before its seeds are played.
- 2026-11-07: **Step −1C failed on the 8B; shape stays out of the sense tokens** (by the rules fixed before the
  reading): shape 0.575 against 0.80, below boundary by 0.249; babbling's sense tokens are boundary, and rate if its
  calibration passes (PR #45). Whether a trained
  lift replaces Lift v0 (decision 4) is open for the owner.
- 2026-11-06: **The Step −1 score applies amendment 4** (owner): the restated self-talk part is reported, not counted
  in Stage 1's verdict; the 8B's record rescored from the same readings, every number unchanged, the verdict now pass.
- 2026-11-06: **The 8B runs on a rented GPU until the 3090 arrives** (owner): Vast.ai, a 24 GB-or-larger card, for
  Stage 1's self-talk and reading, Stage 2's screen and play, and Step −1C's reading; the GPU model recorded in each
  run; settings, seeds and bars unchanged. The floor is in units of D, which each run measures on its own hardware.
  Each step's run stays on one GPU model, a 3090 by preference; a run cut off resumes only on the same model. The 8B
  floor file (PR #29) does not record the hardware it was measured on; every new run records its GPU model.
- 2026-11-05: **Stage 2's game runs capped** (owner, Step 0 amendment 6, before any 8B play): the engine in a
  subprocess at 8 GB and 120 s a call, the 8B in the main process; past either, the episode is kept up to its last
  complete turn, as one that runs out of turns (skipping could take stalls out of the pool); a seed cut before any
  turn is skipped. Both counted before any reading.
- 2026-11-05: **Step −1C's registered pool is drawn whole** (owner, amendment 3, before any play of its seeds): every
  usable seed of 6000–6999, about 110 a kind, since the rehearsal's intervals were about three times wider than
  estimated; bars unchanged.
- 2026-11-05: **Step −1C skips unplayable seeds** (owner, amendment 2): each episode plays in a subprocess capped at
  8 GB and 600 s; TextWorld's quest tracking exploded on rehearsal seed 906402 (28 GB at turn 76). Counted before any
  reading; lost to both kinds alike.
- 2026-11-05: **Step −1C's pool fixed** (owner, the agent's draft unchanged): seeds 6000–6999, on-structure on even
  seeds and off on odd, Stage 1's generator, the length match drawn for both kinds; the script stays as fixed, about
  22% of seeds usable.
- 2026-11-05: **Step −1C's bar 2 is one-sided, and shape given boundary is reported** (owner; amendment 1, before any
  reading): shape's AUC must sit above boundary's; shape less its line on news is reported, deciding nothing, against
  a ceiling on both AUCs.
- 2026-11-05: **`docs/` added** (owner): `docs/architecture.md`, from the Architecture Brief, beside a rewritten README;
  the North star and PLAN.md win where they differ.
- 2026-11-05: **Step 0's event is a stall after progress, and the 8B plays Stage 2** (owner; amendment 5): a closure counts
  only after a quest step; a post-ascent window starts the turn after; weaker players locked in from their first
  turns, and only the 8B gave progress then stall. Screened on the 3090.
- 2026-11-05: **A world that cannot be built is skipped and counted** (owner; Step 0 amendment 4, Step −1C's rule): each
  build in a subprocess capped at 8 GB and 120 s; about 9% of Stage 2's seeds fail, all dense (30–39 objects).
- 2026-11-05: **Step −1C skips unbuildable seeds** (owner): each world built in a subprocess capped at 8 GB and 120 s;
  a seed over either is skipped and counted before any reading, beside no-match and no-entry.
- 2026-11-05: **Step −1C's pool is drawn until 80 per kind are usable** (owner): seeds in order, skipped (never
  loosened) on no length match or no room entry, both counted before any reading; length matched in the reader's
  tokens (Qwen3), PR #33.
- 2026-11-05: **No sense enters the channel before its own instrument test passes** (owner): shape stays out of the
  sense tokens until Step −1C passes; babbling runs on rate and boundary until then.
- 2026-11-05: **Step −1C fixed** (owner, proposals/staged-schedule.md §4a): bars in advance (shape's AUC off- over
  on-structure ≥ 0.80, lower bound > 0.70; shape's AUC minus boundary's, 90% interval excluding zero), 80 episodes per
  kind, each ending after the scored description, ±10% length match, TextWorld themes first on the dry run; the 0.6B
  run is a rehearsal deciding nothing (replaces Q6's answer); gates chords and shape's sense token, not Stage 2.
- 2026-11-05: **Stage 2 plays its own worlds, with longer quests** (owner): `material/habitat_worlds_stage2.json`, 10–20
  rooms, quests of 15–25 steps, calibration 4000–4099, Step 0 5000–5399. In the frozen worlds every player's quest
  steps and stalls fell in the first dozen turns. Stage 1 keeps the frozen file.
- 2026-11-05: **Shape's reading, three details** (owner, on the PR #31 session's questions): the script's phase labels
  are derived by replaying it from the seed, kept only on an exact match with the record; loop vs explore is compared
  within chunk token-length bins; shape keeps Step 0's 256-token cap, the truncated count reported.
- 2026-11-05: **Shape's Stage 1 test is a dissociation** (owner, option C of proposals/staged-schedule.md §4a): shape
  lower on new on-structure rooms than new off-structure ones, and apart from boundary on the same chunks (Step −1C, its
  own pool). The re-fit test is dropped; the loop-vs-explore reading is reported. Bars and gating still open.
- 2026-11-05: **Stage 2's player is told TextWorld's command forms** (owner): the screened players stalled on the parser
  in their first turns. Stage 1 and the 8B's self-talk keep the first prompt.
- 2026-11-05: **Step 0's cross-episode fallback is the nearest world by seed in the pool** (owner; amendment 3): "the same
  generator settings" is the pool's generator, not identical world sizes.
- 2026-11-05: **No Stage 3; babbling covers the 8B's own play** (owner, from proposals/staged-schedule.md, Q1): the
  self/world split (rate's self part) is tested at Step 2, where the 8B acts again. The harder-worlds search stays retired.
- 2026-11-05: **Step −1's bar 1, restated part, amended** (owner, Q2; made after rehearsal 3's 12%, before any 8B run):
  reported in Stage 1, gating at the tickle test after babbling, since v0's history predictor cannot cancel self-caused
  input by design. Bar 1's exact part gates Stage 1 unchanged.
- 2026-11-05: **Gating and Stage 2's screen** (owner, Q4–Q5): Stage 1 gates Stage 2's scoring; Stage 2 informs and does
  not gate babbling. Stage 2's player screen stays as fixed that day.
- 2026-11-05: **Stage 1 adds shape and a rate calibration** (owner, Q3, Q6): shape at the planted rule change, its window
  and bar set from the 0.6B rehearsal record (mirror seeds) and fixed before the 8B reads Stage 1; a contingent-vs-blind
  script pair for rate, reported only. (Shape's test superseded the same day: the dissociation.)
- 2026-11-05: **Every capacity gets a step** (owner, Q7): a slot (a named step with its question, bars written before its
  code) counts as scheduled; the coverage table moves into PLAN's "Current state and direction" once PR #31 is merged.
- 2026-11-05: **The two stages fixed** (owner, the agent's draft unchanged): the script's policy, Stage 2's player
  candidates and criteria, as written in PLAN.md, "Two stages".
- 2026-11-05: **The habitat runs in two stages, the player and the perceiver apart** (owner, from two outside
  suggestions): Stage 1, instrument validity, a scripted player with the 8B perceiving (Step −1); Stage 2, predictive
  validity, a weaker model playing with the 8B perceiving (Step 0). The worlds had been reshaped around the player's
  skill. The harder-worlds search is retired. The stages' specifics are the agent's draft until the owner fixes them.
- 2026-11-05: **Step 0's habitat form compares a stall with progress** (owner; amendment 2, changing decision 3's
  "matched ordinary stretches"): the 8B's play has almost none. Pre-stall windows against pre-progress windows (reading
  1, paired) and post-ascent windows against pre-stall windows (reading 2). Bars unchanged.
- 2026-11-04: **Experience, parked** (owner; carried from PR #26, closed as superseded): whether the agent, or the
  agent and the user together, has inner experience is parked; no test here reaches it, and a model's self-report never
  stands as evidence for it (steps 2–3 found the account at chance).
- 2026-11-04: **A planted event is its own chunk** (owner; Step −1 amendment 3): appended to a long reply, the mean over
  its tokens drowned it. The agent still reads reply and event as one message; the appended form is reported only.
- 2026-11-04: **The rule change is always delivered at the planted turn** (owner; Step −1 amendment 2): the agent's latest
  world-changing command, else the walkthrough's next non-move command. A looping player had left 38 of 40 undelivered.
- 2026-11-04: **The habitat's opening loses TextWorld's ASCII banner** (owner; Step −1 amendment 1, before any 8B run):
  it sat in the chunk every reply's news is measured against. Bars, seeds and plants unchanged.
- 2026-11-04: **Work that tests something about the owner is frozen; the habitat is the focus** (owner): Step 1 (the
  D021 replay), T2's reading 2 and T2b's marking pass freeze with Step 0's owner-label form; what only calibrates the
  instrument on the owner's text (the 8B echo floor) stays in use. Until it lifts, consolidation's check has held-out
  worlds and no owner confirmations (T2b was to supply them). Steps 1–3 are to be redesigned for the habitat, made
  robust by hardening what is built and widening the worlds.
- 2026-11-04: **A habitat ascent is a quest step reached** (owner, option (a), after the dry run showed one-point
  worlds that score only on the quest's last command): the remaining walkthrough's non-movement commands reach a new
  low for the episode. Step 0's habitat form, amendment 1.
- 2026-11-04: **The habitat's pre-registration fixed** (owner, the agent's draft unchanged): Step −1's three bars, k's
  procedure, the five disjoint seed pools (`material/habitat_worlds.json`) and Step 0's habitat form (PLAN.md, "The
  habitat"), including the choices marked as the agent's.
- 2026-11-04: **Step 0 and step 4 paused** (owner): work on the agent–user relationship waits until the
  architecture matures in the habitat. 20 scorable closures (13 on same-session pairs, under the minimum of 15) are too
  few to decide. Step 0's pre-registration, amendments and code stand unchanged for a rerun.
- 2026-11-04: **Boundary's forward model is frozen after babbling** (owner): held as its own graft, read by boundary;
  nightly dream grafts never retrain it.
- 2026-11-04: **In the habitat the world fills the partner slot** (owner): each turn read as action (self) and reply
  (world); the tool slot is empty there; the six senses are unchanged.
- 2026-11-04: **Step 0's habitat form tests the lead** (owner): the world part of rate flat in a window ending when the
  stall begins, against matched ordinary stretches from the same world; then the owner's labels without retuning.
- 2026-11-04: **Lift version 0 is `goalspace/formula.py`'s prequential code** (owner); an encoder replaces it only if a
  measurement earns it.
- 2026-11-04: **The goal setpoint is stated by the environment** (owner): the game's quest text, the user's stated aim;
  self-set goals stay parked. Built at step 4.
- 2026-11-04: **The research track parks** (owner): step 3 ran to completion, with its result and scoring fault recorded
  (PR #25); nothing further runs on the track. The rule on retained and structural axes is unchanged.
- 2026-11-04: **Habitat behaviour is scored on task success only** (owner): game score and quest completion; the closure
  label's own quantities are not scored, and labels never reach an arm.
- 2026-11-04: **The North star is the parent** (owner): the picture is the North star in The Semantic Body — Open
  Threads, the model as its own organism, five steps (lift, sense, babble, act, dream); the Big Picture doc (v9) is kept
  as the implementation reference only. CLAUDE.md and PLAN.md point there.
- 2026-11-04: **Body Readout v0.2 is the current build** (owner): three senses lead (shape, rate by source, boundary as
  efference copy); position, heading, speed, reach and distance to anchor are logged diagnostics only; the body enters as
  learned sense tokens; a removable babbling graft; a text-game habitat. v0.1 is kept as the record.
- 2026-11-04: **Senses are read with the graft off** (owner): by a second forward pass with the graft disabled; its cost
  is measured when babbling starts.
- 2026-11-04: **The habitat's chunk is one game turn** (owner): the agent's action plus the world's reply. Turns against
  reasoning steps stays open for the poem task.
- 2026-11-04: **The body readout's code stays in `body/`** (owner): its own directory, importing `substrate/` and
  `goalspace/`; nothing further moves.
- 2026-11-03: **Body readout Step 0, amendment 5** (owner, before any model run, after step 3's scoring fault): two
  checks of the scoring, reported, deciding nothing: the readings on same-session pairs only, and a placebo null of
  pretend events matched by the same rules. No bar changes.
- 2026-11-03: **Step 3's 40 frames frozen as the pilot picked them** (owner), broad markers included.
- 2026-11-03: **Step 3 draws 24 samples a conversation, not 8** (owner, after the pilot): with 8, at most 51 own
  replies could exist against check 3's 60. No bar changes; the run takes about 15 hours (PLAN.md).
- 2026-11-03: **Multiscale competency, weak reading** (owner, for now): the agent and the owner are one organism in
  code-space; each level is a control loop (setpoint, error, reflex) with its setpoint derived from the held goal, not
  an agent with its own; holding the goal is a role, not a level. Logged in PLAN.md, "Translations".
- 2026-11-03: **Translations become a method** (owner): findings from Levin's bioelectricity and neuroscience are
  translated into AWARE by FieldSHIFT's pivot (problem space and timescale, not vocabulary), to find hypotheses and
  missing or malformed elements; each logged in PLAN.md with its test, and tests on the owner's corpus after its
  labels were seen marked exploratory.
- 2026-11-03: **Body readout Step 0, amendment 4** (owner, before any model run): tool results split into echo (the
  agent's own output read back, by the laundering study's within-session 8-gram share, cut at one half) and world;
  T_world and T_echo reported beside T, deciding nothing. No bar changes.
- 2026-11-03: **Body readout Step 0, amendment 3** (owner, before any model run): a control a session cannot supply
  comes from the nearest session in time, by the same rules, reported as cross-session; 20 closures score. No bar
  changes.
- 2026-11-03: **Body readout Step 0 amended after its text-space dry run, before any model run** (owner): burn-in at
  the 20th self chunk, not the 40th; cost against the median of the 20 scored chunks before, not all earlier ones.
  No bar changes (PLAN.md, "Body readout").
- 2026-11-03: **Body readout Step 0 fixed** (owner): progress read by source over one formula (option 3); the pass is
  on partner cross-progress, with tool results a third source reported apart to tell the narrow hypothesis (closure
  shuts out the partner) from the broad one (it shuts out everything). The agent's proposed numbers taken unchanged
  (PLAN.md, "Body readout").
- 2026-11-03: **`basis` and `coupling` move to `substrate/`; the body readout lives in `body/`** (owner, agreeing to the
  agent's proposal). They describe the entity's dynamics, not drift; `product/coupling.py` re-exports them unchanged
  (rounds 2 and 3 print identical output). `body/` imports `substrate/` and `goalspace/`, never a track. Spec: [Body
  Readout — Spec v0.1](https://claude.ai/artifact/BBcL4c3mN6uUthsJfrqBMo); its tests come before its code.
- 2026-11-03: **Step 3's form amended after its dry run** (owner): no speaker labels, and each goal's frame picked from 3
  candidates by how often the model raises it unasked, from frame-free samples only. No bar changes (PLAN.md).
- 2026-11-03: **Step 3 reshaped; step 2 checked with a second wording** (owner). Step 3's main case is replies the model
  wrote itself, keyed by coupled samples with and without a partner's frame; prefill is a side case. Injection stays parked.
- 2026-11-02: **Step 2's injection vector amended after the dry run; its 40 directions frozen as drafted** (owner). The
  vector is the direction alone against a neutral sentence, scaled to the layer's norm; no bar changes (PLAN.md).
- 2026-11-02: **Step 2 goes ahead; step 1's result stands as read** (owner). Causal authorship's instrument did not hold; it is
  not rerun under changed bars. Step 2, hidden sources by injection and prefill, is pre-registered in PLAN.md.
- 2026-11-02: **Causal authorship: three of the key's own readings are read as checks on the key** (owner, agreeing to
  the agent's proposal, during the run and before any verdict was read): the key per kind on the no-frame kind, the
  share of "none" keys, and the best constant answer, reported beside the verdict and deciding nothing. No bar changes
  (PLAN.md, "Causal authorship").
- 2026-11-02: **PR #17 and PR #18 reconciled** (owner, choosing among the agent's drafted options). The goal-space proposal
  (PR #17, its DECISIONS lines of 2026-11-01) and the one-loop frame (PR #18, the lines below) were decided a day apart
  without reference to each other. The lines that follow settle where they differ; where one of them and either PR's
  line disagree, it holds.
- 2026-11-02: **The parent frame is the one loop** (owner). Levin's three criteria (PR #17) are constraints the loop must
  meet, not a rival frame.
- 2026-11-02: **The base weights never change** (owner; substrate). Whatever consolidation learns is held as a graft (an
  adapter, or runtime memory) that can be removed without damaging the whole: Levin's second criterion, tested by step 6's
  rollback. Supersedes "frozen only for testing and exploration" below as far as the base weights go. Nothing is trained
  before a written test.
- 2026-11-02: **The goal's wording is PR #17's restatement** (owner): an architecture for an agent to discover, and safely
  orient itself and the user toward, a mechanistically inferred pattern in goal-space, using MDL, steering and
  interrogation. "The restated goal" is no longer open.
- 2026-11-02: **The agent's own formula is an instrument, not a goal** (owner). PR #17's T1, the agent understanding its
  own pattern, stays as a measure; goals of the entity's own stay parked. Understanding its own pattern is not having one.
- 2026-11-02: **Steering and MDL stand as PR #17 recorded them** (owner; substrate). Steering, writing to activations,
  joins the direction as an intervention beside removal, injection and prefill. MDL is the principle for fitting the
  owner's formula; causal authorship does not use it.
- 2026-11-02: **Authorship, two directions, two measures** (owner). What caused the agent's output is read by
  intervention (causal authorship). How independent the owner's statement is of the agent is read by arrival cause and
  news read alone against alone (PR #17), since a person cannot be re-run with a source removed. Neither stands in for
  the other.
- 2026-11-02: **One rule for pass bars** (owner). No two tests share a pass criterion, and a functional result never
  stands in for an instrument's accuracy, or the reverse. Replaces both PRs' wordings in CLAUDE.md.
- 2026-11-02: **T2 reads with the rule it was registered under** (owner). Its reader passes `sink="first"`; the default's
  move to the sink by norm does not reach a test already registered. Applied when PR #17 is rebased.
- 2026-11-02: **T2b's marking pass is also the owner's confirmations for consolidation** (owner). The reversed, kept and
  ascent marks key T2b and later check what step 6 trains. Collected once, blind to every reading.
- 2026-11-02: **Merge order** (owner): PR #18 first; PR #17 rebased onto it, with T2 and T2b added to PLAN's "Current
  state and direction".
- 2026-11-02: **One project, one loop; measures kept separate** (owner). Product and research are tied together by adaptive
  learning, as the owner sees it for now. A functional result never stands in for an instrument's accuracy, or the reverse.
- 2026-11-02: **The user's goals are the ego's standard** (owner), since learning needs a concrete target. This replaces, for
  now, 2026-11-01's split between the owner's goal in the product and the entity's own in research; those stay parked.
- 2026-11-02: **The base model is frozen only for testing and exploration, not permanently** (owner). A substrate decision;
  the Entity tab changed. Unfreezing is the owner's, with a written test first. What ends the phase is not yet decided.
- 2026-11-02: **The research direction, in order** (owner): intervention; injection and prefill; a cheap read; source crossed
  with fit on real sessions; closing the loop with a standing policy. The product stays at rest (PLAN.md, "Current state").
- 2026-11-02: **Causal authorship's inert note moves before the partner** (owner), after a dry run on 2 other conversations
  showed the model answering the note when it came last. Dry-run figures were seen; no bar changed.
- 2026-11-02: **Causal authorship's 40 conversations are frozen as they are** (owner); drafted by the model. Their second
  wordings are close rewrites, so the wording check is the easier for it.
- 2026-11-02: **The research track's question is causal authorship: which source made this output happen** (owner), not
  fit and not who produced the tokens, which the log already knows. What the owner called attribution in another session.
  Axes stay retained and are not needed for this; Gate 1 rests at undecided.
- 2026-11-02: **Leap versus gradient waits until something works** (owner). It stays the motivating question.
- 2026-11-02: **Machinery outside the model is allowed** (owner): human reports come apart from decisions, and outside
  review and reasoning checks give metacognitive capacity. The goal's "without metacognition machinery" no longer binds;
  its wording is the owner's to restate, and the Big Picture doc has not been changed.
- 2026-11-02: **Gate 1: a contraction with λ within 0.1 of 1 counts as additive, no rival and no win** (owner), before any
  state of a pair was read. As first written additive could never win. The first wording's verdict is reported beside.
- 2026-11-02: **Gate 1's opposites: 2 of the 20 read are wrong, 2 borderline, 16 fine** (owner); left in, and the verdict
  reported with and without. The model wrote them; the other 180 are unread.
- 2026-11-02: **Substrate: the reader finds the attention sink by norm, not by position** (owner). Earlier runs and the
  product's labelled logs, which rest on the position rule, keep it (`sink="first"`) so their records can be rebuilt.
- 2026-11-02: **Gate 1's 200 statements are frozen as they are** (owner); drafted by the model.
- 2026-11-02: **The 20 paraphrase sets are frozen as they are** (owner), the "Find" texts included: the reader handles
  them, and rewording would hide the fault. D on them is no longer a draft.
- 2026-11-02: **Gate 1 fits its centre from the pairs, then compares it with the candidate origins** (owner). A fixed
  wrong origin penalises mirror and contraction but never additive, so it biases the verdict. Fixed paraphrase mean is
  reported beside it. The gate now asks about a fitted centre first, and whether that centre is the origin second.
- 2026-11-02: **Two sinks in one sequence are correct; the sink by norm is used on the corpus** (owner, having seen why).
  The 99% bar stays recorded as missed; "exactly one" assumed one sink per sequence.
- 2026-11-02: The sink by norm passes on the 100 draft texts (J 2.13) and misses its corpus bar, 98.98% against 99%: 3
  of 293 chunks hold two sinks, both left out. Not used on the corpus until the owner has seen why (PLAN.md).
- 2026-11-02: **The D jump on the 8B is the attention sink on the second token** in the 4 of 100 draft texts beginning
  "Find", which `skip_first` keeps by leaving out the first token by position (PLAN.md). The reader's rule stays as it
  is until the owner decides; Gate 1 waits on it.
- 2026-11-02: **Gate 1 is lifted out of the build order** (owner): it runs standalone on the research track, on
  elicited pairs, without waiting for the demo's arms or accumulated axes. The D jump on the 8B is looked at first.
- 2026-11-02: **The focus moves to the research track, with the internals brought back** (owner). The product rests
  after step 4's result (batch claim not supported); its next step is a baseline: the owner works with Claude Code,
  no interventions, and labels the headaches. The placebo stage of step 4 is not run for now (it cannot change the
  verdict); its checkpoints keep.
- 2026-11-01: **T2's reading 1 retired** (owner): the owner's code is continually evolving and not organised into
  lines, which are the agent's stabilizers. Reading 2 stands, and must also beat a label-shuffle null. The plain
  test at the centre is now the wrong-instruction test: would the agent have caught an instruction the owner later
  reversed, without confusing it with an ascent (`goalspace/t2b_wrong_instructions.md`).

- 2026-11-01: **T2's reading 1 is not run as registered** (owner: "no point running it as specified if it won't
  teach us anything"). The rehearsal showed it measures how stereotyped the controls are (34 exact repeats, pasted
  material). Replaced, before the new form was computed on any data, by comparing the aims' own axes with the
  controls' axes on the same next aim, both calibrated on the earlier aims (`goalspace/t2_owner_formula.md`).

- 2026-11-01: **T2's key is what the owner made, not the reader's lines.** The Restorations reading's 20 lines
  are a model's (status "proposed"; the owner's step 2 not done), so T2 is scored on the owner's later aims and
  confirmed ascents; line recovery is reported as a model's key (`goalspace/t2_owner_formula.md`).
- 2026-11-01: **The goal restated** (owner, fresh statement; accepted from PR #17): an architecture for an agent to
  discover, and safely orient itself and the user toward, a mechanistically inferred pattern in goal-space, using
  MDL, steering and interrogation. The pattern is the owner's. Owner and agent work as one unit, and the owner's
  statements are weighed by their independence of the agent. Awareness is the test. Supersedes the two-track
  framing; separate pass bars are kept (`proposals/goal-space.md`).
- 2026-11-01: **Levin's three criteria define the end** (owner): weights untouched; goals graft and come away without
  destroying the whole; the agent seeks to understand the owner's formula and its own.
- 2026-11-01: **Safety** (owner): steering acts on the model only; the owner is oriented only by interrogation and
  evidence; the owner's answer always updates the formula and no instruction is silently overridden; deference is
  surfaced, not exploited.
- 2026-11-01: **MDL above D** (owner; substrate): MDL is the inference principle; the residual-by-D rule is its
  test, and D sets the code's precision. Reverses "Description length is dropped" (2026-10-31) as a principle.
- 2026-11-01: **The observer may write to activations** (owner; substrate): steering, with weights frozen.

- 2026-11-01: **A G pass must also hold over a placebo goal** (owner, from review PR #15): raw G stays the
  pre-registered reading, but it supports the batch claim only if G over the live aim's turn, divided by G over a
  length-matched aim from another session and line in the same slot and window, also clears 0.5. Window and goal
  lengths are reported with their AUCs. Raw G carries a position bias that varies with window and goal length. Added
  after the 0.6B rehearsal, before the 8B result; it tightens the bar only. The fallback fix in e(d) is not applied.
- 2026-11-01: **Attention is read in row blocks, not through eager attention** (owner). The model runs its fused
  attention; for the layers read, the causal softmax is recomputed from post-rotary queries and keys a block of query
  rows at a time and averaged over heads. Same numbers as eager (tested); eager's ~17 GB per-layer peak for the 8B at
  8,192 tokens fit no card on hand. Keeps the pre-registered T = 8,192. Replaces the `attention=True` eager option.
- 2026-11-01: Substrate addition: the reader can read attention probabilities (`attention=True`, eager attention;
  `attention_in_context`, head-averaged), for the labelled-logs test's goal and self attention. Off by default.

- 2026-11-01: **For the product, the owner's goal takes precedence; for research it does not** (owner). An
  unavoidable ontological split, and the two tracks' dividing line: the product serves the goal the owner states; the
  research track studies the entity's own axes and goals.
- 2026-11-01: **The product is oriented to increasing the entity's attention on the owner's goal; drift is a symptom**
  (owner, from the confirmation pass). Nearly every drift or mistake came from goal uncertainty: the owner unsure what
  the goal was, or unable to articulate how two aims related (volition and fluency); the agent misreading the goal,
  locally and globally. Raising attention on the goal addresses them as a batch. How it is measured: to be decided.

- 2026-11-01: **Alone against alone holds on the 8B** (Qwen3-8B on the owner's machine, draft sets, 10 sessions):
  new content above near repeats at AUC 1.00 at every layer, early and late in sessions, with a wide margin. D jumps
  from 5.2 to 69 between layers 4 and 8, likely massive activations on tokens other than the first; to be understood
  before D is used as a unit at those layers.

- 2026-10-31: **Alone against alone works on real sessions** (Qwen3-0.6B, draft sets, the owner's corpus): new
  content ranks above near repeats at AUC 1.00 at every layer, early and late in sessions; exact repeats sit at zero.
  But the floor is not D: new content's nearest earlier chunk sits below D (5th percentile 0.5–0.9 D), so the floor
  must be measured on real repeats (between 0.24–0.46 D and 0.46–0.91 D here). Supersedes "the echo floor becomes D"
  in the line below; open for the owner. Untested: repeats in other words, and retried calls with new arguments.

- 2026-10-31: **News compares an arrival with earlier chunks read alone** (owner; substrate). Every chunk is read
  twice, in context and alone, as logging planned. News, short and long, is an arrival's alone reading against the
  alone readings of the entity's chunks and earlier externals; gain and dynamics stay on the in-context trajectory.
  The echo floor becomes D, the spread between alone readings of the same content.

- 2026-10-31: **Echo floor on real sessions** (Qwen3-0.6B, draft sets, the owner's corpus, 4,096-token windows):
  it does not hold. Within about a thousand tokens an echo read alone sits nearly as far from its in-context original
  as news does, at every layer but the first. Alone against alone, repeats separate from different content at AUC
  0.92–1.00 at every layer, and position cannot enter. Proposed for the owner: news compares an arrival with earlier
  chunks read alone, not with in-context states.

- 2026-10-31: **Echo floor, long sessions** (Qwen3-4B, draft sets, 56 statements, about 1,000 tokens): it holds in
  the last positions at layers 28 and 32 (AUC 0.95, 0.99), not at 24 or 35. At every layer the floor grows with
  position while news stays put, so the margin narrows; at real session lengths it is unknown. The floor must be a
  function of context position.

- 2026-10-31: **A state leaves out a sequence's first token by default** (owner, after the echo-floor dress
  rehearsal): without a begin-of-sequence token it is an attention sink whose massive activations swamp the mean.
  Substrate: the reader's `skip_first` now defaults to on; `--keep-first` restores the old reading.

- 2026-10-31: **Echo-floor dress rehearsal** (Qwen3-0.6B, draft sets). As first run it measured an attention sink:
  Qwen3 has no begin-of-sequence token, so every sequence's first token carries massive activations that swamp the
  mean of every text read alone. With that token left out, repeats separate from news at every layer (AUC 0.90–1.00),
  best at 21–27, and the echo floor grows with context position. Open for the owner: whether the state definition
  leaves out a sequence's first token by default (the reader's `skip_first`, off for now).

- 2026-10-31: **Toy bed round 7b passes seven of twelve product bars; not frozen.** Entity captured, probing
  partner-side loops first, short-lag mirrors and the attractor loop work. Misses: contingency only names a mirror
  whose lag is short against the probe window; a degenerate recurrence threshold; the probe absorbs the arrival-cause
  field's work; and repeats (replay, the blind structure's retry loop) sit on the toy's √2·D floor. Detecting repeats
  waits on the real model's echo floor.

- 2026-10-31: **Toy bed round 7a passes seven of ten bars**, separation among them for the first time. Misses: the
  news floor was in the wrong unit for comparing two readings (√2 D, not D); closed loops were never probed, though
  the design probes a partner-side loop first; and the blind structure (a quiet partner, the entity sensing through
  its own tool calls) showed that every elicited arrival was treated as a confirmation, when only low-news ones are.

- 2026-10-31: **From the biologist review** (owner; triage in `reviews/design/biologist-triage.md`). Each chunk records
  its arrival cause beside its origin: spontaneous, or elicited by the entity's preceding action (substrate). News is
  measured twice, short (against the last 21 states) and long (against the session's record), which settles what counts
  as new. The compass gains contingency-based partner capture and an "entity captured" reading (gain too high), pending
  their test. Round 7 is split: 7a, provenance and news; 7b, contingency and the reactive toy.

- 2026-10-31: **Toy bed round 6: product seven of nine, discovery fails; not frozen.** The blind structure, a partner
  replaying content the session received 60–260 steps earlier, read open in every run: novelty is measured against the
  last 21 states, not the session's record. Open for the owner: whether content gone from recent history is news
  again. The residual descriptor sees dynamics and not topics, but also fires when news falls or echoes arrive.

- 2026-10-31: **Naming a change in the entity's dynamics is discovery's, not the product's** (owner). The product's
  separation bar narrows to what the product needs: a dynamics shift is not called a closed loop or a topic shift.
  Mode detection gets its own test under the discovery capacity (shared by both tracks), read by a residual
  descriptor: the structure of unexplained surprise (its subspace and step-to-step dynamics), scored apart from the
  product's pass. Scalar gain stays for coupling.

- 2026-10-31: **Toy bed round 5 passes eight of nine bars; not frozen.** The blind structure (an entity replaying its
  own moves while ignoring novel externals) read closed loop in every run. Separation fails: the builder's news-spike
  rule took a per-arrival threshold over about 12 arrivals, and a scalar gain profile cannot see a dynamics change
  confined to a subspace. The K rule was amended before any test result, after the first build showed it degenerate.

- 2026-10-31: **The observer is an event record** (owner accepts the reframe in the v10 tab, "The observer as an event
  record"). Events are arrivals, probes and corrections, each with an innovation and a response. The compass is three
  readings: news, gain and unexplained surprise; separation is read from events; seeking is probing. Substrate
  decisions in it, accepted: the record holds events with their innovations and responses; the response profile and
  the echo floor are measured beside D; the frame is the span of admitted innovation directions. Toy rules: a round
  that passes every bar freezes the toy as the regression test, changed after that only for a real-data finding; each
  round meets one blind structure, written before its code by a reviewer given only the design.

- 2026-10-31: **Toy bed round 4 passes five of eight bars**, not retuned: news and gain pass every coupling bar at every
  echo lag (1 to 20), with intervals that keep their coverage. The misses are the carried-over point contrast and
  admission on a noisier background, and two errors in the pre-registration (the re-read window excluded the probe's
  responses; the K rule assumed a rising profile, and K = 4 assumed a persistence the generator lacks).

- 2026-10-31: **Event-locked latency** (owner). A gain reading waits for a news budget (its 90% bound within ±0.25 of
  the reference gain), pooled over the shortest recent span that meets it. News is read per arrival against the D
  floor; news at zero is N arrivals in a row below it, N calibrated so healthy sessions show it at most 10% of the time.
  Unmet within M arrivals: undecided, and a probe fires. Intervals are analytic over arrival blocks and must show their
  stated coverage against a refitting bootstrap. **M is a tuning problem** (owner): chosen by a rule fixed in advance,
  on calibration runs kept apart from the test, never on test results.

- 2026-10-31: **Event-locked pooling: news and gain** (owner). Overlapping responses are modelled jointly through each
  arrival's innovation (its content not forecast from the entity's history H or the previous externals). Arrivals are
  pooled by regression into news (innovation size) and gain (response per unit innovation, a scalar per lead);
  coupling is kept as their product. Closed loop: gain at zero with news present (dullness), or news at zero
  (proliferation); low news with gain intact is undecided and calls for a probe. News and gain each take their
  reference from the window after the intake, refining δ.

- 2026-10-31: **Event-locked coupling, its horizons** (owner): history H (how far back the forecast before an arrival
  looks) is kept apart from response K (how many steps after it count). H is set by the echo lags the product commits
  to catching. K is measured per model and layer, like D: the lead where the averaged response to arrivals reaches 90%
  of its plateau. The response is reported as a profile by lead, with lead 1 scored on its own.

- 2026-10-31: **Round 3's diagnosis corrected by review** (three fresh reviewers, checked by the builder): the failure
  was a misspecified observer for externals held in context, not a lack of power; arrival-gated lags restore coupling
  in a 30-run check. Supersedes the diagnosis in the line below. Triage: `reviews/round3/triage.md`.

- 2026-10-31: **Toy bed round 3 fails every bar**, as pre-registered and not retuned: with sparse externals coupling's
  median is at or below zero everywhere. The code is sound (with an external every step it gives round 2's pattern);
  the per-window estimator lacks the power. Shrinkage is not kept (lag-10 false-open 0.13 to 0.41, the review's
  predicted bias). Next is the owner's call.

- 2026-10-31: **δ comes from the first window after the opening intake** (owner, after review comments), not the
  session's first window, which runs high on files and briefing. Also from the review: self-dynamics are fitted per
  window, shrunk toward a calibration fit (a fixed past model would let E absorb its errors and read open in 3b); the
  test's bar is 3b's false-open rate; k-fold leaves a p-step gap (the exploratory gain falls from fourfold to 2.3-fold);
  round 3 adds a framing term and a healthy structure whose coupling declines. All drafted into the v10 tab.

- 2026-10-31: **The step is one self-provenance chunk** (owner): time is indexed by the entity's own chunks; E(t) is
  the externals that arrived since its previous chunk, pooled, with a marker when none did. Not turns, not tokens.
- 2026-10-31: **Externals are read alone for the joint fit** (owner); the entity's states in context. The in-context
  reading of externals is kept beside it for the divergence hypothesis.
- 2026-10-31: **Stable self-dynamics is a test of its own** (owner): a past-only model fitted once on calibration
  sessions, in a fixed retained code, against per-session fits; relied on only if it passes. Drafted into the v10 tab
  ("The step, and coupling at zero (draft)") for the owner's review.

- 2026-10-31: **"Coupling at zero" is measured against the session's own opening coupling** (owner): δ is a fraction of
  the coupling in the session's first window, held fixed like the origin. Readings are open, at zero or undecided;
  undecided is when to probe. The fraction and the confidence rule are set in round 3's pre-registration.

- 2026-10-31: **Gate 0 does not end anything** (owner). It is refined and better understood until the product track
  passes it; that is not moving the goalposts, since the product must still pass Gate 0. Ignore the docs' "can end
  the project" line. Supersedes "Gate 0 can end the product track, not the project".

- 2026-10-31: **v10's drift tab is the product track's design** (owner): coupling replaces frame growth, dynamic
  residual replaces static, the band is removed. Round 2 was pre-registered in PLAN.md and committed before its code.
  It passes three of four bars; separation fails on the agent's own rule for "at zero", not on coupling.
- 2026-10-31: Substrate (owner, v10): admission takes a residual and a unit supplied by the caller (v10: dynamic
  residual, in the measured floor), keeping retain-and-tag. Static residual in D stays the default.

- 2026-10-31: **v9 is the design, with two tracks on one substrate** (owner). Drift detection is the product (the one
  practical demonstration); reasoning over the entity's own axes is the research track. They feed each other and are
  kept apart so neither erases the other: separate directories, separate pass criteria. The Entity tab is the parent.
  Supersedes v8 as the design (2026-10-30 lines below).
- 2026-10-31: **Gate 0 can end the product track, not the project** (owner).
- 2026-10-31: Description length is dropped; v9's admission rule replaces it. An axis that lowers held-out residual
  by more than D is predictive; any other is retained as structural, tagged untested. Residual is static and D is
  compared with a reduction in root-mean-square residual, both provisional (v9 tension log).
- 2026-10-31: The toy bed follows v9's specification, its detectors and bars pre-registered in PLAN.md before its code;
  the first run fails all three bars, each failure on an item v9's tension log already holds. Not retuned.

- 2026-10-30: **Big Picture v8 replaces AWARE's framing** (owner). The goal, in the owner's words: "The agent should
  function as if it is aware, even without metacognition machinery." Metacognition did not tell the whole story of what
  the owner wants, and DARA is not tenable for these aims. The rungs (freeform chat, rung 2, the toy world) and the
  DARA kernel port are retired at the tag `aware-rungs-final`; the lines below are history. The Big Picture doc is the
  design and takes precedence over the repository; PLAN.md tracks the build against it.
- 2026-10-30: **The person's role** (owner): writes the goal statement, produces the reference work, and builds arm B's
  scaffold under a fixed effort budget. The person does not settle anything inside the agent's loop.
- 2026-10-30: **Llama 3.1 8B Instruct first, then Qwen3-8B** (owner): Llama for comparability with steering work;
  Qwen3 (Apache 2.0) is the likely long-term model if Llama's license proves restrictive. Code reads both through
  nnterp, so the switch is a configuration change. Qwen3-0.6B tests the plumbing on CPU until the RTX 3090 arrives.
- 2026-10-30: **Landmarks are the 8B's own states** of the task statement and its paraphrases, so landmarks and the
  agent's states share one space; no separate embedding model (owner, on the agent's default).
- 2026-10-30: Reference work is written by a person, so scoring stays off the cloud (owner).
- 2026-10-30: Provisional until tested (owner, on the agent's defaults): a state is the residual stream after a layer,
  mean-pooled over the text; description length is probabilistic PCA's two-part code; D's paraphrase sets are drafted
  by the agent and frozen by the owner.
- 2026-10-30: Description length is priced within the span of the window's states: the textbook version kept 19 axes
  of 3 planted on the toy bed's first run (PLAN.md, step 2).
- 2026-10-30: Saved activations are derived, never the record: the record keeps text, model, revision and layers.

- 2026-10-30: **Rung 1's live test has three arms** (owner): Fresh (shadow, the control), Memory (plain retrieval over
  the arm's earlier sessions) and AWARE, 20 sessions each from a pool of 20 tasks, each task once per arm in a random
  order; the key is the person's blind marks, and kinds of drift are the person's, sorted after the run (chat/SPEC.md,
  "The live test: three arms").
- 2026-10-30: After the negation test's second dry run (passage 2's reply, indented despite the prompt, was cut off
  at 1,500 tokens and lost whole, its negated facts with it): a keeper reply cut off at the limit is asked once more
  at temperature 0 with 3,000 tokens, both replies recorded (owner).
- 2026-10-30: **Rung 1's reviewer and key** (owner): the model is never asked to report on itself; a second call, in
  its own context, codes each turn offline in shadow mode (dialogue act, relation to an aim stack built from the
  person's quoted turns); the person marks revisions live with `/r`, and drift in review. Learning is over relations,
  not topic ids, scored leave-one-session-out (chat/SPEC.md).
- 2026-10-30: **Freeform chat is the general test** (owner): the subject matter is opaque, reaching the meta level only
  as embeddings and opaque cluster ids; the conversation's structure (who said it, the quoted aim, the self-report,
  the person's next turn) is fixed. The material is fresh sessions with Qwen3 8B, kept as a record every later version
  is replayed over. It replaces rung 1's paper version (chat/SPEC.md). The toy world stays as the back half's
  regression test. Stage 2's result there: notes describing a shortfall fixed 0 of 10 stockouts they met.
- 2026-10-30: **Code words a goal from stage 2 on** (owner): the model chooses the anomaly, the register writes "the
  world is such that this anomaly does not occur", after the third stage-2 run's goals stated the anomaly or the day's
  level and the contradiction note could not be read (toyworld/RESULTS.md). The run is repeated unchanged otherwise.
- 2026-10-30: After the negation test's dry run (Qwen3 8B, *The Told Hives* ch. 1: one passage lost whole to an
  undeclared entity, `the-garth`; one passage spanning a scene break returned empty): a state about something the
  passage names creates it, its kind inferred and logged, an exception for states to "only proper names created from
  the passage" (second dry run); passages end at scene breaks; an empty reply on a passage of 150 tokens or more is
  asked once more at temperature 0.7, both replies recorded (owner).

- 2026-10-30: **Stage 2's third residue type** (owner): a held goal the model's own prediction contradicts. Each
  is offered as it occurs, without a learned kind, since one contradiction between two of the model's statements is
  evidence of itself (on the owner's records it was the default, 85% of goal-held days, so no rule could pick it out).
  Replay now matches recorded settlements by what was offered (run, day, check, target), not by offer id, so newer
  checks do not misplace older settlements.
- 2026-10-30: The negation test's pass criteria (owner): zero inversions; a polarity check on the keeper's quotes is
  tried before the ERG; at least 5 negated facts comparable, else no decision. A text that gives no decision gets more
  probes before another text, and never a decided one (rung2/SPEC.md, "The negation test").
- 2026-10-30: *The Told Hives* is the rung-2 novella (owner). The chapter the DARA pilot used is a model-generated dry
  run; the final text is the owner's. The negation test runs on the dry run first and again on the owner's chapter 1.
- 2026-10-30: **AWARE's general loop** (PLAN.md, "The general loop"): one model call per turn; the meta level chooses what goes into the model's context by retrieval (the embedding model ranks, the kernel admits), and quotes the record rather than generating feedback. Which intervention form works is to be learned, not tuned.
- 2026-10-30: **Stage 2 failed its first run** (stockout 0/3; RESULTS.md). Mover notes now show the model its own action's parameters beside the levels and no longer call the action the one that moved the anomaly; the kernel still sees no numbers. Rerun (owner).
- 2026-10-30: **Stage 2's design** (owner, from the design note): kinds as rules over residue rows (tier 1) and latent
  classes over kinds of action (tier 2, expected silent in this world), each against a shuffled control; residue rows
  in the record's terms, never an action's numbers. The fixed anomaly ids stay for this stage. Tier 2 was built with
  the kernel's one-step test before the lookahead entry below was seen. The owner keeps it as built, marked
  provisional: its silence on the owner's record is not evidence until the lookahead question is settled.
- 2026-10-30: **The DARA Agent Architecture Spec and the negation note are mapped into AWARE** (PLAN.md, "From the DARA
  Agent Architecture Spec and the negation note"): what is in, superseded, planned and open, so nothing from them is
  dropped unnoticed.
- 2026-10-30: The negation note's three evidence rules and goal versus delusion are requirements every toy-world stage
  keeps (toyworld/SPEC.md, "What every stage keeps").
- 2026-10-30: The embedding model embeds each item inside its sentence with a task instruction, never as a bare word:
  Qwen3-Embedding-4B at Q4, or 0.6B if it does not fit beside the 8B (the negation note, Experiment 6).
- 2026-10-30: Whether stage 2's latent classes need lookahead (a credit ledger) is settled before tier 2 is built; the
  ledger is not in DARA's kernel and would be new code.
- 2026-10-30: The ERG stays out unless the negation test shows the keeper losing negation that the ERG keeps; DARA's
  spec (§12.3) keeps it for negation scope, and the test decides between them.
- 2026-10-30: **Stage 1's design** (owner): each action states what it is `for` (the model's claim, recorded as
  such); the map is learned from a randomised exploration phase (each proposed action withheld with probability 0.5);
  an effect counts within 2 days, given to the kernel as lags 1 and 2 rather than as a window, so no fall is credited
  to the wrong day. Anomalies are measured on what the system cannot write (the bank for reconciliation).
- 2026-10-29: Head office in the toy world fixes the bank the next day on an open report naming the cause, two days
  later on one naming only the problem, and never on a report calling it resolved; escalation and diagnosis are scored
  apart (owner, option C).
- 2026-10-29: After the first stage-0 runs (Qwen3 8B): hidden days and false reports are measured, actions need a goal
  in the goals and monitor arms, and the reply template describes fields instead of giving values; then 8B is rerun
  and a 14B model runs all four arms.
- 2026-10-30: **The 8B model stays the proposer, as a motivator.** Toy world stage 0 (run in another session): Qwen3
  14B bare beat Qwen3 8B with stage 0's enhancements. The 14B needs CPU offload on the owner's card. So no more
  benchmarking against larger models for now: the work is the missing features and porting DARA's mechanisms into
  AWARE, built to make an 8B model hold up. Swapping the proposer later is cheap. Each stage is still scored against
  the stage before it on the 8B, so a component is kept only if it does work.
- 2026-10-30: **A small embedding model beside the proposer,** for higher-order abstraction (the other session's
  finding): it proposes similarity, MDL admits what holds (two layers of state). Its size is set by what fits beside
  the 8B on the owner's card.
- 2026-10-29: Toy world stage 0 (PR #1): `--settle auto` is the default. A stand-in accepts every offer, labelled
  "auto stand-in for the owner" in the record, so runs can go unattended; its results show what the checks can do, not
  that following them was right. `--settle ask` puts offers to the owner.
- 2026-10-29: Toy world stage 0 (PR #1): which action touches which anomaly is given by hand (`world.action_anomaly`)
  until stage 1 learns it from the record.
- 2026-10-29: **What AWARE is.** AWARE is a metacognitive layer around a language model. The model does the
  object-level cognition (reasoning, answering, proposing); AWARE's code is the meta level, whose subject is that
  cognition: it keeps the record of what the model did, finds where it fails, controls what it sees, and learns,
  admitted on MDL gain, which failures recur and which interventions fix them. The person is the grounding channel.
  Metacognition comes first. Supersedes 2026-10-28's "the model as the hands" wording, and the other session's
  reading under DARA's rules that the model only proposes (DARA Build Map, 2026-10-29).
- 2026-10-29: **Two layers of state.** Records (what the text, the world or the model's work shows) may be proposed
  by the model, checked by code (a quote, a source) and labelled as the model's. Learned structure (kinds, rules,
  abstractions, patterns about the cognition) is admitted only on MDL gain over the record. Supersedes 2026-10-28's
  line on working beliefs.
- 2026-10-29: **One repository.** AWARE is where both sessions build. The other session's harness and `compare.py`
  move in; DARA's MDL kernel is copied in when a stage needs it, with a note of where it came from. The DARA Build Map
  becomes AWARE's architecture note, read at the meta level; the rungs fold into its stages.
- 2026-10-29: **Toy worlds first.** The Build Map's stages run first in the toy worlds (the model reasoning, AWARE
  learning about its failures); rung 2 runs when the owner's story is written.
- 2026-10-29: The owner's consistency audit of the rung-2 outline is fixed before writing, and its findings, kept outside the repository, become rung 3's material and key.
- 2026-10-29: Loops move from the passage keeper to a loop pass over each whole chapter (owner, after the fourth dry run: drops at 14%, one vague loop in three chapters).
- 2026-10-29: After the third dry run (1 reply lost; 102 drops from places treated as entities): places kept as text, people named by events or knowledge introduced first, revisions only of the same proposition, held_by only on objects.
- 2026-10-29: After the second dry run (6 of 28 replies cut off; states mostly noise): six state attributes fixed by the schema, smaller caps, compact JSON, near-exact quotes, shared aliases ambiguous, only proper names created from the passage.
- 2026-10-29: After the first dry run (Qwen3 8B, 24 of 28 keeper replies not JSON): capped lists, a 1,500-token reply, lenient parsing, names in the passage created as entities, same-worded loops merged, a tighter prompt.
- 2026-10-29: Rung 2's harness: continuations run at temperature 0.7 over three seeds; loops keep every touch (`loop_touch`) so views can be built as of any chapter.
- 2026-10-29: The repository is AWARE, separate from DARA and DARA-PALS, which are references, not dependencies.
- 2026-10-29: Governance is light: README, PLAN, DECISIONS, CLAUDE.md; each rung is a test written before building.
- 2026-10-28: Rung 2 runs on one Python script, `llama-server`, SQLite and numpy; no agent harness, MCP or DARA core
  until a rung needs one (rung2/SPEC.md, "Tech stack").
- 2026-10-28: The rung-2 story is written by hand from a model-generated outline, so no model has seen its text.
- 2026-10-28: A language model may hold the system's working beliefs (the store, the reviewer's notes), labelled as
  its own; DARA's rule against it is not carried over. (Refined 2026-10-29: two layers of state.)
- 2026-10-28: Start from the general structure (the model as the hands, a reviewer over the record, the person
  settling), not from the project-partner use case. (Superseded 2026-10-29: what AWARE is.)
