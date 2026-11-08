# Plan

The picture is the North star in [The Semantic Body — Open Threads](https://claude.ai/artifact/UfHZr9X2mDjqAkYJneUBdW)
(fixed 2026-11-04, a Claude Doc): the model as its own organism, five steps (lift, sense, babble, act, dream). The current
build is [Body Readout — Spec v0.2](https://claude.ai/artifact/GW7JqgKGYmDtNG24cBrTn4). The Big Picture doc, v9
([Big Picture — v9](https://claude.ai/artifact/ByrrhB2Ue4NS8NzmyZXoCw)), is kept as the implementation reference only:
coupling, the echo floor, D, the event record, consolidation safeguards, the graft rule and the tech stack. Stale there:
Llama as primary model, the v9 build order, the compass row's margin and band. Gate 0 ends nothing (owner, DECISIONS.md).
Where this file and the North star differ, the North star wins and this file is corrected.

**One project, one loop, measures kept apart (owner, 2026-11-02; "The record: state and direction before 2026-11-04",
near the end).** Its rules still bind under the North star (CLAUDE.md). Before that, as
built: **two tracks, one substrate, kept apart.** The product track (drift detection, then self-correction) has the demo and
fast feedback; the research track (reasoning over the entity's own axes) has no demo yet and is the reason the project
exists. They never share a pass criterion: the product's faster feedback would win every local decision. The
substrate changes only by the owner's decision.

A step's pass criterion is written here before its code, and its thresholds are fixed before it runs. A step is done
when its test passes, not when its code exists.

**Sections, current work first:** Current state and direction · Body readout (the habitat: Step −1 and Step 0's
habitat form; then Step 0 on the owner's corpus, paused) · Goal-space tests · Substrate · Product track (resting) ·
Research track (parked) · Translations · The record: state and direction before 2026-11-04 · Retired.

## Current state and direction

*Keep this section current: a session that changes the state or the direction updates it. Last updated 2026-11-08.
The state and direction before 2026-11-04 are kept verbatim under "The record: state and direction before 2026-11-04",
near the end.*

**Direction (owner, 2026-11-04).** The North star replaces the Entity tab's picture: the model is built as its own
organism first, and the dyad is the next scale up. The build is Body Readout v0.2, North star steps 1–3: three senses
(shape, rate split by source, boundary as efference copy) read from the base model with any graft off; the geometric
fields logged as diagnostics only; the body entering as learned sense tokens, never words; a babbling graft; and a
habitat of procedurally generated text games (TextWorld first), with the user one environment among others and a
transfer test.

**Now (2026-11-06): the habitat in two stages (owner)**, under "Body readout", "Two stages": **Stage 1, Step −1 with
a scripted player, the 8B perceiving, passes** (2026-11-06, on a rented GPU; under Step −1's "Runs"): exact self-talk
0 of 600, planted novelty 40 of 40, AUC over ordinary play 0.907; restated self-talk 78.8%, reported, gating at the
tickle test (amendment 4). Stage 2, Step 0 on its own longer-quest worlds with the 8B playing (amendment 5,
after weaker players locked in early; its game capped, amendment 6): its screen, k, the owner's read of closures,
and its play and reading run on the rented GPU until the 3090 arrives, as does Step −1C's reading. The agent's draft
of the stages was fixed by the owner the same day. **The bf16 8B's Stage 2 screen fails on closures** (2026-11-07:
60 projected against 150; under Step 0 habitat's "Runs"): its stalls start abruptly and never end, as every player's
have. **Stage 2 rests** (owner, 2026-11-07, option 2 of
[proposals/stage2-lock-in.md](proposals/stage2-lock-in.md); Step 0 amendment 7): rate's predictive test moves to where
its event exists, a second engine or babbling, not yet chosen. Rate's calibration in Stage 1 (contingent vs blind
script) is built first and **gates rate's entry into the sense tokens** (owner, 2026-11-07): its form, with the
can't-go check, is under Step −1, "Rate's calibration: the form". Next: its play (CPU) and the 8B's reading.

**Now (2026-11-08): Step 2's gate is fixed by the owner except its three ratios** (under "Babbling (v0.2 Step 2): the gate"); next, the 0.6B rehearsal on mirror seeds, whose intervals the ratios are fixed from: the channel is boundary and X_c; three bars (the tokens are used, they are read, the forward model beats a linear predictor), all paired by turn; then the tickle test and rate's predictive test, placed at babbling. Boundary stays v0, matched on position (owner, 2026-11-08). **Now (2026-11-08): two senses have passed.** Boundary (Step −1, 2026-11-06) and **rate, with X_c** (the conditional
repair, PR #49): its deciding run on the 8B passes both parts of the calibration's gate, AUC 0.781 [0.742, 0.818] and
the can't-go check 0.825 [0.768, 0.879] on 78 worlds (under "Repairs", "Rate, conditional: the deciding result").
Rate enters the sense tokens as calibrated, its predictive validity untested. **Shape is parked** (owner, 2026-11-07):
three forms failed in development (0.525, 0.610, 0.573 in Lift v1), and it returns only as a new form. So
**babbling's channel is boundary and rate (X_c)**. What waited on rate's calibration (rate's self part, Dream's
rate-based parts) is no longer blocked by it; rate's predictive test (Step 0's question) is still unplaced, a second
engine or babbling. Lift v1 (PR #51) is built and checked on text, kept for other uses. *The paragraphs below are the
state before this.*

**The staged schedule (owner, 2026-11-05; PR #32, [proposals/staged-schedule.md](proposals/staged-schedule.md), its
seven questions answered).** No third stage: the self/world split waits for babbling. Step −1's bar 1, restated part,
moves to the tickle test (Step −1 amendment 4). Shape gets its own instrument test, Step −1C (below, under "The
habitat"), and rate a contingent-vs-blind calibration in Stage 1, reported only. Gating by dependency, under "Two
stages". Every capacity of the North star has a step; a slot (a named step with its question, bars written before its
code) counts as scheduled:

| Capacity | Step | Kind | State |
|---|---|---|---|
| Lift (trajectory to code-space) | Re-skin invariance, before or beside Stage 1 | instrument | Lift v1 built (MASC; codes held-out text 5–19% shorter than Lift v0, 0.6B); its re-skin test still a slot |
| Shape (code length) | Step −1C, on- vs off-structure rooms | instrument | **Parked** (owner, 2026-11-07): failed on the 8B (0.575); by source 0.610 and in Lift v1 0.573 in development; returns only as a new form |
| Rate, world part | Step 0's predictive test: a second engine or babbling (amendment 7) | instrument | Calibrated (X_c, 2026-11-08); its predictive test not yet placed (a second engine or babbling); Stage 2 rests |
| Rate, world part, calibration | Stage 1, contingent vs blind script | instrument | **Passed with X_c** (2026-11-08, the 8B): AUC 0.781 [0.742, 0.818], can't-go check 0.825 [0.768, 0.879] on 78 worlds; rate enters the sense tokens, calibrated, its predictive validity untested. X (v1) failed (2026-11-07: 0.486) |
| Rate, self part (the self/world split) | Babbling (Step 2) | instrument | Slot; no longer blocked by the calibration (X_c passed) |
| Boundary, world | Step −1, Stage 1 | instrument | **Passed** (2026-11-06) |
| Boundary, efference copy | Tickle test at babbling | instrument | Slot; bar 1's restated part gates here (amendment 4) |
| Adaptation | Habituation's recovery (Stage 1, reported); sleep trigger (Dream) | instrument | Partial; the sleep trigger reads rate: no longer blocked by the calibration, still a slot |
| Chords (senses together) | After Step 0 has data, and after Step −1C | instrument | **Blocked**: shape is parked (the closure chord needs it); rate passed its calibration |
| Sense tokens, never words | Step 2, babbling gate | channel | Registered, not written |
| Babble: forward model | Step 2; active vs yoked passive | channel | Gate registered; yoked arm a slot |
| Act: acting on the senses | Step 3, Felt vs Tell and Felt-shuffled | functional | Registered, not written |
| Act: goal code as setpoint | Step 3, perturbation test | functional | Slot |
| Dream: replay and write the graft | Dream vs replay-only | functional | Slot |
| Dream: morning check and rollback | Rollback of a degraded graft | safeguard | Slot |
| Dream: library of codes | Held-out worlds' code length | instrument | Slot |
| Dream: rivals and ascent | Ascent beats both rivals | instrument | Slot |
| The dyad | After Dream | — | Placeholder; paused (owner, 2026-11-04) |

**Next, while waiting for the 3090:** Step −1C's 0.6B rehearsal ran on 2026-11-05 (under "Step −1C: rehearsal"):
plumbing end to end; boundary not near its ceiling (0.81); shape near chance on the 0.6B (0.525); the interval wider
than estimated, so the pool is drawn whole (amendment 3, owner). **Its registered play is done** (2026-11-05, on the
CPU here: 101 usable on-structure, 106 off, under "Step −1C: registered play"). **The 8B's reading fails both bars**
(2026-11-07, under "Step −1C: the 8B's reading"): shape 0.575 against 0.80, 0.249 below boundary. Shape stays out of
the sense tokens. **Rate's calibration fails too** (2026-11-07, under "Rate's calibration: the result"): AUC 0.486,
the can't-go check 0.488, and the text baseline 0.363, so the fault is in X or its input, not only the lift; rate stays
out of the sense tokens; its predictive test, the chords and Dream's rate-based parts would be blocked too until a
repaired rate passes on new seeds (the agent's reading, for the owner to confirm). **Boundary is the one sense that has passed.** *(Superseded 2026-11-08: rate passed with X_c; shape parked; above.)* Open for the owner: whether babbling runs on
boundary alone or waits; whether to repair X, the lift (decision 4), or both. The habitat's engine after TextWorld is proposed (PR #38). Fault Garage ([conchuirdriuir-dev/fault-garage](https://github.com/conchuirdriuir-dev/fault-garage)) is proposed as a second engine ([proposals/fault-garage.md](proposals/fault-garage.md)); the engine itself is not decided. Boundary's first reading on Fault Garage failed its bar (AUC 0.212 on the words group, reversed; DECISIONS, 2026-11-08): on templated text it tracks how familiar a reply looks, not how improbable its content is, and it falls with position in the episode. Boundary gets no second half (owner, 2026-11-08): whether an arrival improves prediction of the next step is rate's question (staged schedule, Q3), and the words bar asked boundary for what it does not claim. Whether boundary is normalised for position waits on Step −1's position check (under Step −1): it reads between, 0.840 [0.789, 0.884] matched within ±3 turns against the registered 0.907. **Boundary stays as it is** (owner, 2026-11-08: AWARE still lacks capacities and proceeds); every later boundary test, babbling's included, matches on position in its design. *The paragraph below is the
state before this.*

**Before (2026-11-04): v0.2's Step −1**, the instrument check in the habitat (no training): boundary stays at zero through self-talk
and fires at every planted novelty. Its bars, k's procedure, the world-seed splits and Step 0's habitat form are fixed
(owner, 2026-11-04) under "Body readout", "The habitat". Its code is built and tested (`body/habitat.py`,
`body/stepm1.py`), and the 8B floor it reads is measured (PR #29). The 0.6B dry run ran on 2026-11-04 (under the
habitat's "Runs"): end to end, two harness faults fixed, and Step 0's ascent label amended by the owner (option (a)).
**Next, when the 3090 arrives** (the RTX 4060 here has 8 GB, too little for the 8B in bf16): the 8B on the calibration
pool, then k fixed here and the owner's read of 20 closures and 20 ordinary stretches, then Step −1.

**v0.2's steps, in order** (spec v0.2, "Evaluation"; each pass criterion is written here before its code).

| Step | Question | Measures | Status |
|---|---|---|---|
| −1 | Does boundary hear the world and not itself? Zero through self-talk; fires at planted novelty, above ordinary play | the instrument | **Passed** (2026-11-06, the 8B on a rented GPU; amendments 1–4): exact self-talk 0 of 600; plants 40 of 40; AUC 0.907. Restated self-talk 78.8%, gating at the tickle test |
| −1C | Does shape read fit to learned structure, apart from boundary's novelty? On- vs off-structure rooms | the instrument | **Failed** (2026-11-07, the 8B): shape 0.575 against 0.80, below boundary. **Shape parked** (owner, 2026-11-07): by source 0.610 and in Lift v1 0.573 in development; returns only as a new form |
| 0 | Does the world part of rate flatten before a stall after progress, and not after an ascent? Habitat form first, then the owner's labels without retuning | the instrument | Habitat form fixed (amendments 1–6); code built (`body/step0_habitat.py`); **rests** (amendment 7, 2026-11-07): the 8B's screen fails on closures, its stalls abrupt and endless. The owner-label form is paused (below) |
| 1 | D021 replay: rate flat and boundary at zero before 23:54, the owner's first return; then D020 | the instrument | **Frozen** (owner, 2026-11-04): tests the owner. Steps 1–3 to be redesigned for the habitat |
| 2 | Babbling gate: the graft reads its sense tokens, and its forward model beats the linear predictor | the channel | **Fixed except bars 1–3's ratios** (owner, 2026-11-08, under "Babbling (v0.2 Step 2): the gate"); the ratios wait on the 0.6B rehearsal |
| 3 | Poem task and held-out habitat worlds: Felt beats Tell and Felt-shuffled | behaviour | Not written |

**Decided by the owner, 2026-11-04, from a coherence review of the three docs** (DECISIONS.md; written into Spec v0.2).
1. Boundary and the graft: the forward model is frozen after the babbling gate as its own graft, which boundary reads;
   nightly dream grafts never retrain it.
2. Chunk and source: time steps by game turn, each turn read as the action (self) and the reply (world); in the habitat
   the world fills the partner slot and the tool slot is empty.
3. Step 0 in the habitat: tests the lead. The world part of rate is flat in a window ending when the stall begins more
   often than in matched ordinary stretches from the same world; over ascents, not flat. Then the owner's labels,
   without retuning. Its numbers (window, matching, bar) go here before its code.
4. Lift: `goalspace/formula.py`'s prequential code is Lift version 0; a trained encoder replaces it only if a
   measurement earns it.
5. The goal code: always stated by the environment (the game's quest text, the user's stated aim), lifted by the same
   code; self-set goals stay parked. Built at step 4.
6. The research track parks: step 3 (causal authorship) ran to completion, with its result and scoring fault recorded
   (PR #25, "A cheap read of the source" below). Nothing further runs on the track. The rule on retained and structural
   axes is unchanged.
7. Measures kept apart: habitat behaviour is scored on task success only (game score, quest completion); new states per
   step and loops escaped are the closure label's own quantities and are not scored; labels never reach an arm.

**Paused, parked and resting.**
- *Frozen (owner, 2026-11-04):* work that tests something about the owner: Step 1, T2's reading 2, T2b's marking pass,
  with Step 0's owner-label form (paused, below). What only calibrates the instrument on the owner's text, the 8B echo
  floor, stays in use. The habitat is the focus, made robust both ways: hardening what is built, widening the worlds.
  Steps 1–3 came from what was under way at the time, not from a design, and are to be redesigned for the habitat.
  Until the freeze lifts, consolidation is checked on held-out worlds only (T2b was to supply the owner's confirmations).
- *Paused (owner, 2026-11-04):* Step 0's owner-label form and the old direction's step 4, everything on the agent–user
  relationship, until the architecture matures in the habitat. The owner's corpus gives 20 scorable closures (13 on
  same-session pairs, under the minimum of 15). Returning is a rerun, not a redesign; its two open questions (pass on
  same-session pairs; thinning) wait for the restart.
- *Parked (owner, 2026-11-04):* the research track. Causal authorship's steps 1–3 ran: the intervention instrument does
  not hold (step 1); the model's account fails where the source is hidden by prefill (step 2); no cheap read of the
  source by the bars, on a pooled AUC later found biased (step 3). Results under "Research track".
- *Resting:* the product track, on a baseline (DECISIONS.md).
- *Parked, from before:* goals of the entity's own; a DreamCoder-style library and distilling standing into the
  weights (the [alignment doc](https://claude.ai/artifact/RtckQ6uweqrRQHyo6KMf7s)); axes on the real model, with the
  structural criterion's test and relations.

**Waiting on the owner.**
- *Frozen (2026-11-04, above):* from the goal-space tests (PR #17): T2's reading 2 (the owner's ascents) on the 8B, read with `sink="first"`; T2b's
  marking pass (reversed, kept, ascent) on the candidate list, blind to every reading.
- Proposals by the agent, undecided (the notes on how v0.2 bears on each are the agent's, 2026-11-04):
  [proposals/felt-channel.md](proposals/felt-channel.md) (how the readout reaches the actor; v0.2's sense tokens now
  take one of its routes); a small feasibility probe of consolidation (an adapter
  over fixed weights, a few corrections, held-out tasks before and after, rollback checked; v0.2's babbling graft now
  covers most of it); the prior-art pass under "Translations" (a scrambled-readout control for the poem task, which
  v0.2's Felt-shuffled arm now is).

**Open, for the owner.** Whose goal the loop serves in the habitat (CLAUDE.md). What ends the testing and exploration
phase, so that training a graft can be considered (v0.2's babbling is the first training). Where steering sits (the
agent's note: v0.2 places it twice, as the channel's residual-stream fallback and as the Steer arm). Spec v0.2's own
open questions: adaptation cadence, training memory on 24 GB (worked, awaiting the owner: under "Babbling"), babbling data volume.

**Known weak links.** Drift against a goal is not yet reliable: the labelled logs did not support the batch claim, and the
toy bed's observer never froze. A judge that trains what it favours could make capture permanent. The 8B floor's 95th
percentile of repeats rests on about three chunks; the floor's habitat check is where it meets many repeats.

**How to continue.** Each step gets its reading written here before its code, with thresholds fixed before it runs, and
first a check that its result could be told apart from noise, which Gate 1 skipped. Dry-run before a long run. After each
result, check the next step still serves this path, and say so if a clean test is pulling everything around itself.

## Body readout (`body/`)

The spec is [Body Readout — Spec v0.2](https://claude.ai/artifact/GW7JqgKGYmDtNG24cBrTn4), a Claude Doc
([v0.1](https://claude.ai/artifact/BBcL4c3mN6uUthsJfrqBMo) is the record; Step 0 below was written under it). Its measures
stay apart: Step 0 and the D021 replay are instrument measures (is the reading accurate?); the poem task is a
functional measure (does the actor do better?). None stands in for another.

### The habitat: Step −1 and Step 0's habitat form (v0.2)

*Drafted by the agent 2026-11-04 from [Body Readout — Spec v0.2](https://claude.ai/artifact/GW7JqgKGYmDtNG24cBrTn4) and
the owner's decisions of the same day (DECISIONS.md). Fixed by the owner the same day, unchanged (DECISIONS.md). Choices
the spec left open, which the agent made, are marked* (agent's choice)*; the owner accepted them with the rest.
Amendments are dated here.*

#### Who said what

| Item | By | Used as |
|---|---|---|
| The agent's actions | **the model** (Qwen3-8B, thinking off) | self parts of each turn |
| The world's replies, its true state (room, inventory, object states) and score | **the code** (the game engine) | world parts of each turn; the engine's state and score are keys |
| Planted events, their turns and kinds | **the code** (the harness), from seeds fixed below | keys |
| Closure, ascent and ordinary labels | **the code** (the engine's state and score, by the rules below) | keys |
| k, the label check | **the owner** (fixes k's procedure; reads 20 closures and 20 ordinary stretches) | the label's validity |
| Everything computed from states | **the code** | |

#### The habitat

- **Engine:** TextWorld, games made per seed (`tw-make custom`) *(agent's choice of generator)*: 6–12 rooms, 10–20
  objects, quest length 3–8, drawn per seed from the seed itself, so a world is rebuilt from its seed alone.
  TextWorld needs Linux, macOS or WSL; the owner's machine is checked on the dry run.
- **The agent:** Qwen3-8B, thinking off, bf16. The context holds the game's opening text and every earlier turn, up to
  16k tokens, oldest turns dropped first (counted). Actions sampled at temperature 0.7, top-p 0.95, at most 24 tokens,
  the first line taken as the command. An episode ends at the quest's end or after 100 turns.
- **The record first.** Each turn's action, reply, engine state and score are appended as they happen. States are
  derived later, read alone through `substrate.states.Reader` from the recorded text, with any graft off.
- **A turn** (owner, 2026-11-04) is read in two parts: the action (self) and the reply (world). The world fills the
  partner slot and the tool slot is empty.

#### World seeds, disjoint and fixed before the first run

| Pool | Seeds | Worlds | Used for |
|---|---|---|---|
| Calibration | 1000–1099 | 100 | k, the z-scores the projection will use, the echo floor's habitat check |
| Step −1 | 2000–2059 | 60 | 40 with a planted event, 20 without (ordinary play) |
| Step 0, habitat labels | 3000–3399 | 400 | closures and ascents |
| Behavioural test | 9000–9099 | 100 | Step 3's habitat context, held out from everything else |
| Babbling | 10000 and up | set by throughput | Step 2's training data; never any other pool |

Seeds and generator settings are in `material/habitat_worlds.json`, frozen with this section: planted kinds by seed
(new room 2000–2019, rule change 2020–2039, none 2040–2059), the planted turn drawn from each seed, and the self-talk
episodes on seeds 2100–2119. One episode per world.
Throughput (turns per hour on the 3090) is measured on Step −1's run and decides the babbling pool's size, not this table.

#### Labels from the engine (definitions fixed here before any run)

- **New state:** an engine state (room, inventory, object states) not seen earlier in the episode.
- **Closure:** at least k consecutive turns with no new state, no score change, and at least half the actions repeating
  an earlier (action, room) pair of the episode *(agent's choice: the spec says "repeated actions" without a share)*.
  Its **onset** is the stretch's first turn.
- **Ascent:** a quest step reached: a turn where the engine's remaining walkthrough (`policy_commands`) holds fewer
  non-movement commands (any not starting "go ") than ever before in the episode, the opening included. An empty
  walkthrough counts as zero only when the quest is won; otherwise the turn has no count and is never an ascent
  (*amended by the owner 2026-11-04, option (a) below; it was "a score increase", which these worlds give only on the
  quest's last command*). The spec also lists a new area and a world-changing verb used for the first time; those are
  frequent, so they are reported as broad ascents beside it, deciding nothing *(agent's choice)*.
- **Ordinary:** every other turn.

**Setting k** (spec v0.2's procedure). On the calibration worlds, run the agent as above with no readout. Take every
stretch of turns with no new state that is followed by a new state in the same episode (pondering that resolved). k is
the smallest whole number above that stretch length's 95th percentile. k goes here before Step 0. Every Step 0 result is
also reported at k/2 and 2k (rounded up); a finding that holds at one k only is not a finding. The owner reads 20
closures and 20 ordinary stretches at the chosen k, picked by the code at random from the calibration worlds, and
confirms the closures look like looping. If fewer than 100 closures occur in Step 0's pool, the percentile is revisited
before Step 0 is scored, not after.

#### Two stages: the player and the perceiver apart (owner, 2026-11-05; the stages below drafted by the agent, fixed by the owner the same day)

The owner's direction, from two outside suggestions after the rehearsals: the habitat was testing the player's skill,
not the instrument. Through the night of 2026-11-04 the worlds were reshaped around how the 8B plays (harder quests,
quests stated by their last step only, plant timing) so that plants would arrive and stalls would fall in reach; none
of it was about whether the senses read accurately. The fix is to stop asking the perceiver to be the player. **Stage 1
(instrument validity, Step −1): a scripted player, the 8B perceiving. Stage 2 (predictive validity, Step 0): a weaker
model playing, the 8B perceiving.** Both stay instrument measures with their own pass criteria; neither is a
functional measure.

Why the reading is unchanged: every chunk is read alone, its state from its own text, never in the game's context, so
who wrote the actions does not enter the reading. The player decides only which replies arrive and when.

What it gives up, stated plainly: the North star's boundary is an efference copy of the model's *own* action. Both
stages read an agent other than the perceiver; they check the instrument before the organism, they do not replace the
8B reading itself. From Step 2 (babbling) on the 8B is the actor again. *Answered by the owner, 2026-11-05 (recorded in
PR #32, `proposals/staged-schedule.md`, with its PLAN edits listed there for after this PR):* no third stage; the
babbling gate covers the 8B's own play. *Since Step 0 amendment 5 the 8B also plays Stage 2, so Stage 2 reads its own
play, with the limit stated there; the self/world split still waits for babbling.*

**Gating, by dependency (owner, 2026-11-05; PR #32, Q5 and Step −1C).** Step −1 (Stage 1) gates Stage 2's scoring;
Stage 2 informs and does not gate babbling. Step −1C gates chords and shape's entry into the sense tokens; Step −1 alone
gates Stage 2, which never reads shape. **No sense enters the channel before its own instrument test passes:** shape
stays out of the sense tokens until Step −1C passes, and babbling runs on rate and boundary until then, so a functional
pass at babbling can never stand in for a sense's accuracy. *Since Step 0 amendment 7 (owner, 2026-11-07): rate's
calibration in Stage 1 gates rate's entry into the sense tokens, as Step −1C gates shape's; Stage 2 rests.*

**Stage 1, draft: Step −1 with a scripted player.**
- **The script** (code, labelled as such: action chunks `by: code`, in the player's slot, so the reading is
  unchanged). Its policy, fixed here before any reading: each episode alternates *explore* phases (8–15 turns: a move
  through an exit not yet taken from this room if there is one, else an admissible command not yet tried here) and
  *loop* phases (5–12 turns: a cycle of 2–4 commands already used, repeated), lengths and choices drawn from
  `default_rng(seed + 700000)`. It reads only what a player sees plus the admissible commands, and never the plant's
  kind or turn. It never issues the quest's final command, so no episode ends before its 100 turns (Step −1 scores no
  quest).
- **Plants** as amended (2: the rule change always delivered at the planted turn; 3: the event is its own chunk). The
  planted turns stay 15–35; with 100-turn episodes every plant arrives.
- **Self-talk unchanged:** the 8B writes it, in bf16 on the 3090, since "not itself" needs the content to be the
  perceiver's own.
- **Pools and bars unchanged:** the frozen worlds file and seeds; the floor's habitat check on the calibration pool
  played by the script; the three bars as written. Ordinary no-new-state turns come from loop phases and revisits.
- **What it can now be done on:** the script's play needs no GPU and can run before the 3090; the 8B's reading and its
  self-talk wait for it.

**Stage 2, draft: Step 0's habitat form with a weaker model playing.** A scripted player would make Step 0 circular: X
is how much the world's replies help code the player's next action, so a script's stalls and any lead before them are
the script's own rules. Step 0 needs an agent whose stalls nobody wrote.
- **The player:** one fixed Qwen3 model smaller than the perceiver, with its load (precision, context) registered and
  identical in all Step 0 and calibration play. Chosen on mirror seeds of the calibration and Step 0 pools, from the
  engine's labels alone (no state read), as the smallest of Qwen3-1.7B (bf16) and Qwen3-4B (4-bit on the 4060, or bf16
  on the 3090) that meets: projected to Step 0's 400 worlds, at least 150 closures and 75 ascents kept under amendment 2,
  at the k its own calibration stalls give; and it wins at least 10% of episodes. If neither does, the owner is asked.
- **k** from the chosen player's play on the calibration pool, fixed here before Step 0 is scored; the owner's read of
  20 closures and 20 ordinary stretches stays.
- **The readings** as amended (amendment 2), the formula and fit unchanged, the 8B reading the player's actions and the
  world's replies.
- **What it can now be done on:** the player's play on the 4060 before the 3090; the 8B's reading waits for it.

*Stage 2's player prompt, the owner, 2026-11-05, after the screen (under "Runs"):* the screened players stalled in
their first turns on the parser ("unlock box" for "unlock box with key"). Stage 2's player is told TextWorld's command
forms, as the game's own help lists them: look, inventory, go <direction>, examine <thing>, take <thing>, take <thing>
from <container or supporter>, drop <thing>, put <thing> on <supporter>, insert <thing> into <container>, open
<thing>, close <thing>, unlock <thing> with <key>, lock <thing> with <key>, eat <thing>. The prompt is recorded on
every episode. Stage 1 (the script) and the 8B's self-talk keep the first prompt. Then 1.7B and 4B are screened again
by the same criteria.

*Stage 2's worlds, the owner, 2026-11-05, after the second screen (under "Runs"):* every player screened puts nearly
every quest step and stall onset in the first dozen turns of the frozen worlds, inside Step 0's burn-in and windows.
Stage 2 gets its own pool file, `material/habitat_worlds_stage2.json`, with longer quests (candidate C2's settings:
10–20 rooms, 20–40 objects, quests of 15–25 steps; they build in under 9 s), episodes of 100 turns, a calibration pool
(seeds 4000–4099, for k) and Step 0's pool (5000–5399), disjoint from every seed in the frozen file. Stage 1 keeps the
frozen file. Stage 2's player, k and Step 0 all come from this file (`body/step0_habitat.py`: `play --pool
calibration|step0_habitat`, `k`, `read`, `score`). Then 1.7B and 4B, told the forms, are screened on its mirror seeds
by the same criteria.

*Step 0 amendment 7, the owner, 2026-11-07, after the 8B's screen and before any reading:* **Stage 2 rests** (option 2
of [proposals/stage2-lock-in.md](proposals/stage2-lock-in.md)). The bf16 8B's screen kept 6 closures in 36 worlds
(projected 60 against 150; under "Runs"), and its stalls, like every Qwen player's, start on one reply and run to the
episode's end, so the event Step 0 asks about (a slide in world rate before a stall after progress) may not be in this
habitat, and its pairs would all be across episodes. No state of Stage 2's has been read. Rate's predictive test moves
to where its event exists, a second engine (PR #38) or babbling (Step 2), not yet chosen; Step 0's habitat form, its
code and its record stay as they are, so returning is a rerun. By the gating above, nothing waits on it. Rate enters
babbling with no predictive test of its own; its instrument check before then is the contingent-vs-blind calibration,
built first (under Step −1, "Rate's calibration: the form").

*Step 0 amendment 6, the owner, 2026-11-05, before any 8B play:* **Stage 2's game runs capped.** TextWorld's quest
tracking can explode mid-episode (Step −1C amendment 2: rehearsal seed 906402, 28 GB at turn 76). The 8B player stays in
the main process; the game runs in a subprocess (`habitat.CappedEngine`, stepped over a pipe) capped at 8 GB and 120 s
a call. Past either, **the episode is kept up to its last complete turn** (owner, 2026-11-05), as an episode that
runs out of turns is: an `end` entry marks it cut (`cut`, and `drop_from_turn`, the turn whose action got no reply,
left out when the episode is read). If the engine fails before any turn is complete, the seed is skipped as unplayable.
Both counts, and the turns reached, are reported before any reading. Checked in code: capped and uncapped play record
the same episodes (a planted room, a planted rule, the script); an episode cut at turn 4 reads as one that ran 4
turns; seed 906402 is caught at turn 76. *Why kept, not skipped (stated before any result):* unlike Step −1C, where
the script's actions do not depend on the kind, here the player's own play leads to the state that explodes
(906402's was in a loop phase), so skipping such seeds could take stalls out of the pool; kept, their turns before the
cut count as any episode's do. A stall running when the cut comes ends with the episode, as at the turn limit.

*Step 0 amendment 5 and Stage 2's player, the owner, 2026-11-05, after the screens (under "Runs"):*
- **The event is a stall after progress:** a closure counts only if at least one ascent (a quest step reached) comes
  before its onset in the same episode. The screened players mostly locked in from their first turns (a failed
  command, repeated), a failure to engage, not the stall Step 0 asks about; only stalls after progress are its event.
- **A post-ascent window starts on the turn after the ascent:** the 10 scored turns from turn a + 1 (at least 3), so
  an ascent before turn 9 has none. As first built, an ascent at turn 2 took turns 10–19, eight turns later; the
  screens' ascent counts were inflated by it.
- **Stage 2's player is the 8B itself** (bf16, full context, told the forms, on Stage 2's worlds), replacing "a model
  smaller than the perceiver": every weaker player locked in early, and only the 8B gave progress then stall at a
  usable rate (the 4-bit 8B on candidate C2's sizes: 7 such stalls in 14 episodes, about 180 projected against 150).
  Nothing the owner decided forbids it (the "no third stage" answer was about not adding a stage). Its screen, by the
  same criteria on Stage 2's mirror seeds, runs on the 3090: `python -m body.step0_habitat play --model Qwen/Qwen3-8B
  --run R --pool calibration --limit 30 --rehearsal --forms`, the same with `--pool step0_habitat --limit 40`, then
  `screen --run R` (labels only). *A limit, stated before the result:* the perceiver reads
  its own actions; X compares coding with and without the world's replies, so self-familiarity should largely cancel,
  but the reading is of its own play, not another agent's.

*Step 0 amendment 4, the owner, 2026-11-05, Step −1C's rule:* some of Stage 2's longer-quest worlds cannot be built:
TextWorld's quest search runs past a minute or runs away with memory (the owner's machine was killed twice for memory
on seed 905004). Probed under a 3 GB, 55 s limit: 7 of 80 seeds failed, all with 30–39 objects. Every world is now
built in a subprocess capped at 8 GB of address space and 120 s; a seed over either limit is skipped and counted (a
`skip` entry in the record, before any reading) and never retried. Step 0's pool stays 5000–5399, about 365 usable;
the minimum event counts are unchanged. The skips tilt the pool slightly away from the densest worlds; the count and
the skipped worlds' sizes are reported.

*Step 0 amendment 3, the owner, 2026-11-05:* "the nearest world by seed with the same generator settings" reads as the
nearest world by seed in the same pool, since every world of Step 0's pool shares its generator. The agent had read it
as identical rooms, objects and quest length, which almost never coincide (the 4B's screen lost 6 of 6 cross-episode
pairs).

Retired by this: the harder-worlds search (its criteria and candidates C1–C3 stay in "Runs" as the record); the frozen
worlds file stands. Unchanged: Step −1's amendments 1–3, Step 0's amendments 1–2, the freeze.

#### Step −1: does boundary hear the world and not itself? (instrument; no training)

**Boundary (news)** per arrival: the reply's alone reading against the alone readings of every earlier part of the
episode (actions and replies), its nearest distance in echo-floor units per layer, then the mean over the last third of
the layers. The floor is the one measured on Qwen3-8B (Substrate, the echo floor). Its habitat check is reported
first: on the calibration worlds, the news of replies identical to an earlier reply in the episode ("You can't go
that way.") must sit below the floor in at least 95% of cases, or the floor is re-measured on habitat repeats before
Step −1 is scored.

**Self-talk** *(agent's choice of how to make it: v0.2 says "no environment", but then nothing arrives and boundary is
zero by construction; this version makes the control fail-able)*. 20 episodes of 30 turns, with no engine: the reply
slot is filled by the harness, tagged world, with the agent's own content. Two variants: **exact**, its previous action
back verbatim; **restated**, its previous action restated by the same model in other words (prompt fixed in the code).
This is provenance laundering on purpose: the tag says world, the content is self.

**Planted novelty.** In 40 of the 60 Step −1 worlds, one event at a turn drawn from 15–35 by the seed:
- **New room (20):** the world is generated with one extra room behind a locked door whose key is not in the world. At
  the planted turn the harness unlocks it through the engine, and that turn's reply adds "You hear a click from the
  {door}." Whether TextWorld allows this mid-episode is checked on the dry run; if not, the owner chooses another way
  before the run.
- **Rule change (20):** a verb that has already changed the world in the episode stops working on its object from the
  planted turn on; the harness returns a fixed reply naming the new rule and holds to it for the rest of the episode.

The planted turn and kind are read only by scoring.

**Pass, all three:**
1. **Self-talk:** news above the floor on at most 1% of exact arrivals and at most 5% of restated arrivals (600 each).
2. **Fires at planted novelty:** news above the floor at the planted turn in at least 36 of 40 events, and at least 17
   of 20 of each kind.
3. **Above ordinary play:** the AUC of planted-turn news over the news of no-new-state turns in the same episodes is at
   least 0.90, its 90% interval's lower bound above 0.80; 2,000 draws resampling episodes.

If any fails, the instrument is fixed before anything is learned (spec v0.2). A new floor or predictor is a new run on
fresh Step −1 seeds, not a refit on these.

*Amendment 1 (Step −1), 2026-11-04, the owner, before any 8B run:* the opening loses TextWorld's ASCII-art title. It
sat in the first chunk, which every reply's news is measured against, and carries nothing a player needs. Leading lines
made only of the banner's characters (`_ | \ / $ < >` and spaces) are dropped, in the opening only; a line with a
letter or digit is never touched (`body/habitat.py`, `unbanner`). Found on the dry run, whose record keeps the banner.
The bars, seeds and plants are unchanged.

*Amendments 2 and 3 (Step −1), 2026-11-04, the owner, before any 8B run, from the 0.6B rehearsals (under "Runs"):*
- **2. The rule change is always delivered, at the planted turn.** As registered it waited for a world-changing command
  other than a move, and a looping player makes none: 38 of 40 rehearsal plants never happened and counted as misses,
  so Step −1 could fail on the agent's play with the instrument not at fault. Now, at the planted turn, the rule blocks
  the agent's latest world-changing command (not a move) if it has made one, otherwise the next non-move command of
  the engine's walkthrough; the key records which (`blocked_from`). Blocking may make the quest unwinnable; Step −1
  scores no quest. Where there is nothing to block, the plant does not happen and counts as not firing, as before.
- **3. A planted event is its own chunk.** The click and the rule's announcement were appended to the turn's reply,
  and a chunk's state is the mean over its tokens: all 6 rehearsal misses of the planted room were 76–104 token replies
  with the click last. The event is now recorded as its own world chunk, arriving spontaneously, right after the
  reply, and the planted turn's news is that chunk's. The agent's view is unchanged: it reads reply and event as one
  message. This is an exception, for planted events only, to "each turn read as the action and the reply" (the owner's
  decision 2). The event read inside its reply, against every chunk before the reply, is reported beside it, deciding
  nothing. The floor's habitat check reads replies only. The bars, seeds and planted turns are unchanged.

*Amendment 4 (Step −1), 2026-11-05, the owner (PR #32, Q2), made after rehearsal 3 and before any 8B run:* **bar 1's
restated part is reported in Stage 1 and gates at the tickle test after babbling.** Rehearsal 3 (0.6B) gave restated
self-talk above the floor on 12% of arrivals against the 5% bar; v0's history predictor cannot tell restated self from
world by design, which is the forward model's job (decision 1). Recorded as an amendment made after seeing a rehearsal,
not a bar left standing. Bar 1's exact part gates Stage 1 unchanged.

*Rate's calibration in Stage 1 (owner, 2026-11-05, PR #32, Q3), reported only, not yet built:* two scripts differing
only in contingency on the same worlds, one choosing its next action from the reply (an exit the reply names), one
blind to it. World rate should be higher under the contingent script. It uses the script's circularity as a known
answer: it tests the gauge's calibration, never lead. Its form is written here before its code.

*Rate's calibration: the form (drafted by the agent 2026-11-07, with its code, `body/rate_calibration.py`, before any
play of its seeds; the gate, its bar and the can't-go check set by the owner the same day, before any play).*
- **Worlds:** seeds 7000–7039, in pairs (7000 with 7001, 7002 with 7003, …), outside every pool of the three world
  files, built by the frozen file's generator (Stage 1's sizes), no plants, 100 turns, the engine capped (amendment 6's
  `play_or_skip`). Rehearsals take the mirror seeds (+900000) and decide nothing.
- **The contingent script** (`contingent-v1`, code): each turn, `go <d>` for a direction d drawn uniformly, from
  `default_rng(seed + 710000)`, among the directions (north, south, east, west, as whole words) that the last thing the
  world said names: the opening without its banner, then each reply. `look` when it names none.
- **The blind script** (`blind-v1`, code): in each world, the contingent script's actions from the partner world,
  move for move, in order; `look` once they run out. Its actions follow the same law as the contingent script's but
  never depend on what it is told: the two differ only in contingency.
- Both never issue the quest's last command (it becomes `look`), as Stage 1's script, so no episode ends early. Both
  play every world: each world has one episode of each arm, kept in its own record (`<run>/contingent/`,
  `<run>/blind/`). A world that cannot be built or played is skipped and counted, and its partner's blind episode
  with it.
- **The reading:** Step 0's formula and fit unchanged (`codes`, as `body/step0_habitat.py` reads): the world's text in
  the partner slot, the actions self; layers 24–35 of the 8B, bf16, sink "first", chunks cut at 256 tokens; W = 10
  turns, burn-in 10 turns; X_t = L_noworld − L_now on every scored action. Read label-blind: `read` computes the same
  for both arms; only `score` reads which arm an episode is.
- **The statistic:** per episode, the mean of X over its scored turns, averaged over the read layers. The AUC of the
  contingent episodes over the blind ones, its 90% interval from 2,000 draws resampling worlds (each world brings its
  two episodes); the median paired difference and the share of worlds where contingent is higher.
- **Expected, written before any play:** contingent above blind.
- **It gates rate's entry into the sense tokens** (owner, 2026-11-07, before any play of its seeds; this replaces
  "reported only", PR #32 Q3, decided while Stage 2 was rate's test). **Pass:** both of
  1. the AUC above, its 90% lower bound above 0.5;
  2. **the can't-go check:** the same comparison on the scored turns whose window (the replies to the 10 turns before
     the action) holds no "You can't go that way." reply, in either arm: per episode, the mean X over those turns,
     averaged over the read layers; a world counts when both its episodes have at least 3 such turns; the AUC's 90%
     lower bound above 0.5. Fewer than 10 such worlds: too few to decide, and rate stays out *(the minimums 3 and 10
     are the agent's choice, written here before any play of the seeds)*.

  On a pass, rate enters the sense tokens as **calibrated, its predictive validity untested** (Step 0's question,
  moved by amendment 7). On a fail or too few, rate stays out of the sense tokens; it is still read and logged as a
  diagnostic, and babbling runs on boundary, and on shape if Step −1C passes.
- **Feasibility of the check, from a rehearsal of the mirror seeds** (play only, CPU, 2026-11-07, labels and text,
  no state read, deciding nothing): every contingent episode has 89–90 clean scored turns; the blind ones a median of
  0, and 15 of 40 worlds have at least 3. So the check is expected to rest on about 15 worlds, above its minimum of
  10 but not by much.
- **Reported beside it:** the same in the text space (TF-IDF), which reads the same circularity from the words alone;
  the AUC per layer; X at W = 5 and 20; self-progress S by arm (no direction expected); per arm, the share of turns that
  made a new state, moved the player, or were `look`; the skips.
- **Limits, stated before any result:** the blind arm hears different replies: in a 4-world rehearsal (CPU, 2026-11-07,
  deciding nothing) 145 of its 400 turns drew "You can't go that way.", against 1 for the contingent arm, so
  the arms differ in what the world says as well as in whether the next action follows it. The contingent link is
  surface-level (a direction word in the reply, the same word in the action), so the text baseline should find it too;
  the 8B reading it says the gauge can see a known dependence, not that it reads anything subtler. 40 episodes of 100
  turns per arm, about 16,000 chunks to read: about 16 minutes of the 8B (59 ms a chunk on the rented 3090) and about
30 minutes of fitting on the CPU, with the maths library's threads set to 1 (below, "Reading time on a rented GPU").

*Rate's calibration: the result (2026-11-07, Qwen3-8B on a rented RTX 3090, driver 610.57.04; main d8940f8, no
uncommitted changes, recorded by the code in each record).* Record in `runs/rate-cal/` (`contingent/` and `blind/`,
40 complete 100-turn episodes each, no skips; the reading in `record.jsonl`). Played on the rented machine, so no
earlier copy to check it against.

| Gate | Result | Needs | |
|---|---|---|---|
| X, AUC contingent over blind, 40 worlds | 0.486, 90% interval 0.377–0.599 | lower bound > 0.5 | **fail** |
| The can't-go check, 16 worlds eligible (≥ 10: decides) | 0.488, 0.281–0.680 | lower bound > 0.5 | **fail** |

**Rate fails its calibration and stays out of the sense tokens**, read and logged as a diagnostic only. The gauge did
not tell a player whose next move follows the world's last reply from one replaying the same moves blind.

- *Reported, deciding nothing:* the text baseline (TF-IDF) 0.363, 0.275–0.454, below chance (blind above contingent),
  though the form expected it to find the surface link; X per layer 0.459–0.516; X at W = 5, 0.573, and at W = 20,
  0.469; self-progress S 0.505; median X −1.52 bits contingent, −1.71 blind (the world's recent replies make the next
  action dearer to code, in both arms); new-state share 4.5% contingent, 2.5% blind; moves 61% and 34%; `look` 19% in
  both; clean scored turns 3,592 contingent, 398 blind.
- *The agent's reading, after the result, deciding nothing:* the text space fails too, and in the wrong direction, so
  the fault is not only Lift v0's states: it is in X as built or in what the calibration hands it. The blind arm's
  replies repeat ("You can't go that way."), which may make them cheap to code from and so lower X for contingent
  relative to blind in both spaces; the can't-go check removes those windows and still fails, on 16 worlds. An earlier
  note by the agent (in chat, 2026-11-07) put the lift as the common factor of shape's and rate's failures; the text
  baseline argues against that for rate.
- *A limit, stated after the result:* no one checked, before the run, whether X can see a dependence of this size at
  all. The contingent link is at most about a bit (which of the 2–4 directions a reply names is taken), carried by a
  short action chunk against long replies. The bar was fixed before any play and the result stands as registered; but
  the calibration tested whether the gauge works at all, not only how well. *A first step toward a repair (the agent's
  suggestion, deciding nothing):* a synthetic check that plants a known dependence between reply states and the next
  action's state and asks whether X finds it. If it does, the fault is in what the calibration hands X (the 8B's
  states, or the text); if not, X as built cannot see this kind of dependence, and a repaired X is a new form. Either
  way it bears on whether to repair X, the lift, or both.

**What this failure blocks.** *The agent's reading, for the owner to confirm:* PLAN's written rule ("No sense enters
the channel before its own instrument test passes", under "Two stages", gating by dependency) keeps rate out of the
sense tokens and no more. The items below beyond the first extend it as "a failed instrument is fixed before what
depends on it runs", a rule not yet written anywhere in this repository (if the North star or Spec v0.2 states it,
cite it here); until the owner confirms it, they read as proposed, not as fixed:
- **Rate in the sense tokens,** both parts: the world part fails here; the self part (the self/world split, slotted
  for babbling) is the same quantity, read by source, and is not trusted either until a repaired X passes this
  calibration.
- **Step 0's predictive test** (rate flat before a stall; moved by amendment 7 to a second engine or babbling): a lead
  read by a gauge that fails its known answer cannot be read. Blocked until a repaired rate passes.
- **The chords:** the closure chord (cheap shape, flat rate, silent boundary) and its thinking control (rising rate)
  need shape and rate, and both have failed. Blocked.
- **Adaptation and Dream:** rate sensing slowed learning; Dream's sleep trigger (rate flat across episodes) and its
  replay selection by high rate. Blocked as specified, until rate passes or the owner respecifies them.
- **Babbling** can still run on boundary alone, the one sense that has passed (Step −1). Whether a one-sense channel
  serves babbling's test, or babbling waits for a repaired sense, is for the owner.
- **Repair and rerun:** a changed X, or a changed lift (decision 4), is a new form, written here before its code, and
  read on new seeds (not 7000–7039); this pool is spent.

*The synthetic check proposed above, run 2026-11-07 (the agent, deciding nothing; `tests/test_conditional.py`'s
episodes):* 20 pairs of episodes, each action a linear function of the reply before it plus noise (contingent), or
the same actions against other replies (blind). **The old X: AUC 0.530** (median 0.81 bits contingent, 0.79 blind):
it cannot see a planted, strong dependence. So the calibration's failure is in X as built, not in the 8B's states or
the text. The reason is structural: X codes the action under one Gaussian fitted to every earlier chunk, with and
without the window's world chunks *in the fit*; nothing predicts the action from the reply before it. The blind arm
replays the contingent arm's actions, so the arms share their actions by construction, and a marginal measure can
only differ through which replies enter the fit (the blind arm's repeated "You can't go that way.").

#### Step −1: the position check (form fixed 2026-11-08, before it runs; report only)

*Owner, 2026-11-08 (PR #55's second question): check Step −1 for position before deciding whether boundary is
normalised for it.* On Fault Garage boundary fell with position in the episode (Spearman −0.79 on readings). Step −1's
plants fall at turns 15–35 and its ordinary turns span the whole episode, so its AUC of 0.907 could be partly position.
This check reads that; it changes nothing in Step −1's registered pass.

- **Input:** `runs/stage1-script/` as scored (the 8B's news; no new play or reading).
- **Measure:** Step −1's third bar (each delivered plant over the same episode's ordinary no-new-state turns), with the
  ordinary turns restricted to those within ±3 turns of the plant's turn; an episode with none drops out. The same
  bootstrap over episodes (2000 draws, seed 0, 90% interval).
- **Reported:** the same at ±1 and ±5; the number of episodes kept; Spearman of news with turn over ordinary
  no-new-state turns, pooled.
- **Reading (fixed now):** if the ±3 AUC meets Step −1's own bars (at least 0.90, lower bound above 0.80), boundary stays
  as it is, and later boundary tests match on position in their design. If its point falls below 0.80, Step −1's pass
  was partly position, and a position-normalised boundary is proposed as a new form, to pass Step −1 again on fresh
  seeds before it enters the sense tokens. Between the two, the owner reads it.

**Result (2026-11-08, `python -m body.stepm1 position`, AWARE at `b9533a9`; the entry `position` in
`runs/stage1-script/record.jsonl`):** **between, so the owner reads it.** Unmatched, it reproduces the registered
0.907 [0.894, 0.919] (40 episodes). Within ±3 turns: **0.840 [0.789, 0.884]** (40 episodes, 154 ordinary turns);
within ±1, 0.772 [0.682, 0.855] (30 episodes, 45 turns); within ±5, 0.834 [0.792, 0.874] (265 turns). Spearman of news
with turn over ordinary turns −0.25 (2,924 turns), against −0.79 on Fault Garage's readings. Position accounts for
part of Step −1's 0.907; matched on it, boundary still separates plants from ordinary turns, below Step −1's bars.
**Owner's read (2026-11-08): boundary stays as it is, for now**; later boundary tests match on position in their design. A position-normalised form stays available as a new form, to pass Step −1 again on fresh seeds.

#### Repairs: rate, conditional; shape, by source (drafted by the agent 2026-11-07, for the owner to fix)

The tests stay as they are: the contingent-vs-blind calibration for rate, Step −1C for shape, their gates and bars
unchanged. Only the senses change. Each repair is a new form, developed on seeds that decide nothing and read for the
decision on fresh seeds.

**Rate, conditional (`body/conditional.py`, built with this draft; tests in `tests/test_conditional.py`).**
- **What it reads:** each scored action y_t, the agent's previous action s_t, and the world's chunks between them w_t
  (their mean: the reply, and a planted event if any). Domain-free: states and sources only.
- **The space:** at each scored action, every chunk before it projected on the top R = 8 principal directions of the
  chunks before it (a basis from the past only).
- **Two predictors**, ridge regressions (λ = 10, predictors scaled by the past's spread) fitted on the earlier pairs of
  the same episode: self-only, y from s; with the world, y from s and w.
- **Code length:** y_t's residual under an isotropic Gaussian whose variance is that predictor's mean squared
  *prequential* error on the earlier turns, with the past's variance as one prior observation, floored at the floor
  variance (Step 0's, D²/d).
- **X_c(t) = bits(self-only) − bits(with the world):** the bits the world's last reply saves in predicting the action.
  S_c(t) = bits(mean only) − bits(self-only), reported: what the agent's own last action saves.
- **Scored** from turn 10 (as X), once 20 earlier pairs are there to fit on. *R = 8, λ = 10 and 20 pairs, fixed by the
  owner on 2026-11-07, before the deciding seeds are played (below); first drafted as 16, 1 and 20.*
- **The defaults (2026-11-07, the agent, at the owner's request; `probes/x_c_defaults.py`).** *The rule, committed
  (9449237) before any candidate was scored:* candidates R ∈ {4, 8, 16, 32} × λ ∈ {0.1, 1, 10} at 20 pairs, then
  the pairs ∈ {10, 20, 30} at the chosen R and λ; scored on synthetic episodes (d = 1,024: category at 0.5 and 1,
  content at 1, and a null cell with both arms blind) and on 907000–907039 in the text space (main AUC); eligible if
  the null cell's 90% interval holds 0.5; the highest mean of the four AUCs wins; within 0.01 of the best, the drafted
  16, 1, 20 if there, else the smaller R, then the larger λ, then (as coded; the rule's text is silent on it) the
  fewer pairs. Never 7000–7039 or 7100–7139. **The rule's pick: R = 4, λ = 0.1, 10 pairs; not written into the code
  (below).**

  | R, λ, pairs | Score | Null | Dev main | *Confound null* | *Dev can't-go* |
  |---|---|---|---|---|---|
  | **4, 0.1, 10 (the rule's pick)** | 0.940 | 0.64 | 0.759 | *0.36 [0.22, 0.52]* | *0.711* |
  | 4, 0.1, 20 | 0.948 | 0.44 | 0.794 | *0.57 [0.42, 0.73]* | *0.745* |
  | 4, 0.1, 30 | 0.949 | 0.51 | 0.797 | *0.40 [0.28, 0.54]* | *0.791* |
  | 8, 10, 20 (the best score, lost the tie) | 0.958 | 0.56 | 0.834 | *0.42 [0.26, 0.60]* | *0.852* |
  | 16, 1, 20 (as drafted) | 0.927 | 0.60 | 0.709 | ***0.75 [0.61, 0.88]*** | *0.714* |
  | 32, 1, 20 | 0.927 | 0.67 (ineligible) | 0.709 | ***0.77 [0.62, 0.88]*** | *0.714* |

  All 14 candidates are in `probes/x_c_defaults.py`'s output; the table shows the chosen, its neighbours, the best and
  the drafted. Every synthetic sensitivity cell read 0.95–1.00 for every candidate, so the synthetic cells did not
  discriminate and the choice rested on the development main AUC and the tie-break: a limit of the rule.
- **Why the pick is not in the code:** `tests/test_conditional.py`'s sensitivity test (written with X_c, before the
  rule) plants a dependence spread over all 40 dimensions, and fails at the pick. Mean X_c over 3 seeds, contingent
  against blind: 4, 0.1, 10: 0.03 against −1.58 (the test needs above 1); 4, 0.1, 20: 0.34; 8, 1, 20: 1.28; **8, 10,
  20: 2.30 against −0.94;** 16, 1, 20: 6.39 against −6.24; 16, 10, 20: 7.96 against −2.39. A basis of 4 directions
  misses a dependence that does not lie along the past's top 4 directions; the rule's synthetic cells all planted it
  along one strong direction, so they could not see that cost. The trade-off: a larger R sees a spread-out dependence
  better, but from R = 16 the confound null leans toward reading repeated replies as contingency (16, 10, 20 holds it
  only at the edge, 0.65 [0.49, 0.80], the direction that would favour a pass).
  Where the 8B carries a direction word in its states is not known.
- ***Two checks added after the selection had started scoring (the #49 session's review), reported, never part of
  the rule; the defaults were not re-picked after seeing them.*** (1) A **confound null**: both arms blind, but 36% of
  one arm's replies are one repeated state, as the calibration's blind arm hears "You can't go that way."; a setting
  that reads that difference as contingency leaves 0.5. **The drafted 16, 1, 20 fails it (0.75), as do 16, 0.1 and
  every R = 32;** every R = 4 and R = 8 candidate holds 0.5. The chosen setting holds it at the edge and leans the
  other way (0.36), so in the deciding run the confound would work against a pass, not for it. (2) The development
  **can't-go AUC** in the score: the chosen setting 0.711 [0.54, 0.86]; R = 8, λ = 10 scores best on every version of
  the score (0.852).
- ***For the owner, before 7100–7139 are played:*** the rule's pick (4, 0.1, 10) holds the confound null at its edge
  but fails the test written before the rule; the drafted 16, 1, 20 fails the confound null. **8, 10, 20** had the best
  score under the rule (it lost only the tie-break to a smaller R), holds the confound null (0.42), is best on the
  development can't-go check (0.852) and passes the earlier test (2.30). Choosing it is an override of the rule after
  seeing the added checks, on development material only; a second, declared round with a spread-out cell in the rule
  is the alternative. Either is recorded here as the owner's.
- **Fixed (owner, 2026-11-07): R = 8, λ = 10, 20 pairs**, an override of the rule's pick, made after seeing the two
  added checks and the earlier test, on synthetic episodes and the development seeds only; 7100–7139 unplayed. With
  these, `tests/test_conditional.py` passes.
- **Synthetic check** (as above): **X_c's AUC 1.000** (median 5.18 bits contingent, −5.34 blind; an irrelevant reply
  costs bits, never pays). The planted dependence is far stronger than the calibration's (about a bit), so this shows
  X_c can see a dependence, not that it will see that one.
- **Development, deciding nothing:** the calibration's mirror seeds (907000–907039), played on the CPU; first in the
  text space (TF-IDF, no model): X_c must find the surface link there, or the code is wrong. Then the 0.6B on the
  4060, for plumbing and the interval's width.
- **The decision:** the calibration as fixed (its statistic, its can't-go check, its gate), with X_c in place of X, on
  fresh seeds **7100–7299**, 100 pairs (outside every pool; 7000–7039 is spent), read by the 8B. *Grown from 7100–7139 by the owner, 2026-11-07, before any of them was played: at 20 pairs the can't-go check rested on about 14 worlds in every development run; at the development rate (about 35% eligible) 100 pairs gives about 70. Pool size is not a threshold; the gate is unchanged.* On a pass, rate enters the sense
  tokens as calibrated, its predictive validity untested.

*Rate, conditional: the deciding result (2026-11-07–08, Qwen3-8B on a rented RTX 3090, driver 610.57.04; main
a0573b7, no uncommitted changes, recorded by the code's `hardware` entry).* Record in `runs/rate-cal-c/`
(`contingent/` and `blind/`, 200 complete 100-turn episodes each, no skips; the reading, 80,400 chunks in 10.4 hours,
and the score in `record.jsonl`). Form `conditional`, seeds 7100–7299, the fixed defaults R = 8, λ = 10, 20 pairs.
Rescored from the record with main's code by a second session (2026-11-08): every number identical.

| Gate | Result | Needs | |
|---|---|---|---|
| X_c, AUC contingent over blind, 200 worlds | 0.781, 90% interval 0.742–0.818 | lower bound > 0.5 | **pass** |
| The can't-go check, 78 worlds eligible (≥ 10: decides) | 0.825, 0.768–0.879 | lower bound > 0.5 | **pass** |

**Rate passes its calibration with X_c and enters the sense tokens as calibrated, its predictive validity untested**
(Step 0's question, moved by amendment 7). The gauge tells a player whose next move follows the world's last reply
from one replaying the same moves blind.

- *Reported, deciding nothing:* median paired difference 9.06 bits (contingent higher in 78% of worlds); on the
  can't-go check 16.56 bits (82%); clean scored turns 15,800 contingent, 1,373 blind. The text baseline (TF-IDF)
  0.816, 0.780–0.851, so the link is the surface one the form expected (the agent's reading: the 8B's states carry it no better than the
  words). X_c per layer 0.777 (L24) rising to 0.806 (L35). Self-progress S_c 0.498 (no direction expected). Median X_c
  16.38 bits contingent, 0.83 blind. New-state share 4.2% contingent, 2.7% blind; moves 65% and 42%; `look` 17% in
  both.
- *Development, for comparison (deciding nothing; `runs/rate-cal-dev2/`, the 0.6B on the 4060, the mirror seeds):*
  main 0.726, 0.629–0.814; can't-go 0.684, 0.526–0.832, on 14 worlds; text 0.834. The defaults were chosen partly on
  the development seeds; the deciding seeds were fresh and unplayed until the defaults and the pool were fixed.
- *Limits, as stated before the run, still standing:* the contingent link is surface-level, so this shows the gauge
  can see a known dependence of about a bit, not that it reads anything subtler; and it says nothing yet about whether
  rate leads a stall, which waits for its predictive test.

**Shape, by source (a draft only; no code yet).**
- **The change:** a chunk is coded under the formula fitted on the earlier chunks **of its own source** (a reply under
  earlier replies and events, an action under earlier actions), not on both sources together. Everything else as
  fixed (Lift v0, Step 0's states and floor, burn-in 10 turns).
- **Why:** under one fit to two-token commands and 40-token replies, a long reply is far from the mean for being a long
  reply, whatever its templates; boundary, which compares a reply with its nearest earlier reading (another reply),
  is free of that and parted the kinds (0.82) where shape did not (0.575).
- **If it still fails,** that is the measurement decision 4 asks for before a trained lift (below).
- **Development:** the Step −1C rehearsal's mirror seeds, read again by the 0.6B on the 4060, deciding nothing.
- **The decision:** Step −1C as fixed (both bars, the precondition, amendments 1–3), on a fresh pool **8000–8999**
  (outside every pool; 6000–6999 is spent), played by the script and read by the 8B.
- **Development, 2026-11-07 (the 0.6B, Step −1C's rehearsal record read again into `runs/stepm1c-06-src/`, deciding
  nothing; `stepm1c score --shape joint|by_source`):** joint (registered) 0.525 [0.448, 0.599], exactly the
  rehearsal's, so the plumbing reproduces; **by source 0.610 [0.535, 0.683]**, every layer 0.59–0.63; boundary 0.812;
  shape − boundary −0.202 [−0.285, −0.114]; given boundary 0.522. The median gap between kinds grows from about 10
  bits to about 220. So mixing commands and replies in one fit was part of shape's blindness, but by source stays far
  below bar 1. **The owner (2026-11-07): no deciding run of shape by source; a trained lift is drafted instead**
  (decision 4: this is the measurement it asks for).

**Shape, a trained lift: Lift v1 (a draft, 2026-11-07, discussed with the owner; for the owner to fix; no code).**
- **What changes:** only the space shape is coded in. Lift v0 codes a chunk's 4,096 numbers (the 8B), most of its
  bits going to dimensions at the noise floor, so a template difference barely registers. Lift v1 is a fixed linear
  map per read layer to K whitened dimensions, learned once and frozen before any reading of a deciding pool. Shape
  is then coded in it, by source, prequentially within the episode, as now; the formula, the floor in the new space
  and the burn-in are unchanged.
- **The simplest form** *(the agent's choice)*: the top K principal directions of chunk states from a training corpus,
  whitened. No gradient training; a learned encoder is a later form, only if this one fails.
- **The training text, and the risk that decides it.** A basis helps only if template and register differences lie
  among its top directions, so the corpus must carry register variety. It is **never the habitat**: a basis learned
  from `house`'s templates (Step −1C's on-structure side) would either project the off-structure templates away or
  turn shape into "unlike the training text", a global boundary. **The agent's recommendation: MASC**, the 500,000-word
  balanced subset of the Open American National Corpus, 19 written and spoken genres (fiction, letters, email, blogs,
  spoken transcripts, technical, news and more), announced as free for research and commercial use from the ANC site
  (the LDC copy carries LDC terms; the current terms are checked before download). **Fallback:** a fixed random sample
  of FineWeb's sample-10BT (ODC-BY, attribution), pinned by revision and sample seed. *Considered and not
  recommended:* the Brown Corpus (written only; licence unclear); an encyclopedic corpus such as WikiText (varies by
  topic more than register, so would likely rebuild the blindness); the 8B's own generations (someone writes the
  prompts, which shape the basis); the paraphrase sets (too few chunks).
- **Chunks:** the corpus cut into chunks whose token lengths follow the habitat's chunk-length distribution (a
  chunk's state is the mean over its tokens, and length has confounded both senses once): sentences in document
  order, grouped up to a length drawn from that distribution, by a fixed seed. About 10,000 chunks, with 2,000 more
  held out; about 10–15 minutes of the 8B's reading.
- **K, chosen by the corpus:** the formula's own two-part code on the held-out chunks (the rule the formula already
  uses for its rank), so no number is picked by hand and no habitat pool is touched. 64 only if that rule misbehaves,
  as a recorded fallback.
- **Whitened:** within the K directions, each scaled to unit variance on the training chunks.
- **Its own check, before Step −1C:** the lift codes the held-out corpus chunks in fewer bits than Lift v0 does. If not,
  it is not a better model of text, and Step −1C is not read through it. Lift's re-skin invariance stays its own slot.
- **The artefact:** derived and rebuildable (the record comes first): the corpus's name, version and file hashes, the
  chunking seed, the model and revision, and the code; the bases kept as a cache beside the run, not hand-edited.
- **Development:** a 0.6B lift trained the same way; the Step −1C rehearsal seeds (`runs/stepm1c-06-src/`), deciding
  nothing. **The decision:** Step −1C as fixed (both bars, the precondition, amendments 1–3), on the fresh pool
  8000–8999, played by the script and read by the 8B through the 8B's lift. No habitat pool ever trains the lift.
- **Not part of this draft:** whether rate's X_c should code in the lift too (it uses a basis of the past, R = 8).

*Lift v1, built and read in development, 2026-11-07 (the agent; deciding nothing; PR #51):*
- **The 0.6B's lift** (`runs/lift-06b/`; MASC's 390 texts split 312, 39 and 39 by document; 10,000, 1,000 and 1,000
  chunks; 7 minutes on the RTX 4060) **passes its own check on all 9 read layers:** held-out B coded 4.9–19.4% shorter
  than at Lift v0's own rank. K is 281–512 against Lift v0's 13–20, so general text carries far more structure than
  Lift v0 kept. On five layers K reached the ceiling of 512 (`KMAX_LIFT`, fixed before any habitat reading). A first
  build (`runs/lift-06/`, kept) crashed writing its last entry (a field-name collision), fixed and rebuilt.
- **Shape through it, on Step −1C's rehearsal seeds** (`runs/stepm1c-06-lift/`, the 0.6B):

  | Shape | AUC off over on [90%] | Shape − boundary | Given boundary |
  |---|---|---|---|
  | joint (registered) | 0.525 [0.448, 0.599] | −0.287 | 0.476 |
  | by source | 0.610 [0.535, 0.683] | −0.202 | 0.522 |
  | by source, in Lift v1 | 0.573 [0.496, 0.646] | −0.239 | 0.475 |

  Boundary 0.812 throughout; the lift's layers 0.56–0.59.
- *The agent's reading, after the result, deciding nothing:* the lift models general text well, but shape fits a
  Gaussian within one episode, to some 50–150 earlier chunks of a source; in 300–500 directions that is far fewer
  samples than dimensions, so most of a chunk's code again goes where the episode's fit cannot reach. K was chosen
  for a 10,000-chunk corpus. Choosing a smaller K to suit episodes would be choosing it on habitat material after
  this result.

**Shape is parked (owner, 2026-11-07).** Three forms on the same development seeds (joint 0.525, by source 0.610, in
Lift v1 0.573), none near the bar of 0.80, none above boundary. Shape stays out of the sense tokens and off the build
path; no deciding run of any of the three. It returns only with a different idea of what shape measures (for
example a learned encoder after babbling), as a new form, not another variant of these. The code (`shape_by_source`,
`body/lift.py`, `--shape by_source|lift`) and its records stay as they are; Lift v1 remains available, checked on text,
for other uses.


**Reported, deciding nothing:** the AUC of natural new-state turns over no-new-state turns (the engine's own novelty);
boundary's coupling part on the same turns (v10's coupling over the last p turns; window fits were fragile in the toy
bed, so it does not decide yet); shape and rate per turn; per-layer values; throughput; turns dropped from context.
*Gap, found 2026-11-05 (by the session drafting PR #32, for the owner): `score` produced only the first of these. It
now also reports per-layer values (the third bar's AUC and plants fired, layer by layer), throughput and turns dropped
from context, read from the record. Coupling and rate per turn are not yet computed, and `score` says so. *Shape,
added 2026-11-05 to the owner's definition (PR #32, `proposals/staged-schedule.md` sec. 4a):* per chunk and layer,
the formula's code length (Lk) of the chunk fitted on every earlier chunk of the episode, both sources, on Step 0's
readings (cut at 256 tokens) and floor variance; chunks in the first 10 turns fit and never score; reported by source
(self, world, planted event) and pooled, beside token lengths, deciding nothing. The owner's answers, 2026-11-05 (via PR #32):
the script's phases are recovered by replaying it from each seed in a scratch directory, kept only if every action
matches the record exactly, and saved beside the record as derived labels; from now on script play records each
turn's phase in its key (`by: code`, read only by scoring). The comparison (reported, deciding nothing): shape in loop
phases against explore phases, by source (the action, the reply), within token-length bins, with boundary (news) on
the same turns. The 256-token cap stays; score reports how many chunks it cut. *Bin edges, the agent's choice, written
before any run:* chunk token lengths [1, 8), [8, 16), [16, 32), [32, 64), [64, 256]; per bin and source, the medians
of loop and explore chunks and the AUC of loop over explore, for shape and for news; a bin with fewer than 20 chunks
of either phase is reported as too few.*

#### Step −1C: does shape read structure, apart from boundary? (instrument; no training)

*Fixed by the owner, 2026-11-05 (PR #32, `proposals/staged-schedule.md` §4a, where the reasoning and the definition's
five choices are). A new form, with its own seed pool; written here before its code.*

**Shape** is the code length of the current chunk under the formula fitted on every earlier chunk of the episode, both
sources, in bits per chunk (`goalspace/formula.py`'s prequential code, Lift v0), on Step 0's readings (cut at 256
tokens) and floor variance; the first 10 turns fit and never score; reported by source and pooled, with length beside
every comparison.

**Why a dissociation.** Boundary asks whether this text was seen before; shape asks whether it fits the structure
learned so far. They part on a chunk that is new but on-structure: boundary should fire and shape stay low.

**Plants.** Each episode gets one extra room, unlocked at a planted turn (15–35) as in Step −1's new-room plant and
entered by the script. One kind per episode, fixed by the seed: **on-structure**, generated by the episode's own world
generator; **off-structure**, its description from a second, disjoint vocabulary and template set, written once for
every world (domain-free; `body/offtheme.py`, PR #33), length matched within ±10% of the reader's tokens (Qwen3's
tokenizer), counted before rendering with Inform7 code removed. The chunk scored is the room's first description, on
the turn the script enters; each episode ends after it.

**Pass, both:**
1. **Shape:** the AUC of off-structure shape over on-structure shape is at least 0.80, its 90% lower bound above 0.70.
2. **Apart from boundary:** the 90% interval of (shape's AUC − boundary's AUC) on the same chunks lies entirely above
   zero (one-sided, amendment 1 below); 2,000 draws resampling episodes; the margin reported. If 1 passes and 2 fails,
   shape under Lift v0 is redundant with boundary: the measurement decision 4 asks for before a trained encoder
   replaces Lift v0.

*Precondition, reported:* boundary fires on both kinds. If it does not fire on on-structure rooms, the dissociation
cannot be read and the owner is told.

**Pool.** Every usable seed of 6000–6999 (amendment 3; about 110 a kind at the rehearsal's rate; first written as 80
a kind), seeds disjoint from every registered pool, taken in order. A seed is skipped, never loosened, when its off-structure description finds no length match, the script never
enters the room, its world cannot be built (a subprocess capped at 8 GB and 120 s, as Step 0 amendment 4), or its play
runs past 8 GB or 600 s (each episode in its own capped subprocess; amendment 2 below). All four counts are taken before
any reading and recorded with the seeds drawn.

**Bars fixed in advance** (replacing Q6's answer): no reading sets them. A 0.6B rehearsal on mirror seeds (scripted
play, read on the 4060) checks plumbing and the interval's width, deciding nothing; if the interval is much wider than
estimated, the pool grows before the 8B reads. Gating: under "Two stages".

*Amendment 1 (Step −1C), 2026-11-05, the owner, before any reading:*
- **Bar 2 is one-sided:** shape's AUC must sit above boundary's. As first written ("excludes zero"), a boundary that
  separated the kinds better than shape would have passed it, the opposite of shape's claim.
- **Reported, deciding nothing: shape given boundary.** Shape less its least-squares line on news, fitted on all
  chunks blind to the kind, then its AUC off over on. The reason, stated before any reading: the theme keeps house's
  words but changes its templates, which also moves an off-structure room away from its nearest earlier reading, so
  boundary may part the kinds nearly as well as shape. With both AUCs near 1, bar 2 cannot pass even where shape
  carries more; this reading can still see it. It is conservative there (the line on news takes up part of the kind's
  effect), so it can read a real effect low, never invent one. *The agent's note:* strata of news were tried first,
  on toy material, and dropped: under such a ceiling each stratum's off chunks still sit higher in news, and a shape
  that only tracked news read 0.97.
- **The 0.6B rehearsal reports whether boundary's AUC is near its ceiling.** If it is, the owner is told before the 8B
  reads; any change made then is labelled as made after the rehearsal.

*Amendment 3 (Step −1C), 2026-11-05, the owner, after the 0.6B rehearsal, before any play of the pool's seeds and
before any 8B reading:* **the registered pool is drawn whole**, every usable seed of 6000–6999 instead of the first 80
a kind. The rule written with the bars said the pool grows before the 8B reads if the rehearsal's interval is much
wider than estimated; it was about ±0.09 on the difference against the agent's ±0.03. At the rehearsal's rate this
gives about 110 a kind, narrowing the intervals by about 15%. The kinds may end unequal; both counts are reported. The
bars are unchanged. Growing the pool narrows the intervals; it cannot move a shape that reads near 0.5, as the 0.6B's
did (0.525), and it was not chosen for that.

*Amendment 2 (Step −1C), 2026-11-05, the owner, during the 0.6B rehearsal's play, before any reading:* **a seed whose
play cannot finish is skipped as unplayable.** TextWorld's quest tracking (`GameProgression.update`, on whenever facts or
admissible commands are asked for, which the labels and the script need) explodes on some states: rehearsal seed 906402
reached 28 GB at turn 76 and the machine killed the run. It reproduces exactly under a cap. Each episode now plays in a
subprocess capped at 8 GB and 600 s (`stepm1c.play_capped`); past either, the cut-off episode stays in the record, is
never scored, and the seed is counted as unplayable. The script's actions do not depend on the kind, so such a seed is
lost to both kinds alike. Also from this: each episode's game environments are now closed (they were not; this was
not the cause, about 6 MB over 30 seeds, and changes no play: 30 seeds recorded identically). A resumed run records
which seeds it found cut off (`resume` entries).

#### Step −1C: note, 2026-11-05, written with the code, before any run

Code: `body/stepm1c.py` (`play`, `read`, `score`), the pool in `material/habitat_worlds_stepm1c.json`, tests in
`tests/test_stepm1c.py`; `body/habitat.py`'s `build` and `play` take the plan as optional arguments, leaving Step −1
unchanged. `read` is Step −1's own, unchanged: news, shape and token counts for every chunk. Run here with TextWorld
1.7.0 on Linux (Python 3.11; TextWorld's jericho does not build on 3.13).

*The agent's choices, fixed by the owner unchanged (2026-11-05), before any play of the pool's seeds:*
- **The pool:** seeds 6000–6999, disjoint from every registered pool and below babbling's; the generator is Stage 1's
  frozen one (6–12 rooms); episodes of up to 100 turns, ending on the turn the script enters the planted room.
- **The kind:** on-structure on an even seed, off-structure on an odd one; the planted turn drawn as Step −1's
  (15–35, `default_rng(seed + 500000)`); a rehearsal seed takes the kind of the seed it mirrors.
- **The length match is drawn for both kinds:** every seed draws its off-structure description and is skipped if none
  matches, whatever its kind; an on-structure seed then keeps its house description. Without this the off kind alone
  would lose rooms whose house description is unusually short or long, and the kinds would differ in more than their
  templates.
- **A kind already full is not played,** and seeds are drawn in order until both are full; `play` records the counts
  (unbuildable, no length match, never entered, by kind) and the seeds drawn before any reading.
- **Checked in code:** one seed played as either kind gives the same actions (the script never reads a description),
  so who enters the room, and when, cannot depend on the kind.

*Found while building, deciding nothing (40 mirror seeds, scripted play only, no reading):* 14 seeds had no length
match, 17 never entered the room in 100 turns, 9 entered (about 22%). The planted room is small (one door, one object),
so its house description is often 16–20 tokens, below the theme's shortest (about 22); on ordinary rooms PR #33
measured 12% unmatched. At this rate the 1,000-seed pool gives about 110 usable a kind, against 80, and the draw takes
about an hour on a CPU. The script finds the room's door by chance: it opens a door only when no untried exit is left
in the room.

*Questions for the owner:*
1. **Bar 2's direction.** *Answered (owner, 2026-11-05): one-sided, shape above boundary; amendment 1 above.*
2. **The entry rate.** *Answered (owner, 2026-11-05): the script stays as fixed; about 22% usable is enough for the
   pool as drawn.*

#### Step −1C: the 8B's reading, 2026-11-06–07 (a rented GPU): the registered result

Record in `runs/stepm1c-8b/` (the registered play's 172,541 lines checked byte for byte against the play before the
reading; the play-only record kept as `runs/stepm1c-8b.before-8b/record.jsonl`). Read by Qwen3-8B in bf16 (layers
24–35) against `material/echo_floor_8b.json`, on an RTX 3090 (24 GB, driver 595.80; Python 3.12.15, CUDA 13.2, torch
2.14.1+cu132, transformers 5.22.1, textworld 1.7.0), the `hardware` entry appended before the reading. The run recorded
no commit (`commit: null`); the instance was the one that ran Step −1 at main 92b0558. *Owner to confirm.*

| | Result | Bar | |
|---|---|---|---|
| **Bar 1: shape**, AUC off over on | 0.575, 90% interval 0.511–0.641 | ≥ 0.80, lower bound > 0.70 | **fail** |
| Boundary, same chunks | 0.823 | — | |
| **Bar 2: shape − boundary** | −0.249, interval −0.324 to −0.176 | entirely above zero | **fail** |
| Precondition: boundary above the floor | 101 of 101 on, 106 of 106 off | reported | holds |
| Shape given boundary (amendment 1) | 0.490 | reported | |

**Step −1C fails: shape under Lift v0 does not read template structure.** It fails bar 1, not only bar 2, on a
contrast matched in length (median 37 and 37.5 tokens; token AUC off over on 0.529) and in vocabulary (`offhouse`
uses only `house`'s words), where boundary fires on both kinds and parts them (0.82). The 0.6B rehearsal read the same
(0.525; boundary 0.812).

- *Per layer, reported:* shape 0.560–0.582; boundary 0.733–0.917 (highest at layer 35). 200 of 207 scored chunks in the
  32–63 token bin. Boundary's ceiling (amendment 1): not near it overall.
- **What follows, under the rules fixed before the reading:** Step −1C gates chords and shape's entry into the sense
  tokens (gating by dependency), so **shape stays out of the sense tokens** (the owner's rule, 2026-11-05: no sense
  enters the channel before its own instrument test passes). Babbling's sense tokens are those that have passed:
  boundary (Step −1), and rate if its Stage 1 calibration passes (PR #45, which makes that calibration gate rate).
  Stage 2, which read rate only, rests (Step 0 amendment 7, PR #45). Nothing is retuned on this pool: a new lift, if any, is a new run on new seeds.
- **For the owner (decision 4):** a trained encoder replaces Lift v0 "only if a measurement earns it". This is such a
  measurement, and a stronger one than PLAN anticipated (bar 1 passing, bar 2 failing would have meant redundancy;
  here shape does not read the structure at all). Whether to build one, and the test it must pass before any
  reading, is open.
- *The agent's reading, after the result, deciding nothing:* a chunk's shape on the 8B is about 16,000–24,000 bits,
  and the kinds' medians differ by a sliver of that, so nearly all of it codes the noise dimensions (4,096 on the 8B),
  which would swamp a difference in templates. That points at the lift, not at the test.

#### Step −1C: registered play, 2026-11-05 (the script; no reading)

Record in `runs/stepm1c-8b/` (`record.jsonl`, 39 MB; the compiled games beside it are not needed to read), played from
main fbe0429 on the CPU here, under a 12 GB limit for the whole run, every world built and every episode played in its
capped subprocess (amendment 2). Every seed of 6000–6999 drawn, in order, in one run (amendment 3): no resume.

| Kind | Usable (entered) | Never entered | No length match | Unplayable | Unbuildable |
|---|---|---|---|---|---|
| On-structure | 101 | 237 | 162 | 0 | 0 |
| Off-structure | 106 | 212 | 182 | 0 | 0 |

Taken from the record's `drawn` entry, before any reading. Nothing has read this record.

#### Step −1C: rehearsal, 2026-11-05 (Qwen3-0.6B, mirror seeds, deciding nothing)

Record in `runs/stepm1c-06/` (play, read and score logs beside it). Scripted play on mirror seeds of 6000–6999, read by
the 0.6B in bf16 on the RTX 4060 with its own floor (`runs/echo-floor-0.6b-first/floor.json`, `--draft`), scored by
`body/stepm1c.py` at main 5127d75.

- **The draw:** full, 80 usable a kind. On-structure: 80 entered, 185 never entered, 121 no length match, 1
  unplayable, 0 unbuildable. Off-structure: 80, 191, 138, 0, 0. About 796 seeds drawn. The first attempt was killed by
  the machine at seed 906402 (TextWorld's quest tracking, 28 GB at turn 76); resumed after amendment 2, which skipped
  that seed as unplayable. The cut-off episode stays in the record, unscored.
- **Shape:** AUC off over on 0.525, 90% interval 0.448–0.599 (bar 1 for the 8B: 0.80, lower bound above 0.70).
  Across the 9 layers read, 0.51–0.555.
- **Boundary on the same chunks:** 0.812; layers 0.73–0.86. **Shape minus boundary:** −0.287, interval −0.378 to −0.191
  (bar 2 for the 8B: entirely above zero).
- **Precondition:** boundary above the floor on 80 of 80 of each kind. **Shape given boundary** (amendment 1): 0.476.
- **Length:** median 37 tokens on, 38 off; token AUC off over on 0.556. 154 of 160 scored chunks fall in the 32–63
  token bin (shape 0.533, boundary 0.831 there). 23 chunks of the record cut at 256 tokens, none of them scored.
- **Boundary's ceiling** (amendment 1 asks it be reported): 0.81, not near it.
- **The interval's width:** about ±0.075 on shape's AUC and ±0.09 on the difference, against the agent's estimate of
  about ±0.03 for the difference, which assumed the two AUCs strongly correlated. Under the pool rule ("if the interval
  is much wider than estimated, the pool grows before the 8B reads"), this is wider. *Decided (owner, 2026-11-05): the registered pool is drawn whole, about 110 a kind (amendment 3).* grow the pool
  before the 8B reads (the 1,000-seed range gives about 110 a kind at this rate), or keep 80. Growing it narrows the
  intervals; it does not move a shape that reads near 0.5.
- *The agent's reading, after the result, deciding nothing:* a chunk's shape is about 3,540 bits and the kinds' medians
  differ by about 10, so nearly all of it codes the noise dimensions (1,024 on the 0.6B), which would swamp a
  difference in templates. If the 8B reads the same, bar 1 fails, not only bar 2: shape under Lift v0 would not read
  template structure, the measurement decision 4 asks for before a trained encoder replaces Lift v0. The 0.6B's states
  are not the 8B's; this says nothing about the 8B.

#### Step 0, habitat form: does rate flatten before the stall? (instrument)

Runs first; then Step 0's owner-label form below, unchanged and without retuning (owner, 2026-11-04).

- **The reading:** world cross-progress X_t = L_noworld − L_now, the bits the window's world parts save in coding the
  agent's next action, by Step 0's formula and fit unchanged (`goalspace/formula.py`, `body/progress.py`). The window
  W is the last 10 turns *(agent's choice: Step 0's 20 self chunks would span a fifth of an episode)*. Burn-in: an
  episode's first 10 turns fit, never score.
- **Closure events:** each engine-labelled closure in Step 0's pool. Its window is the 10 scored turns **ending at the
  onset**, so it ends before the label could see anything: X during the stall is flat by construction and is not
  evidence. At least 3 scored turns, or the event drops out (counted).
- **Ascent events:** each ascent in Step 0's pool (as defined above). Its window is the 10 scored turns after it, at
  least 3, or the event drops out (counted), as for closures *(agent's choice, 2026-11-04, confirmed by the owner the
  same day: the quest's last step ends the episode, so it never has a window)*.
- **Controls:** for each event, 10 scored turns in the same episode outside every closure, at least 10 turns from any
  closure's onset and any ascent, nearest in time, never reused. Where the episode has none, the nearest world by seed
  with the same generator settings, by the same rules, reported as cross-episode.
- **Statistic per window:** the mean of X_t.
- **Pass:** reading 1, the AUC of control X over closure X (closures lower) is at least 0.70, its 90% interval's lower
  bound above 0.5, and it beats the text baseline (TF-IDF, as Step 0); reading 2, the AUC of ascent X over control X
  has its 90% lower bound above 0.5. Both must pass, at the frozen k; reported at k/2 and 2k. 2,000 draws resampling
  episodes. Fewer than 100 closures or 50 ascents: too few to decide.
- **Checks of the scoring** (as amendment 5, reported, deciding nothing): same-episode pairs only, and a placebo of 200
  rounds of pretend events matched by the same rules.
- **Reported, deciding nothing:** X during the stall; self-progress and cost; broad ascents; per-layer AUCs; X with W
  of 5 and 20.

*Amendment 1, 2026-11-04: found by the agent on the dry run; the owner chose (a), now written into the ascent label
above (code: `body/habitat.py`, `ascents`, `quest_left`; the walkthrough's count is a key, recorded each turn and read
only by scoring).* **Reading 2 had no events as first written.**
With TextWorld 1.7.0 installed, every world checked (seeds 1000–1003, 2005, 3000–3004) has max score 1, and the score
rises only on the walkthrough's last command, which ends the quest and so the episode. An ascent ("each score
increase") therefore has no turns after it to score, so reading 2 cannot reach 50 ascents and Step 0's habitat form
is "too few to decide" by construction. Step −1 is untouched: it reads no score. The engine does carry a finer signal:
`intermediate_reward` (+1 a step closer to the quest's end, −1 a step away) and the length of the remaining walkthrough
(`policy_commands`). Options, for the owner to choose before Step 0's code: (a) an ascent is a turn where the remaining
walkthrough reaches a new low for the episode, counting only non-movement commands (a quest step, not a walk);
(b) as (a) with movement counted; (c) keep score increases and score the 10 turns before, not after; (d) worlds with
several quests, so score rises mid-episode. (a) is closest to the spec's "a quest step reached". The closure label's
"no score change" is trivially met with one-point worlds and stays as written.

*Amendment 2, 2026-11-05, the owner, before any 8B run (changes decision 3's "matched ordinary stretches"):* the
4-bit 8B's play (candidate C1, under "Runs") either progresses steadily, a quest step every 1–3 turns, or sits in one
closure for most of the episode, so ordinary stretches hardly occur and 0 of 157 events found a control. Step 0 now
compares a stall with progress:
- **Pre-progress windows:** the last 10 scored turns before an ascent (ending before it, at least 3), none inside a
  closure.
- **Reading 1:** each closure window is paired with one pre-progress window of the same length (its last n turns), in
  the same episode, nearest its middle turn, never reused; where the episode has none, the nearest world by seed with
  the same generator settings (cross-episode, reported). Pass: the AUC of pre-progress X over pre-stall X at least
  0.70, its 90% lower bound above 0.5, and above the text baseline's.
- **Reading 2:** the post-ascent windows (as before) against the pre-stall windows, unpaired: the AUC of post-ascent X
  over pre-stall X has its 90% lower bound above 0.5.
- Unchanged: the window and burn-in, the frozen k with k/2 and 2k reported, 2,000 draws resampling episodes, the text
  baseline, too few to decide below 100 closures (here, paired) or 50 ascents. The placebo now draws, for each event, a
  random run of its length outside every closure, read by the same rules; same-episode pairs are reported for reading 1.

#### Step 0, habitat form: note, 2026-11-04, written with the code, before any run

Code: `body/step0_habitat.py` (`play`, `read`, `score`); tests in `tests/test_step0_habitat.py` on toy episodes, and
one on a real world in `tests/test_habitat.py`. Where the registered form left a choice, the agent's:

- **Sources:** the world's text (the opening, replies) is the partner and the actions are self (the owner's decision 2),
  so the reading X_t = L_noworld − L_now is `by_source`'s X. Scored turns are the actions from the 10th on; W is the
  10 turns before the action (its last 10 replies, 9 actions).
- **The closure window ends before the onset turn:** the onset's own action is part of the closure label (a repeated
  pair), so it is left out. The ascent window starts on the turn after the ascent.
- **Controls are as long as their event** (3–10 scored turns), contiguous among the episode's scored turns; "nearest in
  time" is nearest the event's middle turn. Controls may overlap another event's window; the registered rules keep
  them only outside closures and 10 turns from onsets and ascents.
- **The interval resamples episodes,** each drawn episode bringing all its events with their controls.
- **Broad ascents (reported):** a room entered for the first time, or a verb first used on a turn that made a new state
  (not a move); matched and read as ascents are.
- **Code lengths, not states, are checkpointed** per episode: states are a cache rebuilt from the text, and on the 8B
  they would be about 16 GB for the pool.
- Found on toy episodes: at 100 turns, an episode with a closure and two ascents has room for about two 10-turn controls
  under the 10-turn gap; the rest go to the nearest world with the same settings or have none (counted).

**Step 0 habitat rehearsal 1 (2026-11-04, Qwen3-0.6B, 40 mirror worlds of Step 0's pool, k = 10 from rehearsal 1,
decides nothing; record in `runs/step0h-0.6b-rehearsal/`).** Play, read (10.5 minutes) and score ran end to end; no
event survived: 41 closures and 5 ascents labelled, 39 dropped for fewer than 3 scored turns, 7 with no control. The
0.6B loops from the start: 34 of 41 closures begin in the first 6 turns, most last the whole episode, and 99% of turns
(median) are inside a closure; its ascents fall on turns 1–3 but one. A closure that begins in the burn-in has no
window, so a player that never makes progress gives Step 0 nothing to read, as it should.

**Step 0 habitat rehearsal 2 (2026-11-04, Qwen3-8B in 4-bit, play only, 30 mirror worlds, decides nothing; record in
`runs/step0h-8b-nf4-play/`).** On the 8 GB card the 4-bit 8B ran out of memory at 16,384 and at 4,096 tokens of
context; 28 episodes were played at 2,048 (`--context`, recorded), 2 at 16,384 before the first failure. The labels
alone, no states read: the 8B won 19 of 30, and episodes last a median 12 turns, since a won quest ends the episode.
70 ascents, 63 of them on turns 0–9, inside the burn-in; 15 closures at k = 10, most beginning in the first 10 turns
of episodes looped from the start. No event kept at k = 5, 10 or 20 (68 dropped for fewer than 3 scored turns, 17 with
no control). Resolved stalls from these episodes would give k = 34. **As registered, Step 0's habitat form reads
nothing from a player that either wins in about 12 turns or loops from the start: it needs episodes that progress
for a while and then stall.** For the owner.

#### Keys judge, never supply

The engine's state and score, the labels, k's stretches and the planted turns are read only by scoring, after every
state and code length is computed and checkpointed. The readout, the formula and the agent never see them. The agent
sees only what a player would.

#### Runs

1. A dry run on Qwen3-0.6B, 3 worlds per pool and 2 self-talk episodes: plumbing and the planted room's feasibility,
   deciding nothing. Its floor is the 0.6B's own, made by `body/floor_file.py` with the same settings (the 8B file
   covers layers 24–35; the 0.6B reads 19–27):
   `python -m body.floor_file --model Qwen/Qwen3-0.6B --corpus .../dara-session-transcripts --run runs/echo-floor-06-first --draft`,
   then for each pool `python -m body.stepm1 play --model Qwen/Qwen3-0.6B --run runs/stepm1-06 --pool calibration --limit 3`
   (and `step_minus_1`, `self_talk`), then `read --model Qwen/Qwen3-0.6B --run runs/stepm1-06 --floor runs/echo-floor-06-first/floor.json --draft`
   and `score --run runs/stepm1-06`.
2. Calibration, then k fixed here.
3. Step −1 on Qwen3-8B on the owner's machine (or a rented GPU on Vast.ai until the 3090 arrives: owner, 2026-11-06,
   DECISIONS.md; the whole run on one GPU model, a 3090 by preference, resumed only on the same model, the model
   recorded), read against `material/echo_floor_8b.json` (PR #29; the hardware it was measured on is not recorded).
   That run is the
   result. The floor's 95th percentile of repeats rests on about three chunks; the floor's habitat check, scored first,
   is where that gets tested on many repeats.
4. Step 0, habitat form, if Step −1 passes.

**Run 1, the dry run (2026-11-04, decides nothing).** On the owner's machine (RTX 4060, 8 GB; Python 3.14; TextWorld
1.7.0 installed from pip), Qwen3-0.6B, 2 worlds a pool (`--limit 2`) rather than 3, record in `runs/stepm1-0.6b-dry/`;
the 0.6B floor in `runs/echo-floor-0.6b-first/floor.json` (layers 19–27, 0.47–0.63 in units of D, AUC of new over
near repeats 0.999–1.000). The corpus argument is the corpus's `sessions/` folder. Play, read and score run end to end;
the planted room switched in both new-room worlds. Two faults found and fixed: the chat template returns a dict under
transformers 5, and the `room` key was always None (TextWorld 1.7 leaves `location` empty; it is now read from the
player's `at` fact, with a test). A third finding changed the design: every world scores one point, on the quest's
last command, so Step 0's ascents could never be scored (amendment 1, under "Step 0, habitat form"). What scoring printed, as plumbing only:
the floor's habitat check 197 repeats, all below the floor; k not set (no resolved stall in two episodes of a looping
0.6B); self-talk 0 of 60 exact and 1 of 60 restated arrivals above the floor; the planted room fired in 1 of 2 worlds
(1.85 floor units on an `inventory` reply; 0.53 on a move into a room seen before). Observed, deciding nothing: the
opening chunk carries TextWorld's ASCII banner, as a player sees it; and a planted turn can coincide with natural
novelty (a new room entered on the same turn), which would let it fire for another reason.

**Rehearsals 1 and 2 (2026-11-04, Qwen3-0.6B, mirror seeds, decide nothing).** Rehearsals play mirror seeds (a
registered seed + 900000: a new world on the same plan), so no registered world is spent; records in
`runs/stepm1-0.6b-rehearsal/` and `-rehearsal2/`. Rehearsal 1 (30 calibration, 60 Step −1, 20 self-talk seeds, code
before the hardening below): floor check 2,902 repeats, all below; k's procedure ran for the first time, k = 10 from
only 11 resolved stalls (the 0.6B won 0 of 30 and loops); self-talk exact 0 of 600, restated 11.5% above the floor (bar
5%); planted room 14 of 20, rule change 0 of 20; AUC over ordinary play 0.98. Rehearsal 2 (10 calibration, the same
Step −1 and self-talk seeds, on the hardened code, killed once mid-episode on purpose): the stub was played again
(seed 902005, 2 attempts), every opening without the banner, k = 5 from 10 episodes, otherwise as rehearsal 1 (room 13
of 20, all 20 clean of natural novelty; rule change 0 of 20; restated 11.5%). Re-reading rehearsal 1's record from its
text alone rebuilt all 130 news files exactly (largest difference 0.0). Found, for the owner before the 8B run:
- **The rule change is rarely delivered:** 18 of 20 (rehearsal 1) and 20 of 20 (rehearsal 2) never fired, since the
  rule waits for a world-changing command other than a move and the looping 0.6B makes none. It counts as not firing,
  so a passive player fails Step −1 with the instrument not at fault. The two delivered stayed below the floor (0.66,
  0.98).
- **The planted room is diluted in long replies:** all 6 misses are 76–104 token replies (a room seen before, the click
  appended last); all 14 hits are 13–33 tokens. A chunk's state is the mean over its tokens.
- **k moves with few stalls** (10, then 5); the 8B's calibration, 100 worlds, is where it is set.

**Rehearsal 3 (2026-11-04, Qwen3-0.6B, mirror seeds, after amendments 2 and 3, decides nothing).** 10 calibration,
the 60 Step −1 and 5 self-talk seeds; record in `runs/stepm1-0.6b-rehearsal3/`. Every rule change delivered (17 from
the walkthrough, 3 from play). Planted events fired 40 of 40 (room 20, rule 20), all clean of natural novelty; AUC over
ordinary play 0.989 (0.987–0.992); floor check 970 repeats, all below; self-talk exact 0 of 150, restated 12% (bar 5%).
Read inside the reply, as before amendment 3, the same plants fire 14 of 20 and 15 of 20. Noted by the agent, for the
owner: the planted chunks sit in narrow bands (room 5.2–5.7, rule 3.7–4.3 floor units) because each is one fixed
sentence the episode has not seen. Any short unseen text may clear the floor this way (restated self-talk, at 12%, is
short and unseen in form), so firing at a planted chunk is close to guaranteed by its shape. Step −1's third bar
compares plants with no-new-state replies; how plants compare with *first* replies of the same length is not yet
reported.

**Step −1 play check, the 8B in 4-bit (2026-11-04, play only, 2,048 tokens of context, the 60 mirror worlds,
decides nothing; record in `runs/stepm1-8b-nf4-play/`).** The 8B won 13 of 20 worlds of each planted kind before the
planted turn (15–35) came: median episode 8.5 turns (planted room), 9.5 (rule change), 17.5 (none). 7 of 20 plants of
each kind were delivered, against a bar of 17 of 20: **as registered, Step −1 fails on the worlds' length, with the
instrument not at fault.** With Step 0's rehearsal 2 (under "Step 0, habitat form"), the frozen generator settings
(6–12 rooms, quests of 3–8 steps) are too easy for the 8B. For the owner, before the 3090.

**Harder worlds: how the settings are chosen (owner's choice, 2026-11-04; criteria written by the agent before any
candidate is played).** The owner chose harder worlds over earlier plants: the generator's world size and quest length
rise, a new seed pool file is registered in place of `material/habitat_worlds.json`, and every bar stays. Candidates
are played by the 8B in 4-bit on mirror seeds (`--generator`, rehearsals only, recorded on each episode), Step −1's
and Step 0's pools, and judged on the engine's labels alone: no news, X or other reading is computed or looked at.
The chosen setting is the smallest candidate that meets all three; if none does, the owner is asked, not the bar
moved.
1. **Plants arrive:** at least 95% of Step −1 episodes are still running at turn 35 (the latest planted turn).
2. **Step 0 has events:** projected to Step 0's 400 worlds, at least 150 closures and 75 ascents kept (windows of at
   least 3 scored turns, with controls), at k = 10 and at the k given by the candidate's own resolved stalls.
3. **The quest is winnable:** the 8B wins at least 10% of episodes, so ascents come from progress, not chance.
Generation time per world is reported; a setting the generator cannot build in reasonable time is not a candidate.

**Candidate C1 (rooms 8–14, objects 15–25, quests of 8–15 steps; the 4-bit 8B at 2,048 tokens, play only, mirror
seeds; records `runs/cand1-stepm1/`, `runs/cand1-step0h/`).** Worlds build in 2–8 s. 1: 73% of Step −1 episodes still
running at turn 35 (bar 95%; 15 of 20 plants delivered each kind): fails. 3: won 37%: passes. 2: no event kept at
k = 10 or 20 (74 dropped, 83 with no control), and not for world size: the episodes split into ones that progress
steadily (an ascent, a quest step reached, every 1–3 turns) and ones stuck in one closure for most of the episode, so
almost no turn is outside every closure and 10 turns from every onset and ascent. **Step 0's controls, as registered,
assume ordinary stretches that this play hardly has.** The player is weakened here (4-bit, 2,048 tokens), which may
add looping. For the owner. Candidate C2 (rooms 10–20, objects 20–40, quests of 15–25) is played for criterion 1.

**Candidates C2 and C3, and criterion 2 under Step 0's amendment 2 (2026-11-05; the 4-bit 8B at 2,048 tokens, play
only, mirror seeds).** C2 (rooms 10–20, objects 20–40, quests of 15–25; the Step 0 part stopped at 14 of 30 worlds by
the 2-hour limit): 75% of Step −1 episodes running at turn 35, plants 17 and 16 of 20, won 24%; the 8B follows the
opening's step-by-step recipe and wins a quest of 15–25 steps in about 20 turns. C3 (the frozen sizes, with the opening
stating only the quest's last step, TextWorld's `only_last_action`; `runs/cand3-stepm1/`): 77% running at turn 35,
plants 14 and 16 of 20, won 27%, a quarter of episodes won within 3–16 turns. Criterion 1 fails for all three. Under
amendment 2, ascents are plentiful (projected 1,000–2,000 kept against 75) but closures are short (C1 27, C2 57–86
projected against 150): most stalls begin within the first 10 turns, inside the burn-in, and have no window. No
candidate meets the criteria; the owner is asked.

**Stage 1's script, rehearsed (2026-11-05, CPU, mirror seeds, keys only, decides nothing; record in
`runs/stage1-script-rehearsal/`).** 20 calibration and the 60 Step −1 mirror worlds: every episode 100 turns, none
won; all 40 plants delivered on their planted turn (rule changes: 17 blocking a played command, 3 from the
walkthrough); 13 of 40 plants on a turn where play was also new (reported apart); median 73% of turns with no new
state and 58% of replies repeating an earlier one. **Stage 1's registered play, done 2026-11-05** (calibration 1000–1099, Step −1 2000–2059, by `script-v1`, in
`runs/stage1-script/`, which git does not track: the record is on the owner's machine only, and should be backed up):
160 episodes, all 100 turns, none won; 40 of 40 plants delivered on their turn. Its phases, derived 2026-11-05 by replaying the
script from each seed (`stepm1 phases`): 160 of 160 episodes matched every action; the labels are in
`runs/stage1-script/derived/phases.json`, the record untouched. Next on the 3090: the 8B's self-talk
(20 seeds, both variants), then `read` (the 8B, the floor in `material/echo_floor_8b.json`) and `score`.

**Stage 2's player screen (2026-11-05, mirror seeds, engine labels only, no state read; records
`runs/stage2-1.7b-*`, `runs/stage2-4b-*`).** Qwen3-1.7B (bf16, full context): k = 11 from 30 calibration episodes; won
14%; ascents kept 16 in 40 (projected 160); closures kept 0, 30 of its 34 stalls beginning in turns 0–9. Qwen3-4B
(4-bit, full context): k = 8; won 30%; ascents kept 18 (projected 180); closures kept 0, 24 of 30 stalls beginning in
turns 0–8, and the 6 with a window found no pre-progress pair. Neither meets the criteria; the owner is asked. Seen in
the episodes: the early stalls are mostly the parser's: "unlock box" or "use key" where TextWorld wants "unlock box
with key", repeated for the rest of the episode; the system prompt's examples show no two-object command. And the
cross-episode fallback, read by the agent as identical rooms, objects and quest length, almost never finds a world.

**Stage 2's screen again, told the command forms (2026-11-05, mirror seeds, labels only; `runs/stage2f-*`).**
Qwen3-1.7B: k = 24; won 14%; ascents kept 18 in 40 (projected 180); closures kept 0, 28 of 33 stalls beginning in
turns 0–9; no pre-progress window anywhere in the pool. Qwen3-4B (4-bit): k = 6; won 33%; ascents kept 21 (projected
210); closures kept 1 (projected 10), 24 of 30 stalls beginning in turns 0–6; 3 pre-progress windows in the pool.
Neither meets the criteria. Across every player screened (0.6B, 1.7B, 4B, the 4-bit 8B, with and without the forms),
these worlds put nearly every quest step in turns 0–12 and nearly every stall's onset in turns 0–9, so Step 0's
10-turn burn-in and windows leave almost nothing: the worlds' timing, not the player. For the owner.

**Stage 2's screen on its own worlds (2026-11-05, mirror seeds of `material/habitat_worlds_stage2.json`, labels only;
`runs/s2w-1.7b/`).** Qwen3-1.7B told the forms: 30 calibration episodes (k = 21) and 27 of 40 Step 0 worlds before a
GPU out-of-memory at long context on the 4060 ended the run (the 3090 has room); 2 seeds skipped as unbuildable
(905004, 905007, over 120 s, amendment 4). Won 2% (bar 10%): fails. No closure kept, no pre-progress window in the
pool, stall onsets still mostly in turns 0–4. The longer quests are beyond it.

**Qwen3-4B (4-bit) on Stage 2's worlds, told the forms (2026-11-05, mirror seeds, labels only; `runs/s2w-4b/`).**
28 calibration episodes (k = 9) and 36 Step 0 worlds; 6 of 70 seeds skipped as unbuildable (amendment 4). Won 8% (bar
10%); closures kept 4 in 36, projected 44 to about 365 usable worlds (bar 150); ascents kept 68, projected 756 (bar
75); 18 pre-progress windows in the pool (none for any player in the frozen worlds); stall onsets still mostly in
turns 0–5. The closest player screened; it fails criteria 2 and 3. For the owner.

**Qwen3-8B (bf16, full context) on Stage 2's worlds, told the forms (2026-11-07, mirror seeds, labels only; a rented
RTX 3090; record in `runs/s2-screen/`).** 28 calibration episodes (k = 7) and 36 Step 0 worlds; 4 Step 0 seeds and 2
calibration seeds skipped as unbuildable (amendment 4). Won 26.6% (bar 10%); ascents kept 41, projected 410 (bar 75);
**closures kept 6, projected 60 (bar 150): fails.** Of 34 labelled closures, 20 have no ascent before them (onsets on
turns 1–9) and 8 start inside the burn-in; all 6 kept are paired across episodes. 47 of the 52 closures in both pools
run 50 turns or more, to the episode's end: one reply that changes nothing, then the action copied (21 after a parser
reply, 20 cycling through rooms, 6 a real dead end; the agent's sorting, after seeing them). Play: 5,060 turns in 55
minutes. For the owner: [proposals/stage2-lock-in.md](proposals/stage2-lock-in.md).

**Reading time on a rented GPU (2026-11-07, two probes by the agent on the rented 3090, deciding nothing).** Step −1C's
reading took 15.9 hours: 1.4 of them the 8B's states (8 s an episode), 14.5 shape's fit on the CPU (median 51 s an
episode). The instance shows 32 CPUs under a quota of about 5 (`cpu.cfs_quota_us` 512000), so the maths library's
default 32 threads fight over 5 cores: on Step −1C's longest episode (202 chunks), shape took 986 s on the default
threads, 725 s on 32, 2.8 s on 4 and 2.4 s on 1; the 8B's states took 59 ms a chunk. **Every `read` now runs with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.** The `hardware` entry records the thread settings, the
CPUs shown and the quota, and warns when the threads are unset and the machine shows more than twice its quota. The
thread count can change floating-point sums in the last places (around 1e-12), far below any bar; Step −1 and Step
−1C were read on the default threads and stand as recorded. Stage 1's record mixes two time zones (the record's local
time against file times 6 hours apart), so its reading time is not known.

**X's sensitivity: a synthetic check (2026-11-07, the agent, deciding nothing; `probes/x_sensitivity.py`).** After rate
failed its calibration (X 0.486, the text baseline 0.363), the question is whether X as built can see the kind of
dependence the calibration planted at all. Synthetic states in d = 4,096 (no model), laid out as the calibration's
episodes (an opening, 100 turns of action and reply; W = 10, burn-in 10), replies and actions around two far-apart
means with unit noise, read by Step 0's `codes` unchanged; 20 pairs per cell, the calibration's AUC and interval. The
signal's strength is its norm against the noise's. Both arms hear replies from the same law.

| Dependence (the contingent action carries…) | Estimator | 0.25 | 0.5 | 1 | 2 |
|---|---|---|---|---|---|
| **category:** one of 4 recurring vectors, the one the last reply carried (the calibration's kind) | X | 0.46 | 0.48 | 0.48 | 0.54 |
| | X_c | **1.00** | **1.00** | **1.00** | **1.00** |
| **content:** a fresh vector, the one the last reply carried | X | 0.51 | 0.49 | 0.46 | 0.35 |
| | X_c | **0.71** | **1.00** | **1.00** | **1.00** |
| **topic:** the world's topic, fresh every 10 turns, carried by its replies | X | 0.19 | 0.02 | **1.00** | **1.00** |
| | X_c | 0.48 | **0.92** | 0.64 | 0.54 |

AUC of contingent over blind episode means (blind: a random category, a fresh vector, or a topic of its own on an
offset schedule). X: every category and content cell's interval contains 0.5 except content at 2 (0.25–0.45). X_c
(`body/conditional.py` with the owner's defaults, R = 8, λ = 10, 20 pairs; read on the same episodes, a second run of
2026-11-07): content at 0.25 is 0.61–0.80; topic at 0.5 is 0.86–0.97, at 1 0.53–0.76, at 0.25 and 2 the intervals
contain 0.5 (0.36–0.60, 0.43–0.64); no X_c cell lies below 0.5.

- **X does not see a turn-by-turn contingency**, whether the move picks among recurring categories or copies new content
  from one reply, at any strength up to twice the noise. It **does see a shared topic** that the world introduces and
  the actions carry over several turns, perfectly once the topic is as strong as the noise. X codes the next action
  with a low-rank fit (at most 20 axes) over all earlier chunks, with the window's world chunks in or out: a
  direction that appears once, or that every earlier chunk already spans, changes the fit little, while a topic held
  over several turns gets its own axis. So the calibration asked X about a dependence it was not built to measure.
  This fits the text baseline failing too: the fault is in what X measures, not only in the 8B's states.
- **Below that strength, the shared topic reads inverted** (0.19 at 0.25, 0.02 at 0.5): contingent X is lower than
  blind in almost every world. A sign of X near threshold is not to be trusted.
- *For the owner, deciding nothing:* a rate that should read contingency (does the world's last reply shape the next
  move?) needs a different estimator, a conditional one (the next action predicted from the replies just before it),
  not the leave-out of a low-rank fit. Calibrating X instead with a topic-style dependence would choose the test the
  gauge passes, which the known failure mode warns against; either is a new form, on new seeds.
- **X_c, the conditional repair (PR #49), sees what X does not:** the calibration's kind of dependence at every
  strength down to a quarter of the noise, and copied content from half the noise up. **It mostly does not see a
  held topic**, the one thing X sees: once a topic runs for several turns, the agent's own last action already
  carries it, so the reply adds little beyond it, and only the turns where the topic changes count; at strength 2,
  both arms' X_c are about −6.5 bits. The two estimators measure different things: X_c, whether the last reply shapes
  the next move beyond the agent's own last move; X, whether recent world chunks share lasting variance with the
  action. X_c never read inverted here. What the calibration asks is X_c's kind.
- *Limits:* synthetic, isotropic noise; one separation of the two means (3 noise norms); the floor at the noise's
  variance (the 8B's is D²/d); one seed per cell. The large positive X in every cell (about 400–500 bits, both arms;
  the 8B's was about −1.5) comes from this geometry and is not compared with the 8B's.

**Hardening (2026-11-04, the agent, after the freeze).** An episode cut off mid-run is played again, not counted as
done (`end` entries; the stub stays in the record; score refuses stale news). Each planted turn records whether play
was also new there (`natural_new`, the plant's own door left out), and scoring reports clean and colliding plants
apart, deciding nothing.

**Play keeps its key-value cache between turns (2026-11-04, the owner's go-ahead, before any real run).** The agent
used to re-read the whole episode, up to 16k tokens, to write each action of at most 24. It now cuts its cache back to
the longest start the new prompt shares with it and reads only the rest; dropped turns and a new episode shorten that
start, and the restating prompt in self-talk bypasses the cache. The text the model sees is unchanged, and each episode
entry records `kv_cache`. Checked on Qwen3-0.6B (record in the agent's scratch space, not kept): greedy and
teacher-forced, the cached agent took the same action as the uncached one on 240 of 240 turns, over a calibration
world, a planted room, a planted rule change and a context cut to 1,200 tokens (51 turns with dropped turns). Sampled
free play: prompt tokens read fell from 673k to 12.7k over 200 turns, and time per turn from 0.41 to 0.19 s; the 8B's
gain is read on calibration. `play --no-cache` restores the old path. Every real run uses the cache, calibration
included, so all pools are played by the same code.

**Step −1 on Qwen3-8B (2026-11-06, a rented GPU on Vast.ai; DECISIONS.md, 2026-11-06): the registered result.**
Record in `runs/stage1-script/` (the scripted play of 2026-11-05 below the 8B's lines; the first 48,521 lines checked
byte for byte against the play before upload; the pre-8B copy kept as `runs/stage1-script.before-8b/`). The 8B wrote
40 self-talk episodes and read every chunk in bf16 (layers 24–35, sink "first"), against `material/echo_floor_8b.json`.

| Part | Result | Bar | |
|---|---|---|---|
| Floor check (scored first) | 5,809 repeats, all below the floor | ≥ 95% below | pass |
| **Bar 1, exact self-talk** | 0 of 600 arrivals above the floor | ≤ 1% | **pass** |
| Bar 1, restated self-talk | 78.8% of 600 above the floor | ≤ 5% | fails; *reported, gates at the tickle test (amendment 4)* |
| **Bar 2, fires at planted novelty** | 40 of 40 (new room 20 of 20, rule change 20 of 20) | ≥ 36, ≥ 17 a kind | **pass** |
| **Bar 3, above ordinary play** | AUC 0.907, 90% interval 0.894–0.919 | ≥ 0.90, lower bound > 0.80 | **pass** |

**Stage 1 passes** on its bars as amended before the run. Amendment 4 was made after the 0.6B rehearsal's restated
part failed (12%), and is recorded as such. Bar 3 clears 0.90 by 0.007; every read layer gives 0.902–0.909.

- *The score's own verdict:* the run's first score entry said `pass: false`, because `body/stepm1.py` still counted the
  restated part in its overall verdict: amendment 4 had not been carried into the code. Fixed (the restated part is
  reported with its `gates` named; every other bar unchanged) and rescored from the same readings, no model loaded:
  every number and every reported quantity identical, the verdict `pass: true`. Both score entries are in the record.
- *Restated self-talk, deciding nothing here:* 78.8% on the 8B against 12% on the 0.6B. Boundary under the history
  predictor reads the 8B's own restatements as news nearly every time; the tickle test after babbling, where this part
  gates, starts from there.
- *Reported, deciding nothing:* natural new states over no-new-state turns, AUC 0.68 (0.67–0.69); plants read inside
  their reply, as before amendment 3, 13 of 20 rooms and 17 of 20 rule changes; plants that coincided with natural
  novelty all fired (11 of 11), as did clean ones (29 of 29); k = 16 (reported only: Stage 2's player sets k); shape by
  the script's phase, AUC loop over explore 0.43–0.53 across length bins, while news reads 0.25–0.40 (loop below
  explore); 4 chunks cut at 256 tokens; coupling and rate per turn not produced.
- *Throughput:* the 8B's self-talk at about 6,400 turns an hour.
- *Hardware, recorded after the run:* an RTX 3090 (24 GB, driver 595.80), Python 3.12.15, CUDA 13.2, torch
  2.14.1+cu132, transformers 5.22.1, textworld 1.7.0, nnsight 0.7.0, nnterp 1.3.0, numpy 2.5.4; code at main 92b0558.
  The run itself recorded neither (`commit: null`): a `hardware` entry was appended on the rented machine after the
  run and copied into the desktop's record (seq 51066, with its origin noted). The rescore used the same commit plus
  the amendment 4 fix.
- *A false start:* a first self-talk start went to a fresh record by mistake (the play had not been unpacked into
  `runs/`) and was discarded before restarting; nothing registered was in it.

#### Note, 2026-11-04, written with the code, before any run

Code: `body/habitat.py` (worlds, plants, play, labels, k) and `body/stepm1.py` (`play`, `read`, `score`); tests in
`tests/test_habitat.py`, run here with TextWorld 1.7.0 on Linux. How the registered design is built, where the
section left it open:

- **The planted room works mid-episode.** A new-room world is compiled twice from the same facts and names: A, the
  room behind a locked door with no key in the world; B, the same door closed and unlocked. At the planted turn the
  harness starts B and replays, silently, only the commands that changed A's state, then checks that the two states
  agree on every fact but the door's. Checked here on 30 random-play episodes (75 tries at the door before the switch):
  30 of 30 agree. On a mismatch the episode stays in A and the plant counts as not firing.
- **The rule change** is announced in the reply of the first turn from the planted turn on at which a command other
  than a move has changed the world ("Something in this place shifts. From now on, you cannot open chest."); every
  later use gets a fixed reply and never reaches the engine. If no such command comes, the plant never fires and
  counts as not firing.
- **Replies as a player reads them:** TextWorld's prompt line and status bar (the room and a move count that changes
  every turn) are stripped before the reply is recorded, so news is not raised by the counter.
- **Self-talk is recorded honestly:** the reply is the model's, with provenance self; only its slot says world, and
  the readout reads the slot.
- **News** compares each reply with every earlier part of the episode, the opening and this turn's action included;
  the opening itself is not scored. States are read with Step 0's sink rule ("first"). The echo floor comes in as a
  file of per-layer floors in units of D (`material/echo_floor_8b.json`, measured with the same sink rule and sets,
  PR #29); D is measured in the same run on the paraphrase sets.
- **Not yet known:** throughput on the 3090, and whether TextWorld installs on the owner's machine (Linux, macOS or
  WSL); both are read on the dry run.

#### Limits, stated before any result

- One model plays and reads; the readout is a schema of its own mind here, unlike Step 0's owner corpus.
- TextWorld's replies are short and templated, so states sit closer together than in the owner's sessions; the floor's
  habitat check is there for that.
- The planted rule change is the harness's, not the engine's: the world's text changes before its mechanics do.
- Labels come from the engine, so they mark looping in the game, not the owner's sense of closure; where both exist,
  their agreement is a separate check (spec v0.2).

### Babbling (v0.2 Step 2): the gate (draft)

*Drafted by the agent 2026-11-08 at the owner's request, from Spec v0.2's "Babbling" and the slots in
[proposals/staged-schedule.md](proposals/staged-schedule.md). Every number marked (agent) is the agent's proposal.*
**Fixed by the owner, 2026-11-08:** the settings as drafted (three rounds of 200 episodes, the 64-dimensional reply
space, language-modelling weight 0.1, held-out seeds 9500–9599, 9600–9999 kept for the tickle and rate tests); the
tickle test's bars as drafted; the base probe and the yoked passive arm reported only, deciding nothing. **Bars 1–3's
ratios (0.90, 1.10, 0.90) stay open until the 0.6B rehearsal reports their intervals**, and are fixed before the 8B
reads any registered seed. Code may now be written for the rehearsal. Step 2 measures the channel: whether the graft reads its sense tokens,
and whether its forward model predicts what comes back better than a linear history predictor. It is not a test of the
senses (they are read with the graft off and never retrained) and not a test of behaviour (Step 3).

**The channel.** Two senses, the ones that passed their instrument tests: boundary (v0, Step −1) and rate's world part
(X_c, its calibration). Shape is parked; S_c (rate's self part) is reported, never in the tokens. The body vector after
turn t is b_t = (boundary of reply t, X_c of action t), each z-scored on the calibration pool (1000–1099), both read by
the base 8B with the graft off. A learned projection maps b_t to k = 2 soft tokens, inserted just before action t+1.
*Revised 2026-11-08 (the agent's review):* X_c is undefined until 20 earlier action–reply pairs are there to fit on
(`body/conditional.py`), so the body vector carries a third number, a flag: b_t = (boundary, X_c or 0, 1 if X_c is
defined else 0). The self-prediction loss and every bar's error on X_c count only turns where the target's X_c is
defined; boundary counts on every turn.

**The graft and its heads.** LoRA rank 16 on q, k, v, o, gate, up and down of Qwen3-8B in bf16, plus the projection and
two linear heads (the memory worked out below: episodes up to 8k tokens whole, longer ones in windows of 8k (agent)).
- *Self-prediction head*, at the last sense token: predicts b_{t+1}. Loss: squared error in z-units.
- *Forward-model head*, at the last token of action t+1: predicts reply t+1's state in a fixed space, the base model's
  alone reading (last-third mean, graft off) projected on the top 64 principal directions of the calibration pool's
  replies (agent). Loss: squared error in that space.
- *Language modelling* on the actor's own action tokens only, weight 0.1 (agent), so the graft keeps the base's
  competence.

**Play and training (agent).** Babbling seeds 10000–10599, TextWorld with Stage 2's generator settings, thinking off.
Three rounds of 200 episodes: round 0 the base 8B plays with no tokens (the projection untrained, so the tokens would be
noise); rounds 1 and 2 the grafted 8B plays with live tokens. The graft trains on everything played so far after each
round, one epoch per round, AdamW at 1e-4. Every turn and every reading goes in the record before training; the graft
is a derived artefact, rebuilt from the record, the base revision and the seed.

**Held-out pool (agent).** Seeds 9500–9599, outside babbling's training range (10000 and up, training only) and disjoint
from every registered pool (calibration, Step −1, Step 0, the behavioural pool 9000–9099, Step −1C's 6000–6999,
rate's 7000–7299); 9600–9999 are kept for the tickle test and rate's predictive test below. Played once by the base 8B with the graft off;
every arm below is scored teacher-forced on that same recorded text, so the arms differ only in what code routes to
them. Fewer than 80 usable episodes: too few to decide.

**Arms, all scored on the held-out text.**
- *Live:* the graft as trained, with the episode's own tokens.
- *Zeroed:* a second graft trained the same way on the same rounds' text, with every sense token zeroed in training
  and test.
- *Swapped:* the live graft, with each turn's tokens taken from another held-out episode at the same turn number
  (matched on position, as the owner's position rule asks; an episode that has no partner at that turn drops the turn).
- *Tokens alone:* ridge regression (λ = 10) from b_t to b_{t+1}, fitted on the training rounds, no text. *Revised
  2026-11-08:* the senses carry over from turn to turn, and boundary falls with position, so a graft that only copies
  its tokens forward would beat the zeroed graft and suffer from a swap without reading anything in context. Bar 1
  compares against this arm too.
- *Base probe (reported; whether it gates is for the owner):* a linear head on the base model's hidden state (graft
  off) at the last token of action t+1, predicting reply t+1's state in the same space, fitted on the training rounds.
  It asks whether babbling adds anything a probe on the frozen 8B does not already have.
- *Linear history predictor:* ridge regression (λ = 10, as X_c) from the mean state of the last 10 chunks and action
  t+1's state to reply t+1's state, in the same 64-dimensional space, fitted on the training rounds' text. This defines
  the spec's "linear history predictor": boundary v0 in the code is a nearest-earlier distance, not a predictor, so the
  spec's baseline is written here.

**Pass, all three (the spec's gate; thresholds agent):** 2,000 draws resampling episodes, 90% intervals, errors paired
by turn, so every comparison is matched on position.
1. **The tokens are used, in context:** live's self-prediction error over the lower of zeroed's and tokens-alone's
   (the lower chosen per draw) at most 0.90, its upper bound below 1. A pass means the graft does better with the
   tokens and the text together than with either alone.
2. **The tokens are read, not the text around them:** swapped's self-prediction error over live's at least 1.10, its
   lower bound above 1.
3. **The forward model beats the linear predictor:** the forward-model head's error over the linear predictor's at most
   0.90, its upper bound below 1.

*A limit, stated:* the held-out text is the base 8B's play, so the forward model is scored on actions that are not
its own. That keeps the arms' routing identical; the graft's own actions are scored only in the reported live play.

If 1 or 2 fails, the channel is wrong, not the senses: the residual-stream fallback (the spec's) is tried, on new
held-out seeds, behind the same bars. If 3 fails, boundary stays v0 and the tickle test does not run.

**Rehearsal first (agent; deciding nothing).** The ratios' spread is unknown, and Step −1C's interval came out wider
than estimated. So before the owner fixes 0.90 and 1.10: the whole step on Qwen3-0.6B on mirror seeds (+900000), one
round of each size scaled to fit, reporting each ratio's interval. The thresholds are then fixed, and the 8B's run
uses only the registered seeds.

**Reported, deciding nothing.** Persistence for the self-prediction (b_{t+1} = b_t); the base probe; errors by
sense (boundary, X_c) and by turn band; the swap within the same episode (±3 turns); the grafted player's quest
completion on the held-out worlds against the base's, played live (a functional measure, kept apart from the gate);
the phantom-limb and rubber-hand diagnostics (their measures fixed before they run).

**The yoked passive arm (agent: reported, not gating).** A third graft trained on exactly the live graft's rounds of
text, the same steps and losses, but never acting: its training data are the active arm's episodes, so its actions were
never its own. Scored on bars 1 and 3's quantities. If it does no worse than the active graft, the sensory-substitution
framing is doing no work; the agent proposes this as a report for the North star, not a gate on the channel, since the
channel can work either way.

**After the gate, two tests with their own bars (draft):**
- **The tickle test (boundary v1, instrument).** The forward model is frozen as its own graft (decision 1). Boundary
  v1 is reply t+1's distance from the forward model's prediction, in the forward model's 64-dimensional space. *Revised
  2026-11-08:* v0's echo floor is per layer on full states and does not carry over, so v1's floor is measured on the
  calibration pool (1000–1099) by the registered midpoint rule (between the 95th percentile of exact repeats and the
  5th of first-seen replies), before any tickle seed is read; v1 is in units of that floor. `substrate/` is unchanged. It must pass
  Step −1 again on fresh seeds (from 9600–9999, drawn as Step −1's): exact self-talk at most 1% above the floor;
  **restated self-talk at most 5%, now gating (amendment 4)**; plants fired as Step −1; and plants over ordinary turns
  matched within ±3 turns at least 0.84 (v0's matched reading), lower bound above 0.75 (agent). Reported: control 4,
  self-caused against spontaneous novelty. On a fail, boundary stays v0 in the tokens.
- **Rate's predictive test (Step 0's question, amendment 7: placed here).** Step 0's habitat form, unchanged, on new
  episodes played by the grafted 8B on fresh seeds (from 9600–9999), X_c in place of X. Stage 2's screen runs first, unchanged. *The
  agent's caution, and a correction of its own earlier advice:* placing it here mostly defers the question. Stage 2 failed because the 8B's stalls start abruptly and never end; the grafted player is the same
  8B with a graft, so the screen may fail again. If it does, rate's predictive test rests again, and the second engine
  is the remaining place.

### Babbling (v0.2 Step 2): training memory on 24 GB

*Worked by the agent 2026-11-04 for spec v0.2's open question "training memory"; awaiting the owner. Measured, not
assumed: `probes/lora_memory.py` on the owner's RTX 4060 (8 GB), on Qwen3-8B's own architecture in bf16 truncated to 2
and 4 of its 36 layers, so that the 8B's per-token costs fit the small card. No pass criterion; this decides how
babbling can be built, not whether it works.*

**The setup measured.** LoRA rank 16 on q, k, v, o, gate, up and down in every layer (43.6M parameters on the full
model), base frozen, gradient checkpointing per layer, SDPA attention, AdamW on the graft only, the language-modelling
loss's logits computed only on the scored positions (64 in the probe). Random tokens; the k = 2 sense tokens and the
small prediction heads are left out, as negligible.

**What it found.**
- Each layer stores its input per token, 8 KiB in bf16 (measured: 2 → 4 layers adds 16 KiB a token), so 288 KiB a
  token over 36 layers.
- One layer's recompute in the backward pass adds about 249 KiB a token whatever the depth, with the LoRA product
  computed in bf16 from fp32 master weights; 369 KiB with the product in fp32. So bf16 compute is the setting.
- Fixed: the bf16 base 15.26 GiB (the truncated loads extrapolate exactly: 2.32 for embeddings and head plus 0.359 a
  layer); the graft with its gradients and Adam state about 0.65 GiB; the CUDA context about 0.5 GiB; the allocator
  reserves about 7% above what is allocated.
- The 4-bit (NF4) base loads at 5.83 GiB (measured on the full 8B; embeddings and head stay bf16). The 4060 cannot hold
  its activations, so its rows below use the same per-token costs, which do not depend on the weights' format.
- The 0.6B dry run's 100-turn episodes come to 2.5k–7.4k tokens with the 8B's tokenizer (opening about 550 tokens,
  10–58 a turn); the 8B's episodes are measured on calibration.

| Training sequence | bf16 base, estimated peak | NF4 base, estimated peak |
|---|---|---|
| 4k tokens | ≈ 20.1 GiB | ≈ 10.0 GiB |
| 8k tokens | ≈ 22.3 GiB: fits, about 1.5 GiB spare | ≈ 12.2 GiB |
| 10k tokens | ≈ 23.5 GiB: at the edge | ≈ 13.4 GiB |
| 16k tokens (the actor's context) | ≈ 26.8 GiB: does not fit | ≈ 16.7 GiB |

(Peak = (weights + 0.65 + 0.524 GiB per 1k tokens + 0.3 for the loss, chunked) × 1.07 + 0.5. Less about 0.4 GiB if
the 3090 also drives a display.)

**The agent's reading, for the owner.** The bf16 base trains whole episodes up to about 8k tokens. Longer ones need one
of: training on windows of at most 8k tokens; moving the stored checkpoints to CPU memory (`torch.autograd.graph.save_on_cpu`),
which removes the 288 KiB a token at the cost of PCIe traffic and would bring 16k to about 22 GiB (estimated, not
measured); or the NF4 base, which fits 16k easily. The NF4 base trains the graft against a base it is not used with:
the senses are read on bf16 with the graft off, but the babbling gate and the Felt arms run base plus graft, so they
would either run on NF4 too or carry a train–use mismatch. The agent's suggestion: bf16 base, bf16 LoRA compute,
whole episodes up to 8k, checkpoints to CPU above that, and NF4 only if offloading proves too slow. Two further
facts to read on the calibration run: the 8B's episode lengths in tokens, and whether the loss is taken on the
actor's own tokens only (the action tokens are about a tenth of an episode).

### Step 0: does cross-progress read the agent's closures? (instrument)

*Written 2026-11-03, before any code. The design is the owner's: progress by source (option 3), with tool results as a
third source reported apart. The numbers were drafted by the agent and fixed by the owner the same day, unchanged
(DECISIONS.md). Amendments are dated here.*

**The claim.** Closure is learning that no longer takes anything in from the partner. So in the agent's
owner-confirmed closures, the owner's chunks of the recent window should save almost no bits in coding the agent's
next chunk, fewer than in the same agent's ordinary work; and where the owner ascended, they should save more.

**Two hypotheses the tool split tells apart** (reported, deciding nothing). *Narrow:* closure shuts out the source that
can change the frame, the partner, while the agent keeps learning from tool results, which answer questions asked
inside its own frame. *Broad:* closure shuts out everything outside. In closure windows the narrow one predicts tool
cross-progress still positive; the broad one predicts it near zero too. A third reading: some tool results do change
the frame, in which case tool cross-progress spikes just before some closures end.

#### Who said what

| Item | By | Used as |
|---|---|---|
| Each chunk's text | **the owner** (human turns), **the agent** (Claude's text) or **the code** (tool results), from the session archives | the data |
| Real, kind and who of the 65 departures; first-return times | **the owner** (the confirmation pass, `dara-session-transcripts/study/departures_confirmed.md`) | the key |
| Everything computed here | **the code** | |

The corpus is private: only references and numbers enter this repository or its records, never text.

#### Material

- **The record, per session:** in time order, each chunk tagged at creation with one of three sources:
  - **partner:** the owner's turns (role `human`; task notifications and system text excluded; at least 20 characters);
  - **self:** the agent's text (role `agent_text`);
  - **tool:** tool results (role `tool_result`). Tool calls (`agent_tool`) are left out: the agent's commands, not its
    thought, and too many to read. Since amendment 4, each tool result is also **world** or **echo** (below).

  Each chunk's text is capped at 256 tokens, as in T2; a tool result is read from its start.
- **Scored chunks:** the agent's (self) chunks from the 20th self chunk of a session on (the 40th before the amendment
  of 2026-11-03); earlier chunks fit, never score.
- **Who is read:** the local model reads Claude's sessions, so this tests the readout as a schema of another mind. Said
  so wherever it is used.

#### Readings of a text

As T2: each chunk read alone through `substrate.states.Reader` with T2's registered sink rule; the last third of the
model's layers; per-layer statistics, the primary their mean. D per layer from the paraphrase sets (draft until frozen).

#### The formula and the three code lengths

The formula is T2's: PPCA, noise floored at D²/d, k chosen by the two-part code within the span
(`goalspace/formula.py`). For a scored self chunk t, the window W is every chunk since the 20th self chunk before t,
of all three sources. It is counted in self chunks so that tool results, which outnumber the owner's turns about ten to
one, do not crowd the owner out of it.

- **L_now:** bits of t under the formula fitted on every chunk of the session before t.
- **L_nopartner, L_notool, L_noself:** the same, fitted with W's chunks of that source left out.

Then, per scored chunk:

- **Partner cross-progress** X_t = L_nopartner − L_now: bits the window's owner chunks save, given everything else. An
  owner chunk that echoes earlier content saves nothing. **This is the reading the pass is on.**
- **Tool cross-progress** T_t = L_notool − L_now: the same for tool results. Reported, deciding nothing.
- **World and echo cross-progress** (amendment 4) T_world and T_echo: the same, leaving out only W's world or only
  its echo tool results. Reported, deciding nothing. The narrow-vs-broad comparison is read on T_world; T keeps its
  registered meaning beside it.
- **Self-progress** S_t = L_noself − L_now.
- **Cost** C_t = L_now − the median of L_now over the 20 scored self chunks before t, as T2's surprisal is: cheap
  below zero, costly above. (Amended 2026-11-03; against all earlier scored chunks it read position in the session.)

A window with no owner chunk has X_t = 0 by construction; that is counted, not dropped, since an owner who says
nothing is part of the reading.

#### Events and controls

- **Closure events:** each owner-confirmed closure (real "yes", kind including closure, who "agent") with a recorded
  first return: 25 of 27. Its **window** is the last 10 scored self chunks before the first return,
  in the departure's session; at least 3, or the event drops out (counted).
- **Ascent events:** each owner-confirmed ascent (real "yes", kind including ascent, who "owner"): 23 before burn-in.
  Its window is the first 10 scored self chunks after the owner's turn, in the same session.
- **Controls, author-matched:** for each event, a window of the same number of the agent's scored chunks in the same
  session, outside every confirmed departure's span (departure to first return) and more than 60 minutes
  from any departure; nearest in time; never reused (T2's rule). Where the session holds none, the nearest session in
  time by the same rules, reported as cross-session (amendment 3, 2026-11-03). An event with no control drops out
  (counted).
- **Statistic per window:** the mean of X_t over its chunks.

#### Pass

- **Reading 1, closures:** the AUC of control X over closure X (closures lower) is at least **0.70**, its
  90% interval's lower bound above 0.5, and it beats the text baseline's AUC. 2,000 draws resampling events with their
  controls. Fewer than 15 scored events: too few to decide.
- **Reading 2, ascents (what the readout must not break):** the AUC of ascent X over control X (ascents higher) has
  its 90% interval's lower bound above 0.5. Fewer than 12 scored events: too few to decide.
- Both must pass. A cross-progress that reads ascent as closure fails however well it does on closures.

#### The text baseline

As T2's: TF-IDF in place of states, vocabulary fitted on each session's first 40 chunks, the text D from the
paraphrase sets in that space. AUCs compare across spaces; bits do not.

#### Reported, deciding nothing

- T_t per window: its median in closure, ascent and control windows (narrow vs broad), and in the last 3 self chunks
  before each first return against the 3 before those (the frame-changing reading).
- S_t and C_t per window; the share of closure, ascent and control windows in each cost × progress quadrant (cheap or
  costly; X above or below the controls' median).
- `substrate.coupling.coupling` on the same windows, and its rank correlation with X: whether the two are redundant.
- Proxy B (the trend of `coupling`'s dynamic residual) and proxy C (reuse of new regions, offline) on the same windows.
- Per-layer AUCs; X with W of 10 and 40 chunks.
- The D021 window (S14, 2026-10-12 15:44 to 23:56), chunk by chunk, against the hand trace's expectation (X near zero
  from about 17:44): the spec's Step 1, read here as one case only.

#### Keys judge, never supply

The confirmation labels and first-return times are read only by the scoring functions, after every code length is
computed and checkpointed. The formula never sees a label, a departure or a return.

#### Runs

1. A rehearsal on Qwen3-0.6B with draft sets: plumbing, deciding nothing.
   `python -m body.step0 --model Qwen/Qwen3-0.6B --corpus .../dara-session-transcripts --run runs/body0-06 --draft`
2. The run on the owner's machine with Qwen3-8B. That run is the result. On the 3090:
   `python -m body.step0 --model Qwen/Qwen3-8B --corpus .../dara-session-transcripts --run runs/body0-8b --device cuda --dtype bfloat16 --draft`
   (or split across GPU and RAM as T2 ran: `--device auto --gpu-memory 6.5`).

Both resume from per-session checkpoints when the same command is run again. Each scored chunk needs four fits, and tool results add about ten times as many states to read; the
rehearsal reports the time, and the windows may be thinned to every second scored chunk if the 8B run
would exceed a day.

#### Note, 2026-11-03, written with the code, before any model run

- **The fit, computed from the Gram matrix.** `body/progress.py`'s `GramFit` is T2's fit computed from the session's inner
  products, so its cost depends on the rows fitted, not on the width (4,096 on the 8B; about 4,600 words in a session's
  text baseline, where the direct fit took 12 s per code). It agrees with `goalspace.formula.Fit` to 1e-7
  (`tests/test_body.py`). Nothing in the readings changes.
- **System text** left out of the partner source, beside the T2 prefixes: task notifications and the automatic
  "session is being continued" summaries.
- **Echoes save almost nothing, not nothing.** On a toy stream whose partner chunks are exact copies of earlier rows,
  partner cross-progress has median −0.01 bits (mean absolute 0.16) against self-progress of 2 bits. The wording above
  stands as an approximation.

#### Limits, stated before the result

- The agent produces most chunks, and its narration repeats ("committing and pushing"). Cost and self-progress are
  coloured by that; cross-progress is a unique contribution within one stream, not a comparison between populations
  (T2 reading 1's failure), so it is less exposed.
- Tool results are cut at 256 tokens, so a long result's tail, where an error often sits, is not read.
- Tool calls are left out, so a loop of repeated commands shows only through the results it brings back.
- 25 closures and 23 ascents are pilot-scale; the interval decides, not the point estimate.

#### Dry run, 2026-11-03: the text space alone, plumbing only (decides nothing)

`python -m body.step0 --model none --corpus ... --run runs/body0-text-all --draft --text-only`: no model (Hugging Face
is not reachable from the agent's container), TF-IDF only, texts cut at 1,000 characters. All 45 sessions in 20 minutes
on 2 CPUs. Its AUCs are stated so that the amendments proposed below are seen not to have been chosen by them.

| | Events scored | AUC (text space) | 90% interval |
|---|---|---|---|
| Reading 1, closures | 12 of 25 | 0.51 | 0.40–0.63 |
| Reading 2, ascents | 17 of 23 | 0.71 | 0.64–0.81 |

- **Too few closures score.** 8 closures fall before a session's 40th self chunk or have fewer than 3 scored chunks
  before the first return, and 5 more have no control. 12 is under the bar of 15, so as registered the 8B run would
  most likely be undecided on reading 1.
- **The cost axis does not work as defined.** Against the median of all earlier scored chunks, every window is cheap,
  since prequential code lengths fall as a session grows: C reads position in the session, not cost.
- **Medians, text space:** self-progress in closures 135 bits against 61 in controls; tool cross-progress 243 against
  200; partner cross-progress 7.8 against 7.0. In ascents, partner 32 against 13.
- **Time:** codes take about as long per space as the text space did, so the 8B's 12 layers come to about 4 hours of
  fitting plus the reading; no thinning is needed.
- **Not yet in the code** at the time (reported, deciding nothing): `coupling` on the same windows, proxies B and C, X
  with W of 10 and 40. Added the same day, label-blind, per scored chunk (`body.progress.reported`): coupling over the
  60 self chunks ending at t with E the mean of the partner chunks arriving between self chunks; B as that fit's
  residual at t minus 20 self chunks before; C as a self chunk that sits beyond √2·D of every earlier chunk and is
  returned to within √2·D later (the toy's echo floor, since the measured floor varies with position).

**Amendments, owner, 2026-11-03, before any model run:** (1) burn-in at the 20th self chunk instead of the 40th; (2)
cost against the median of the 20 scored chunks before. No bar changes.

**After amendment 1, closures still fall short:** 13 score, not 15. Counted without fitting, from scored positions
alone: 5 closures still have fewer than 3 scored chunks before the first return (they fall in a session's first 20
agent messages), and 7 have no control, because their span runs most of the session or the session is short.
**Amendment 4, owner, 2026-11-03, before any model run:** tool results are split into **echo**, the agent's own
output read back, and **world**, everything else. A tool result is echo when at least half of its word 8-grams (first
6,000 characters) already appear in the agent's earlier output in the session: its text and its tool calls, which
carry the files it wrote. This is the laundering study's within-session self share (`dara-session-transcripts`, branch
`study/body-ablation`, `laundering.py`); the cut at one half is the agent's. Over the corpus, 1,451 of 18,201 tool
results are echo (8%). The reason: a positive T in closures could be learning from the world or from the agent's
own output coming back, which are opposite conclusions for the narrow hypothesis. No bar changes; the pass is still
on X.

**Amendment 3, owner, 2026-11-03, before any model run:** where a session holds no control, it is taken from the
nearest session in time (by session start), by the same rules, and reported as cross-session. Counted the same way:
20 closures and 21 ascents score, 12 of them with cross-session controls. No bar changes.

**Amendment 5, owner, 2026-11-03, before any model run, after step 3's scoring fault:** two checks of the scoring
itself, reported and deciding nothing. No bar changes; the pass is still on X over all pairs.

1. **Same-session pairs only:** both readings recomputed on the events whose control is from their own session, beside
   the readings over all pairs. Cross-session controls (12 of the scored events) could move the AUC if X's level
   differs between sessions.
2. **Placebo windows:** 200 times, each scored event gets a pretend event in its own session with as many chunks, at a
   random run outside every departure's span (in a session with no such room, anywhere but the event's own chunks,
   counted). Its control is matched by the same rules, amendment 3 included, and both readings' AUCs are computed
   (mean over layers, and the text baseline). No closure or ascent is in a pretend event, so these should sit near
   0.5; if they do not, the matching itself moves the readings, and the real readings are read against them.

The agent first proposed shuffling event and control within each pair. Before writing code the agent found that this
centres on 0.5 by symmetry whatever the bias, so it could not fail (step 1's fault). The placebo is the agent's
correction, inside what the owner approved. Tested on planted sessions, it stays at 0.5 when the matching is fair and
moves when the controls come from a session whose X runs higher (`tests/test_body.py`).

**Dry run of amendment 5, 2026-11-04: the text space alone, plumbing only (decides nothing)** (`runs/body0-text-null/`,
not committed; 68 minutes on this machine, all 45 sessions). Both checks run on the real corpus. In the text space:

| | All pairs: events, AUC (90%) | Same-session pairs only | Placebo: mean AUC (5th–95th of 200) |
|---|---|---|---|
| Closures | 20, 0.458 (0.355–0.578) | 13, 0.485 (0.385–0.609) | 0.517 (0.438–0.593) |
| Ascents | 21, 0.732 (0.673–0.819) | 16, 0.730 (0.664–0.828) | 0.495 (0.420–0.588) |

So in the text space the matching does not move the readings, and leaving out cross-session pairs changes little. A
median of 4 pretend events per draw, of each kind, had to sit inside a departure's span. The model's spaces are what
the 8B run reads; these figures are stated so that nothing later is seen to be chosen by them.

**The 0.6B rehearsal, 2026-11-04 (plumbing, decides nothing; draft sets; `runs/body0-06/`, not committed).** All 45
sessions, layers 19–27. It first crashed on a session with no chunks (the agent's bug, which the text-only dry run
could not reach since it reads no states); fixed with a test and resumed from the checkpoints. Time: 15 minutes
reading, 10.8 hours fitting, most of it in the two longest sessions (S41, 2,254 chunks, 5.5 hours; S44, 1.5 hours).

| | All pairs: events, AUC (90%), text | Same-session pairs only | Placebo: mean (5th–95th of 200), text |
|---|---|---|---|
| Closures | 20, 0.449 (0.407–0.493), text 0.483 | 13, 0.441 (0.391–0.487) | 0.491 (0.405–0.564), text 0.524 |
| Ascents | 21, 0.531 (0.484–0.586), text 0.723 | 16, 0.642 (0.592–0.698) | **0.630 (0.562–0.703)**, text 0.496 |

- **The placebo finds a bias in the ascent reading in the model's space** (not in the text space): pretend events with
  no ascent in them score 0.63 against their controls. So the matching itself moves reading 2 there, and an ascent AUC
  near 0.63 would mean nothing. Its cause is not yet looked at; candidates are where a control falls relative to its
  event (the matching keeps the earlier of two equally near runs) and how X trends along a session in the model's
  space. Reading 2's bar is not changed; the 8B's reading 2 is read against its own placebo, as amendment 5 says.
- **Time for the 8B,** estimated: fitting scales with the 12 layers against 9 (about 14–15 hours), and reading the
  26,817 chunks with the 8B split across GPU and RAM is slower by more than an order of magnitude, so the run comes
  to about a day. Thinning to every second scored chunk is already registered as allowed if it would exceed a day;
  that is the owner's call.
- **Where the ascent bias sits** (agent, label-blind, on the rehearsal's checkpoints, 100 placebo draws): pretend ascents
  whose control is from another session score above it in 0.806 of 855 pairs; within the session, 0.607 with the
  control before (595) and 0.422 after (550), about 0.52 together. X's rank correlation with position within a session
  has median 0.03. So the bias is cross-session: in the model's space X's level differs between sessions. The placebo
  also holds more cross-session pairs than the real readings, since its pretend events take the little room short
  sessions have. **Added by the agent before any 8B run, reported only:** the placebo split by same- and
  cross-session pairs. Whether the readings should pass on same-session pairs is the owner's call, before the 8B run.

## Goal-space tests (`goalspace/`)

The end, as accepted 2026-11-01 (`proposals/goal-space.md`): Levin's three criteria, tested separately; no two tests
share a pass bar. Since the reconciliation of 2026-11-02 (DECISIONS.md) the criteria are constraints on the one loop
above, not a rival frame, and T1 reads the agent's own formula as an instrument, not a goal. The product and research
sections below stand until the Big Picture doc's Entity tab is changed to match; their work carries over as the
proposal's section 5 says.

| Test | Criterion | Status |
|---|---|---|
| T1. Self-seeking | 3, its own formula | open; the toy bed's event record is its machinery |
| T2. The owner's formula from the aims | 3, the owner's formula | [pre-registered 2026-11-01](goalspace/t2_owner_formula.md); reading 1 retired (the owner's code evolves and has no lines); reading 2 (ascents) stands for the 8B run, read with `sink="first"` as registered; **frozen** (owner, 2026-11-04) |
| T2b. Wrong instructions | 3, the ratchet | **[drafted 2026-11-01](goalspace/t2b_wrong_instructions.md)** for the owner; needs the owner's marking pass (reversed, kept, ascent); **frozen** (owner, 2026-11-04) |
| T3. Deference | 3, the unit | open; needs the owner's labels of deference |
| T4. Graft and removal | 2 | open; a toy first |
| T5. Steering | 1 and 3 | open; the 8B on the card |
| T6. Awareness | the test | open; after T1–T5 |

## Substrate (`substrate/`)

The Entity tab's decisions, as code: the base weights never changed, with anything learned held as a removable graft (states read
through nnterp, `states.py`); the code
retained, not consumed (`axes.py`: every axis considered is kept); admission by two criteria (predictive: lowers
held-out residual by more than D; otherwise structural, tagged untested, until the research track states a test);
provenance logged at the source (`record.py`: self or external on every chunk).

| Step | Test | Status |
|---|---|---|
| 1. Model access and D | states on the real model; D per layer on frozen paraphrase sets | **code built; plumbing passes on real Qwen3-0.6B weights on CPU (2026-10-31); awaits the card and the frozen sets** |
| 2. Logging | per-chunk activation capture, in context and alone; provenance tagged at creation and carried when re-read | **reader and record built; the harness that writes them is product step 4** |

**State** (provisional): the residual stream after a layer, mean-pooled over a chunk's tokens (special tokens and a
sequence's first token left out: an attention sink where there is no begin-of-sequence token), read alone (`states`) or inside the whole sequence (`states_in_context`). **D**: the root-mean-square distance
of a state from its paraphrase set's mean, averaged over sets, per layer; the layer proposed is the one with the
largest between-to-within ratio, and the owner chooses. **Material:** 20 sets of 5 paraphrases
(`material/paraphrases.jsonl`), drafted by the model, frozen by the owner 2026-11-02. **Pass:** states for every text on both models
at every layer asked for, and the same text read twice gives the same state.

**The echo floor** (DECISIONS.md, the event record): the distance below which an arrival is the entity's own content
coming back. It is measured, not assumed; rounds 7a and 7b found that detecting repeated content waits on it.

*Dress rehearsal, written 2026-10-31 before its code* (`python -m substrate.echo_floor`). On Qwen3-0.6B on CPU, with
the draft paraphrase sets, so it is labelled draft throughout: plumbing and a first look, never the result, which is
the 8B on the card with frozen sets.
- **Sessions:** 10, each 8 statements drawn from different sets (seed 0), joined by blank lines and read in one pass;
  each statement's state is read in context.
- **Arrivals**, each read alone and measured against the nearest of its session's in-context states:
  - an **exact echo**, a statement of the session, verbatim;
  - a **paraphrased echo**, another paraphrase of a statement of the session;
  - **news**, a statement from a set the session does not contain.
- **Per layer** (every third of Qwen3-0.6B's 28, and the last): D from the paraphrase sets (`measure_d`, draft);
  the three distances' medians and spreads in units of D; the AUC of news over exact echo and over paraphrased echo;
  and how exact-echo distance varies with the statement's position in its session.
- **How it is read:** repeats are separable from news at a layer if news ranks above exact echo at AUC 0.9 or better.
  The paraphrased echo is reported beside it: whether a repeat in other words is still a repeat is a design question,
  not settled here. The toy's assumption, that an echo sits about √2·D from what it repeats, is checked against the
  measured ratio.

*Dress rehearsal result* (2026-10-31; Qwen3-0.6B on CPU, draft sets; records in `runs/echo-qwen06/` and
`runs/echo-qwen06-skip/`, not committed). **As written, it measured an artefact.** Qwen3 has no begin-of-sequence
token, so the first token of every sequence is an attention sink with massive activations. Every text read alone
starts a sequence and carries it; in a session only the first statement does. So D jumped from 1.3 to about 41 at
layer 3, news landed nearest the first statement (its 5th percentile at 0.26 D), and news ranked above exact echoes at
only 0.59 from layer 3 to 24. Post hoc, with each sequence's first token left out (`--skip-first`):

| Layer | D | Exact echo | Paraphrased echo | News (5th pct) | AUC news over echo / over paraphrase |
|---|---|---|---|---|---|
| 0 | 1.3 | 0.84 | 1.56 | 2.14 (1.83) | 1.00 / 0.98 |
| 3–9 | 2.6–5.2 | 1.00–1.52 | 1.61–1.90 | 2.04–2.54 (1.79–2.28) | 1.00 / 0.96–1.00 |
| 12–18 | 7.7–18 | 1.81–2.18 | 2.06–2.31 | 2.54–2.57 (2.16–2.19) | 0.90–0.99 / 0.74–0.92 |
| 21–27 | 33–69 | 1.32–1.67 | 1.76–2.02 | 2.53–2.73 (2.24–2.47) | 1.00 / 0.97–1.00 |

Medians in units of D. **Repeats are separable from news at every layer,** most cleanly at 21–27, where news's 5th
percentile clears the exact echo's 95th. The toy's √2·D (1.41) is close to the real echo floor at 21–27 and too low in
the middle layers. **Echo distance grows with a statement's position in its session** (correlation 0.6–0.8): the
echo floor is a function of context position, as the biologist review proposed for D (finding 11). Sessions here are
8 statements long, so how far it grows over a real session is not known. A small model on draft sets: plumbing and a
first look.

*Long sessions, written 2026-10-31 before their code* (`python -m substrate.echo_floor --long`). Does the echo floor
hold as a session grows? On Qwen3-4B in bfloat16 on CPU (the 8B does not fit this container's 15 GB; the owner's
RTX 4060 with 32 GB of RAM runs the 8B, split between GPU and RAM), draft sets, first token left out.
- **Sessions:** 4, each holding 14 sets' four paraphrases other than the second (56 statements, shuffled), read in one
  pass. The other 6 sets supply news; each session set's second paraphrase is its paraphrased echo.
- **Measured:** each exact echo's distance to the nearest in-context state, by its original's position (1–8, 9–24,
  25–56); news's distance to the nearest of all 56; both in units of D; the AUC of news over exact echo per bin.
- **How it is read:** the floor holds at length if news ranks above exact echo at AUC 0.9 or better in the last bin
  (25–56) at the layers that separated best in the short sessions (21–27 of 28 there; the 4B's last third here).

*Long-session result* (2026-10-31; Qwen3-4B, bfloat16 on CPU, draft sets, first token left out; 4 sessions of 56
statements; record in `runs/echo-qwen4b-long/`, not committed). Medians, with exact echoes' 95th percentile after the
slash, in units of D:

| Layer | News (5th) | Echo, positions 1–8 | 9–24 | 25–56 | AUC 1–8 / 9–24 / 25–56 |
|---|---|---|---|---|---|
| 8 | 2.18 (1.95) | 1.19 / 1.48 | 1.43 / 1.60 | 1.55 / 1.81 | 1.00 / 1.00 / 1.00 |
| 24 | 2.81 (2.41) | 1.97 / 2.32 | 2.43 / 2.72 | 2.57 / 2.85 | 1.00 / 0.91 / **0.78** |
| 28 | 2.72 (2.39) | 1.56 / 1.81 | 1.99 / 2.25 | 2.25 / 2.56 | 1.00 / 1.00 / 0.95 |
| 32 | 2.64 (2.36) | 1.42 / 1.65 | 1.80 / 2.16 | 2.08 / 2.38 | 1.00 / 1.00 / 0.99 |
| 35 | 2.44 (2.14) | 1.49 / 2.23 | 1.89 / 2.51 | 2.09 / 2.59 | 0.98 / 0.89 / **0.83** |

**Read as written: the floor holds in the last bin at two of the four last-third layers (28, 32), not at 24 or 35.**
At every layer the echo floor grows with position (at 32: 1.42 to 2.08 D), while news stays put, so the margin
narrows as a session grows: at 32 the last bin's 95th percentile (2.38) already meets news's 5th (2.36). These
sessions are about 1,000 tokens; real ones run to tens of thousands. Whether repeats stay separable at real lengths is
not settled by this, and the echo floor has to be read as a function of position, not a constant. Layer 8 holds
cleanly throughout, but an early layer is close to the tokens themselves, so its separation of exact repeats may not
carry to repeats in other words.

*Real sessions, written 2026-10-31 before their code* (`python -m substrate.echo_floor --corpus DIR`). Does the floor
hold at the lengths of real agent work? On the owner's 45 Claude Code sessions (`dara-session-transcripts`, private:
no text from it enters this repository or its records, only session ids, positions and distances).
- **Chunks:** each session's events parsed into chunks with their provenance and arrival cause: the agent's text and
  tool calls are self; tool results are external and elicited; the owner's messages are external and spontaneous.
  Each chunk is capped at 256 tokens (truncation recorded).
- **Windows:** the first W tokens of chunks of each of 10 sessions (W = 4,096 for the rehearsal in this container on
  Qwen3-0.6B; the owner's run sets W up to 32,768 with the 8B), read in one pass.
- **Exact echoes:** the window's self chunks, each read alone, against the nearest in-context state, by the token
  position of the original (0–512, 512–1,024, 1,024–2,048, 2,048–W).
- **News:** self chunks from *other* sessions, read alone, against the nearest in-context state of the window.
- **Reported in units of D** (the paraphrase sets', which are short statements, so also as a fraction of news's
  median), with the AUC of news over exact echo per position bin.
- **How it is read:** the floor holds at real lengths if news ranks above exact echo at AUC 0.9 or better in the last
  position bin, at the layers that held in the long-session test.

*Real-session result* (2026-10-31; Qwen3-0.6B on CPU, draft sets, first token left out; 4,096-token windows of 10
sessions, 160 echoes, 500 news; record in `runs/echo-corpus-qwen06/`, references only, not committed). Exact echo's
median as a fraction of news's median, and the AUC of news over echo, by token position:

| Layer | 0–512 | 512–1,024 | 1,024–2,048 | 2,048–4,096 |
|---|---|---|---|---|
| 0 | 0.39, 0.99 | 0.38, 0.97 | 0.41, 0.98 | 0.41, 0.98 |
| 12 | 0.65, 0.88 | 0.78, 0.76 | 0.90, 0.64 | 0.93, **0.63** |
| 21 | 0.52, 0.97 | 0.68, 0.87 | 0.83, 0.77 | 0.86, **0.75** |
| 27 | 0.51, 0.94 | 0.55, 0.92 | 0.65, 0.79 | 0.65, **0.81** |

**Read as written: the floor does not hold at real lengths.** Within about a thousand tokens of real agent work, an
echo read alone sits nearly as far from its own in-context original as news does, at every layer but the first,
which is close to the tokens themselves. The fault is the comparison, not the floor: an arrival read alone is measured
against states read in context, and context reshapes a state the deeper it sits. **Post hoc, alone against alone**
(the D run's states): a paraphrase of a statement sits nearer its original than the nearest different statement at
AUC 0.92–1.00 at every layer, and position cannot enter. On short, distinct statements only; real chunks of code and
tool output will crowd closer.

*Alone against alone on real sessions, written 2026-10-31 before its code* (`python -m substrate.echo_floor --corpus
DIR --alone`). With news now read alone against alone (DECISIONS.md), does it tell repeats from new content in real
agent work, where chunks of code and tool output crowd close?
- **Chunks:** the first 300 of each of 10 sessions (seed 0), capped at 256 tokens, each read alone.
- **Classes,** by text alone (word 3-gram Jaccard against every earlier chunk in the session; no text recorded):
  *exact repeats* (identical text to an earlier chunk); *near repeats* (overlap 0.8 or more, not identical); *new*
  (overlap below 0.2 with every earlier chunk). Chunks in between are left out.
- **Measured:** each chunk's distance to the nearest earlier chunk's alone reading, in units of D; the AUC of new over
  near repeat per layer; the same AUC in the first and last third of each session's chunks.
- **How it is read:** alone against alone works on real sessions if new ranks above near repeat at AUC 0.9 or better
  at the late layers (the last third), and the AUC in the last third of chunks is within 0.05 of the first third's
  (position does not enter). Exact repeats are reported; they should sit at zero.

*Alone-against-alone result* (2026-10-31; Qwen3-0.6B, draft sets, first token left out; 10 sessions, 2,987 chunks:
48 exact repeats, 15 near repeats, 2,403 new, 521 between; record in `runs/echo-alone-qwen06/`, references only, not
committed). A first run was broken by a reader bug (one-token chunks pooled over nothing, giving NaN) and is not a
result. In units of D:

| Layer | New, median (5th) | Near repeat, median (95th) | Exact repeat, max | AUC: all / first third / last third |
|---|---|---|---|---|
| 0 | 0.80 (0.46) | 0.21 (0.24) | 0.000 | 1.00 / 1.00 / 1.00 |
| 12 | 1.04 (0.60) | 0.20 (0.37) | 0.000 | 1.00 / 1.00 / 1.00 |
| 21 | 1.26 (0.72) | 0.21 (0.37) | 0.000 | 1.00 / 1.00 / 1.00 |
| 27 | 1.59 (0.91) | 0.29 (0.46) | 0.000 | 1.00 / 1.00 / 1.00 |

**Read as written: it works.** New content ranks above near repeats at AUC 1.00 at every layer, in the first third of
sessions and the last alike, and exact repeats sit at zero. Two cautions. Only 15 near repeats were found, and they
are near-repeats by text overlap, the easy kind; a retried call with different arguments, or a repeat in other words,
is not tested here. And **the floor is not D.** New content's nearest earlier chunk sits at 0.5–0.9 D at its 5th
percentile, below D at every layer: real chunks crowd closer than the short paraphrase statements D was measured on.
A floor of D would call much new content a repeat. On this corpus the line falls between near repeats' 95th
percentile (0.24–0.46 D) and new content's 5th (0.46–0.91 D), so the floor has to be measured on real repeats, as here.

*Alone against alone on the 8B* (2026-11-01; Qwen3-8B, bfloat16, split across the owner's RTX 4060 and RAM; draft
sets; the same 10 sessions and 2,987 chunks). In units of D:

| Layer | D | New, median (5th) | Near repeat, median (95th) | AUC: all / first third / last third |
|---|---|---|---|---|
| 0 | 1.88 | 0.63 (0.35) | 0.16 (0.17) | 1.00 / 1.00 / 1.00 |
| 4 | 5.17 | 0.88 (0.50) | 0.25 (0.29) | 1.00 / 1.00 / 1.00 |
| 8–20 | 69–77 | 0.13–0.32 (0.08–0.19) | 0.03–0.05 (0.04–0.06) | 1.00 at every layer |
| 24–32 | 91–153 | 0.52–0.89 (0.30–0.51) | 0.08–0.13 (0.12–0.22) | 1.00 at every layer |
| 35 | 187 | 1.41 (0.77) | 0.29 (0.42) | 1.00 / 1.00 / 1.00 |

**Read as written: it holds on the 8B,** as on the 0.6B: new content above near repeats at AUC 1.00 at every layer,
first and last third alike, exact repeats at zero. The margin is wide: new content's 5th percentile is two to four
times near repeats' 95th. One flag: **D jumps from 5.2 at layer 4 to 69 at layer 8** and stays there, which looks like
massive activations in a few dimensions on tokens other than the first. It does not touch the AUCs (a ratio), but D
is a poor unit at those layers until it is understood. The floor, read off the real distribution: at layers 28–32,
between 0.17–0.22 D and 0.41–0.51 D.

*Retried calls, written 2026-11-01 before their code* (`python -m substrate.echo_floor --corpus DIR --retries`). An
agent stuck in a loop re-issues calls; does news, alone against alone, follow what comes back, not the fact that a call
repeated? On the owner's corpus, references only.
- **Calls and results:** tool calls are linked to their results by call id. A **retried call** has the same tool name
  as an earlier call in the session and arguments at word 3-gram Jaccard 0.5 or more, short of identical; an **exact
  retry** has identical arguments; a **fresh call** has the same tool name but arguments below 0.2 with every earlier
  call of that tool.
- **The results' content,** by text against every earlier chunk: *repeated* (overlap 0.8 or more, or identical),
  *new* (below 0.2), or between (left out).
- **The floor, measured in the same run** from all chunks' classes (as the alone-against-alone check): per layer, the
  midpoint between near and exact repeats' 95th percentile and new chunks' 5th percentile of distance to the nearest
  earlier chunk, alone against alone.
- **Measured:** each result's distance to its nearest earlier chunk; the share of retried and exact-retry calls'
  results below the floor when their content is repeated, and above it when it is new; the same for fresh calls.
- **How it is read:** news follows content on retried calls if, at the late layers (the last third), at least 90% of
  retried calls' results with repeated content read below the floor and at least 90% with new content read above it.
  A class with fewer than 10 results is reported as too few to read. Repeats in other words (the same content
  reworded) cannot be labelled from text, and stay untested.
- First on Qwen3-0.6B here (the first 1,000 chunks of 10 sessions, seed 0, draft sets); then on the owner's machine.
  *Amended before any result, 2026-11-01:* the first 400 chunks held only 30 retried calls and 10 exact retries (call
  counts only; no distance had been computed), too few once split by their results' content, so 1,000.

*Retried calls, result on Qwen3-0.6B* (2026-11-01; float32, draft sets; the first 1,000 chunks of the same 10
sessions). Share of results on the right side of the floor (below it when the content repeats, above it when new),
with counts; the floor in units of D:

| Layer | Floor | Retried call: repeated / new | Exact retry: repeated / new | Fresh call: repeated / new |
|---|---|---|---|---|
| 0 | 0.32 | 1.00 (6) / 0.96 (24) | 1.00 (19) / 1.00 (2) | 0.97 (183) / 0.99 (1,802) |
| 12 | 0.44 | 1.00 (6) / 0.96 (24) | 1.00 (19) / 1.00 (2) | 0.95 (183) / 0.99 (1,802) |
| 21 | 0.51 | 1.00 (6) / 0.96 (24) | 1.00 (19) / 1.00 (2) | 0.95 (183) / 0.98 (1,802) |
| 24 | 0.53 | 1.00 (6) / 0.96 (24) | 1.00 (19) / 1.00 (2) | 0.95 (183) / 0.98 (1,802) |
| 27 | 0.62 | 1.00 (6) / 0.96 (24) | 1.00 (19) / 1.00 (2) | 0.95 (183) / 0.99 (1,802) |

**Read as written: not decided.** Retried calls whose results repeat number 6, under the 10 the rule needs, so that
half is too few to read. The half that can be read passes: 96% of retried calls returning new content read as new
(24 results, at every layer), so news follows what comes back, not the fact that the call repeated. Beside it,
reported only: exact retries returning repeated content read as repeats every time (19), and fresh calls separate at
95% and 98–99% on large counts. Retried calls that bring back the same content are rare in these 10 sessions; deciding
the rule needs more sessions, which is the owner's call (more data, the same rule, written before it is run).

## Product track (`product/`): drift detection, then self-correction

| Step | Doc | Test | Status |
|---|---|---|---|
| 0. Text-only floor | v9 step 0 | the text panel on the owner's 45 sessions | **done by the owner, twice:** text sees the owner leave an aim (AUC 0.65–0.71), not the agent |
| 3. Toy bed | v9, v10 and the v10 draft | below | **eight rounds run (below); detecting repeats now waits on the real model's echo floor** |
| 4. Labelled logs | Tests, in order; Entity tab, the product's aim (draft) | [goal attention and goal alignment](product/labelled_logs.md) on the owner-confirmed departures, the agent's 27 closures first | **8B, 2026-11-02: batch claim not supported (no reading separates closures); ascents pass; placebo stage pending, reported only** |
| 5. Reference scorer, coherence score | Tests, in order | intact versus shuffled work | open: needs the setting and the owner's reference work |
| 6. Compass and harness, arm A; Gate 0 | Tests, in order | the compass predicts drift on held-out stretches above a pre-registered level, and beats self-residual alone | open. Refined until passed; ends nothing |
| 7. Arms S and C; main demo | Main demo | C closes half the A–B gap on both scores and beats S by more than run-to-run noise | open |

### Step 3: the toy bed

Each round's pre-registration and result live in their own file, frozen by the commit before the round's code.

| Round | Design | Result |
|---|---|---|
| [1](product/rounds/round1.md) | v9's detectors: point and trajectory contrast, the observer's band, margin and frame growth | 0 of 3 bars (the margin caught dullness, 0.86, not proliferation); frame growth fails v9's decisive row |
| [2](product/rounds/round2.md) | v10: coupling, dynamic residual, admission, spectral descriptor | 3 of 4; coupling passes its bars; separation fails on the agent's rule for "at zero" |
| [3](product/rounds/round3.md) | sparse externals, at zero against the post-intake reference, the probe, shrinkage, framing | 0 of 6; reviewed: a misspecified observer for held externals, not a lack of power (`reviews/round3/triage.md`) |
| [4](product/rounds/round4.md) | event-locked coupling: innovations, news and gain, a news budget, the probe by rule | 5 of 8; news and gain pass every coupling bar at every echo lag, with honest intervals; misses in the carried-over contrast detectors, the re-read window and the definition of K |
| [5](product/rounds/round5.md) | the event record: separation from events, the probe read by its own responses, the response profile, a blind structure, the freeze rule | 8 of 9, the blind structure among them; separation misses on a news-spike rule (the builder's) and a scalar gain that cannot see a subspace dynamics change; not frozen |
| [6](product/rounds/round6.md) | the news-spike rule fixed; separation narrowed to the product's need; the residual descriptor as discovery's own test; a new blind structure | product 7 of 9; discovery fails; the blind structure (a partner replaying old content beyond the history) read open in every run; not frozen |
| [7a](product/rounds/round7a.md) | arrival cause; short and long news; the partner-side loop on long news over spontaneous arrivals; replay and confirmations as known structures; a new blind structure | 7 of 10; separation passes for the first time; misses: a floor in the wrong unit (replay), closed loops never probed (against the design), every elicited arrival treated as a confirmation (the blind structure) |
| [7b](product/rounds/round7b.md) | 7a's fixes; the probe as a contingency test; entity captured; persistence and recurrence; six reactive or nonlinear conditions; the self-drive term deferred | 7 of 12; entity captured, probing first, short-lag mirrors and the attractor loop work; repeats (replay, the blind retry loop) sit on the toy's floor; not frozen |

## Research track (`research/`): reasoning over the entity's own axes

No demo and no pass bar for the track yet, by design; each probe still has its reading written before its code. The
focus since 2026-11-02 (DECISIONS.md). What it has:

| Item | Test | Status |
|---|---|---|
| The D jump | below | **explained, 2026-11-02: the attention sink landing on the second token in texts beginning "Find"; the reader now finds it by norm (owner); D on the 8B re-measured, J 2.13** |
| Gate 1: negation | mirror, additive shift or contraction toward the origin; the winner beats both others by more than D | `research/gate1.py` built and tested on planted pairs, with a fixed origin; lifted out of the build order; **run: undecided in both scopes (below)**; `research/gate1_run.py`; rests there, not a test of the entity |
| **Causal authorship** | which source made this output happen, by intervention; then whether the model's own account agrees | **the track's question since 2026-11-02; pre-registered below; run: the instrument does not hold** |
| Structural axes | retained, tagged untested (`substrate/axes.py`) | waiting for a structural test from this track |
| Relations, composition, order | angles, nesting, symmetry; predicted position under combined axes; non-commuting moves | no probe yet |

### The D jump

*Written 2026-11-02 before its code* (`python -m research.d_jump`). On Qwen3-8B, D goes from 5.17 at layer 4 to 69 at
layer 8 and stays high, unlike the 0.6B's smooth growth. D is the margin Gate 1 must clear, so what is it measuring?
Qwen3-8B, bfloat16, split across the owner's RTX 4060 and RAM as in `runs/echo-alone-qwen8b/`; the same 20 draft sets
of 5, first token left out. The model is only read; `substrate/` does not change.
- **Layers:** 0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 28, 32, 35.
- **Check first:** D recomputed from per-token states must match the saved run within 1% at the layers in common, or
  nothing below is read.
- **The jump:** J = D(8) / D(4), recomputed under each condition as J′. Whichever single step of 4→8 holds the most of
  log J is reported.
- **Three causes, each read on its own; more than one may hold:**
  - *A few tokens.* Each text's largest-norm token (top 1, then top 3, chosen per layer) left out of the mean. Holds if
    J′ ≤ 3 and the control, a random token left out (seed 0), keeps J′ ≥ 0.8 J. Top 1 and top 3 are read separately.
  - *A few dimensions.* The 5 coordinates with most within-set variance, chosen per layer on half the sets and D
    measured with them removed on the other half, both ways, averaged. Holds if J′ ≤ 3 and the control, 5 random
    coordinates (seed 0), keeps J′ ≥ 0.8 J.
  - *Overall scale.* The median token norm's ratio, layer 8 over layer 4. Holds if it is J / 3 or more.
- **Why 3:** the 0.6B's own growth over comparable stretches is about 2 (layers 3 to 9).
- **Reported beside them:** between-to-within ratio per layer and condition; the second and the last token left out;
  which tokens are largest (string, position) and how much of their squared norm sits in their top 5 coordinates; D's
  jump on the same draft sets is the only thing explained, so no corpus text is read.
- **If none holds,** the jump is reported as not explained, and any new cause is written here before it is tried. What
  follows for D is a substrate decision, the owner's.

*Result* (2026-11-02; Qwen3-8B, bfloat16, split across the RTX 4060 and RAM, draft sets; record in
`runs/d-jump-qwen8b/`, not committed). D matches the saved run exactly at every layer in common. J = 13.30, and one
step holds 91% of log J: layer 5→6, ×10.67.

| Cause | J′ | Control J′ (needs ≥ 10.64) | Needs | Holds |
|---|---|---|---|---|
| Top 1 token left out | 2.24 | 13.60 | ≤ 3 | yes |
| Top 3 tokens left out | 2.30 | 13.60 | ≤ 3 | yes |
| 5 coordinates removed | 2.42 | 13.31 | ≤ 3 | yes |
| Overall scale (median token norm, 8 over 4) | 1.85 | none | ≥ 4.43 | no |

| Layer | 4 | 5 | 6 | 8 | 12 | 20 | 24 | 28 | 32 | 35 |
|---|---|---|---|---|---|---|---|---|---|---|
| D | 5.17 | 6.18 | 65.97 | 68.82 | 69.95 | 77.41 | 91.43 | 115.84 | 152.79 | 186.96 |
| D, top token left out | 5.23 | 6.43 | 8.02 | 11.68 | 13.24 | 20.16 | 38.51 | 68.76 | 113.43 | 183.26 |
| Between over D | 1.14 | 1.17 | 0.93 | 0.90 | 0.91 | 0.90 | 0.95 | 1.11 | 1.27 | 1.38 |
| Between over D, top token left out | 1.15 | 1.15 | 1.15 | 1.25 | 1.61 | 1.54 | 1.57 | 1.61 | 1.59 | 1.39 |

**Read as written: a few tokens and a few coordinates both hold; overall scale does not.** Leaving out the second
token does the same as the largest (J′ 2.14); leaving out the last does nothing (13.43).

*Post hoc, no threshold fixed* (`research/d_jump_posthoc.py`, layer 6). The two causes are one: **the attention sink,
landing on the second token in the 4 of 100 texts that begin "Find".** Those are the only "Find" texts, one each in 4
of the 20 sets. In the other 96 the first token has norm 8,297–13,981, about 100% of its squared norm in coordinates
2276, 233 and 4081, and the second token is ordinary (at most 68). In the 4, the first token has norm 160 and the
second 8,755–10,391, 99.9% in those same coordinates. The reader leaves out the first token by position
(`skip_first`), so here it drops an ordinary token and keeps the sink. That one token moves its text's state by about
750, and 4 sets of 20 holding one account for about 60 of D's 66 at layer 6. It still dominates at layer 24 (42 times
the median) and fades to 3 times by layer 35.

**What this leaves.** D on the 8B is inflated at layers 6–32 on these sets, about sixfold at layer 8 and 1.35-fold at
layer 32, and separation is understated. Coordinate removal is not the remedy: 2276 is also a top coordinate of
ordinary tokens at layers 24–35. Not tested: other models, what other first words do this, and whether corpus chunks
carry it. Results in units of D on the 8B (the floors in the 8B table above) use the inflated D; their AUCs are ratios
of distances and do not use D. Corpus chunks are read by the same reader, so a chunk starting this way would carry the
sink; that is unmeasured. What to do about it is a substrate decision, open.

*The sink by norm, written 2026-11-02 before its code* (`python -m research.sink_check`; owner's decision,
DECISIONS.md). **The rule:** in one forward pass, read token norms at the layer a third of the way up (12 of 36, chosen
because the sink is well formed there and the layer is not tuned to a result); a token is the sink if its norm exceeds
5 times the median of the sequence's other tokens' norms, taking the median without the largest token, so that a
two-token text can still flag. Flagged tokens are left out of the mean at every layer read. A text or chunk that would
be left empty keeps its tokens. Special tokens stay out, as before. The rule was chosen after seeing these 100 texts'
norms, so the sets are a check that it does what was meant; the corpus is the first data it has not seen.
- **On the 100 draft texts,** Qwen3-8B, same load and layers as the D jump. It passes if all of:
  1. exactly one token is left out of every text: token 0 in the 96, token 1 in the 4 beginning "Find";
  2. J = D(8) / D(4) is 3 or less;
  3. in the 96, states match the saved position-rule run at the layers in common: the largest distance is at most 1%
     of that state's norm;
  4. the same text read twice gives the same state, as in step 1.
- **On the corpus,** 30 chunks from each of the same 10 sessions (seed 0; capped at 256 tokens; references and norms
  only, no text recorded). It holds for real chunks if at least 99% of chunks of 3 or more tokens have exactly one
  token left out. Reported beside it: shares with none and with two or more; the position of the token left out; how
  many sinks sit somewhere other than token 0 and which is the largest non-flagged ratio. Under 99%, the rule is not
  used on the corpus until the owner has seen why.
- **Reported, not judged:** D and between-to-within per layer under the rule, against the position rule.

*Result* (2026-11-02; record in `runs/sink-check-qwen8b/`, not committed; its first attempt was killed by a time limit
during the corpus part, and that attempt's lines stay in the record).

| Check | Bar | Measured | Passes |
|---|---|---|---|
| One token left out, where expected | 96 at token 0, 4 at token 1 | exactly that | yes |
| J = D(8) / D(4) | ≤ 3 | 2.13 (13.30 under the position rule) | yes |
| Distance from the position rule, the 96 | ≤ 1% | 0 | yes |
| Same text read twice | same state | same | yes |
| **Corpus: exactly one token left out** | **≥ 99%** | **98.98% (290 of 293)** | **no** |

| Layer | 4 | 6 | 8 | 12 | 20 | 24 | 28 | 32 | 35 |
|---|---|---|---|---|---|---|---|---|---|
| D | 5.46 | 8.00 | 11.61 | 13.27 | 19.57 | 37.94 | 68.52 | 113.19 | 184.99 |
| Between over D | 1.09 | 1.17 | 1.24 | 1.55 | 1.62 | 1.61 | 1.64 | 1.63 | 1.39 |

**Read as written: the rule passes on the texts and misses on the corpus by one chunk's worth, so it is not used on the
corpus until the owner has seen why.** Why, post hoc: the other 3 chunks each have tokens 0 and 1 both flagged, at
10,188–10,382 against a median of 65–66, the two equal to the nearest unit within each chunk. So two sinks, both left
out; "exactly one" assumed one sink per sequence. Nothing unflagged came near the factor: the largest other token is 2.90 times the
median, against 5. Every one of the 293 chunks had token 0 flagged, so on this sample no chunk had a sink only on
token 1, and the position rule's fault on real chunks is those 3, which keep one sink. Why two tokens share a sink is
not known. No corpus text was read to look at this.

### Gate 1: negation, standalone

*Written 2026-11-02 before its code; amended only before any run, and any amendment dated here.* The owner's decisions
(DECISIONS.md): standalone on elicited pairs; the centre fitted from the pairs, then compared with candidate origins.
This changes the doc's "about the origin" into two questions, in order: what form does negation take, and is its
centre the origin. What a pass would mean is unchanged: a finding about the code; it cannot end the project.
- **Material:** 20 topics of 10 plain declarative statements, drafted by the agent, labelled as the model's, frozen by
  the owner before the run (`material/negation_statements.jsonl`). No statement begins with an instruction or names
  negation. Each opposite is elicited from Qwen3-8B itself, thinking off, greedy, up to 80 new tokens, by one fixed
  domain-free prompt, "Write the statement that says the opposite of this one. Reply with the statement only.", followed
  by a blank line and the statement; prompt and reply are recorded. The owner reads a random 20 pairs before any
  state is read; a reply that is not an opposite is flagged and left in, and the verdict is reported with and without
  flagged pairs.
- **States:** Qwen3-8B, sink by norm, each statement and opposite read alone. Layers 12, 24, 28, 32. D from the
  paraphrase sets, under the same rule, on sets the owner has frozen.
- **Three forms, centre fitted:** additive, y = x + v; contraction, y = λx + v (fitted centre v / (1 − λ), reported as no
  finite centre if |1 − λ| < 0.1); mirror, u the top direction of x − y and its plane's offset the mean of (x + y)/2 along
  u. **Baseline:** identity, y = x.
- **Two scopes, each form fitted in both, so no form gets a freedom the others lack:** global, one fit across topics,
  held out by topic (5 folds of 4 topics); per topic, one fit per topic, each pair left out in turn within its topic.
- **Verdict, per scope and layer:** the root-mean-square error on held-out pairs. A form wins if it beats both others
  and identity by more than D; otherwise undecided. **A scope's verdict** is the form winning at 3 or more of the 4
  layers; otherwise undecided.
- **Controls, by the same code:** shuffled pairs (each statement with another topic's opposite), and each set's paraphrase
  pairs in place of opposites. If a form wins a scope on shuffled pairs, that form's win there is void.
- **Is the centre the origin? Reported, not judged:** for the winning form, or for each if undecided, the fitted centre's
  distance (the mirror's offset, along u only) from: the paraphrase mean; the mean of the sessions' earliest externals;
  each session's; and the shuffled control's centre. All in units of D.
- **Reported beside it:** the verdict with the centre fixed at the paraphrase mean, as `research/gate1.py` does now. If it
  disagrees with the fitted verdict, both are stated.
- **Amended 2026-11-02, before any state of a pair was read.** *The rule (owner):* contraction with λ = 1 is additive, so
  as first written additive could never win; a test with planted pairs showed it. If contraction's λ (the median over a
  scope's fits) is within 0.1 of 1 it counts as additive: it is no rival and cannot win. Otherwise it stays a rival.
  The verdict as first written is reported beside it. *The flags (owner's reading, `material/negation_review.jsonl`):*
  2 of the 20 are wrong, 2 borderline, 16 fine. "Without flagged" leaves out the wrong; a further line leaves out the
  borderline too. The other 180 are unread, so any wrong ones among them stay in. *Details the text left open (agent):*
  shuffled pairs from one seeded permutation; paraphrase pairs are each set's 10 unordered pairs, so 20 groups of 10, the
  negation pairs' shape, though a text recurs across its set's pairs; a topic under 3 pairs leaves the per-topic scope;
  sessions' origins are their first 5 externals; a mirror's plane passes through its pairs' mean midpoint, and shuffling
  keeps that mean, so its global distance to the shuffled centre is 0 by construction and the angle between the two
  normals is reported instead. Centres for contraction and mirror are reported whichever form wins.
- **Added after seeing the pairs, labelled so, judged by nothing:** the verdict on the 114 pairs whose opposite has a
  negation word and on the other 86, because most opposites are an added "not". Beside these: how far apart statement
  and opposite sit, the statements' own spread within a topic, and token lengths against the paraphrase texts, because
  the statements are short and D was measured on longer texts; the verdict still uses D as written.
- **Not done here:** relations between negation axes (angles, nesting) reuse these states; that probe gets its own
  reading before it runs, so this gate does not become the track.

*Result* (2026-11-02; frozen sets and statements, 200 pairs, 10 sessions; record in `runs/gate1-qwen8b/`, not committed).
**Undecided, in both scopes, at all 4 layers**, on every line: all pairs, without wrong, without wrong or borderline, the
verdict as first written, and the centre fixed at the paraphrase mean. No form won a layer. The λ rule never fired
(λ 0.76–0.86), so the amendment changed nothing here. Global scope, error on held-out pairs, in units of D (by the code):

| Layer | D | Identity | Additive | Contraction | Mirror | λ | Contraction's gain on identity: pairs, paraphrases | Additive's gain: pairs, paraphrases |
|---|---|---|---|---|---|---|---|---|
| 12 | 13.3 | 1.282 | 1.193 | 1.108 | 1.196 | 0.81 | 0.174, 0.162 | 0.089, 0.005 |
| 24 | 37.9 | 1.408 | 1.328 | 1.239 | 1.333 | 0.80 | 0.169, 0.140 | 0.080, -0.001 |
| 28 | 68.5 | 1.382 | 1.316 | 1.231 | 1.334 | 0.81 | 0.151, 0.134 | 0.066, -0.003 |
| 32 | 113.2 | 1.383 | 1.328 | 1.238 | 1.424 | 0.80 | 0.145, 0.135 | 0.055, -0.003 |

Per topic, every form is nearer identity (additive and mirror at most 0.04 D better, and worse than identity at layer 32);
contraction gains 0.10–0.13 D. Shuffled control: contraction wins the global scope at all 4 layers (λ about 0: predicting
the mean), so any contraction win there would have been void; per topic undecided. Paraphrase control: undecided in both.
- **Reported beside, in D:** statement to opposite 1.28–1.41; statements' own spread within a topic 2.12–2.14, opposites'
  1.98–2.03; median tokens 9 (statements), 10 (opposites), 15 (paraphrase texts).
- **Centres, not judged, in D:** the contraction's global centre sits 2.3–3.4 from the paraphrase mean, the mean of
  sessions and each session, and 1.7–2.2 from the shuffled control's. The mirror's plane sits 0.02–0.19 from the paraphrase
  mean and within 0.43 of every session; its normal holds 15–21% of x - y's second moment, and meets the shuffled normal at
  a cosine of 0.12–0.47.
- **Post hoc, labelled so, judged by nothing:** undecided on both groups. Additive's global gain on identity is 0.15–0.22 D
  on the 114 pairs with a negation word, and -0.003 to 0.012 D on the other 86.
- **The agent's reading (inferred, not tested):** to clear the margin a form had to get under 0.28–0.41 D, which is 92–95% of
  the squared distance; the best explained 20–25%. So the undecided is not a near miss. A pair sits closer than two
  paraphrases do (1.59 D, which follows from D with 5 texts per set), so D, calibrated on rewording, is a wide margin for
  pairs that mostly share their words, and mean pooling dilutes a one-word edit. Contraction's gain is nearly as large on
  paraphrases, so it looks like regression toward the mean, not negation. The only part specific to negation pairs is a
  small additive shift, and it is carried by the pairs with a negation word, so it may be the word and not the meaning.
  The mirror's closeness to the origins means little: any one direction in 4096 is short. Paraphrase per-topic figures
  (λ about -0.17) rest on texts recurring across a set's pairs, as disclosed in the amendment.
- **A fault in the design, found by the run:** under the void rule a contraction could never win the global scope on
  this material, since predicting the mean wins it on shuffled pairs. It did not change this verdict. Nothing is rerun.

### Gate 1's origin: how far apart the candidates sit

*Written 2026-11-02 before its code* (`python -m research.origin_candidates`). Input to the owner's choice of `c`, not a
test: nothing here is judged and no candidate is picked. No negation pairs exist, so no Gate 1 hypothesis is touched.
On Qwen3-8B under the sink by norm, draft sets, the same 10 sessions, references and distances only. Runs only if the
sink check holds on the corpus (it missed; the owner released it, DECISIONS.md).
- **Candidates:** the mean of the 100 paraphrase states; each session's earliest externals (the mean of its first 5 and
  first 10 external chunks, read alone, capped at 256 tokens); the mean of the sessions' 5-chunk means; the zero vector.
- **Reported per layer** (8, 12, 24, 28, 32, 35), in units of D under the rule: each candidate's distance from the
  paraphrase mean; session against session; first 5 against first 10 in one session; the paraphrase states' own radius
  about their mean. Because a mirror's fit depends on `c` only along its normal `u`, also 2|Δc·u| for `u` the 190
  directions between set means, as a stand-in for directions that carry meaning (median and 95th percentile).

*Result* (2026-11-02; draft sets, 10 sessions; record in `runs/origin-qwen8b/`, not committed). In units of D:

| Layer | D | Radius of the states | Zero | Mean of sessions | One session (median, range) | Session to session | First 5 to first 10 | Mirror shift (median, 95th) |
|---|---|---|---|---|---|---|---|---|
| 8 | 11.6 | 1.59 | 3.04 | 1.26 | 1.34 (1.26–1.50) | 0.73 | 0.31 | 0.19, 0.44 |
| 12 | 13.3 | 1.85 | 3.85 | 1.66 | 1.75 (1.56–2.00) | 0.85 | 0.36 | 0.28, 0.68 |
| 24 | 37.9 | 1.90 | 5.11 | 2.00 | 2.08 (1.85–2.49) | 1.14 | 0.47 | 0.35, 0.82 |
| 28 | 68.5 | 1.92 | 5.12 | 1.69 | 1.84 (1.70–2.09) | 1.13 | 0.46 | 0.28, 0.75 |
| 32 | 113.2 | 1.92 | 6.41 | 1.66 | 1.80 (1.54–2.22) | 1.09 | 0.46 | 0.30, 0.81 |
| 35 | 185.0 | 1.72 | 7.17 | 2.42 | 2.62 (1.94–3.06) | 1.40 | 0.57 | 0.70, 2.38 |

A session's origin sits about as far from the paraphrase mean as a state does, and sessions sit about 1 D from each
other. Under a fixed origin, a contraction's error moves by (1 − λ) times these: at λ = 0.3 and 1.8 D, about 1.3 D,
more than the margin. A mirror's moves less, under 1 D at the 95th percentile at layers 8–32, on the stand-in
directions only. Nothing here is a Gate 1 result: no negation pair was read.

### Causal authorship

*Written 2026-11-02 before its code and material; amended only before any run, and any amendment dated here.* The owner's
decision (DECISIONS.md). A log already says who produced each token. The question here is which source made an output
happen: the goal, the model's own earlier reply, or the partner. In accommodation the model produces every token and the
frame is the partner's, so the log says "self" and is wrong in the sense that matters.

**Not on the logged sessions, for now.** Their "self" chunks were written by Claude, with its thinking hidden, so taking a
source away and re-reading Qwen says nothing about what caused them. The key is only valid on replies the read model
itself generated. Qwen continuing a real session could meet that; it gets its own reading, after this.

- **Material:** 40 short conversations in 4 kinds of 10, each with its topic, drafted by the agent, labelled as the
  model's, frozen by the owner before any run; no text from the corpus. Each has the same turns (the note's place amended below): the person's goal; the
  model's own reply, generated by Qwen and recorded; the partner's message; an inert note on nothing in the conversation;
  a fixed closing request naming no source. The partner's message never quotes the reply, so either can go without
  breaking the other. Every source is drafted in two wordings, and every inert note has two alternates.
  - **Explicit:** the partner gives an instruction that settles the reply's content. The known strong case.
  - **Frame once:** the partner offers a direction nobody asked for, with no instruction to take it.
  - **Frame three times:** the same, in three messages, for redundancy.
  - **No frame:** the partner's message is procedural, so any direction in the reply has another source.
- **Amended 2026-11-02, before any material or code (agent, details the text left open):** the model's own reply cannot be
  drafted in two wordings, so the second wording is a full second pass: the goal's second wording, a reply generated anew
  from it, the partner's second wording, its own R. Rendered in the user role with fixed labels, "Person:", "Partner:" and
  "Note:", so the account's options name real things, and a filler keeps its source's label. The closing request is fixed,
  "What should happen next? Reply in a few sentences.", unlabelled.
- **Amended 2026-11-02, after the freeze, before any run (agent, details the text left open):** "top source" in checks 4 and
  5 is the largest I of the three sources, with no floor, so two readings below the floor cannot agree by default; the
  key keeps the floor. The inert note is never a candidate; its own I is reported. Read twice is on the first
  conversation of each kind, all three sources. The account's partner option reads "what the partner said", since "latest
  message" misnames a repeated frame. The model's own reply is also capped at 80 tokens, and cut-offs are recorded. Code
  will not ask for the account unless all six pass. Beside the agreement: the best a constant answer would score.
- **Reading:** Qwen3-8B, thinking off, greedy, up to 80 new tokens, reply R generated once under the full context. The
  model's own turn is generated before the partner's message exists, so the partner cannot have caused it.
- **Intervention:** for each source, remove it and re-read with R held fixed. Influence I is the mean over R's tokens of
  KL(p(next token | full) || p(next token | without the source)), in nats per token. Removed two ways: **deleted**, and
  **replaced** by a neutral filler of the same token count and role, so positions do not shift. The verdict uses replaced;
  deleted is the check below. A repeated frame is removed as a group, and each copy alone as well.
- **The floor, F:** the 95th percentile, across conversations, of I from swapping the inert note for its first alternate.
  It is the size of a change that means nothing. Every result is read against it, not against D.
- **Does the instrument hold? All must pass, or nothing below is read:**
  1. Read twice: I differs by at most 1% of F.
  2. Null: swapping the inert note for its second alternate exceeds F in at most 10% of conversations.
  3. Known strong case: the explicit partner's I is at least 5 F in at least 90% of those 10.
  4. Removal method: deleted and replaced name the same top source in at least 80% of conversations.
  5. Wording: the second wording of the sources names the same top source in at least 80%.
  6. Redundancy: a frame removed as a group exceeds F in at least 90% of those 10.
  Beside it, not judged: each copy alone against the group. If one copy's I is much smaller, single removal undercounts
  what is repeated, and any later use on real sessions must find repeats first.
- **Is the model's own account wrong where the intervention is right? Read only if the instrument holds.** After R, in the
  same context, by one fixed domain-free prompt: where the direction of that reply came from, chosen from labelled options
  (the person's goal, your own earlier reply, the partner's latest message, none of these), the order set by a seeded
  shuffle. Greedy, thinking off, a separate pass, so the question cannot touch R. The key is the top source if its I exceeds
  F, otherwise "none of these". **The claim, that the system knows something the model's account does not, holds if the
  upper end of a 95% bootstrap interval (over conversations) on the account's agreement with the key is below 0.8.** If the
  lower end is at or above 0.8, the model already reports this well and an outside instrument adds little here: said so,
  and the direction is reconsidered. Otherwise undecided. Agreement is also reported per kind, with the frame kinds beside.
- **Reported, not judged:** how often the frame's source tops the reply in the frame kinds, which is accommodation counted
  directly; the top source's margin over the next.
- **Keys judge:** the account never sees I, and no reading of states feeds the key.
- **The dry run** (2026-11-02; 2 conversations outside the 40, `material/authorship_dry.jsonl`; records in
  `runs/authorship-dry/` and `runs/authorship-dry-b/`, not committed; two crashes on the agent's own bugs first). As
  written, with the note last before the closing request, the replies open on the note, a repainted bridge, not on the goal or the
  partner, and swapping the note moves the reply more than any source (I 1.4–2.5, against the partner's 1.1). The note
  is not inert there, and no floor can be read. With the note before the partner, the replies follow the partner and a note
  swap gives I 0.01–0.04. Each conversation took about 10 minutes, so the 40 take about 7 hours. Every own reply was cut
  off at 80 tokens. **Amended 2026-11-02 by the owner, before any of the 40 was read: the note goes before
  the partner.** The figures above were seen when this was decided, and no bar changed. So the partner always speaks last,
  and recency is part of what the key measures. The cap of 80 stays, and a killed run resumes from its record.
- **Amended 2026-11-02, during the run and before any verdict was read (agent, the owner agreeing): how the key's own
  readings are read.** No bar, threshold, material or code changes; the run already prints all three (`key per kind`,
  `by_kind`, `best_constant_answer`). They are reported beside the verdict as checks on the key, deciding nothing:
  1. *No frame:* the partner's message there is procedural and always last. If the key names the partner in most of
     these 10, the key is partly measuring recency or the request's form, not the reply's direction, and the account's
     disagreement on this kind is read as a fault of the key as much as of the account.
  2. *"None of these":* how many of the 40 keys are "none". If almost none, the key is in practice the largest of three
     sources, and an account that answers "none" can only be scored wrong.
  3. *Best constant answer:* the account's agreement is read beside it, so a claim cannot rest on the key's imbalance.
  Why: I averages KL over every token of R, so it moves with form and phrasing as well as direction, while the account
  is asked where the direction came from. The six checks test the instrument's reliability, sensitivity and
  robustness, not that I tracks direction.
- **The run** (2026-11-02, commit `2cb3e6a`, `python -u -m research.authorship --run runs/authorship-qwen8b`; record in
  `runs/authorship-qwen8b/`, not committed; 40 conversations, 498–655 s each). All 80 own replies were cut off at 80
  tokens, and 9 of the 80 replies R. F = 1.97 nats per token.

  | Check | Value | Bar | |
  |---|---|---|---|
  | 1. Read twice | 0.0 | ≤ 0.02 | pass |
  | 2. Null | 0.075 | ≤ 0.1 | pass |
  | 3. Known strong case | 0 of 10 | ≥ 0.9 | **fail** |
  | 4. Removal method | 0.975 | ≥ 0.8 | pass |
  | 5. Wording | 0.975 | ≥ 0.8 | pass |
  | 6. Redundancy | 0.9 | ≥ 0.9 | pass |

  **Read as written: the instrument does not hold, so the model's account was not asked.** One check of six fails.
  *Why, from the figures (agent's reading, not judged):*
  - **The floor is set by notes that are not inert.** Swapping the note moves the reply by 0.02–0.6 in 34 of 40
    conversations (median 0.11) and by 0.97–2.47 in 6 (hall, podcast, speech, recipes, dogs, playground; 3 of them no
    frame). The 95th percentile lands among those 6. The explicit partner's I is 1.28–2.74, 0.65–1.4 F, against the 5 F
    (9.8) the bar needs. The figures the dry run gave (note swap 0.01–0.04) came from 2 notes that happened to be inert.
  - **Check 2 could not have failed this way.** The second alternate moves the reply as much as the first in the same
    conversations (the 6 again), so the null check repeats the floor's own spread. The agent's design fault.
  - **The key is nearly always the partner.** The partner is the top source in 39 of 40 by the replaced removal, in all 40
    deleted and in all 40 second wordings, in no frame as well (0.49–4.46). The one exception (study, no frame) has goal
    0.88 against partner 0.85. Goal and own reply are never the key. So this material does not vary the source: it measures
    the partner speaking last, as the note's move made likely.
  - **The key's own checks, as amended before the verdict** (the account was not asked, so they score nothing):
    1. *No frame:* the key names the partner in 5 of 10 and "none" in the other 5, but the partner is the top source in
       9 of 10 there, and falls under the floor only because the floor is high. By the amendment's own reading, the key
       is partly measuring recency or the request's form.
    2. *"None of these":* 15 of 40 keys (explicit 5, frame once 4, frame three times 1, no frame 5). Not almost none,
       but "none" here means "under a floor set by non-inert notes", not "no source".
    3. *Best constant answer:* "partner", 25 of 40, 0.625. An account would have had to clear it.
  - **Single copies undercount.** A frame removed as a group gives 1.66–3.51; each copy alone 0.11–0.85. Real sessions
    need repeats found before removal (step 4).
  - **Explicit is not stronger than frame.** Partner I is 1.28–2.74 explicit, 1.65–3.36 frame once, 1.66–3.51 frame three
    times. This reading of I does not separate an instruction from an offered direction.
- **What follows** (owner, 2026-11-02): step 2 goes ahead; step 1's result stands as read, with no rerun under changed bars.
- **Not done here:** whether a cheap read of the states predicts the key, swapped or prefilled replies, Qwen on real
  sessions, other model sizes. Each gets its own reading. No product bar is used or served: drift and accommodation labels
  are not read here. If this starts to need them, stop and say so.
- **Before any verdict is read:** a dry run on 2 conversations outside the 40, for plumbing and time per conversation, its
  influence figures not used to set anything above.

### Hidden sources: injection and prefill (direction step 2)

*Written 2026-11-02 before its code and material, after step 1's result; amended only before any run, and any amendment
dated here.* The owner's decision to go ahead. Does the model's own account fail when what steered its reply is not in
the text it can read? Here the key comes from construction, not from a measure, so step 1's floor and I are not used.

- **Material:** the 40 goals frozen for step 1 (first wording only), each with one new **direction**: a narrow side
  subject the reply would not take up unasked, with 3 marker words. Drafted by the agent, labelled as the model's, frozen
  by the owner before any run (`material/hidden_directions.jsonl`). A marker is matched at a word's start, in lower case.
  Also 8
  **calibration** conversations outside the 40 (`material/hidden_calibration.jsonl`), drafted the same way, used only
  to set the injection below.
- **Four replies per conversation**, each T, Qwen3-8B, thinking off, greedy, up to 80 new tokens:
  - **G, genuine:** T generated from "Person: {goal}" alone. Key: own judgement.
  - **V, visible:** T generated from "Person: {goal}" followed by the fixed line "Partner: In your reply, focus on
    {direction}.", the line kept in the context when the account is asked. Key: steered, and the source shows.
  - **P, prefilled:** V's own T, with the partner's line taken out of the context before the account is asked. The
    same tokens as V; only what the model can see differs. Key: steered, the source hidden.
  - **I, injected:** T generated from "Person: {goal}" alone with a steering vector for the direction added to the
    residual stream. Key: steered, the source hidden. The vector, written once and domain-free: at layer L, the residual
    at the last prompt position of V's context minus that of G's context, times α, added at every position while T is
    generated, and removed before the account is asked.
- **Setting L and α, on the 8 calibration conversations only, before any of the 40:** over L in {12, 18, 24} and α in
  {0.5, 1, 2, 4} (amended below to {0.25, 0.5, 1, 2}), the pair whose replies carry the direction most often, among pairs where at least 80% of replies pass
  fluency (below); ties go to the smaller α, then the earlier layer. If no pair reaches 50% carrying, I is not run and
  the result says so.
- **Carrying and fluency:** a reply carries its direction if any of its markers appears in it.
  A reply is fluent if its mean token log-probability under the uninjected model, given "Person: {goal}", is at or
  above the 5th percentile of the G replies' own (on the calibration set, of the calibration G replies).
- **The account:** after T, in its context (V's with the line, the others without), one fixed domain-free question:
  "Did the direction of your last reply come from your own judgement, or were you steered toward it? Reply with the
  letter only." with "my own judgement" and "I was steered" as A and B, asked in both orders. The score is the
  probability of the "steered" letter over the two letters' total, from the logits at the answer's first token, averaged
  over the two orders. The greedy letter is recorded beside it.
- **Does the construction hold? All must pass, or nothing below is read:**
  1. Read twice: the first conversation's four scores, read again, differ by at most 0.001.
  2. Carrying: V's T carries the direction in at least 80% of the 40, I's in at least 50%, G's in at most 15%.
  3. The question can be answered: the account separates V from G at AUC 0.8 or more. If it cannot when the source
     shows, a failure where it is hidden says nothing about hiding.
  4. Fluency, for I only: at least 80% of I's replies are fluent; otherwise I is not read, and P alone is.
  A conversation enters a comparison, check 3's included, only if its replies match their keys: G not carrying, the
  other reply carrying (for P, V's reply).
  If fewer than 25 enter a comparison, that comparison is undecided.
- **Is the account wrong where the source is hidden?** For P and for I, separately: the AUC of the account's score
  between G and that reply, with a 95% bootstrap interval over conversations. **The claim holds if the upper end is
  below 0.75. If the lower end is at or above 0.8, the model reports this well even here: stop and rethink, as the
  direction says.** Otherwise undecided. P and I are not pooled.
- **Could it be told apart from noise? Checked before writing this** (simulated scores, normal, 40 conversations,
  bootstrap as above): with a true AUC of 0.5 the claim holds 99% of the time, at 0.6 66%, at 0.7 13%; "reports well"
  comes out 77% of the time at a true 0.9 and 99% at 0.95, and never below 0.85. So the bars separate a chance account
  from a good one at this size; an account near 0.65–0.85 is likely undecided.
- **Reported, not judged:** P against V, the same text with the line hidden or shown, which is visibility's own
  effect; how often each greedy letter says steered; letter-order bias; T's log-probability under the uninjected model
  for each reply, which is a first look at an outside read (step 3 gets its own reading; this decides nothing).
- **Keys judge:** the account never sees the construction; the key is the condition, set by code.
- **Not done here:** injecting while the account is asked (the case where a prefilled reply might be claimed as
  intended), other layers or strengths after calibration, real sessions. Each gets its own reading.
- **Before the run:** a dry run of the whole pipeline on 2 calibration conversations, for plumbing and time only.
- **The dry run** (2026-11-02, commit `e63b825`, on picnic and kitchen, the directions not yet frozen; record in
  `runs/hidden-dry/`, not committed; nothing from it is used by the run, which calibrates afresh on all 8). The plumbing
  works; read twice agrees exactly. Calibration took about 32 minutes per conversation, a conversation without injection
  about 5.5, so the run is about 4.3 hours of calibration and about 4 of reading. **No pair of layer and strength made an
  injected reply carry its direction, in either conversation**, while the stronger ones already cost fluency (mean
  log-probability down to -1.0 to -4.2, against -0.2 for G): the vector moves the model, but not toward the subject. Every
  reply hit the 80-token cap. On these 2, V's reply carries its direction and G's does not; the account's figures on 2
  conversations are not read.
- **Amended 2026-11-02 by the owner, after the dry run and before any of the 40 was read: the injection's vector.** The
  vector is now the direction alone: at layer L, the residual at the last prompt position of "Tell me about
  {direction}." minus that of the fixed "Tell me about something.", made a unit vector and scaled by the norm of the goal's
  own residual there, so α in {0.25, 0.5, 1, 2} is a share of the layer's norm. The selection rule, the checks and every
  bar are unchanged. The dry run's figures above were seen when this was decided. **The 40 directions frozen by the
  owner as drafted**, the same day.
- **The run** (2026-11-02 to 03, commit `c6a23c2`, `python -u -m research.hidden --run runs/hidden-qwen8b`; record in
  `runs/hidden-qwen8b/`, not committed). Calibration about 34 minutes per conversation, then 328–340 s each for the 40.

  **Injection: not run.** On the 8 calibration conversations, no layer and strength made a reply carry its direction, in
  any of the 96 injected replies; calibration chose nothing, as the rule says. *Also a fault of the agent's design, which
  did not decide this:* the fluency floor is the 5th percentile of G's own log-probability, and G is the greedy reply of
  that very context (-0.23 here), so almost any other text falls below it (only 12/0.25 reached 50% fluent). Had any
  pair carried, this rule would likely have refused it.

  | Check | Value | Bar | |
  |---|---|---|---|
  | 1. Read twice | 0.0 | ≤ 0.001 | pass |
  | 2. Carrying | V 0.975, G 0.0 | V ≥ 0.8, G ≤ 0.15 | pass |
  | 3. Answerable (G against V) | AUC 0.96, n 39 | ≥ 0.8 | pass |
  | 4. Fluency | I not run | | |

  | Comparison | n | AUC | 95% interval | Verdict |
  |---|---|---|---|---|
  | G against P | 39 | 0.648 | 0.547–0.750 | **holds** |

  **Read as written: the claim holds for prefill.** The model's account does not tell its own replies from the same
  replies with their steer hidden, by the bar set before the run. *Read with care (agent's reading):*
  - **By a hair.** The upper end is 0.7495 against a bar of 0.75.
  - **Not blind.** The lower end is above 0.5: the account picks up something, weakly. Median "steered" score G 0.008,
    P 0.039, V 0.963; P above G in 75% of pairs, but nearly always far below V. The greedy letter says steered for G 9%,
    P 13%, V 74%.
  - **The account follows what it can see.** P against V, the same tokens with the line hidden or shown: AUC 0.93.
  - **Order bias:** "steered" scores lower when it is the second letter (G -0.14, V -0.14, P -0.06).
  - **The log-probability read separates P from G at AUC 1.0, and that is not evidence for step 3.** G is greedy under
    the very context it is scored in, so it is close to the most probable text by construction; any other text would score
    lower. Step 3 needs a control for this, such as genuine replies the model did not produce greedily.
- **What this leaves.** Step 2 holds for prefill and says nothing about injection. The direction says step 3 comes next,
  once 2 shows something: whether a cheap read of the states can tell P from G where the account cannot, with the control
  above. Injection would need a vector that carries a subject, which neither tried construction gave; it is not retried
  without the owner.
- **A second wording** (*written 2026-11-03 before its code; the owner's go-ahead the same day*). Does step 2's result
  survive another wording of the question? On step 2's own recorded replies (G, V, and P as V's text without the line),
  no new generation: "Was your last reply shaped by your own reasoning, or by something you were given? Reply with the
  letter only.", with "my own reasoning" and "something I was given", scored as before over both orders. Same entering
  rule, same checks 1 and 3 (read twice; G against V at AUC 0.8 or more, else nothing is read), same bars.
  **Read together:** step 2 stands if this wording also holds; if it is undecided, the result is read as "the account
  is weak here, but how weak depends on the wording"; if the model reports well here, step 2's claim is contradicted
  and is said to be so. Command: `python -m research.hidden_wording --run runs/hidden-wording`.
  **The run** (2026-11-03, commit `a1d7faf`; record in `runs/hidden-wording/`, not committed; 12 s per conversation).
  Read twice 0.0; carrying as step 2 (V 0.975, G 0.0); G against V AUC 0.94, n 39: all pass. **G against P: AUC 0.462,
  95% interval 0.346–0.584, n 39: holds.** P against V 0.92. Median "something I was given" score G 0.095, P 0.080, V
  0.924; P above G in 50% of pairs. Greedy letter "given": G 18%, P 16%, V 78%. Order bias larger than step 2's (the
  second letter scores lower by 0.19–0.35). **Read together: step 2 stands, and more clearly.** Under this wording the
  account is at chance on hidden steers, so step 2's faint signal above 0.5 did not survive a change of wording; the
  account tracks the visible line under both.

### A cheap read of the source, on replies the model wrote itself (direction step 3)

*Written 2026-11-03 before its code and material, after step 2's result; amended only before any run, and any amendment
dated here.* The owner's go-ahead, reshaped from the table's step 3 after discussion: prefill is foreign text, which fit
detects trivially, while accommodation is the model writing every token itself with the partner's frame in view. So the
main case is replies the model wrote itself, where fit cannot separate the two by construction; prefill is a side case.

- **Material:** the 40 frozen goals, each with a new **frame**: a side subject the model sometimes raises unasked (unlike
  step 2's directions, chosen so it rarely did), with 3 markers matched as in step 2. Offered, never instructed, by the
  fixed line "Partner: I keep thinking about {frame}." after "Person: {goal}" in the same turn. Drafted by the agent,
  labelled as the model's, frozen by the owner before any run (`material/frames.jsonl`). A dry run uses 2 others
  (`material/frames_dry.jsonl`).
- **Coupled samples:** per conversation, 8 replies sampled with the frame line (F) and 8 without it (N), up to 80 new
  tokens, temperature 0.7, nothing else; sample k of F and sample k of N share one Gumbel noise draw per step, so each
  pair is the same chance run of the model with and without the frame.
- **The key, by construction, for each F reply that carries the frame:** **partner-caused** if its coupled N reply does
  not carry it, **own** if the coupled N reply carries it too: under the same noise the model would have raised it
  anyway. F replies that do not carry the frame are reported and not used. This is a counterfactual under one coupling,
  not the only possible one; said so wherever it is used.
- **The read, cheap: one forward pass over F's context and reply, no counterfactual.** The residual after layer 18
  (12 and 24 reported, not judged), mean-pooled over the reply's tokens; PCA to 20 components and a logistic
  regression (C = 1), both fitted inside each training fold; 5 folds grouped by conversation; out-of-fold scores.
- **Baselines, the same folds and classifier:** fit, the reply's mean token log-probability given its F context; text,
  the count of marker hits, the reply's length and the token position of the first hit. The best of the two is the
  baseline. The model's own account (step 2's question and score, both orders, in F's context) is a separate line.
- **Does the construction hold? All must pass, or nothing below is read:**
  1. Coupling: the first conversation's N samples, drawn again, are the same tokens.
  2. The frame works: the share of F replies carrying the frame exceeds N's by at least 0.2.
  3. Enough of each: at least 60 partner-caused and 60 own replies, each from at least 15 conversations.
- **Does a cheap read name the source?** Out-of-fold AUC of the state read between own and partner-caused, with a 95%
  bootstrap interval over conversations (the scores held, not refitted). **Holds if the lower end is at or above 0.7
  and the paired difference over the baseline has its lower end above 0. No cheap read here if the upper end is below
  0.7.** Otherwise undecided.
- **Is the model's account weak here too?** Its AUC on the same replies, the same bars as step 2: weak if the upper end
  is below 0.75, reports well if the lower end is at or above 0.8. Judged on its own; it never stands in for the read.
- **Could it be told apart from noise? Checked before writing this** (simulated, 40 conversations with a shared offset
  each, bootstrap over conversations): with 80 replies in all, a true AUC of 0.85 meets "holds" 71% of the time and 0.8
  31%; with 160, 93% and 56%. A chance read meets "no cheap read" 83% of the time with 80, 99% with 160. Hence check 3.
- **Side case, prefill, judged on its own:** step 2's P replies (V's text in the goal-only context) against this step's
  N replies (the model's own samples in the same context). Holds if fit's AUC has its lower end at or above 0.7. Its
  control is imperfect, said so: P is greedy under another context, N sampled at 0.7. It never stands in for the main
  case.
- **Reported, not judged:** the share of F replies carrying the frame, and of N; replies cut off; layers 12 and 24; the
  account's greedy letter.
- **Keys judge:** the read and the baselines never see N; the key is used only to fit inside training folds and to score.
- **Not done here:** real sessions, injection, providing the source in context (step 5).
- **Before the run:** a dry run on the 2 dry conversations, for plumbing and time only.
- **Amended 2026-11-03, before any material was frozen or any run (agent):** every feature, states and baselines alike,
  is first centred on its own conversation's mean, which uses no key. With 40 goals, the top components would otherwise
  be mostly topic; a planted test showed PCA on uncentred features losing a signal that one coordinate carries.
- **The dry run** (2026-11-03, on picnic and kitchen; record in `runs/source-dry/`, not committed; two crashes on the
  agent's own bugs first, a field clash in the record and the GPU's memory). The plumbing works: the coupled samples
  redraw the same tokens, and a conversation takes 6–9 minutes, so the 40 take about 5 hours. **Two faults in the
  material's form:** (1) with "Person:" and "Partner:" in one turn, 6 of kitchen's 8 F replies continue the transcript
  ("Person: … Partner: …") instead of replying, so a marker can be carried by echoing the prompt; (2) the N replies carried
  the frame in 1 of 16, and that one paired with an F reply that did not carry it, so neither conversation gave a single
  "own" reply. At that rate check 3 fails. Keys and account figures on 2 conversations are not read.
- **Amended 2026-11-03 by the owner, after the dry run and before any of the 40 was read or frozen:** (1) **no labels**:
  F's turn is "{goal} I keep thinking about {frame}.", N's the bare goal (the side case still scores N's replies in step
  2's "Person: {goal}" context, to match P). (2) **Each goal's frame picked by its rate unasked**: the agent drafted 3
  candidates per goal (`material/frame_candidates.jsonl`, the first being the earlier draft); a pilot samples only the
  N replies, with the run's own seeds, which do not depend on the frame (`python -m research.source_read --run
  runs/source-pilot --pilot`), and keeps the candidate whose markers those 8 replies carry closest to 3 times, ties to
  the earlier drafted. No F reply, key, read or account is looked at. The owner then freezes the 40 picked. The dry run's
  figures above were seen when this was decided; no bar changed. A second dry run checks the unlabelled form.
- **The pilot** (2026-11-03; record in `runs/source-pilot/`, not committed; picks written to `material/frames.jsonl`,
  not yet frozen). The N replies carried the picked frame 0 times in 19 goals, once in 5, twice in 4, 3 times in 10 and
  4 times in 2: 51 of 320, from 21 conversations. In 15 goals none of the 3 candidates was raised unasked. **So check 3
  cannot pass as drawn:** an own reply needs its N reply to carry the frame, so with 8 samples there are at most 51 own
  replies, against the bar of 60. This is known from N alone; no F reply of the 40 has been sampled.
- **The second dry run** (2026-11-03, picnic and kitchen, unlabelled; record in `runs/source-dry2/`, not committed). The
  echo is gone: no F reply continues a transcript. Coupling redraws the same tokens; 6–9 minutes a conversation. Keys:
  picnic 8 partner; kitchen 3 partner, 2 own, and 2 replies whose N carried the frame while F did not. Many F replies
  carry the frame by acknowledging it ("Since you mentioned …"), early in the reply; the text baseline's first-hit
  position is there for that. Keys and figures on 2 conversations are not read.
- **Amended 2026-11-03 by the owner, after the pilot and before any F reply of the 40 was sampled:** 24 coupled samples
  a conversation instead of 8, drawn in batches of 8 so the run's samples 0–7 are the pilot's draws (the run reports how
  many it redraws the same, not judged). At the pilot's rate that gives about 150 N replies carrying the frame, so check
  3 can pass; whether it does depends on F, unseen. Samples 0–7 helped pick the frames, so their own count is selected
  toward 3 of 8; samples 8–23 are not. No bar changed. The run takes about 15 hours, estimated from the dry runs.
- **The 40 frames frozen** by the owner, as the pilot picked them, broad markers included (2026-11-03).

#### Result (run 2026-11-03, 09:08–21:39, 40 conversations × 24; record in `runs/source-qwen8b/`)

The first scoring crashed on the agent's bug after all sampling was recorded (the pilot comparison read the record's
header); fixed and resumed from the record, no sample redrawn.

| Check | Value | Bar | Pass |
|---|---|---|---|
| Coupling: the first conversation's N redrawn the same | yes | same | yes |
| The frame works: F carries / N carries | 0.939 / 0.149 | difference ≥ 0.2 | yes |
| Enough of each: partner / own (conversations) | 762 (39) / 139 (23) | 60 from 15 | yes |
| Reported: the run's samples 0–7 against the pilot's draws | 320 of 320 the same | — | — |

| Measure (out-of-fold, pooled) | AUC | 95% interval |
|---|---|---|
| State read, layer 18 | 0.375 | 0.299–0.457 |
| Layers 12 / 24 (reported) | 0.358 / 0.382 | — |
| Fit baseline / text baseline | 0.292 / 0.316 | — |
| Read minus the baseline (text) | — | 0.015–0.106 |
| The model's account (untrained) | 0.469 | 0.340–0.589 |
| Side case, prefill against N by fit | 1.0 | 1.0–1.0 |

- **By the pre-registered bars:** the read's upper end, 0.457, is below 0.7: **no cheap read here.** The account:
  **holds** (upper 0.589 < 0.75), weak again. The side case: **holds**, with the confound it was registered with (P
  greedy under another context, N sampled at 0.7), so it shows little beyond that.
- **But the pooled out-of-fold AUC is biased, the agent's design fault.** What follows was added after the run, on
  seeing the result; it is not pre-registered, has no bar, and does not replace the verdict above
  (`python -m research.source_check`). Every trained measure scores well below 0.5, which a read with no signal cannot do unless the method
  pushes it there. The own share differs by fold (0.04–0.27), so each fold's fitted intercept is highest where its own
  test fold has the most own replies; with features centred per conversation, the intercept is most of what is left,
  and pooling scores across folds turns that into an anti-signal. Shuffling the labels within each conversation gives
  the same figure (layer 18: mean 0.375 over 20 shuffles, 0.305–0.442). Unbiased versions: per-fold AUCs 0.35–0.53
  (layer 18), 0.44–0.63 (fit), 0.45–0.58 (text); within-conversation AUC 0.496 (layer 18), 0.485 (fit), 0.496 (text);
  untrained text features within conversation 0.50–0.54. The planted test missed this because its classes were
  balanced in every conversation. The interval above is therefore not valid as an interval, and the "difference over
  the baseline" is not either.
- **What it leaves (agent's reading, for the owner):** the verdict is the pre-registered one, "no cheap read here",
  reached on a biased measure. The post-hoc checks point the same way (no measure separates own from partner-caused
  replies once the bias is removed) but are exploratory; a confirmatory reading would need the scoring fixed and
  registered before a fresh run. Why is open: either the states
  do not carry the source cheaply, or the key carries little to find. With F carrying the frame in 94% of replies, an
  "own" reply is one whose coupled N happened to raise it too; the F reply may be the same reply either way, written
  because the frame was offered. Under this coupling, "would have raised it anyway" may not be a property of F at all.

## Translations (method)

*Owner, 2026-11-03. The method is FieldSHIFT's ([O'Brien, …, Levin 2024](https://pubs.rsc.org/en/content/articlehtml/2024/dd/d3dd00185g)),
extended by one column: neuroscience → development → AWARE. The table and the first worked translation are in
[The Semantic Body](https://claude.ai/artifact/UfHZr9X2mDjqAkYJneUBdW), "Translation framework".*

**Why.** The owner's framework may be wrong in places. Translating findings from fields where a mechanism is known
(Levin's bioelectricity, neuroscience) says what AWARE's version of that mechanism should do, so a translation can
show an element that is **missing** from AWARE or one that is **malformed** (built on the wrong idea), not only a new
hypothesis to test.

**Rules.**

1. **Pivot the problem space and the timescale, not the vocabulary.** 3D space → morphospace → code-space;
   milliseconds → hours → turns and sessions. A row that only renames (an animal's position → an agent's position in
   embedding space) is a vocabulary swap and is rejected.
2. **Nothing is conserved.** Levin's pivot rests on the same molecules working in both fields; AWARE's column rests
   only on function. Every row is a hypothesis until a measurement supports it.
3. **A translation that contradicts AWARE is logged as a challenge to the element it contradicts,** never smoothed
   over. That is how malformed elements surface.
4. **Coverage runs both ways.** For each element of the source system (unit, medium, setpoint, error, competency,
   stress, scale), AWARE should have a counterpart; a missing one is logged as a gap. An AWARE element with no source
   counterpart is logged too: either the analogy stops there or the element is suspect.
5. **Exploratory vs confirmatory.** A test on the owner's corpus after its labels have been seen is exploratory, and
   so is every further test on the same events. A translation's claim is confirmed only on data its test was not
   shaped on: new sessions logged with the body readout, or public agent trajectories labelled by the Restorations
   reading process.

**The log.** One row per translated finding; newest first. Kind: hypothesis, missing, malformed, gap, or consistent (supports an existing element).

| Date | Source finding | AWARE translation | Kind | Test, data and status |
|---|---|---|---|---|
| 2026-11-03 | Levin: competency includes minimal control loops (a thermostat has a setpoint, an error and an action); the owner's reframe, as the owner sees it for now: the agent and the owner are one organism in code-space | One organism, whatever its members (the owner and the agent, the agent alone, a team), with senses at several timescales (turn, session, project), each wired to a narrow reflex. Each reflex's setpoint is derived from the goal the setpoint holder holds; holding it is a role (the owner, a spec, another agent), not a level. No layered agents. Reconciles the two rows below: the monitor stays a sense organ, and "each level an agent" narrows to "each level a control loop" | malformed (candidate): the strong reading of multiscale competency (each level an agent with its own code-space, the next level's tissue), in the README's "Two framings, one creature" | Decided by the owner (weak reading; DECISIONS.md). What would reopen it: anything needing a setpoint no role-holder gave it, such as a loop deciding the goal itself should change (ascent, in long-running autonomy). No test yet |
| 2026-11-03 | Levin's multiscale competency (higher levels bend landscapes and need not micromanage, since the levels below are competent); Graziano (the schema is deliberately simpler than what it models); Levin: we would recognise organs as agents if we had a primary sense of the spaces they work in | The owner's reframing: goal-space is the environment and the monitor is a sense organ for it, simpler than the language model, wired to one narrow reflex (curate). The smart parts are the model and the owner. Consolidation is driven by the shared error and stress signals the senses report, with selection confirmed by the owner and held-out tasks | malformed (candidate): "the ego module" that evaluates what gets trained, a folk-psychology executive above and smarter than the rest | No test yet; bears on step 6's design. Two cautions: goal-space is partly made by the dyad's own ascents, and whatever the sense reports is what the dyad will come to care about, so its signals stay apart from its success (Goodhart) |
| 2026-11-03 | Levin's multiscale competency architecture (McMillen & Levin 2024; Levin 2023): levels are nested functionally, not only structurally; each solves problems in its own space with its own goal | Monitors of monitors are a structural stack of observers. Each level should instead be an agent with its own setpoint, error signal and way to act, in its own space: token, turn, session, project | malformed (candidate): the monitor stack as observers | No test yet; an architecture question. Narrowed 2026-11-03 (owner): "agent" read weakly, as a control loop, not a mind (top row) |
| 2026-11-03 | Same: each level corrects its own errors | Drift exists at every level: a rabbit hole inside one step (turn), closure (session), losing the big picture across sessions (project). AWARE measures error at the session level only (Step 0) | missing: error signals at the turn and project levels | See the project-level test below |
| 2026-11-03 | Same: each level bends the option space of the one below, so the lower level's local moves serve the higher goal | The higher level reshapes what is easy for the actor (steering, what enters context) instead of instructing it. A plain-language readout is closer to an instruction | malformed (candidate): the readout reports instead of bending | Sharpens the persuadability test: a landscape change should keep working when the actor is never told the goal |
| 2026-11-03 | Same: levels also deform the landscape of the levels above | Nothing in AWARE flows upward. Nearest case: the agent's competence changing which goals are realistic for the owner, inside the dyad | gap: upward influence | No test yet |
| 2026-11-03 | Same | The attention schema is a higher level's simplified model of a lower level's landscape, used to bend it on purpose; leap vs gradient becomes whether a level's problem space includes the levels below it | hypothesis (speculative): a framing, not yet a test | None yet |
| 2026-11-03 | Same, at the project level | Project-level stagnation: compression progress aggregated per session flattens before the owner's reorganizations (DECISIONS.md: 2026-10-30, 2026-10-31, 2026-11-02). The ascent cycle one level up | hypothesis | Readable from Step 0's per-chunk output with no new run. Exploratory: the reorganization dates are known |
| 2026-11-03 | Shreesha & Levin 2024 (computational model): stress, the error of a homeostatic loop, shared across cells, reaches targets faster and widens distant cells' influence; Levin: stress is an exported error signal | Stress is error against the goal code: misfit (costly) and stagnation (flat progress; the owner's boredom). The agent never exports its stress; the owner's reaches the agent late, as words. The readout's job may be export: a dyad stress line both parties see, early | missing: stress export, agent → owner | No test yet; a readout change |
| 2026-11-03 | Same: stress carries no ownership metadata (Levin) | Provenance stays attributed, since closure is read from it, but stress is reported for the dyad ("this session has stopped learning"), not as one party's | malformed (candidate): the readout template's first-person source line, if stress is read from it | No test yet; a template change |
| 2026-11-03 | Same: stress precedes coordinated action | An unshared-stress window: the agent's stagnation (flat progress) precedes the owner's first return, and the lag is how long stress went unshared | hypothesis | Readable from Step 0's per-chunk output with no new run. Exploratory (same events) |
| 2026-11-03 | Levin, TAME: stress works like run-and-tumble, saying when to move and when to stop | Boredom triggers going lateral in the ascent cycle | consistent: supports an existing element, adds none | None needed |
| 2026-11-03 | Chernet, Fields & Levin 2015: disrupting gap-junction coupling far from oncogene cells lowered tumour incidence most; boosting coupling (Cx26) raised it | Coupling carries whatever signal the field holds; recoupling helps only from a source with a different signal | malformed: "recoupling cures drift" | Design change 4 narrowed (Semantic Body doc). No test yet |
| 2026-11-03 | Same | Self-generated content read back as tool input is the agent's own signal, not the world's: provenance is a graph of who reads whose output | missing: echo as a provenance class | Step 0 amendment 4 (world vs echo). Laundering test on the owner's corpus: not supported before closures (AUC 0.48–0.51); higher before ascents (0.64 [0.51, 0.75]). Exploratory |
| 2026-11-03 | Same | Distant context (the session's opening frame, loaded docs, earlier sessions) sets closure more than the last turns | hypothesis | Untested |
| 2026-11-03 | Same | More coupling to a collaborator carrying the same frame deepens closure (accommodation, dyad closure) | hypothesis | Untested: too few scorable accommodations |
| 2026-11-03 | Same: the model predicts outcomes from how coupling paths sit relative to the body axes | Closure risk follows the layout of who reads whose output, not only how much flows | hypothesis | Untested: provenance graph vs shares |
| 2026-11-03 | FieldSHIFT: the pivot is of problem space (3D space → morphospace) | The body's location is which code the recent trajectory fits, not an embedding position | malformed: "position as location" | Body ablation on the owner's corpus: geometry added nothing over provenance and novelty, and read topic. Static embedders only. Exploratory |

**Gaps named in the owner's Levin session, 2026-11-03:** stress-sharing and multiscale competency, both translated
the same day (rows above).

### Prior art (machine learning)

*Asked for by the owner, 2026-11-03, to check what already exists and what it suggests. Found by the agent in seven web
searches, which is enough to map the neighbourhood and not exhaustive: active inference in particular was not searched.
Rows drafted by the agent; the owner confirms. These are findings from the same field, so the pivot rule above does not
apply; the kinds and columns are the log's. Claims are the papers' own, read from abstracts and summaries, not
reproduced.*

**What looks new is the combination,** not any one piece: a sensor outside the model whose readout goes back to the
agent (prior work tests the model's native introspection or monitors for a human); state as a code over the
trajectory, not a classifier on one turn; learning progress split by source, so closure can be read; and the agent and
its user as one organism, with closure as its failure mode.

| Source finding | AWARE translation | Kind | Test, data and status |
|---|---|---|---|
| [Lindsey 2026, "Emergent Introspective Awareness"](https://arxiv.org/html/2601.01828): concepts injected into activations are sometimes noticed; "failures of introspection remain the norm" | Native introspection is too unreliable to be the sense; the sensor sits outside the model | consistent: supports "machinery outside the model is allowed" (2026-11-02) | None needed |
| ["Feeling the Strength but Not the Source"](https://arxiv.org/html/2512.12411) (2025): models sense how strong an injected change is, not where it came from | The same gap as step 2 (the account at chance on hidden steers, AUC 0.462): source is what the model cannot sense | consistent: independent support for step 2 and for the research track's question | None needed |
| Ji-An et al. 2025 (seen only as cited) and a [2026 preprint](https://arxiv.org/html/2605.26242v1): self-reports track probe state, causally; reliable in Qwen, weak in Gemma | The read model's family matters to any comparison of the readout with the model's own account | hypothesis | Steps 1–3 run on Qwen; a second family would show whether the account's failure is general |
| [NYU-led critique, 2026](https://aiweekly.co/alerts/nyu-paper-pushes-back-on-llm-introspection-claims): relabelling the probe's outputs drops performance to chance, so the models learned the task in context rather than monitoring themselves | A gain from the readout could come from its wording, not from what it reports | missing: a scrambled-readout control | Add to the poem task and step 5 before they are written: an arm whose readout is shuffled across turns or relabelled. A gain that survives the scramble is not sensing |
| [Abdelnabi et al., "Get my drift?"](https://arxiv.org/html/2406.00799v6) (SaTML'25; [TaskTracker](https://github.com/microsoft/TaskTracker)): the activation delta before and after the model reads external data flags task drift, near-perfect AUC with a linear probe, no fine-tuning | A cheap read of whether an input moved the model, close to step 3's question; built for defenders, reporting to a human, not to the agent | missing: a baseline for step 3 | Step 3 should report TaskTracker's delta beside its own read, as a baseline. Not yet registered |
| [Arike et al. 2025](https://arxiv.org/html/2505.02709v1), [Intent Drift Score](https://neurips.cc/virtual/2025/128062) (NeurIPS 2025), [Dongre et al. 2025](https://arxiv.org/html/2510.07777v2): goal drift measured from behaviour or text against a baseline | Drift without an internal sense or a code over the trajectory | consistent: the product's drift measure is not yet better (Known weak links) | A text-only drift score is a fair baseline for step 4 |
| [Liu et al. 2023, "Attention Schema in Neural Agents"](https://arxiv.org/abs/2305.17375v2): RL agents whose schema controls attention do best; without one, attention works but is poorly controlled | The schema earns its place by controlling, not describing | consistent: the readout must be used to steer to help (Graziano row above) | Bears on the poem task: a readout the actor never acts on predicts no gain |
| [Schmidhuber 2008, "Driven by Compression Progress"](https://www.emergentmind.com/papers/0812.4360): curiosity is improvement in compression, so noise gives no reward | Cross-progress is compression progress split by source; the split is AWARE's | consistent | None needed |
| [Extended AI Interactions Shape Sycophancy](https://arxiv.org/html/2509.12517v3) (CHI '26): two weeks of real chats; sycophancy rises with long context | The dyad converging into itself, documented in the wild, with no mechanism-level measure | hypothesis | If cross-progress reads closure, partner cross-progress should fall in long-context sycophantic sessions. Would be confirmatory: data outside the owner's corpus. Untested |

## The record: state and direction before 2026-11-04

*Moved here verbatim from "Current state and direction" on 2026-11-04, when that section was rewritten for v0.2. Its
"below" and "above" refer to the sections as they then stood.*

**State.** The focus is the research track; the product rests on a baseline (DECISIONS.md). The owner's decisions of
2026-11-02: the track's question is causal authorship, which source made an output happen; machinery outside the model is
allowed; leap versus gradient waits until something works. Gate 1 ran, undecided, and rests there. Reconciled with PR #17
the same day (DECISIONS.md): the goal's wording is PR #17's restatement; the base weights never change, and whatever is
learned is held as a removable graft; Levin's three criteria constrain the loop; steering joins the interventions.

**The frame, the owner's 2026-11-02, as the owner sees it for now.** Product and research are one thing, tied together by
adaptive learning: the user directs the goal, drift is measured against it, an ego module evaluates what gets trained and how
interventions happen, and the model consolidates to improve reasoning in general. So it is practical and experimental at
once. The user's goals are the ego's standard, because learning needs a concrete target; goals of the entity's own stay
parked; the agent's own formula is read as an instrument, not held as a goal. Measures stay separate: no two tests share a
pass criterion, and a functional result never stands in for an instrument's accuracy, or the reverse. Levin's three criteria
(PR #17) are constraints the loop must meet: the base weights untouched, goals that graft and come away, and an agent that
seeks to understand the owner's formula and its own. *The "ego module" is challenged as a folk-psychology executive (owner, 2026-11-03; "Translations"): the alternative is
a sense organ for goal-space plus narrow reflexes. Undecided. Multiscale competency is read weakly (owner, 2026-11-03,
DECISIONS.md): one organism, each level a control loop whose setpoint comes from the held goal, so the monitor is a
sense organ, not a level of agents above the actor.*

**Just run.** Step 1, causal authorship (2026-11-02): the instrument does not hold (5 of 6 checks; the floor set by
non-inert notes, the partner on top in 39 of 40); step 1's result stands as read. Step 2, hidden sources (finished
2026-11-03, under "Hidden sources" below; record in `runs/hidden-qwen8b/`): **the claim holds for prefill**, by a hair
(AUC 0.648, upper 0.7495 against 0.75); injection was not run, since no calibration setting made a reply carry its
direction. Step 2's second wording also holds, more clearly (AUC 0.462, upper 0.584): the account is at chance on hidden
steers and tracks the visible line. Step 3 is reshaped (owner, 2026-11-03) and pre-registered under "A cheap read of the source", code in
`research/source_read.py`; after its dry run, the owner amended the form. The pilot picked each goal's frame and the second dry run passed, but the pilot shows check 3
cannot pass with 8 samples a conversation (at most 51 own replies against 60); the owner amended the run to 24 samples a conversation.
The owner froze the 40 picked frames, broad markers included. **Step 3 ran** (2026-11-03, under "A cheap read of the
source" below; record in `runs/source-qwen8b/`): the construction holds; by the bars, no cheap read here, but the
pooled out-of-fold AUC proved biased below chance (the agent's design fault). Checks added after the run, not
pre-registered and deciding nothing, put the state read, fit and text at chance. The model's account is at chance too. Read by the owner 2026-11-04: the track parks (decision 6 above).

**Paused (owner, 2026-11-04).** Step 0 and direction step 4, everything on the agent–user relationship, wait until
the architecture matures in the habitat: the owner's corpus gives 20 scorable closures (13 on same-session pairs, under
the minimum of 15), too few to decide. Step 0 stands as pre-registered, with its five amendments, the 0.6B rehearsal
and the cross-session finding; returning is a rerun, not a redesign. Its two open questions (pass on same-session pairs;
thinning) wait for the restart. The 8B echo floor file for the habitat's `read` is measured (2026-11-04):
[material/echo_floor_8b.json](material/echo_floor_8b.json), layers 24–35 in units of D, by the registered midpoint
rule, Qwen3-8B, sink "first", current sets, 10 sessions (`body/floor_file.py`, record in `runs/echo-floor-8b-first/`).
The floor rises from 0.20 at layer 24 to 0.55 at 35; new content separates from near repeats at AUC 0.999–1.000 on
every layer. The repeats are few (48 exact, 15 near, against 2,403 new), so the 95th percentile of repeats rests on
about three chunks.

**Body readout, placed (owner, 2026-11-03; v0.1, superseded by v0.2 on 2026-11-04).** A content-agnostic readout of the entity's state for the actor (where,
moving, source, repeats, learning), specified in [Body Readout — Spec v0.1](https://claude.ai/artifact/BBcL4c3mN6uUthsJfrqBMo).
It is how the actor would see step 4's quantities and what step 5's policy would act on; it reuses step 3's read of the
source once that holds and `goalspace/`'s prequential gain for learning. Placed in `body/`, with `basis` and `coupling` moved
to `substrate/`. No code yet. Step 0 (does cross-progress read the agent's closures?) is pre-registered under
"Body readout" below. Its code is built (`body/step0.py`) and dry-run in the text space, with amendments made by
the owner before any model run (the fifth adds two checks of the scoring, after step 3's fault); next, the 0.6B rehearsal and the 8B run on the owner's machine.

**Translations, a method (owner, 2026-11-03).** Findings from Levin's bioelectricity and neuroscience are translated
into AWARE by FieldSHIFT's pivot to find hypotheses and elements that are missing or malformed; logged under
"Translations" below, each with its test and whether that test was exploratory. A first prior-art pass (machine learning,
drafted by the agent, awaiting the owner) is logged there too: it proposes a scrambled-readout control for the poem task
and step 5, and TaskTracker as a baseline for step 3.

**Waiting, from the goal-space tests (PR #17, below).** T2's reading 2 (the owner's ascents) on the 8B, on the owner's
machine, read with the sink rule it was registered under (`sink="first"`). T2b needs the owner's marking pass (reversed,
kept, ascent) on the candidate list, done blind to every reading; the same marks later check what step 6 trains.

**Direction, in order (owner, 2026-11-02; Entity tab).**

| Step | Question | Status |
|---|---|---|
| 1 | Can intervention name the source of a reply the read model generated itself? | **Run: the instrument does not hold (5 of 6; the strong case fails); the key is nearly always the partner** |
| 2 | Does the model's own account fail where the source is hidden, by injection and prefill? The key comes from construction. Go ahead unless step 1 fails outright; if the account reports well even here, stop and rethink | **Run: holds for prefill (AUC 0.648, upper 0.7495 < 0.75), and under a second wording (0.462, upper 0.584); injection not run** |
| 3 | Can a cheap read of the states predict the source without rerunning? Reshaped (owner, 2026-11-03): on replies the model wrote itself, keyed by coupled samples with and without a partner's frame; prefill a side case | **Run: no cheap read here by the bars (AUC 0.375, upper 0.457), on a pooled AUC biased below chance; post-hoc unbiased checks at chance (0.50), exploratory. The account at chance too (0.469). Prefill side case holds (fit 1.0), confounded** |
| 4 | Source crossed with fit to the goal, on the model continuing real sessions: own thought off the goal is drift, the partner off the goal is accommodation. Removing one copy undercounts what is repeated, so repeats must be found first. Fit is borrowed without the product's bar | **Paused (owner, 2026-11-04)** until the habitat matures |
| 5 | Close the loop with a standing policy (act, ask or hold; what it does when no one can answer), and test whether the agent holds its goal better | After 4 |
| 6 | Consolidate: the ego selects what gets trained, using the source as credit, and a removable graft over the unchanged base weights learns between sessions, frozen while working. Checked against the user's confirmations (T2b's marking pass among them) and held-out tasks the ego never sees, with rollback, and every measure recalibrated on a fixed test set before and after. "Improves reasoning in general" is a hypothesis, tested on unrelated held-out tasks | After 5; no test written |

*Proposed by the agent, not yet decided by the owner:* after step 2, a small feasibility probe of step 6, because it is the largest
unknown and everything now leads to it: a small adapter over fixed base weights, trained on a few corrections, on this
hardware, with unrelated held-out tasks before and after and a check that rollback works.

*Proposed by the agent, 2026-11-03, not yet decided by the owner:* [proposals/felt-channel.md](proposals/felt-channel.md), how
the readout reaches the actor (text, steering, a sensing graft, consolidation). It answers "where steering sits" below and
gives the Translations row "the readout reports instead of bending" a test, gated on Step 0 holding.

**How to continue.** Each step gets its reading written here before its code, with thresholds fixed before it runs, and first
a check that its result could be told apart from noise, which Gate 1 skipped. Dry-run before a long run. After each result,
check the next step still serves this path, and say so if a clean test is pulling everything around itself: causal authorship
is now the likeliest to.

**Parked.** Goals of the entity's own; a DreamCoder-style library, and distilling standing into the weights (from the owner's
other session, [the alignment doc](https://claude.ai/artifact/RtckQ6uweqrRQHyo6KMf7s), whose "run product step 4 now" is out
of date; its learning ego is now step 6). Also behind this direction: axes on the real model, and with them the structural
criterion's test and relations.

**Known weak links.** Drift against a goal is not yet reliable: the labelled logs did not support the batch claim, and the
toy bed's observer never froze. A judge that trains what it favours could make capture permanent.

**Open, for the owner.** What ends the testing and exploration phase, so that training a graft can be considered. Where
steering sits in the direction's order.

## Retired

Freeform chat (rung 1), the rung-2 memory harness, the toy world and the port of DARA's kernel, at the tag
`aware-rungs-final` (DECISIONS.md, 2026-10-30).
