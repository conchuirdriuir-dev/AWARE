# Decisions

Dated, one line each, newest first. The owner decides; an agent proposes.

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
