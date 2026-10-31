# Decisions

Dated, one line each, newest first. The owner decides; an agent proposes.

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
  the first run fails three of four bars, each failure on an item v9's tension log already holds. Not retuned.

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
