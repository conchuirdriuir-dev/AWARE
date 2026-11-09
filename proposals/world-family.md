# Proposal: a family of worlds, with Fault Garage as the first

**Accepted by the owner, 2026-11-09** (DECISIONS.md): all five of section 8's questions, as recommended. The reach
bar of section 6a is the agent's draft, for the owner to fix before the measurement runs.

Drafted by the agent, 2026-11-09, from the owner's direction of
2026-11-08 ("bespoke worlds with more granular control, possibly under different paradigms, that more specifically
test different components; Fault Garage can act as one of those worlds"). It takes up
[habitat-engine.md](habitat-engine.md) (merged as a proposal in PR #38, no decision recorded), whose option B, an engine of our own, it widens into a family.
Fault Garage's design is [Fault Garage — Habitat Spec v0.1](https://claude.ai/artifact/R1tVctU5dJnvFk3qinsrLq).

## 1. Why

**A failure on TextWorld cannot be traced to its cause.** When one model acts and perceives, a failure could be
the sense (mismeasuring), the player (producing nothing worth sensing), the world (too hard) or the graft (changing the
player). Step 2's two 0.6B rehearsals took two runs and a collapse check to find that the player was the cause
(PLAN, "Step 2: the code, and what building it found"). The engine's one label, a new state, measured it:

| Player | Worlds | Turns reaching a new state |
|---|---|---|
| Qwen3-0.6B | TextWorld, Stage 1 and Stage 2 | 1.1–1.9% |
| Qwen3-4B | Stage 2 | 4.1–5.4% |
| Qwen3-8B | Stage 1 | 9.9–17.2% |
| The script | Stage 1, Step −1C | 26–29% |

TextWorld exposes a few knobs (rooms, objects, quest length). It does not let us set how surprising a reply is, whether
a stall ends, whether an event was self-caused, or where information lives (words or numbers).

**What worked was control.** Boundary passed and rate was calibrated where a script made the events known (Stage 1).
Fault Garage adds what no TextWorld run had: an oracle that holds the exact posterior over faults and labels every turn
in bits, independently of the player and the senses. With it, "the player is stuck", "the sense registered it" and
"the world was hard" are separate questions.

## 2. The contract every world meets

From habitat-engine.md's requirements, with Fault Garage's practice added:

1. **Worlds from seeds**, rebuilt exactly; the same actions give the same text.
2. **An oracle's labels**, read by scoring only and never by any arm or sense (the "one rule" of Fault Garage's spec:
   the organism must not learn to feel the oracle instead of the world). At least: progress (new state, or
   information gain), surprise, self-caused events, closure and ascent.
3. **Controls as data:** every dial a world exposes is a setting recorded on every episode, never a code edit.
4. **Text under our control:** templates and phrasings as data, so wording can be varied apart from content.
5. **Scripted players**, at least a greedy one (the oracle's best action) and a random one, beside model players.
6. **A stated goal** by the environment (decision 5).
7. **Scale:** an engine step costs far less than the reader's forward pass.
8. **Its own repository**, importing nothing from AWARE; only records cross, each stamped with the world's commit and
   the reader's commit and model revision (boundary's Fault Garage reading recorded neither: DECISIONS, 2026-11-08).

## 3. Components and the controls they need

What each component of the North star needs from a world, and where Fault Garage stands. *Has*: built. *Planned*: in
Fault Garage's spec, not built. *Gap*: needs a change Fault Garage's spec does not name, or another world.

| Component | The control it needs | Fault Garage |
|---|---|---|
| Boundary (news) | Novelty in the **wording** set apart from novelty in the **content** | **Gap.** Templates make surprising replies look like earlier ones (the 0.212 reversal); phrasings as data with a wording dial would close it |
| Boundary, efference copy (the tickle test) | Self-caused events against external ones | **Has:** `unplug` sets self-caused codes, flagged by the oracle |
| Rate (X_c) | Learnable, contingent structure; stalls that end | **Has:** information gain by source; episodes end in a diagnosis or the cap |
| Closure and ascent | Labels of epistemic progress | **Has:** closure as near-zero gain while the posterior is wide; oracle ascent; agent ascent from the fork probe |
| A closure trap | A misleading prior that keeps the obvious hypothesis tested | **Has:** the misleading-prior fault class |
| A rule change | The world contradicting a settled reading | **Has:** intermittent faults (cold, bumps). **Planned:** wrong wiring pages |
| Numbers | Information carried in values, not words | **Has** the world; the senses cannot see it (rewording moves the 8B's state 10–180 times further than 1 V). The readings sense is **planned** |
| The partner channel | A partner true about what it saw, wrong about what it means | **Has:** the customer filter with its vagueness dial |
| Babbling | **Reach set to the player:** play neither all closure nor trivial | **Partly.** Three levels and the load line in the prompt; the reach of a model player is not yet measured (the rate test's player screen measures the 4B's and 1.7B's) |
| Dream: the library of codes | Patterns that compress many worlds | **Has:** shared-cause faults ("one ground feeds these modules"), fresh wiring per world |
| Transfer | A world not built for the sense | **Planned:** real garage cases in the same schema; TextWorld for senses checked on Fault Garage, and the reverse |

So Fault Garage covers most of the table. Its two gaps that matter now are wording control (for boundary) and a
measured reach for the 8B (for babbling).

## 4. The transfer rule (the guard on shaping)

A world built for a component can be built so that the component passes: habitat-engine.md's risk B, and close to
CLAUDE.md's known failure mode. Held three ways:

- **Fixed before reading.** A world, its labels, its controls' settings for a pool and every bar are committed before
  any sense reads that pool. Nothing in a world is tuned to a reading.
- **No hand-building in discovery.** A world may plant an event; it may not encode a sense's computation.
- **Certification is per world; generality needs transfer.** A sense that passes in a world built for it is accurate
  *in that world*. The claim that it is accurate in general needs it to read a world not built for that sense: for
  senses checked on Fault Garage, TextWorld and later real garage cases; for senses checked on TextWorld, Fault Garage.
  Each transfer reading has its own bars, written before it runs. Measures stay apart: a functional result in a world
  never stands in for a sense's accuracy there.

## 5. How the family grows

- **Fault Garage is world one.** Its own repository, decisions and build order stand.
- **TextWorld stays** for every registered pool already played on it (Stages 1 and 2, Step −1C, rate's calibration),
  and as a transfer world. Nothing registered moves.
- **A new world only when a component needs a control Fault Garage cannot give,** argued in a proposal of its own
  against section 3's table. The likeliest first candidate is none: wording control fits in Fault Garage (phrasings as
  data), and reach is a dial there (levels, prompt, the customer's vagueness). The agent would not build a second world
  before Fault Garage's gaps are closed or shown not to close.
- **A paradigm, not only a domain.** Spec v0.2's paradigm table (self-talk, pure reasoning, tasks with verifiers, text
  games, two copies) is the menu a later world is chosen from. Fault Garage is a task with a verifier.

## 6. What it changes next

- **Babbling's world is chosen by measured reach,** before Step 2 runs on the 8B: the new-state share on TextWorld, or
  the gain per turn on Fault Garage, of the 8B as the player, against the base model's and the script's. The agent's
  proposal: measure the 8B's reach on Fault Garage level 1–2 with the load line (20 mirror scenarios each; it played
  20 level-1 scenarios without the line and gained 4.2 bits in 367 steps) and on TextWorld Stage 1's generator; choose
  the world where it reaches most, with the bar for "enough" fixed before the measurement.
- **Step 2's collapse criterion becomes reach against the base player** in the chosen world (PLAN, the re-run's
  correction).
- **Fault Garage's wording gap** becomes an item in its build order (phrasings as data, a wording dial), in its own
  repository, before boundary is read there again.

## 6a. The reach measurement (the agent's draft, for the owner to fix before it runs)

- **What it decides:** babbling's world only. No sense is read; nothing here is evidence about a sense.
- **The player:** Qwen3-8B, thinking off, as Step 2's player would be, graft off. *Where it runs, for the owner:*
  in bf16 on the 3090 or a rented GPU, or now in 4-bit (NF4) on the 4060, as the 8B's earlier TextWorld plays here
  were (`runs/step0h-8b-nf4-play`, `runs/stepm1-8b-nf4-play`). NF4 is a rehearsal setting in AWARE; the agent would
  accept it for this choice only, since it decides a world, not a reading.
- **The worlds (agent):** TextWorld with Stage 1's generator (`material/habitat_worlds.json`), on mirror seeds
  910600–910619, beyond every rehearsal's; Fault Garage at level 1 and at level 2, with the load line in the prompt,
  20 scenarios each, on seeds 4000–4019 (level 1) and 4020–4039 (level 2), registered in Fault Garage first. Each
  world's own episode cap (TextWorld 100 turns; Fault Garage 100 accepted actions).
- **One measure across worlds, from each world's own label:** a *progress turn* is a turn whose reply reaches a new
  state (TextWorld) or gains at least 0.05 bits (Fault Garage's `NO_CHANGE_BITS`). A *stall* is a run of at least 5
  turns without progress (agent); it *ends* if a progress turn follows it in the episode.
- **Enough reach (agent):** the progress share is between 10% and 50% of turns (at least as much as the 8B's low end on
  Stage 1, at most well short of always progressing), and at least a quarter of episodes hold a stall that ends. Both
  stuck stretches and stretches that come unstuck are what babbling learns from.
- **The choice:** among the worlds meeting the bar, the one with the larger share of episodes holding a stall that
  ends; reported for all: the progress share, stalls per episode, the share that end, and (Fault Garage) correct
  diagnoses. If none meets it, the owner is asked; the bar is not moved.

## 7. Risks

- **More worlds, more to validate.** Each world is an engine, an oracle, labels and scripted players. Held by section
  5's rule: one world until a component needs another.
- **Worlds shaped for the senses.** Held by section 4.
- **The application pulls the research.** Fault Garage is also an application probe (the garage log). Its research
  pools stay registered and fixed as any other; application work stays in its repository.
- **Numbers.** Much of Fault Garage's information is in readings the current senses cannot see; until the readings
  sense passes its own test, a sense's failure there may mean blindness to numbers, not inaccuracy. Readings and words
  are scored apart, as boundary's first reading already did.

## 8. Questions for the owner

- **Q1.** The family as framed: one contract, Fault Garage as world one, a new world only on a component's need?
- **Q2.** The transfer rule of section 4, written into PLAN as a rule beside "measures kept apart"?
- **Q3.** Babbling's world chosen by measured reach (section 6), with the 8B measured on Fault Garage levels 1–2 and on
  TextWorld Stage 1 before Step 2 runs? And what counts as enough reach (the agent would draft a bar for the owner)?
- **Q4.** habitat-engine.md (merged in PR #38 with no decision): marked as taken up here, or kept for an own
  TextWorld-like engine?
- **Q5.** Fault Garage's wording gap: into its build order now, before boundary's next reading there?
