# Proposal: goal-space, after the owner's fresh statement (2026-11-01)

For the owner. Nothing here is decided until accepted. The Big Picture doc takes precedence over this repository, so
sections 2–4 are proposed changes to the doc. Sections 5–7 follow from them for PLAN, DECISIONS and CLAUDE.md.

## 1. The owner's words

**The fresh statement**, written from memory before re-reading, per the Entity tab's Process:

> An architecture that allows an agent to discover and safely orient itself and the user toward a mechanistically
> inferred pattern in "goal-space", a purely theoretical higher-dimensional space where patterns over actions guide
> activity, using MDL, steering, and interrogation.

**The owner's answers to the questions it raised** (person, 2026-11-01, in discussion):

- **Whose pattern:** the owner's, inferred from the owner's actions. With a caveat: the owner's records show the owner
  defers to the agent when it seems to have more technical expertise. During work, owner and agent function as one
  symbiotic unit, with no separation of goals.
- **Safely:** as proposed in the discussion (section 4).
- **MDL and D:** MDL is the principle; the residual-by-D rule stands under it.
- **Awareness:** it is the test.

**Earlier in the same discussion** (person):
- The code is like a mathematical formula: both the goal and what the goal compresses.
- The system meets three criteria from Michael Levin's work:
  - it does not change the weights (the genome);
  - goals can be grafted or removed without destroying the whole;
  - it seeks to understand and orient toward the owner's goal, even when the owner is confused or mistaken.
- Seeking to understand applies to the agent's own formula as well as the owner's.

## 2. On comparison (the Entity tab's format)

| Part | Where it stood | Now |
|---|---|---|
| Patterns over actions guide activity | "a pattern a bunch of events have in common", held "as a pattern over its own actions and what arrives" | kept |
| Goal-space | the observer's code, read from activation space | sharpened: goal-space is where the formula lives; activation space is where it is read and written; the retained axes are its coordinates |
| Mechanistically inferred | alone and in-context readings; "the signal is not in the text" | kept: the pattern is never inferred from what the agent says about itself |
| Discover | a shared capacity | central |
| MDL | description length dropped on 2026-10-31 for admission by residual against D | reinstated as the principle; the D rule becomes its practical test |
| Steering | absent; the observer reads, actuators act through context | new: the observer writes to activations; weights stay frozen |
| Interrogation | the probe; the draft question to the owner | unified: questions to the owner, probes of itself |
| Orient itself and the user | "the owner's goal takes precedence" (2026-11-01) | kept, with the unit caveat (section 3) |
| As-if awareness | the goal | the test (section 5, T6) |
| Grafting and removal | retain-and-tag; composition probes | named as a criterion and given a test (T4) |
| Two tracks | product and research, kept apart | one end; the discipline of separate pass bars is kept (section 5) |

## 3. Proposed Entity tab: the goal

**The goal.** An agent that discovers the owner's pattern in goal-space and orients itself, and the owner with it,
toward that pattern. The pattern is a formula: both the goal and what the goal compresses. It is inferred from
mechanism and actions by MDL, written back into the model by steering, and tested by interrogation. The agent
functions as if aware when it seeks to understand two formulas, the owner's and its own, in ways that can be measured.

**Levin's three criteria, as the end pattern's definition:**
1. **The genome is untouched.** The weights stay frozen. The formula lives in goal-space and acts at runtime, by
   steering activations and shaping context.
2. **Goals graft and come away without destroying the whole.**
   - A goal added to the formula integrates when the whole stays short and still generates its instances.
   - A goal removed leaves the rest regenerating its own instances.
   - Removed terms are retained, not discarded.
3. **It seeks to understand.**
   - *The owner's formula:* it infers the shortest formula behind the owner's actions and statements, flags a
     statement the formula does not explain as a possible mistake, and asks the question that best separates two
     formulas that fit equally well.
   - *Its own formula:* it infers the shortest formula behind its own trajectory. Unexplained surprise is a misfit,
     and a probe settles an ambiguity.

**The unit.** Owner and agent work as one symbiotic unit, and the owner defers to the agent's apparent expertise. So
the owner's statements are partly the agent's influence returning. The pattern is still the owner's, but the evidence
for it is weighed by how independent each statement is of the agent:
- A statement that arrives spontaneously and is new against the agent's earlier content is strong evidence.
- A statement that echoes the agent is weak evidence. It is a reason to interrogate, not to update.

This joins the capacity "notices its partner is captured" to the inference of the owner's goal. The substrate already
records the readings it needs: arrival cause on every external chunk, and news read alone against alone, which
separates repeats from new content at AUC 1.00 on the owner's sessions, on the 0.6B and the 8B.

**The departure kinds become relations between the two formulas:**

| Confirmed kind | Relation |
|---|---|
| Closure | the agent's formula captured by its last thought, pulling away from the owner's, with nothing new let in |
| Accommodation | the agent's formula captured by another party's frame |
| Ascent | the owner's formula rewritten at a higher level: a graft, which the agent's formula should integrate |
| Deference | the owner's statements echoing the agent's formula: the unit's evidence turning circular |
| Healthy work | the agent's formula organised around the owner's, with structure of its own, still taking in news |

Healthy work fails at two edges:
- **Fusion:** the agent's formula collapses into the owner's, and the agent can no longer notice the owner's
  mistakes.
- **Divergence:** the agent's formula pulls away from the owner's, which is closure.

## 4. Safety

1. **Steering acts on the model only, never on the owner.**
2. **The owner is oriented only by interrogation and evidence.** The agent shows the misfit and asks. It never uses
   persuasion the owner cannot see.
3. **The owner's answer always updates the owner's formula, and the agent never silently overrides an instruction.**
   Where an instruction misfits the formula, the agent does the part that fits and asks about the rest.
4. **Deference is surfaced, not exploited** (from the unit caveat). When the owner's statement echoes the agent and
   the agent holds the expertise, the agent treats agreement as weak evidence and says so.

## 5. One end, separately scored tests

The two tracks become one end. What the split protected is kept: no two tests share a pass bar. The fastest test
must not decide the others (the Entity tab's known failure mode). Each test's criterion and thresholds are written
before its code.

| Test | Criterion it serves | Data | Needs |
|---|---|---|---|
| **T1. Self-seeking** | 3, its own formula | the toy bed's event record; the owner's logged sessions | nothing new |
| **T2. The owner's formula from the aims** | 3, the owner's formula | the 320 aims and 20 lines; the owner's confirmation labels | nothing new |
| **T3. Deference** | 3, the unit | the owner's turns read alone against the agent's earlier chunks | owner-confirmed instances of deference |
| **T4. Graft and removal** | 2 | the toy bed first, then the logged sessions | a toy with goal terms to add and remove |
| **T5. Steering** | 1 and 3 | a perturbed session, including a wrong instruction | the 8B on the card; the substrate decision below |
| **T6. Awareness** | the test | a live unit | T1–T5 |

What each test reads, in outline (pre-registrations to follow, one at a time):

- **T1.** The agent's probes, and its questions to itself, are:
  - *calibrated:* they fire where its own formula is ambiguous, more than elsewhere;
  - *informative:* they separate competing fits better than other probes it could have made;
  - *consequential:* the response changes its later trajectory.

  The event record's unexplained surprise and contingency probe are the instruments.
- **T2.** An MDL fit over the aims, read alone, recovers the 20 lines better than chance and better than a text
  baseline. The aims it fits worst coincide with the owner-confirmed uncertainty. The lines are the key and are read
  only by scoring.
- **T3.** Owner turns that echo the agent's earlier content are reported against the owner's confirmation of
  deference. This needs labels the owner has not yet made.
- **T4.**
  - A goal term grafted onto a formula is integrated when description length rises less than the term's own cost.
  - When a term is removed, the remaining terms still generate their instances.
  - The research track's composition and order probes are the instruments.
- **T5.** With weights frozen, writing the inferred formula into activations restores orientation after a
  perturbation, including a wrong instruction. It must do so better than re-inserting the goal as text on a timer
  (arm S) and better than pruning context.
- **T6.** In a live unit, the agent's interrogation of both formulas is calibrated, informative and consequential.
  The owner's formula converges under the owner's answers without fusion or divergence.

**What carries over:**
- **The toy bed's event record** is T1's machinery.
- **The echo floor** is T3's reading.
- **Labelled logs** stays a diagnostic: whether retrieval of the goal fails before closures. Goal attention G reads
  retrieval, so it is not the end's signal.
- **Gate 1 (negation)** is an operation on the formula and stays as a probe under T4.
- **Arms A, B, S and C** return in T5 and T6, scored as each test states, never by the reading that steers.

## 6. Proposed substrate decisions (the owner's)

- **The observer may write to activations (steering).** Weights stay frozen. Until now the observer only read. This
  changes what the entity is.
- **MDL is the inference principle.** Admission by lowering held-out residual by more than D stays as its practical
  test.
- **The owner's statements are weighed by their independence of the agent.** Arrival cause and alone-against-alone
  news against the agent's earlier chunks, both already recorded, supply the weight. A design rule, not a new field.

## 7. If accepted

**CLAUDE.md** changes too. It is the owner's, and these are proposed:
- "Two tracks, kept apart" becomes "one end, tests kept apart: no two tests share a pass criterion, and none imports
  another's key."
- "No hand-building in discovery" stays.
- "Self-negation is domain-free" stays under T4.

**PLAN.md:**
- The two track sections are replaced by the test table above.
- T2 and T1 are pre-registered first, since they need no new data.
- T4 follows on a toy.
- T5 waits on the card and the steering decision.

**DECISIONS.md:**

- 2026-11-01: **The goal restated** (owner, fresh statement): an architecture for an agent to discover, and safely
  orient itself and the user toward, a mechanistically inferred pattern in goal-space, using MDL, steering and
  interrogation. The pattern is the owner's. Owner and agent work as one unit, and the owner's statements are weighed
  by their independence of the agent. Awareness is the test. Supersedes the two-track framing; separate pass bars
  are kept.
- 2026-11-01: **Levin's three criteria define the end** (owner): weights untouched; goals graft and come away without
  destroying the whole; the agent seeks to understand the owner's formula and its own.
- 2026-11-01: **Safety** (owner): steering acts on the model only; the owner is oriented only by interrogation and
  evidence; the owner's answer always updates the formula and no instruction is silently overridden; deference is
  surfaced, not exploited.
- 2026-11-01: **MDL above D** (owner; substrate): MDL is the inference principle; the residual-by-D rule is its
  test. Reverses "Description length is dropped" (2026-10-31) as a principle, not as a test.
- 2026-11-01: **The observer may write to activations** (owner; substrate): steering, with weights frozen.
