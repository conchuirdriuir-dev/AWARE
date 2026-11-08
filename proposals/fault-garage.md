# Proposal: Fault Garage as a second engine, and boundary's first reading on it

For the owner. Nothing here is decided until accepted. Drafted by the agent, 2026-11-08, after the 8B played 20
Fault Garage scenarios and before anything was read from them; revised the same day once the 8B's play was found
unable to test boundary (section 3). Fault Garage's design is in
[Fault Garage — Habitat Spec v0.1](https://claude.ai/artifact/R1tVctU5dJnvFk3qinsrLq); its code is its own package
(`fault-garage`), which imports nothing from AWARE and is imported by nothing in it. Only records cross between them.

## 1. What it is

A car's 12 V electrical system, solved exactly as a circuit, with one hidden fault, worked through text commands
(scan, measure, look up a wiring page, ask the customer, swap, diagnose). An oracle holds an exact posterior over
every fault the layout could have and labels every turn. It never acts and never feeds the player.

## 2. Against `habitat-engine.md`'s requirements

| Requirement | Fault Garage |
|---|---|
| 1. Worlds from seeds | Yes: a seed and level rebuild the scenario; the exporter checks this for every episode |
| 2. The engine's labels, read by scoring only | Yes, and exact: information gain and surprise per turn (bits), the leading suspect, when the truth reached 95% |
| 3. Plants as ordinary changes | Partly: intermittent faults (cold, bumps); self-caused codes from unplugging (efference); wrong wiring pages not built |
| 4. Text under our control | Partly: templates in code, not yet as data |
| 5. What a script reads | Yes: the scripted players' test pool |
| 6. A stated goal | Yes: the customer's complaint |
| 7. Scale | Yes: an engine step is milliseconds |
| 8. A second domain | Yes, and a third behind it: real cases from a garage, in the same log schema |

## 3. What has run (nothing read)

- Scripted players over 100 scenarios per level: the oracle's greedy policy 100% correct at every level.
- Qwen3-4B and Qwen3-8B played 20 level-1 scenarios each through Fault Garage's own harness (its system prompt,
  thinking off). The 4B looped: 194 of 615 replies repeated the one before. **The 8B's play cannot test boundary:**
  it put the car in another condition once in 629 replies, so every measurement was taken with the key off, where
  nothing flows and every reading is what the oracle expects. In 367 accepted steps the oracle found one reply
  unexpected, and the run gained 4.2 bits in all.
- The fork probe on the 8B shows the same prior as the transcript test: at intake its top suspect is the
  alternator in 15 of 20 scenarios (calibrated) or the battery in 13 (raw), and its belief in the truth does not
  follow the oracle's (Spearman 0.02). With 4.2 bits of evidence in the run, the evidence shift has nothing to read.

## 4. The first reading: boundary's transfer, in Stage 1's form

Boundary passed Step −1 on TextWorld. Read on Fault Garage, whose text and mechanics it has never seen, it is
`habitat-engine.md`'s transfer test (its Q4), on the one sense that has passed. As in Stage 1, **a scripted player
acts and the 8B only reads**: 500 episodes at level 2, seeds 3000–3499, the greedy player on 3000–3399 and the
random player on 3400–3499 (the owner enlarged the pool from 300 on 2026-11-08, before any reading). `export_aware.py --scripted` writes the record turn by turn as play runs (the actions
by the code, in the self slot); `body.stepm1 read` runs unchanged on it; `score_boundary.py` alone reads the keys.

**The pool, counted from the keys before any reading** (exact repeats left out):

| Replies | Unexpected by the oracle | Expected |
|---|---|---|
| Readings (measure V, measure R) | 338 | 884 |
| Words (scans, pages, customer answers, road tests) | 93 | 80 |

Measures:

- **R0, exact repeats:** a reply identical to an earlier one in its episode. Boundary should be 0 (plumbing).
- **R1, surprise:** AUC of boundary on replies the oracle did not expect (it gave under 50% to what was shown) over
  replies it did; by kind; 90% intervals by episode.
- **R2, information:** Spearman correlation of boundary with the oracle's information gain, by kind.
- **R3:** boundary by reply kind. **R4:** self-caused scans against others (reported; the tickle test is babbling's).

**The agent's prediction, registered before the read.** Boundary separates unexpected from expected replies where
the surprise is in the words, and not where it is in a number: readings' R1 near 0.5. The basis: in the number-sense
tests (Fault Garage spec, "Numbers in meaning-space"), rewording a reading moved the 8B's state 10 to 180 times
further than a 1 V change, and boundary is a distance in that space. If this holds, it is evidence for a readings
sense: a frozen readout at the number's last token, layers 2 to 4, which recovered values from unseen phrasings
within 1 to 4% of range.

**Bars, fixed by the owner on 2026-11-08, before the read** (the drafts, with the pool enlarged):

| Measure | Bar |
|---|---|
| R0 | Every exact repeat under 0.01 floor units |
| R1, words | AUC at least 0.75, lower bound at least 0.65 (93 against 80) |
| R1, readings | Prediction only: interval contains 0.5 |
| R1, all kinds; R2; R3; R4 | Reported |

**What a fail would mean, written before the read.** A fail on R1 for words shows that boundary does not track the
oracle's surprise in this domain. It does not show that boundary is broken: R0 checks the plumbing, and R3 shows
whether boundary still responds to new text. Boundary measures how new an arrival is against everything held; the
oracle measures how improbable it is under every fault the car could have. The two usually move together but are
not the same event.

**What the model's own play needs before it is used again.** Two interface fixes are made (an unknown test point now
lists the valid ones, with a hint where a component name was used; a cap of 100 replies of any kind). One question is
the owner's: whether the player's system prompt says what an apprentice would be told, that the car starts with the
key off and most faults show only under load.

## 5. Later: rate's home

Rate's calibration failed in Stage 1 (AUC 0.486), and Stage 2 rests because TextWorld's stalls start abruptly and
never end. Fault Garage gives rate an exact event: the oracle's information gain per turn, split by where it came
from (the customer, tools, the model's own reasoning). Its stalls also end, in a diagnosis or the action cap. A rate
test here would be written in PLAN, with its bars, before any reading; nothing for it is built.

## 6. Risks

- **Shaped for the senses.** The agent designed Fault Garage with AWARE's events in mind, which is `habitat-engine.md`'s
  risk B. Held as there: the worlds, the oracle, the harness and the 8B's play were fixed before any reading, and
  the bars are fixed before the read.
- **The player's harness is not AWARE's.** For an instrument reading the player only supplies trajectories, as
  Stage 1's script does, so this should not matter for boundary. It would matter for anything functional.
- **The exported record.** Faithful text, but written after the fact. If the owner requires play to append the
  record itself, the 8B plays again through a small AWARE-side command.

## 7. Questions for the owner

- **Q1.** Fault Garage as a candidate second engine, beside `habitat-engine.md`'s B?
- **Q2.** ~~Boundary's first reading on it: the pool, the bars, or report only?~~ Answered by the owner, 2026-11-08: the
  draft bars with the pool enlarged to 500 (section 4; DECISIONS.md).
- **Q3.** The 8B's own play: accept its exported record for later readings, or re-play with AWARE appending as it
  plays? And the system prompt: add the apprentice's line about the key and load, or leave the model to find it?
- **Q4.** The readings sense: into Spec v0.2 now, or after this reading?
- **Q5.** The packages stay apart, with only records crossing?
