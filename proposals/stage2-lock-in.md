# Proposal: Stage 2 after the 8B's screen — the players lock in, they do not slide

**Accepted 2026-11-07: option 2** (owner; Step 0 amendment 7, DECISIONS.md). Where rate's predictive test goes, a
second engine or babbling, is not yet chosen.

For the owner. Nothing here is decided until accepted. Drafted by the agent, 2026-11-07, after the bf16 8B's Stage 2
screen (Step 0 amendment 5) failed on closures. It is read against PLAN.md on `main` after PR #44. It asks one
question: is Step 0's event in this habitat at all, and if not, where does rate's predictive test go?

## 1. The screen

Qwen3-8B, bf16, 16,384 tokens of context, told the command forms, on the mirror seeds of
`material/habitat_worlds_stage2.json`; a rented RTX 3090 (driver 595.58.03); labels only, no state read; record in
`runs/s2-screen/` (owner's machine). Play took 55 minutes for 5,060 turns (about 5,500 turns an hour).

| Criterion | Screen | Projected (×10: 36 played, 4 skipped, about 360 usable of 400) | Bar | |
|---|---|---|---|---|
| Closures kept (after progress, paired) | 6 | 60 | 150 | fails |
| Ascents kept | 41 | 410 | 75 | passes |
| Won | 26.6% (of 64 episodes) | — | 10% | passes |

k = 7 from 28 calibration episodes. Even read generously the closures do not reach Step 0's own floor: 6 events as a
Poisson count have a 90% range of about 2.6–11.8, projected 26–118, against 100 below which Step 0 is too few to
decide. Pooling the calibration play (10 kept in 28) gives about 16 in 64, projected about 90.

Where the 34 labelled closures of Step 0's mirror pool went:

| Step | Closures |
|---|---|
| Labelled at k = 7 | 34, in 31 of 36 episodes |
| No ascent before the onset (amendment 5) | −20, onsets on turns 1–9 |
| After progress, but fewer than 3 scored turns before the onset (the 10-turn burn-in) | −8 |
| Kept, each with a window | 6 |
| Paired with a pre-progress window | 6, **all 6 from another episode** (calibration: 8 of 10) |

Reported only: 7 kept at k = 3, 6 at k = 14. k barely matters, because the closures are long.

## 2. What the closures are

**They do not end.** 47 of the 52 labelled closures in both pools last 50 turns or more, most 71–99: they run to the
episode's end. Only 5 are shorter. So no closure resolves, and the same episode has no pre-progress window after the
stall to pair it with; reading 1's pairs come from other worlds.

**They start abruptly.** Sorted by a rule the agent wrote after seeing the episodes (labels and text only; the
commonest kind of reply and action during the stall; not registered):

| Kind | Long closures | Example |
|---|---|---|
| A parser reply, then the same command repeated | 21 | "Which do you mean, the box or the soap scented box?" → `unlock box with keycard`, 98 times |
| — of which the parser asks or half-understands ("Which do you mean", "I only understood you as far as", "I didn't understand") | 13 | |
| — of which "You can't see any such thing" (a naming mismatch or a real mistake; not told apart) | 8 | |
| A fixed cycle through the same rooms | 20 | `go west` 50, `go east` 49 |
| A real dead end, retried | 6 | "That doesn't seem to fit the lock." → the same key, 90 times |

On average the commonest action fills 55–61% of a long closure's turns. The picture: one reply that changes nothing,
then the model copies its own last action, or its last few, for the rest of the episode.

**It is not the model's size.** Every player screened shows it (PLAN.md, Step 0 habitat "Runs"): the 0.6B loops from
the start; the 1.7B and 4B, with and without the forms, on both world files, mostly lock in on turns 0–9 on the
parser; the 4-bit 8B wins in about 12 turns or loops from the start; the bf16 8B, here, the same. The bigger the
model, the more it progresses before locking in, but the lock-in itself looks the same.

## 3. Why it matters for Step 0

Step 0 asks whether the world part of rate flattens *before* a stall after progress: whether, over the 10 turns before
the onset, the world's replies stop helping code the player's next action. Section 2 suggests the stalls in this habitat
have no such lead: a single reply triggers them and copying sustains them. That is a hypothesis about the habitat, not
a reading: no state has been read on Stage 2, and none should be read to choose between the options below, since that
would choose the design on the result.

If the hypothesis holds, Step 0's habitat form can fail for a reason that says little about rate. Its pairs would also
compare a stall in one world with progress in another, so a difference could come from the worlds, not the stall.

## 4. Three options

### Option 1: treat the parser's traps as a harness defect, and screen again

TextWorld names objects so that two can share a noun ("the box or the soap scented box"), and its parser then asks a
question the player never answers. Treated like the banner stripped in Step −1 amendment 1, as the engine's defect and
not the player's skill: Stage 2's worlds are built so that no two objects in a world share a noun, as a new pool file
fixed before any play, with its seeds disjoint from every seed already used. Then the 8B is screened again by the same
criteria. The screen's play is about an hour on a 3090.

- **Addresses:** the 13 closures where the parser asks or half-understands; perhaps some of the 8 "can't see" ones.
- **Leaves:** the 20 room cycles and the 6 dead ends, and the copying that sustains every lock-in. The stalls may still
  start abruptly and run to the end.
- **Against it:** it reshapes the worlds around how the player plays, which the two-stage fix of 2026-11-05 set out to
  stop. Whether shared nouns are a defect or a difficulty is the owner's call.

### Option 2: move rate's predictive test to where the event exists

Stage 2 rests (as Step 0 on the owner's corpus rests) and its row in the staged schedule becomes a slot, with this
proposal as its record. Two places it could go:

- **A second engine** (PR #38, `proposals/habitat-engine.md`), whose worlds are built for the event: quests whose later
  steps are harder, so that stalls after real progress happen, and replies that are never parser questions.
- **Babbling (Step 2)**, where the 8B acts with its senses as the actor, and the self/world split is tested anyway.

Nothing waits on it: by the gating in PLAN ("Two stages"), Stage 2 informs and does not gate babbling, and Step −1
alone gates Stage 2. One consequence, stated plainly: rate enters babbling with no instrument test of its own. Its
only check before then would be Stage 1's contingent-vs-blind calibration (reported only; decided, not built). This
option should build that calibration first.

### Option 3: run Stage 2 as registered, with a larger pool

The worlds, rules, k's procedure and bars unchanged; Step 0's pool grows until the projected closures reach 150. At the
screen's rate (6 in 36) that is about 900 usable worlds, about 2.5× the pool; at the pooled rate (16 in 64), about 600,
about 1.7×. New seeds, disjoint and fixed before any play, as an amendment made after the screen.

- **Cost:** play about 15 hours for 900 worlds at the screen's rate. Reading at Step −1's rate (about 0.6 s a chunk on
  the rented 3090) would take more than a day, less if the read-speed probe finds the cause.
- **What it reads:** mostly abrupt lock-ins, paired across episodes. Its limits would be written before the reading:
  the event as section 2 describes it, and cross-episode pairs throughout (same-episode pairs, a registered check,
  would be almost none).
- **For it:** it changes nothing the instrument measures, and the hypothesis in section 3 gets tested, not assumed.

## 5. The agent's suggestion

Option 2. Option 3 spends several days of GPU on a test whose fail would be hard to read, and whose pass would rest on
cross-episode pairs. Option 1 fixes about a quarter of the lock-ins and leaves the copying that makes them all abrupt.
Option 2 costs nothing downstream by PLAN's gating, and points the work where it is cheapest: building Stage 1's
contingent-vs-blind calibration, so rate has an instrument check before babbling. Options 1 and 2 can be combined: if
the second engine comes, its worlds would avoid shared nouns from the start.

## 6. What is asked of the owner

1. Which option, or another.
2. Under option 1: whether shared nouns count as the engine's defect.
3. Under option 2: the second engine or babbling as the home of rate's predictive test, and whether the
   contingent-vs-blind calibration is built next.
4. Under option 3: the pool's new size, and the limits written before any reading.

Whichever is chosen goes into PLAN as an amendment made after the screen and before any reading, as Step 0's
amendments 2 and 5 were.
