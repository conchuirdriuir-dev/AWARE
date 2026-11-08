# Proposal: rate's predictive test on Fault Garage

*Drafted by the agent 2026-11-08 at the owner's request, for the owner to fix. Nothing here is built or run. Every
number marked (agent) is the agent's proposal; the bars are Step 0's, unchanged. Follows
[fault-garage.md](fault-garage.md) section 5 ("Later: rate's home").*

## 1. Why this, and how it sits with PR #56

Rate (X_c) passed its calibration (DECISIONS, 2026-11-08) and enters the sense tokens with its predictive validity
untested. Step 0 asks the predictive question: does the world part of rate flatten before a stall after progress, and
not after an ascent? It rests (amendment 7) because TextWorld's stalls start on one reply and never end.

PR #56 (the babbling gate, fixed by the owner except its ratios) places rate's predictive test after the gate: Step 0's
habitat form, unchanged, on the grafted 8B's TextWorld play (seeds from 9600–9999). Its own caution: the grafted
player is the same 8B, so Stage 2's screen may fail again, and then "the second engine is the remaining place".

This proposal is that remaining place, written now so the owner can choose where the test runs. It is **the same
test, not a second one**: Step 0's question and pass criterion. Under "no two tests share a pass criterion", it runs in
one place:

- **(A) Here, now** (the agent's recommendation): Step 0 in Stage 2's form on Fault Garage, a weaker model playing and
  the base 8B perceiving, as Step 0 was registered. It does not wait on babbling. PR #56's rate bullet becomes a
  reported rerun on the grafted player, deciding nothing.
- **(B) Here, after babbling:** the grafted 8B plays Fault Garage instead of TextWorld. PR #56's placement stands; the
  engine changes.
- **(C) Fallback only:** PR #56 as fixed; this form is used only if Stage 2's screen fails on the grafted player.

What Fault Garage gives Step 0 that TextWorld does not: an exact label on every turn (the oracle's posterior), stalls
that can end (the agent finds a lead, a diagnosis, or the cap), and progress that is measured in bits rather than read
off a walkthrough.

## 2. What rate claims, and what this does not ask

X_c(t) is the bits the world's last reply saves in predicting the agent's next action (`body/conditional.py`): whether
the world is shaping what the agent does. It does **not** claim to measure how informative a reply is about the hidden
fault. The oracle's information gain is the key for *when* progress and stalls happen; it is never a target X_c must
track turn by turn. Boundary's words bar asked boundary for something it does not claim (DECISIONS, 2026-11-08); this
form avoids repeating that. The correlation of X_c with gain is reported only (section 6).

## 3. Why the scripted pool cannot carry it

A scripted player makes Step 0 circular (PLAN, Stage 2: its stalls are its own rules), and in Fault Garage X_c could
hardly be read on one anyway. Measured on Fault Garage's scripted-baseline seeds 0–99 at level 2 (no reading,
labels only):

| | Greedy | Random |
|---|---|---|
| Accepted turns per episode, median (max) | 3 (7) | 16.5 (63) |
| Turns at index ≥ 20, where X_c is defined (`MIN_PAIRS` = 20) | 0 of 285 | 1,361 of 2,586 |
| Of those, turns with gain above 0.1 bits | 0 | 33 |

The random player is blind as well, so its X_c has nothing to follow. Rate needs a model's own play, and episodes long
enough to score.

## 4. The form (Step 0 amendment 2's form; definitions adapted to the engine)

**The player (Stage 2's rule):** one fixed Qwen3 model smaller than the perceiver, chosen by a screen on mirror seeds
(+900000) from the engine's labels alone. Candidates: Qwen3-4B, then Qwen3-1.7B (agent). The 8B reads, in bf16, as in
Step −1. The player's prompt waits on the owner's open question (fault-garage.md Q3): whether it is told that the car
starts with the key off and most faults show only under load. Without that line the 8B's play gained 4.2 bits in 367
steps, so there would be no ascents to score. Recorded on every episode.

**Episodes (agent):** level 2. The cap rises from 60 to 100 accepted actions (and 150 replies) for these pools only, so
that turns 21 to 100 can be scored. A Fault Garage harness setting with its own DECISIONS line there, before play.

**Seeds (agent; Fault Garage's registry, not yet registered):** calibration (for k) 3500–3599; Step 0's pool
3600–3999; the screen on their mirrors (903500–903999). Level 2. Inside Fault Garage's model-run range (1000+),
after the first AWARE pool (3000–3499); no run has used them.

**Labels, from the oracle (keys, read only by scoring):**
- *Gain* at a turn: the drop in the oracle's posterior entropy from the world's reply to that turn's action
  (`oracle_gain_bits`).
- **Ascent:** a turn whose reply brings the oracle's entropy to a new low for the episode, at least 0.25 bits (agent)
  below the previous low, intake included. The analogue of "a quest step reached". A self-caused result (an `unplug`
  code) is never an ascent.
- **Closure:** at least k consecutive accepted turns each gaining under 0.05 bits (Fault Garage's `NO_CHANGE_BITS`),
  with at least half the actions repeating an earlier (action, car condition) pair of the episode. The analogue of
  TextWorld's (action, room). Its onset is the stretch's first turn.
- *Reported only:* the oracle-sure closure, where the agent goes on acting after the oracle holds 95% on the truth
  (the spec's first closure measure).
- **k:** on the calibration pool, from every run of sub-0.05-bit turns that is followed by a gaining turn in the same
  episode: the smallest whole number above that run length's 95th percentile. Fixed here before Step 0 is scored;
  reported at k/2 and 2k; the owner reads 20 closures and 20 ordinary stretches.

**The reading:** X_c per scored action, the 8B reading each chunk alone (layers, sink and chunking as in rate's
calibration), `body/conditional.py` unchanged (R = 8, λ = 10, `MIN_PAIRS` = 20). An action is scored only once X_c is
defined, which replaces Step 0's burn-in of 10.

**Windows and pairs (amendment 2, unchanged except for position):**
- *Pre-stall:* the last 10 scored turns ending at a closure's onset; at least 3, or the event drops out (counted).
- *Pre-progress:* the last 10 scored turns before an ascent, none inside a closure; at least 3.
- *Post-ascent:* the 10 scored turns after an ascent; at least 3.
- *Position (agent, added):* each pre-stall window is paired with a pre-progress window whose end lies within ±3 turns
  of its own, in the same episode where one exists, else the nearest seed (cross-episode, reported); never reused. X_c
  is fitted on more pairs as an episode goes on, and boundary fell with position here (Spearman −0.79), so every
  comparison is matched on position, as the owner asked of boundary.

**Pass (Step 0's, unchanged):** reading 1: the AUC of pre-progress X_c over pre-stall X_c is at least 0.70, its 90%
interval's lower bound is above 0.5, and it is above the text baseline's (TF-IDF, as Step 0). Reading 2: the AUC of
post-ascent X_c over pre-stall X_c has its 90% lower bound above 0.5. Both must pass, at the frozen k. 2,000 draws
resampling episodes. Fewer than 100 paired closures or 50 ascents: too few to decide.

**The screen (Stage 2's criteria):** projected to Step 0's 400 worlds, at least 150 closures and 75 ascents kept
(scorable, at index ≥ 23 for a window of 3), at the k its own calibration play gives; the player diagnoses correctly in
at least 10% of episodes. If no candidate meets it, the owner is asked, not the bar.

## 5. What it needs built

- **Fault Garage:** the cap setting for these pools (with a DECISIONS line); the closure and ascent labels from the
  oracle's rows (`faultgarage/log.py` already writes gain, entropy and a stuck run); a screen report; the export of
  model-played episodes, which exists (`export_aware.py --run`), stamped with the repo's commit. The reading machine
  also records AWARE's commit and the model revision, which boundary's reading missed.
- **AWARE:** a reader for Fault Garage records in `body/step0_habitat.py`'s form (`read`, `score`), labels taken from
  the record's keys; nothing in `substrate/`.

## 6. Reported, deciding nothing

The oracle-sure closure; X_c during the stall; S_c (the self part); X_c by the reply's source (tool, customer, page);
Spearman of X_c with the gain of the reply before it, overall and within turn bands; Spearman of X_c with turn; the
placebo (200 rounds of pretend events matched by the same rules); same-episode pairs only; W of 5 and 20.

## 7. The agent's prediction (written before any play; not to be edited after)

- The screen fails on ascents with the current prompt (no load line): under 75 projected.
- With the load line, the 4B passes the screen on closures (it looped: 194 of 615 replies repeated the one before) and
  is borderline on ascents, because most diagnoses that gain come early, before X_c is defined.
- If it is read, reading 1 lands between 0.55 and 0.70: above chance, below the bar. Repeated actions in a stall are
  predicted well by the agent's own last action, so X_c falls, but pre-progress turns at level 2 are often single
  probes whose reply changes the next action only weakly.

## 8. Risks

- **Shaped for the sense.** The labels are adapted by the agent who knows X_c's form. Held as in fault-garage.md:
  definitions, seeds, the cap and the screen fixed before any play; the bars are Step 0's, unchanged.
- **Long episodes are a selected sample.** Only episodes past 23 turns score; a player that solves fast contributes
  nothing. Reported: how many episodes, and which faults, reach scoring.
- **The cap is not a stall ending.** An episode cut at 100 can end inside a closure; it is counted as Step 0 counts a
  closure running to the episode's end.

## 9. Questions for the owner

- **Q1.** Where rate's predictive test runs: (A), (B) or (C) in section 1.
- **Q2.** The player's prompt: the load line or not (fault-garage.md Q3). (A) and (B) need an answer first.
- **Q3.** The adapted labels (ascent at 0.25 bits, closure over (action, condition)), the cap of 100, and the seeds.
- **Q4.** Position matching within ±3 turns, added to Step 0's pairing.
