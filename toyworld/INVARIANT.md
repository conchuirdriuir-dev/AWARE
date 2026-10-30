# The toy world: a hidden invariant

Can DARA's law find a regularity the model misses, and does showing it change what the model does? In stage 0, Qwen3
8B never worked out that oat milk's demand is sold + unfilled, about 40 a day: it ordered 20 to 30 and the stockout
stayed (1 of 18 acting runs fixed it). The demand is in every report. This test asks whether code finds it from the
record by compression, and whether the model acts on it once shown. No larger model is the benchmark: the measure is
whether the capability appears (DECISIONS.md, 2026-10-29).

## What is DARA's here, and what is not

DARA's kernel cannot state this invariant: it reads relational structures of opaque symbols and never what a number
means (DARA `docs/KERNEL_v0_1.md`, the renaming test). What is taken from DARA is the law (DARA `CORE.md`, "The law,
precisely"): one total code length; a posit admitted only when it shortens the record by more than m = 2 bits; rivals
within the margin kept as rivals; a held posit retired when its record codes worse than the reference by more than the
margin. The posit family and its codes are declared here, in [keeper.py](keeper.py), and are AWARE's, not DARA's.

| Part | Declared as |
|---|---|
| Posits | for two report fields y, x and s = ±1: y − s·x stays at a constant. At most one per target; no field coded from a field coded from it |
| Reference | y around its own level |
| Data code | prequential: each day's residual from the mean of the days before, signed Elias gamma |
| Prior S | log2 of the target fields + log2 of the other fields + 1 bit for s, about 6.8 bits |
| Admission, retirement | DARA's m = 2 bits |

The keeper reads the reports, which the model also sees, never the key. Its posits are records of the keeper's
inference, labelled as such (`said_by: keeper`), and are shown to the model as inferred, not stated. Nobody settles
them: learned structure is admitted on MDL gain (DECISIONS.md, 2026-10-29, "Two layers of state").

## Arms

| Arm | What the model has |
|---|---|
| goals | stage 0's goals arm (v2: option C, actions need a goal) |
| keeper | the same, plus the keeper's held posits in the view, with their daily values and saving |

The scorer also runs the keeper over the goals arm's reports without showing them, so each run says whether the
invariant was there to find.

## Procedure

Qwen3 8B, plain condition, 5 runs per arm, 8 days, so there are days left to act after the invariant appears:

`python -m toyworld.run run --url URL --run runs/inv-8b --arms goals,keeper --runs 5 --days 8`

About 80 calls; about 8 minutes at stage 0's speed.

## Measures

| Measure | What it counts |
|---|---|
| found | the keeper held "unfilled + (sold, ordered or delivered) stays at about D", D within 3 of the key's mean demand |
| found day | the day it was first held |
| other posits | posits about anything else, such as a coincidence between unfilled requests and till raises |
| ordered enough | an order of at least the mean demand less 3 |
| after found | an order that large on or after the found day |
| stockout fixed | stage 0's measure, on the last day |

## Pass criteria

| Outcome | Reading | Next |
|---|---|---|
| found in at least 2 of 3 runs where sales varied, by day 5 | the law finds the invariant from the record | — |
| keeper: after found ≥ 3 of 5; goals (found, not shown): ≤ 1 of 5 | showing the invariant gives the model the capability | the next missing capability |
| found but keeper's after found ≤ 1 of 5 | the model does not act on a shown regularity | try offering its consequence (an order of about D) as a question to the owner |
| sales never vary, so nothing is found | the model never explores, and the pattern stays hidden | exploration is the missing capability, not pattern-finding |
| the keeper arm acts on an other posit | a coincidence misled the model | a stricter prior, or longer records before admission |

On stage 0's 48 recorded runs, the keeper found the invariant in 20 of the 26 runs where sales varied, from day 2 to
day 6, and admitted at least one coincidence in 11 of the 48.
