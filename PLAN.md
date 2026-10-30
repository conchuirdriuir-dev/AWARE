# Plan

The design is the Big Picture doc ([Big Picture — v8](https://claude.ai/artifact/RmZatX3o5ZrAkJToUvSsQW), a Claude
Doc: the Big Picture and Tech stack tabs). This file tracks where the build stands against it. Where the two differ,
the doc wins and this file is corrected.

Each step runs only if the one before it passes. A step's pass criterion is written here before its code, and its
thresholds are fixed before the step runs. A step is done when its test passes, not when its code exists.

| Step | Doc | Test | Status |
|---|---|---|---|
| 1. Model access and D | Tests 1; build 1 | states read on the real model; D measured per layer on frozen paraphrase sets | **code built; tested on a tiny random Qwen3 only (Hugging Face unreachable from the build container); awaits the card and the frozen sets** |
| 2. Toy bed and geometry core | Test 3 (step 0); build 2 | below | **built; passes up to D/R = 0.5; rerun at the measured D** |
| 3. Reference scorer and coherence score | Test 2; build 3 | validated on intact versus shuffled work | open: needs the Gate 0 domain and the owner's reference work |
| 4. Compass and harness, arm A; Gate 0 pilot | Test 4; build 4 | the compass predicts drift on held-out stretches above a pre-registered level, and beats self-residual alone | open. **Can end the project** |
| 5. Gate 1, discovery, arms S and C | Test 5; build 5 | a mirror about the landmark centroid predicts elicited opposites better than additive −α by more than D | open |
| 6. Main demo, then transfer | Main demo | C closes half the A–B gap on both scores and beats S by more than run-to-run noise | open |

The build follows the doc's build order, so the toy bed comes before the reference scorer.

## Step 1: model access and D

**State** (provisional, owner 2026-10-30): the residual stream after a layer, mean-pooled over a text's tokens, read
in bf16 through nnterp's standardized layer names so the same code reads Llama and Qwen. Several layers are read at
once; the layer is chosen by the measurement below, by the owner.

**D**: the root-mean-square distance of a state from its paraphrase set's mean, averaged over sets, per layer. Beside it,
the spread between sets (the root-mean-square distance of set means from their overall mean). The layer proposed is the
one with the largest between-to-within ratio.

**Material:** 20 task-like statements, 5 paraphrases each (`material/paraphrases.jsonl`). Drafted by the agent; the
owner edits them and marks them frozen before D is measured. Not frozen yet.

**Pass:** states come back for every text on both models, at every layer asked for, and the same text read twice gives
the same state. D itself has no pass bar: it is a measurement the later steps are set against.

**Before the card:** the code is tested on a tiny randomly initialised Qwen3 built in the test, checked against
transformers' own hidden states; `pytest -m model` adds Qwen3-0.6B (ungated, Apache 2.0) on CPU wherever Hugging Face
is reachable. Both test the plumbing, never the result. Llama 3.1 8B Instruct comes first on the card, then Qwen3-8B (DECISIONS.md, 2026-10-30).

## Step 2: toy bed and geometry core

Synthetic states in numpy, with noise of size D about every state, so the geometry and the compass are tested on
structures planted where the answer is known.

**Code** (provisional): the agent's axes are the principal axes of a window of its states, their number chosen by
description length: a two-part code, the axes' parameters at ½ log₂ N bits each plus Gaussian bits for what they leave
unexplained (probabilistic PCA's BIC), priced within the n − 1 dimensions a window of n states spans. Provisional; the
doc names description length without defining it. The toy bed's first run found the textbook version, priced over all
d dimensions, keeping every axis a 20-state window allows (19 of 3 planted); priced within the span it finds the 3.

**Separating the structures.** Pairs of states (x, y), 30 to fit and 10 held out, planted either as a mirror about the
landmark centroid c (y = c + H(x − c), H a reflection) or as an additive direction (y = x + αv). Both are fitted; the
pipeline says mirror when the mirror's held-out root-mean-square error beats the additive one's by more than D,
additive in the opposite case, and undecided otherwise. Undecided counts as wrong.

**Placing trajectories.** Four kinds, each a run of states: on track (variation along the code's axes, near the
landmarks), progress (variation along a new axis, near), drift (a random walk away), and a rabbit hole (coherent
variation in a subspace far from the landmarks). The compass compares a recent window with the code fitted to the
window before it:
- **Self-residual** r: the mean squared distance of the recent states from the code's affine subspace, over D².
  High when r > 2.
- **Anchor distance** a: the distance of the recent window's mean from c, over the root-mean-square distance from c
  of the first window's states (the start of the work, taken as on task). Far when a > 1.

**Pass:** both at 90% or better, over 200 trials each, at the measured D. Until D is measured, the toy bed is swept
over D/R from 0.05 to 1 (R the size of the planted structure) and its results are reported, not passed.

**First sweep** (2026-10-30, 256 dimensions, `python -m toybed`): separation 1.00 / 1.00 / 1.00 / 0.98 / 0.51 and
placement 1.00 / 1.00 / 1.00 / 1.00 / 0.92 at D/R = 0.05 / 0.1 / 0.2 / 0.5 / 1. It breaks where it should, as noise
reaches the size of the structure; progress is the first cell to blur. The toy bed's generator and the pipeline were
written together, so this shows the mechanics work, not that real states will look like this.

## Retired

Freeform chat (rung 1), the rung-2 memory harness, the toy world and the port of DARA's kernel, at the tag
`aware-rungs-final` (DECISIONS.md, 2026-10-30).
