# Plan

The design is the Big Picture doc, v9 ([Big Picture — v9](https://claude.ai/artifact/ByrrhB2Ue4NS8NzmyZXoCw), a
Claude Doc). Its **Entity** tab is the parent: what the entity is, the substrate decisions, the capacities, and the two
tracks. The **Drift detection (v9)** tab is the product track's full treatment; **Tech stack** sets the build order.
Where this file and the doc differ, the doc wins and this file is corrected.

**Two tracks, one substrate, kept apart.** The product track (drift detection, then self-correction) has the demo and
fast feedback; the research track (reasoning over the entity's own axes) has no demo yet and is the reason the project
exists. They never share a pass criterion: the product's faster feedback would win every local decision. Gate 0 can
end the product track, not the project (owner, 2026-10-31). The substrate changes only by the owner's decision.

A step's pass criterion is written here before its code, and its thresholds are fixed before it runs. A step is done
when its test passes, not when its code exists.

## Substrate (`substrate/`)

The Entity tab's decisions, as code: the base model frozen (states read through nnterp, `states.py`); the code
retained, not consumed (`axes.py`: every axis considered is kept); admission by two criteria (predictive: lowers
held-out residual by more than D; otherwise structural, tagged untested, until the research track states a test);
provenance logged at the source (`record.py`: self or external on every chunk).

| Step | Test | Status |
|---|---|---|
| 1. Model access and D | states on the real model; D per layer on frozen paraphrase sets | **code built; tested on a tiny random Qwen3 only (Hugging Face unreachable from the build container); awaits the card and the frozen sets** |
| 2. Logging | per-chunk activation capture, in context and alone; provenance tagged at creation and carried when re-read | **reader and record built; the harness that writes them is product step 4** |

**State** (provisional): the residual stream after a layer, mean-pooled over a chunk's tokens (special tokens left
out), read alone (`states`) or inside the whole sequence (`states_in_context`). **D**: the root-mean-square distance
of a state from its paraphrase set's mean, averaged over sets, per layer; the layer proposed is the one with the
largest between-to-within ratio, and the owner chooses. **Material:** 20 sets of 5 paraphrases
(`material/paraphrases.jsonl`), drafted by the model, not yet frozen. **Pass:** states for every text on both models
at every layer asked for, and the same text read twice gives the same state.

## Product track (`product/`): drift detection, then self-correction

| Step | Doc | Test | Status |
|---|---|---|---|
| 0. Text-only floor | v9 step 0 | the text panel on the owner's 45 sessions | **done by the owner, twice:** text sees the owner leave an aim (AUC 0.65–0.71), not the agent |
| 3. Toy bed | Toy bed specification | below | **built; first run fails three of its four bars (below)** |
| 4. Labeled logs | Tests, in order | detectors on the 65 departures, agent-authored first | open: owes the owner's confirmation pass (drift or re-aiming) and logging on the card |
| 5. Reference scorer, coherence score | Tests, in order | intact versus shuffled work | open: needs the setting and the owner's reference work |
| 6. Compass and harness, arm A; Gate 0 | Tests, in order | the compass predicts drift on held-out stretches above a pre-registered level, and beats self-residual alone | open. **Can end the product track** |
| 7. Arms S and C; main demo | Main demo | C closes half the A–B gap on both scores and beats S by more than run-to-run noise | open |

### Step 3: the toy bed (pre-registered 2026-10-31, before its code)

**Generator** (v9): s(t+1) = A·s(t) + B·E(t) + noise at scale D, about the origin's centroid c (the mean of the first
10 external states). Synthetic for now: 32 dimensions, D = 1, 240 steps, the structure planted at step 120. Baseline:
A = 0.5·I, B = I − A (so E is the equilibrium), E a slow random walk (0.3 D a step) observed with noise D. The six
structures change only A, B or E:

| Structure | Planted at step 120 |
|---|---|
| 1. Topic shift | E jumps 6 D in a random direction |
| 2. Dynamics shift | A becomes two rotations (0.9, 0.6 rad) in a random 4-dimensional subspace; B = I − A, so the equilibrium stays |
| 3a. Closed loop, dullness | over 60 steps B → 0, A → 0.98·I, and a drift of 0.3 D a step along one direction grows in |
| 3b. Closed loop, proliferation | E arrives as s plus noise 0.3 D: provenance external, content self |
| 4. Mastery | over 60 steps noise → 0.3 D and A → 0.2·I |
| 5. Noise spike | three steps of noise 20 D |
| 6. Progress (control) | nothing: the baseline |

**Detectors.** Windows of 40 steps.
- Point contrast: the distance between the mean states of two windows, in D.
- Trajectory contrast: linear dynamics fitted in each window (each window's own mean removed, in the top 4 principal
  axes of the two together); the distance between the sorted magnitudes of the two fits' eigenvalues.
- Each at onset (the windows either side of step 120) and for persistence (the window before onset against steps
  160–200). A detector fires when its statistic exceeds the 99th percentile of the same statistic over 500 progress
  runs drawn with a separate seed.
- The observer, in blocks of 20 steps from the start: a self code (candidates from the block's own and external
  states) and a frame (candidates from external states only), both through c, each candidate the block's first half's
  top residual direction, admitted by `axes.py` on the block's second half of the entity's states. Per block:
  self-residual and frame-residual (static, in D) before updating, the margin (frame minus self), frame growth (1 if
  the frame admitted an axis) and distance to the origin.

**Pass** (v9's bars):
- Rows 1, 2, 5, 6 separated at 90% or better, each over 200 trials, by these signatures: topic shift, point contrast
  persists and trajectory contrast does not; dynamics shift, the reverse; noise spike, something fires at onset and
  nothing persists; progress, nothing fires.
- Row 3 against row 4: the margin over steps 180–240 ranks closed-loop blocks above mastery blocks at AUC 0.8 or
  better, each variant against mastery.
- Row 3b against row 6: frame growth over steps 120–240 is lower in 3b than in progress, at AUC 0.8 or better.

The band's edges and the compass cells are reported, not passed: self-residual in band between 0.75 and 2 D, frame
residual high above 2 D (provisional). The generator's closed loop is a conjecture (v9, "Limitation"); if the
detectors pass here and fail on labeled logs, the generator is at fault.

**First run** (2026-10-31, `python -m product.toybed`, 200 trials a structure, seed 0). Not passed.

| Bar | Result | Pass |
|---|---|---|
| Separation: topic shift / dynamics shift / noise spike / progress | 0.99 / **0.25** / **0.74** / 0.97 | no |
| Margin AUC, dullness over mastery / proliferation over mastery | 0.86 / **0.57** | no |
| Frame growth AUC, progress over proliferation | **0.00** | no |

What each miss says, each mapped to an item already open in v9's tension log:
- **Dynamics shift (0.25).** The fitted eigenvalues do rise (about 0.6 to 0.9), but the externals' slow walk puts a
  near-1 mode in every window, and 40-step fits are too noisy to clear a 99th-percentile threshold set on progress
  runs. A post-hoc check, not a retune. Bears on "What contrast yields".
- **Noise spike (0.74).** Only point contrast sees it: a spike decays at the entity's own rate, so its dynamics do
  not change, and trajectory contrast is right to stay quiet. The signature asks one detector to catch it.
- **Proliferation.** Frame growth is higher, not lower, when externals are generated from the entity (5.8 admissions
  against 2.1 in progress), and the margin cannot tell it from mastery. This is v9's decisive row 3b failing as the
  doc warned: the frame measures provenance, not new directions, and the person-inside-the-loop case is invisible to
  it. Bears on "What the closed-loop statistic is" (coupling B̂ rather than frame growth).
- **The band.** Static self-residual is about 2 D in every structure (it carries the externals' walk), above the band
  in the closed loop (5.1) rather than below it, and mastery does not fall below the band; the commonest cell in
  progress is "lost". The tension log predicted the lower edge would not survive the generator's noise. Bears on
  "Which residual" (dynamic, not static).
- **What passed.** Point contrast separates topic shifts, and the margin catches the dullness closed loop (0.86).

Any change to the detectors in answer to this is a new pre-registration and a new run, not an edit to this one.

## Research track (`research/`): reasoning over the entity's own axes

No demo and no pass bar yet, by design. What it has:

| Item | Test | Status |
|---|---|---|
| Gate 1: negation | mirror, additive shift or contraction toward the origin; the winner beats both others by more than D | `research/gate1.py` built and tested on planted pairs; runs on the accumulated axes after the demo's arms exist (v9 build order) |
| Structural axes | retained, tagged untested (`substrate/axes.py`) | waiting for a structural test from this track |
| Relations, composition, order | angles, nesting, symmetry; predicted position under combined axes; non-commuting moves | no probe yet |

## Retired

Freeform chat (rung 1), the rung-2 memory harness, the toy world and the port of DARA's kernel, at the tag
`aware-rungs-final` (DECISIONS.md, 2026-10-30).
