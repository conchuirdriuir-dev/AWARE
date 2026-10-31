# Toy bed, round 1

The implementation record for one round of the product track's toy bed (PLAN.md, product step 3): its
pre-registration, frozen by the commit before its code, then its result and any post-hoc checks. The design it tests
is the Big Picture doc (v9's Toy bed specification). Run with `python -m product.toybed`.

## Step 3: the toy bed (pre-registered 2026-10-31, before its code)

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
