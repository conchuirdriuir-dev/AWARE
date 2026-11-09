# Proposal: area, a sense of order (shape's new form)

For the owner. Nothing here is decided until accepted. Drafted by the agent, 2026-11-09, from the owner's doc
[Meaning as Transport](https://claude.ai/artifact/CVZseb49aUE4WFkiwpRseW) ("One quantity: signed area") and its toy
checks (PR #64, `proposals/toy_area.py`). Shape is parked and "returns only as a new form" (DECISIONS, 2026-11-07);
this is offered as one. No code is written and nothing has been run for it.

## 1. Why shape failed, and what area is instead

Shape was the code length of a chunk under a Gaussian fitted to the episode's earlier chunks. That reads only where
the chunk lands: shuffle the earlier chunks and it does not change. Boundary reads the same thing, the nearest earlier
chunk's distance. So the two were two estimators of one quantity, the novelty of a position, and with 50–150 earlier
chunks in hundreds of dimensions the Gaussian is the worse estimator. Step −1C asked shape to part from boundary;
in that form it could not (on the 8B, 0.575 against boundary's 0.82, and 0.49 given boundary).

The doc's order is a different quantity: the signed (Lévy) area of the path, level 2 of its log-signature. It is
zero for any order-free reading, and it has its own symmetries, from which its test follows:

| Change to the episode | Boundary (position) | Area (order) |
|---|---|---|
| Shuffle the visits, same states, ends held | unchanged | changes |
| Reverse the path | — | flips sign |
| Rotate the basis | unchanged | its size unchanged (read as a norm) |

(The bag and shuffle rows are checked exactly on toy paths in PR #64's checks 1–2.)

## 2. The sense

**Area of the world's path.** Over a window of turns, the world's replies (slot `world`, elicited) as states on the
read layers, projected into the episode's own R = 8 principal directions fitted on world chunks up to the window's
end (as rate's X_c; causal). A is the 8 × 8 antisymmetric Lévy-area matrix of that path. The sense is

    rho = ||A||_F / sum_t |dx_t|^2

a scale-free number: about constant while the path keeps circling the same way, falling as 1/sqrt(n) while it
wanders without a consistent turn. Mean over the read layers, as the other senses.

Read on the world's path alone, never on the interleaved command–reply path: alternation makes a sawtooth whose area
comes from turn-taking, not from the world. The command path's area, and the cross area between commands and replies
(a lead–lag reading, a model-free relative of X_c), are later forms, not in this proposal.

## 3. The test: a loop against a dither (a dissociation, as Step −1C)

**The plant.** A scripted player, in Fault Garage, plays the greedy player for 10 turns (the burn, as every sense),
then a plant window of 24 turns using four commands that leave the car's state unchanged (taken from the episode's
test pool by the seed). One kind per episode, fixed by the seed:

- **Loop:** the four commands in one fixed cyclic order, six times round.
- **Dither:** the same four commands, six times each, in a shuffled order with no command twice in a row.

The episode ends after the window. Both kinds visit the same states the same number of times; only the order
differs. Boundary should not part them; area should. The kind is in the seed's key, read only by scoring.

**Why Fault Garage:** its commands can be taken in any order without moving through a map (TextWorld's rooms can't
be cycled and shuffled with the same visits), its replies are rebuilt from seeds, and it is world one of the family.
The script lives in its repository; seeds are registered there first; records cross stamped with the world's commit,
the reader's commit and the model revision.

**Pass, both (bars for the owner to fix):**

1. **Area:** AUC of loop over dither, rho on the window, at least 0.80, its 90% lower bound above 0.70.
2. **Apart from boundary:** the 90% interval of (area's AUC − boundary's AUC) on the same windows, boundary as the
   window's mean news, lies entirely above zero.

80 episodes (40 each kind), 2,000 bootstrap draws resampling episodes; a rehearsal on the 0.6B decides nothing; the
deciding reading on the 8B in bf16. Reported, deciding nothing: area given boundary; rho on the interleaved path;
rho by layer.

**What a pass means:** area reads the order of the world's replies, apart from their novelty, in a world built for
it. By the transfer rule, generality needs a world not built for it (TextWorld) with its own bars. Whether a loop
sense helps babbling is a functional question, asked separately.

## 4. Before any text: one toy check of power (stage A)

Area per episode is estimated from 24 points in 8 dimensions, the small-sample setting that sank shape. Before
stage B is built, `toy_area.py` gains a check at this size:

- Four reply points in 8 dimensions, plus per-visit noise and a slow drift per turn, both **set from the 8B's
  Step −1C states (replayed from the record if not cached) before the check runs**: the noise as the echo floor against the median distance between
  replies to different commands; the drift from the distance between replies to the same command four turns apart.
  Recorded with the check.
- 40 loop and 40 dither windows as in section 3; rho on the world path; the nearest-distance analogue of boundary.
- **Bar (for the owner):** rho's AUC loop over dither ≥ 0.90 on the world path. Below it, stage B is not built and
  the owner is told; the window is not lengthened to suit the result. Reported: the boundary analogue's AUC
  (expected near 0.5) and rho on the interleaved path (expected to be swamped by alternation).

## 5. Risks

- **Too few samples.** Held by stage A, with the noise from records.
- **Context drift.** The reader sees the whole episode, so a repeated command's reply is not the same state twice;
  a drift strong enough to dominate the four-point cycle would hide the loop. Stage A sets the drift from records.
- **Fault Garage's templates.** Repeated commands give near-identical text; this suits the test (the states differ
  by context, not wording) but means the pass says nothing about wording. The wording gap (world-family.md, Q5)
  is not needed here.
- **A world shaped for the sense.** Held by the transfer rule: the plant, seeds and bars are fixed before any read,
  and a general claim needs TextWorld.
- **Keys supply.** No part of the sense is fitted to a label; rho is fixed in advance from the symmetry. Fitting a
  readout of signature terms to oracle labels would let the key supply the sense, and is ruled out.

## 6. Questions for the owner

- **Q1.** Area as shape's new form, with its test as a loop–dither dissociation from boundary?
- **Q2.** The sense as defined: world path only, R = 8 causal basis, rho = ||A||_F / sum |dx|^2?
- **Q3.** Stage A first, with its bar (AUC ≥ 0.90) and the noise and drift set from the 8B's Step −1C records?
- **Q4.** Stage B in Fault Garage: the plant (10-turn greedy burn, 24-turn window, four state-preserving commands),
  80 episodes, and bars 1–2 as Step −1C's?
- **Q5.** The command path's area and the lead–lag cross area left for later forms?
