# Proposal: area, a sense of order (shape's new form)

**Accepted by the owner, 2026-11-09** (DECISIONS.md): section 6's questions as revised in discussion. Area is shape's
new form; its test uses replies reworded on every visit; the sense reads the largest singular value of the area
matrix; the toy power check (stage A) runs first, its noise from the 8B's paraphrase spread; stage B waits for Fault
Garage's wording as data, its sizes and bars as drafted, fixed before any reading.

Drafted by the agent, 2026-11-09, from the owner's doc
[Meaning as Transport](https://claude.ai/artifact/CVZseb49aUE4WFkiwpRseW) ("One quantity: signed area") and its toy
checks (PR #64, `proposals/toy_area.py`). Shape is parked and "returns only as a new form" (DECISIONS, 2026-11-07);
this is one. No code is written and nothing has been run for it.

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
| Rotate the basis | unchanged | its singular values unchanged |

(The bag and shuffle rows are checked exactly on toy paths in PR #64's checks 1–2.)

**Each chunk is read on its own** (the habitat's reading rule; Fault Garage's `HFBackend.chunk_states`): a state
carries no trace of the chunks before it. So the doc's split holds exactly here: position is order-free, and order
lives only in the path.

## 2. The sense

**Area of the world's path.** Over a window of turns, the world's replies (slot `world`, elicited) as states on the
read layers, projected into the episode's own R = 8 principal directions fitted on world chunks up to the window's
end (as rate's X_c; causal). A is the 8 × 8 antisymmetric Lévy-area matrix of that path. The sense is

    rho = sigma_max(A) / sum_t |dx_t|^2

the largest singular value of A, scale-free: about constant while the path keeps circling the same way, falling as
1/sqrt(n) while it wanders without a consistent turn. A cycle puts its area mostly in one plane; rewording noise
spreads over all 28 area coordinates, so the largest singular value keeps the signal and sheds most of the noise
(owner, Q2). The Frobenius form, ||A||_F in place of sigma_max(A), is reported beside it. Mean over the read layers,
as the other senses.

Read on the world's path alone, never on the interleaved command–reply path: alternation makes a sawtooth whose area
comes from turn-taking, not from the world. The command path's area, and the cross area between commands and replies
(a lead–lag reading, a model-free relative of X_c), are later forms (owner, Q5).

## 3. The test: a loop against a dither (a dissociation, as Step −1C)

**Why the replies must be reworded.** Read on its own, the same reply text gives the same state. With templates, a
loop would trace one exact polygon and a dither would hop between four exact points: area would part them and
boundary (zero distance to an earlier repeat in both kinds) could not, by arithmetic alone, as PR #64's identities.
That tests nothing about the reading. With each visit's reply reworded, the content cycles while the states vary, and
the question is real: does a cycle in meaning survive the wording's noise? (Owner, Q1.)

**The plant.** A scripted player, in Fault Garage, plays the greedy player for 10 turns (the burn, as every sense),
then a plant window of 24 turns using four commands that leave the car's state unchanged (taken from the episode's
test pool by the seed), each reply drawn from that command's phrasings without repeating a phrasing within the
episode. One kind per episode, fixed by the seed:

- **Loop:** the four commands in one fixed cyclic order, six times round.
- **Dither:** the same four commands, six times each, in a shuffled order with no command twice in a row.

The episode ends after the window. Both kinds visit the same contents the same number of times; only the order
differs. Boundary should not part them; area should. The kind is in the seed's key, read only by scoring.

**Waits for wording as data** (owner, Q4): the phrasings are Fault Garage's build-order item "Wording as data"
(world-family.md, Q5), written once for every world, never for this test. The plant's sizes and bars below are fixed
before any reading of its pool.

**Why Fault Garage:** its commands can be taken in any order without moving through a map (TextWorld's rooms can't
be cycled and shuffled with the same visits), its replies are rebuilt from seeds, and it is world one of the family.
The script lives in its repository; seeds are registered there first; records cross stamped with the world's commit,
the reader's commit and the model revision.

**Pass, both:**

1. **Area:** AUC of loop over dither, rho on the window, at least 0.80, its 90% lower bound above 0.70.
2. **Apart from boundary:** the 90% interval of (area's AUC − boundary's AUC) on the same windows, boundary as the
   window's mean news, lies entirely above zero.

80 episodes (40 each kind), 2,000 bootstrap draws resampling episodes; a rehearsal on the 0.6B decides nothing; the
deciding reading on the 8B in bf16. Reported, deciding nothing: area given boundary; rho on the interleaved path;
the Frobenius form; rho by layer.

**Rehearsal (2026-11-09, the 0.6B on the mirrors 905000-905199; decides nothing, recorded before the 8B reads).**
Built as DECISIONS records it (the burn: random-player tests, owner). Area's AUC 0.788 [0.696, 0.869], just under
bar 1's 0.80; area minus boundary +0.366 [0.216, 0.508], clear of zero; boundary 0.422; area given boundary 0.786;
the interleaved path 0.589; the Frobenius form 0.780; area by layer 0.73-0.84, rising with depth. The within/between
ratio on Fault Garage's phrasings is 0.63 (0.62-0.65 by layer), above stage A's 0.508 from the paraphrase sets. No
setting, bar or pool changes; the 8B in bf16 reads `runs/area-b-8b` (seeds 5000-5199) next.

**Deciding result (2026-11-09, the 8B in bf16, revision b968826, seeds 5000-5199; 40 loop and 40 dither, none
skipped): fails bar 1.** Area's AUC 0.759 [0.666, 0.848], below 0.80 and its lower bound below 0.70; bar 2 holds,
area minus boundary +0.308 [0.172, 0.446], boundary 0.451. Reported, deciding nothing: area given boundary 0.764; the
interleaved path 0.609; the Frobenius form 0.745; area by layer 0.67-0.82, rising with depth to 0.819 at layer 34;
boundary 0.45-0.47 at every layer. The within/between ratio on Fault Garage's phrasings is 0.73 (0.71-0.78 by layer),
higher than the 0.6B's 0.63 and stage A's 0.508: the rewording noise this test met is larger than the toy's power
check assumed. Stage B fails as fixed; nothing is re-run or re-scored. What follows is the owner's to decide.

**What a pass means:** area reads the order of the world's replies, apart from their novelty, in a world built for
it. By the transfer rule, generality needs a world not built for it (TextWorld) with its own bars. Whether a loop
sense helps babbling is a functional question, asked separately.

## 4. Before any text: one toy check of power (stage A, runs now)

Area per episode is estimated from 24 points in 8 dimensions, the small-sample setting that sank shape. Before
stage B is built, `toy_area.py` gains a check at this size (owner, Q3):

- **Noise from the 8B's paraphrase spread,** measured before the check runs and recorded with it: on the frozen
  paraphrase sets (`material/paraphrases.jsonl`, 20 sets of 5), the 8B's states on the read layers, the ratio of the
  spread within a set (paraphrases of one content) to the distance between set means (different contents). The toy's
  four contents are points in 8 dimensions at that between-content distance; each visit adds noise at the
  within-set spread. No drift term: chunks are read on their own.
- 40 loop and 40 dither windows as in section 3; rho on the world path; the nearest-distance analogue of boundary.
- **Bar:** rho's AUC loop over dither ≥ 0.90 on the world path. Below it, stage B is not built and the owner is told;
  the window is not lengthened to suit the result. Reported: the boundary analogue's AUC (expected near 0.5), the
  Frobenius form, and rho on the interleaved path (expected to be swamped by alternation).

**Result (2026-11-09, PR #64, `material/area_stage_a.json`): pass.** The noise reading first used the habitat's
"first" sink rule; the four paraphrases beginning "Find" put the sink on their second token and made the ratio
meaningless (layer 24: between-set distance 0). The owner fixed the norm rule before any re-read number was seen
(`runs/area-noise-8b-first` kept). Under it, within / between is 0.508 (0.48–0.62 by layer).

| AUC loop over dither [90%] | Toy, mean ratio | Toy, largest layer's ratio | The 8B's paraphrase states (reported) |
|---|---|---|---|
| Area, sigma_max (the sense) | **0.979** [0.955, 0.996] | 0.964 [0.929, 0.992] | 1.000 |
| Area, Frobenius | 0.976 [0.949, 0.995] | — | 1.000 |
| Boundary's analogue | 0.473 [0.367, 0.579] | 0.471 | 0.527 |
| Area on the interleaved path | 0.609 [0.509, 0.715] | — | — |

As expected: boundary near chance, alternation swamping the interleaved path. The real-state variant's 1.000 is
likely because the paraphrase noise in 4,096 dimensions lies mostly outside the episode's 8 directions, so the
isotropic toy is the harder case; it decides nothing. Stage B waits for Fault Garage's wording as data.

## 5. Risks

- **Too few samples.** Held by stage A, with the noise from the 8B's own paraphrases.
- **Toy against text.** Paraphrase sets are short statements, not Fault Garage's replies; the spread may differ.
  Stage A's ratio is recorded, and stage B's rehearsal reports the same ratio on Fault Garage's phrasings.
- **A world shaped for the sense.** Held by the transfer rule: the plant, phrasings, seeds and bars are fixed before
  any read, the phrasings are written for every world and not for this test, and a general claim needs TextWorld.
- **Keys supply.** No part of the sense is fitted to a label; rho is fixed in advance from the symmetry. Fitting a
  readout of signature terms to oracle labels would let the key supply the sense, and is ruled out.

## 6. Questions for the owner (answered 2026-11-09)

- **Q1.** Area as shape's new form, with its test as a loop–dither dissociation from boundary? *Yes, with replies
  reworded on every visit (identical replies would pass by arithmetic).*
- **Q2.** The sense as defined? *World path only, R = 8 causal basis, the largest singular value of A; the Frobenius
  form reported.*
- **Q3.** Stage A first? *Yes, bar AUC ≥ 0.90, noise from the 8B's paraphrase spread, no drift.*
- **Q4.** Stage B in Fault Garage? *After wording as data; plant and bars as drafted, fixed before any reading.*
- **Q5.** The command path's area and the lead–lag cross area? *Later forms.*

## 7. After stage B: what cost area its signal, and a new form for the owner (2026-11-10)

**Report only, by the agent. Stage B's deciding result stands as fixed: nothing here re-scores it.** The readings below
are on the 80 episodes already read (seeds 5000–5079; `runs/area-b-8b/states`, the states the sense received), and
their list was fixed before the states were opened. `python -m proposals.area_stage_b_diag`; numbers in
`material/area_stage_b_diag.json`, computed from the states' packed copy, which reproduces the deciding result exactly.

| Basis the window's replies are projected into | AUC loop over dither [90%] | By layer (24–35) |
|---|---|---|
| As run: every world reply up to the window's end, R = 8 | 0.759 [0.666, 0.848] | 0.67–0.82 |
| The window's replies only, R = 8 | **0.949** [0.902, 0.985] | 0.94–0.97 |
| The four contents' means, R = 3 (a ceiling: it uses the commands) | 0.996 [0.988, 1.000] | 0.99–1.00 |
| As run, R = 4 / 12 / 16 | 0.683 / 0.887 / 0.943 | |
| Window only, R = 4 / 12 / 16 | 0.777 / 0.951 / 0.951 | |

**What cost the signal: the basis, not the noise and not the layers.**

- *Not the noise's size.* The toy at the measured ratio (0.73) expects 0.93, and at the worst layer's (0.78) 0.92.
  The window-only reading gives 0.95: the toy was right about the noise.
- *The burn takes the basis.* The basis is fitted on 35 world replies, 11 of them before the window (the opening
  and the burn), and those are 1.5 times as spread as the plant's four contents. As run, the 8 directions hold 75%
  of the window's variance; fitted on the window, 89%.
- *What is pushed out is the circling.* The four contents' variance is unequal over its three directions: 76%, 17%
  and 8%. As run, the basis keeps all of the first, 85% of the second (21% at the 10th percentile) and 23% of the
  third. Fitted on the window it keeps 100%, 100% and 97%. A loop needs at least two of them.
- *The rise with depth was a symptom.* With the window's basis every layer reads 0.94–0.97. Deeper layers looked
  better only because the burn took less of the basis there.
- *Why stage A missed it.* The toy has 8 dimensions and keeps 8, so its projection loses nothing, and its four
  contents are equally spaced. Giving it more dimensions, or burn replies up to 6 times as spread, does not
  reproduce the failure (0.96 or above): the unequal contents are what matter.

**The same failure as shape's, one level along.** A basis fitted on everything so far is set by how varied the past
was, which is position. It would also get worse as an episode grows. Area needs the basis of its own window.

**Is the window's basis fair?** The basis is a principal-direction fit, which ignores order, and a loop and a dither
hold the same four contents six times each. So the basis has the same distribution in both kinds and cannot favour
one. It is causal in the same sense as before: nothing after the window's end.

**A new form, for the owner.** Chosen after this pool was seen, so it needs a reading of its own.

- *The sense:* as section 2, with one change: the R = 8 basis is fitted on the window's world replies only.
- *The test:* stage B again, as section 3, on episodes not yet played or read, with its bars fixed before the read.
- *Bars (agent):* the same two: AUC loop over dither at least 0.80 with a lower bound above 0.70; area minus
  boundary's interval above zero.
- *The agent's prediction, written before any such read:* about 0.95, clear of boundary.
- *Not proposed:* more directions on the old basis (16 reaches 0.943, but it treats the symptom), or a choice of
  layers.

**Risks.** The form was picked on the deciding pool, which is why it is read on fresh seeds and why one form is
named and not several. In live play the window is not given by a plant: a rolling window of the last 24 world replies
is the natural reading, and is untested. Twenty-four points in 8 directions is still a small sample; stage B's
second read is its test.

**Questions for the owner.**

- **Q6.** The new form: the basis on the window's replies only?
- **Q7.** Its seeds. Stage B registered 5000–5199 and played 5000–5079; 5080–5199 have no episodes. Use them
  (60 of each kind if none is skipped), or register a new range in Fault Garage?
- **Q8.** The bars as drafted, the same as stage B's?
- **Q9.** Stage A's toy: add unequal contents and a basis that can lose them, so the power check can fail for this
  reason, before the second read or after?

**The owner's answers (2026-11-10: "I'll take your recommendations", the agent's in chat).**

- **Q6.** Yes: the basis is fitted on the window's world replies only, R = 8. The interleaved path, reported, takes
  the window's own basis likewise (its chunks in the window).
- **Q7.** Seeds 5080–5199, registered for stage B and never played: every usable seed, up to 60 of each kind. A
  rehearsal on the 0.6B on their mirrors 905080–905199 first, which decides nothing.
- **Q8.** Stage B's bars unchanged: AUC loop over dither at least 0.80 with its 90% lower bound above 0.70; the 90%
  interval of (area − boundary) above zero; 2,000 draws resampling episodes.
- **Q9.** Before the second read: stage A2, below.

### Stage A2: a toy that can fail the way stage B did (fixed 2026-11-10, before its code)

The toy of section 4 with what it lacked: more dimensions than the basis keeps, four contents of unequal spread, and
burn replies more spread than the contents.

- **World:** d = 64. The four contents are drawn in 3 directions with variance shares 0.76, 0.17 and 0.08 (stage B's
  medians), turned into the 64 dimensions at random, and scaled to median distance 1. Eleven replies before the
  window (the opening and the burn) are drawn at random directions with 1.5 times the contents' RMS spread. Every
  reply adds isotropic noise of expected norm 0.73 (stage B's measured ratio on the 8B). Loop and dither as section
  4; 40 episodes of each kind; 2,000 draws; seed 51000.
- **Bar (gates the second read):** the sense with the window's basis, R = 8, AUC loop over dither at least 0.90.
- **Check of the toy itself (gates too):** the as-run basis (every reply up to the window's end) must read at least
  0.05 lower than the window's basis. If it does not, the toy still cannot see this failure, and the owner is told
  before the second read.
- Reported: both bases at R = 4, 12, 16; boundary's analogue.

**Stage A2's result (2026-11-10; `python -m proposals.toy_area --stage-a2`, `material/area_stage_a2.json`): fails
both.** The window's basis reads 0.846 [0.768, 0.919], below the bar's 0.90; the as-run basis reads 0.807, only 0.039
lower, short of the 0.05 the toy needed to show it can see stage B's failure. At R = 4, 12 and 16 the two bases read
0.80–0.85; boundary's analogue 0.528. So the toy is more pessimistic than the 8B's own states with the window's basis
(0.949 in section 7's diagnostic) and still does not reproduce the failure (0.759 against 0.949 there). As fixed, the
second read waits, and the owner is told. Nothing in the toy's settings is changed.

**Why the toy missed (report on PR #72, 2026-11-10; exploratory, on the 80 episodes already read).** Two faults of
the toy, not of the sense. (1) Its contents: section 7's 76/17/8 are medians of each episode's own split, and four
points drawn from those variances are far flatter (third direction's median share 0.008 against 0.082 real; checked
by the agent). With the real contents and the toy's noise the window's basis reads 0.98. (2) Its noise: real
rewording noise is low-dimensional (participation ratio about 7) and tied to content, and competes with the faint
content directions for the basis; isotropic noise in 64 dimensions does not, so the toy barely reproduced the failure.
Replanting the real replies as new loops and dithers reads 0.935 with the window's basis and 0.829 as run.

**The owner (2026-11-10): A2 is recorded as failed, for these reasons, and its gate is lifted.** The second read on
5080–5199 keeps its bars and is the deciding test. A toy matched to the real geometry would be fitted to the data.

**The second read's rehearsal (2026-11-10, the 0.6B on the mirrors 905080–905199, 60 of each kind, none skipped;
decides nothing; recorded before the 8B reads).** `body/area_fg.py --form b2`. Area with the window's basis 0.940
[0.902, 0.972]; area minus boundary +0.514 [0.415, 0.609]; boundary 0.426. Reported: the as-run basis on the same
episodes 0.821; area given boundary 0.940; the Frobenius form 0.932; the interleaved path, now in its window's basis,
0.586; area by layer 0.91–0.95, flat; the noise ratio 0.72. No setting, bar or pool changes; the 8B in bf16 reads
`runs/area-b2-8b` (seeds 5080–5199) next.
