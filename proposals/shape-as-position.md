# Proposal: shape as position (a new form)

**Not adopted as shape (owner, 2026-11-09, DECISIONS.md).** By the trajectory mathematics a path's signature,
its shape, does not see where the path sits, so position is what shape discards; shape's new form is area
([area-sense.md](area-sense.md)). Section 3's pairing test is kept as a candidate for Lift's re-skin slot, to be
decided there; section 6's report is optional and decides nothing.

*Drafted by the agent 2026-11-09 at the owner's request, for the owner to fix. Nothing here is built or run. Every
choice marked (agent) is the agent's proposal; the bars follow Step −1C's convention and are for the owner to fix
before any code. The idea came from a conversation with another session (the owner's report of it, 2026-11-09);
the form, the test and the risks below are the agent's.*

## 1. Why a new form, not a fourth variant

Shape is parked (owner, 2026-11-07): three forms of one idea (code length under the code fitted so far: joint 0.525,
by source 0.610, in Lift v1 0.573 in development; 0.575 on the 8B, against a bar of 0.80) and none above boundary. It
"returns only with a different idea of what shape measures … as a new form, not another variant of these" (PLAN.md,
"Shape is parked").

This is that different idea. Two observations, both from the record:

- **Cost is what boundary already reads.** Shape (bits for this chunk under the Gaussian fitted to all earlier chunks)
  and boundary (distance of this chunk to its nearest earlier reading) are both one number for how surprising the
  current chunk is against the episode's history. Step −1C was built to part them, on rooms new but on-structure. On
  the 8B, shape given boundary (amendment 1) read **0.490**: once its line on boundary is taken out, nothing of shape
  parts the kinds. Whether shape was a weaker copy of boundary or simply blind (the plan's reading: one Gaussian fitted
  to 50–150 chunks in hundreds of dimensions) the data cannot say, but either way a fourth cost-valued shape is
  likely to re-derive boundary.
- **The counterpart is proprioception, a sense of where, not of how hard.** README's table gives shape as "bits needed
  to describe this turn" and its counterpart as proprioception. Proprioception reports posture: which state the body
  is in. A cost answers "how hard is this moment?"; the question asked of shape is "what state is this?". That is a
  place in code-space, a value with coordinates, not a number of bits.

So the new form changes what shape is, not how it is estimated: **shape is the body's position in code-space over a
short window, a vector; it carries no bits, no distance and no novelty.**

## 2. The form

- **Lift:** Lift v1 (PR #51; the top K principal directions of chunk states from MASC, whitened, frozen before any
  habitat reading; K chosen by the corpus). An 8B lift is built the same way before any deciding reading. No habitat
  pool trains it (Lift v1's own rule).
- **Window (agent):** the last W = 3 turns ending at turn t. Proprioception is a standing sense, so shape is read over
  a stretch, not chunk by chunk; this also keeps its timescale apart from boundary's per-arrival one. W = 1 and 5
  reported.
- **Position (agent):** by source, as the by-source repair taught (a two-token command and a 40-token reply do not
  belong in one fit): the mean of the window's reply chunks and the mean of its action chunks, each in the lift's
  whitened coordinates, concatenated; read layers as Step −1's, averaged over layers as the other senses are. The
  mean is the simplest code of a window (its first moment); a window's fitted subspace is a later form, only if this
  one fails.
- **Read** by the base 8B with any graft off, as every sense is.
- **Nothing is fitted within the episode.** The failure the plan named for shape (a per-episode Gaussian with far fewer
  samples than dimensions) cannot recur here.
- **No distance term.** Shape does not report how far the window is from anything. A distance to the nearest earlier
  position, or to the nearest library code, is a scalar novelty: boundary at a coarser grain. Any such number is
  computed only by scoring, never output by the sense.

### The library, until Dream

The fuller idea is shape as *which code in a library the window sits in*. The library of codes is a Dream slot
(PLAN.md's table: "Dream: library of codes", held-out worlds' code length), after babbling and acting. Three ways to
stand in for it now:

- **(A) No library: coordinates** (the agent's recommendation). Position is the vector above. Whether it carries the
  state is what the test asks; the identity against a library comes with Dream, as a later form tested again.
- **(B) A fixed codebook:** k-means centroids of Lift v1 windows from MASC, the index of the nearest. The codebook
  comes from general text, so its cells may not cut the habitat at its joints; a codebook from habitat play would be
  hand-building the library before Dream.
- **(C) Wait for Dream.** Shape stays parked until a library exists.

## 3. The instrument test: Step −1P, does shape say where the body is? (no training)

*Written before any code; the bars are for the owner to fix before any play.*

**The question.** Do two windows that end in the same engine state sit at the same position, and windows in different
states apart, when the text of the rooms differs? A sense of where must hold when the words change and the place does
not; matching identical text is boundary's and the words' job.

**Play.** Stage 1's scripted player (Step −1's script, no plants), episodes of 100 turns. Every seed is built and
played twice: rendering **A** with `house`'s room descriptions, rendering **B** with every room's description from
`offhouse` (`body/offtheme.py`; written once, domain-free, the same words, other templates). The script never reads a
description (checked in code for Step −1C), so both renderings take the same actions and pass through the same engine
states turn for turn; this is checked in code on every seed (a seed whose states diverge is recorded and skipped).
Other replies (taking, opening, "You can't go that way.") are the engine's own in both.

- *Length (agent):* no length match is required for B. The match in Step −1C served a comparison of costs, which this
  test does not make, and a whole world rarely matches room by room (Step −1C: 14 of 40 single rooms unmatched).
  Tokens per window are reported beside every comparison.

**The key (read only by scoring).** The engine's state at turn t: `facts_key` of the game's facts, as Step −1's
new-state labels use it. Coarser, reported: the room alone.

**Pairs.** Each pair is a window from A ending at turn t and a window from B ending at turn t′ of the same seed, with
|t − t′| ≥ W, so the two windows share no turn (pairs at the same turn would share every action and score the script,
not the place).
- *Same:* the engine state at t equals that at t′ (the agent is back in a state it was in).
- *Different:* the states differ, drawn so that each same pair has different pairs whose gap |t − t′| lies within ±3
  turns of its own (position matched, as every boundary test now is).
- *Hard different, reported:* the same room with other facts (an object taken, a door opened).

**Similarity:** cosine of the two positions (scoring only).

**Pass, both:**
1. **Shape:** the AUC of similarity, same over different, is at least 0.80, its 90% lower bound above 0.70.
2. **Beyond the words:** the AUC minus the text baseline's AUC (TF-IDF of the same windows, by source, cosine, as
   Step 0's text baseline), on the same pairs, has its 90% interval entirely above zero (one-sided).

2,000 draws resampling seeds; 90% intervals. *Too few (agent):* fewer than 30 seeds with at least one same pair, or
fewer than 300 same pairs: too few to decide, and the owner is told; the pool is not grown after any reading.

*Why bar 2 is the text baseline and not boundary.* Boundary is a scalar of one window and cannot score a pair.
Rate passed with its text baseline equal to it, reported only; for shape the words are the central risk, since an
identity of text is not an identity of state, so the baseline decides here.

**Reported, deciding nothing:** the same within one rendering (A with A), expected far easier; the room-only key; hard
different pairs; W = 1 and 5; each source alone; each layer; Lift v0 against Lift v1; tokens per window; the AUC of
boundary's absolute difference between the two windows (does novelty alone pair states?); the partial correlation of
shape-as-cost with boundary on the Step −1C record (section 6).

**Seeds (agent).** The deciding pool: the first 200 usable seeds of **8000–8999**. These were named for shape's
by-source and Lift v1 deciding runs, which the owner chose not to run; none has been played. Skipped, never loosened, as
Step −1C's amendment 2: unbuildable, unplayable (8 GB, 600 s per episode, each rendering), renderings that diverge.
Counts taken before any reading. *Development, deciding nothing:* their mirrors 908000–908039 (+900000), the 0.6B on
the 4060, for plumbing and the interval's width.

## 4. What it would change, if it passes

- **The channel.** b_t carries boundary, X_c and its flag (three numbers). Position has hundreds of coordinates. How it
  enters (the learned projection over all of it, or a fixed few of the lift's leading coordinates) is babbling's
  question, decided before shape enters, and would need Step 2's gate re-run with it. Shape does not enter the sense
  tokens before Step −1P passes (the owner's rule, 2026-11-05).
- **The stall chord.** README's chord is "cheap shape, flat rate, silent boundary"; cheap shape assumes a cost. With
  shape as position it would read *a position revisited* (the body returning to where it was), flat rate, silent
  boundary. The chords are blocked anyway (PLAN.md's table); the chord's new wording is written, with its own test,
  only if shape passes.
- **Step −1C stands as recorded.** Its question was about cost; it is not re-run and its result is not reread.
- **README's table** gives shape as position, its counterpart unchanged.
- **Lift's re-skin slot.** Step −1P also asks whether the lift keeps a place when its words change. The two must not
  share a pass criterion: Lift's slot asks whether the body (the window's code) stays put under a re-skin of the whole
  trajectory; Step −1P asks whether positions identify engine states across renderings. Which goes first is the
  owner's call (section 8, Q5).

## 5. What it needs built

- `body/lift.py` run for the 8B on MASC (as drafted for Lift v1: about 10,000 chunks; the 0.6B took 7 minutes).
- `body/offtheme.py`: a whole-world rendering, every room's description drawn from `offhouse` by the seed, without
  the length match.
- `body/stepm1p.py` (`play`, `read`, `score`) and `tests/test_stepm1p.py`: the two renderings, the state check, the
  windows, the pairs, the bars. Reading reuses Step −1's reader; positions are derived from the recorded chunk states
  and can be rebuilt.
- Nothing in `substrate/`.

Cost: play on the CPU (two renderings of about 230 seeds drawn); the 8B reads about 2 × 200 × 200 chunks, close to
rate's calibration (80,400 chunks in 10.4 hours on the rented 3090).

## 6. First, a cheap report (deciding nothing)

Before any of this is built, from the Step −1C record (`runs/stepm1c-8b/`, news and shape for every chunk; no play,
no reading): the partial correlation of shape (as cost) with boundary, given tokens and turn, within episodes, over
every scored chunk. High says shape-as-cost was boundary again; low says it was blind. Neither changes Step −1C's
result or this proposal's test; it informs how much the redefinition is needed. Length and position are partialled out
because both have confounded these senses before.

## 7. The agent's prediction (written before any play; not to be edited after)

- Within one rendering, same over different reads above 0.90: identical room text makes it easy.
- Across renderings, bar 1 lands between 0.70 and 0.85. Mean-pooled states carry much of their text's surface form,
  and B changes every room description; what holds the place is the room's and objects' names and the replies the
  engine shares across renderings.
- Bar 2 is the hard one. The text baseline sees the shared names too; the agent expects the difference to be small and
  its interval to include zero. A fail there says the position is the words' position, not the body's.

## 8. Risks and the known failure mode

- **Re-deriving boundary.** Held by the form: the sense outputs coordinates, never a distance or a cost. Any scoring
  that turns positions into a scalar novelty and feeds it to the channel would re-derive boundary; this draft does not.
- **A functional result standing in for the instrument.** Step −1P keys on engine state, not on stalls or progress. If
  shape later helps act or babble, that never stands in for this test, and this test never stands in for a sense
  that leads stalls.
- **Hand-building.** `offhouse` was written once for Step −1C, domain-free, and checked in code; the key is the
  engine's own facts. No habitat text trains the lift or any codebook.
- **The key's grain.** `facts_key` treats every fact alike, so two states that differ only in a closed drawer count as
  different. The room-only key and the hard pairs are reported so the owner can see where the line falls.
- **Words carry the place.** If renderings A and B share most of a window's tokens (names, engine replies), bar 1 can
  pass on words; bar 2 is there for this.

## 9. Questions for the owner

1. Does shape return as position (the vector, no bits, no distance), with the library stood in by (A), (B) or (C)?
2. The window: W = 3, by source, the mean in Lift v1?
3. Step −1P's bars as drafted: bar 1 at 0.80 with the lower bound above 0.70, bar 2 one-sided above the text baseline;
   the too-few rule.
4. The pool: 8000–8999 (first 200 usable), mirrors 908000–908039 for development; no length match for rendering B.
5. Order: Step −1P before, after or beside Lift's re-skin slot, and whether either waits on babbling.
6. The cheap report (section 6): run it first, from the Step −1C record?
