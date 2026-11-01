# Proposal: goal attention G checked against a placebo goal, before the 8B run

For the owner, against `product/labelled_logs.md`. Nothing here is decided.

**Where it stands.** The test is pre-registered, and the 0.6B rehearsal has since reported G between closures and
controls (AUC 0.64, interval 0.47–0.81, in the predicted direction; labelled plumbing, decides nothing). So this can
no longer be an amendment made before any data. What section 2 proposes only makes G's bar *harder* to pass: raw G
stays the pre-registered reading, and a check is added that it must also meet. A change in that direction cannot
have been chosen to help the claim. The mechanism in section 1 was found from the code and a synthetic check, not
from the rehearsal's numbers.

**The rehearsal is consistent with the bias.** G sits above 1 in both groups (closures 1.22, controls 1.53), where
attention that ignores the goal should give 1. On a real model, the goal draws more than outside content at the same
distance whatever it says.

## 1. What was found

**G is not 1 when attention ignores the goal, and how far it misses depends on window length and goal length.**

G divides the attention self tokens give the goal by `e(d)`: the attention outside tokens receive at the same
query–key distance d. The goal always sits at the front of the sequence. So at any distance d, the goal's keys are
paired with *earlier* queries than the outside keys that make up `e(d)`. Softmax rows sum to 1, and an earlier query
spreads its attention over fewer keys, so each key it sees gets more. The goal comes out ahead on position alone.

The branch's unit test misses this because its synthetic attention is a function of distance only and is not
normalised per row. `proposals/goal_attention_null.py` repeats the test with rows that sum to 1 (half of each row on
the sink) and a random content salience per key that has nothing to do with being the goal:

| | Window 512 tokens | 1,024 | 2,048 |
|---|---|---|---|
| Sharp locality, goal 32 tokens | 1.24 | 1.38 | 1.50 |
| Sharp locality, goal 128 tokens | 1.17 | 1.36 | 1.45 |
| Broad locality, goal 32 tokens | 1.93 | 2.09 | 2.12 |
| Broad locality, goal 128 tokens | 1.23 | 1.37 | 1.38 |

These are null G values (median of 20 windows); every one should read 1. Without row normalisation the same check
reads 1.01.

**Why it matters for the bar.** A constant offset would cancel in the AUC, but this offset is not constant:
- A window is cut to T tokens only when its 100 chunks exceed T. A window of short chunks runs shorter.
- The goal turn's length varies from aim to aim, up to the 256-token cap.

If closures differ from controls in either length (a stretch of many short tool calls before a closure, say), G's AUC
can pass or fail on length alone. Real models also favour early positions beyond what normalisation does, in the same
direction, and the distance baseline does not remove that either.

S is not affected. The entity's own tokens are interleaved with outside tokens, so both sit at the same positions (S
reads 0.95–1.05 in the same check).

**The fallback (minor; this is what was first flagged).** At distances where no outside key exists (the far end of
the window, behind the goal), `e(d)` falls back to the mean over all keys, the goal's own included. That reintroduces
the self-feeding the earlier amendment removed, pulling G toward 1. It touches about goal length / (2 × window
length) of the goal's pairs: about 3% at a 1,024-token window with a 64-token goal, and under 2% at T = 8,192. It is
small and biased toward the null, but worth fixing in the same amendment.

## 2. Proposed amendment (tightening only)

**Raw G stays the pre-registered reading, and its bar is unchanged.** Two things are added before the 8B run:

1. **A length check, reported.** Each anchor's window length in tokens and goal-turn length in tokens are recorded,
   with each one's AUC of closures against controls. If either separates the groups, raw G's AUC partly reads length.
2. **A placebo check that a G pass must also meet.** G counts toward the batch claim only if G(live) / G(placebo)
   (below) also has its 90% interval's lower bound above 0.5 in the predicted direction. If raw G passes and the
   ratio does not, the result reads "G's pass is not distinguishable from position and length", and G does not
   support the claim. S and A are untouched.

**The placebo reading: G over a placebo goal in the same slot.** For each window, run a second attention pass that is identical except
for the goal slot. That slot holds a **placebo aim** instead:
- an owner turn holding an aim from a *different* session and a different line;
- the one nearest in token length (after the 256-token cap) to the live aim's turn;
- ties broken by seed 0.

The window's chunks are cut to the budget left by the longer of the two turns, so both passes read the same chunks.

**The placebo reading is G(live) / G(placebo).** Position, window length, goal length and row normalisation are the same in
both passes, so they cancel. What is left is how much the entity attends to *this* goal over another goal-shaped
turn in the same place.

In the same check, with attention on the goal raised, raw G reads 2.04, 3.12 and 2.01 across three window and goal
sizes. The ratio reads 1.39, 1.45 and 1.44. (Raising the pre-normalisation weight two-fold does not double the
normalised attention, so the ratio is the honest size of the effect.)

**Also:**
- The fallback in `e(d)` averages over non-goal keys only. This touches raw G by under 2% at T = 8,192. It is a
  change to the pre-registered reading, so it is the owner's call; dropping it costs little.
- A unit test with rows summing to 1: under the null the ratio reads 1 within 0.05.

**Unchanged:**
- the bars (AUC ≥ 0.7 with the interval's lower bound above 0.5, beating the best text statistic);
- the events and controls;
- S and A, and the ascent check on S.

**Not proposed: replacing raw G with the ratio.** That would read more cleanly, but after the rehearsal it would
swap a primary reading once its direction had been seen. If the ratio is the better reading, it becomes the primary
in a later test, pre-registered fresh.

**Cost.** One more attention pass per window. That roughly doubles the attention time on the owner's machine; the
alone readings are unchanged.

**What it does not change.** G is still read in a constructed context, with the aim's turn placed directly before
the window. The result is worded as "attention the model gives the goal when it is in view", not as the attention
the entity gave it in the session.

## 3. If accepted

A dated amendment in `product/labelled_logs.md`, after the rehearsal and before the 8B run, saying it tightens
G's bar and why. And in DECISIONS.md:

- 2026-11-01: **A G pass must also hold over a placebo goal** (owner): raw G stays the pre-registered reading, but it
  supports the batch claim only if attention on the live aim's turn over attention on a length-matched aim from
  another session, in the same slot and the same window, also clears 0.5. Window and goal lengths are reported with
  their AUCs. Raw G carries a position bias that varies with window and goal length (null 1.2–2.1 in a synthetic
  check; 1.22 and 1.53 in the 0.6B rehearsal). Added after the rehearsal; it tightens the bar only.
