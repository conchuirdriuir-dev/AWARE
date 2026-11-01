# Proposal: an amendment to goal attention G, before the 8B run

For the owner, against `product/labelled_logs.md` on `big-picture-v8`. The test is pre-registered, so this is an
amendment before data. It holds only if no comparison of G between closures and controls has been seen yet, on any
model. The session running the test should confirm that the smoke runs looked at plumbing only. Nothing here is
decided.

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

## 2. Proposed amendment

**G over a placebo goal in the same slot.** For each window, run a second attention pass that is identical except
for the goal slot. That slot holds a **placebo aim** instead:
- an owner turn holding an aim from a *different* session and a different line;
- the one nearest in token length (after the 256-token cap) to the live aim's turn;
- ties broken by seed 0.

The window's chunks are cut to the budget left by the longer of the two turns, so both passes read the same chunks.

**G is reported as G(live) / G(placebo).** Position, window length, goal length and row normalisation are the same in
both passes, so they cancel. What is left is how much the entity attends to *this* goal over another goal-shaped
turn in the same place.

In the same check, with attention on the goal raised, raw G reads 2.04, 3.12 and 2.01 across three window and goal
sizes. The ratio reads 1.39, 1.45 and 1.44. (Raising the pre-normalisation weight two-fold does not double the
normalised attention, so the ratio is the honest size of the effect.)

**Also:**
- The fallback in `e(d)` averages over non-goal keys only.
- A unit test with rows summing to 1: under the null the ratio reads 1 within 0.05.

**Unchanged:**
- the bars (AUC ≥ 0.7 with the interval's lower bound above 0.5, beating the best text statistic);
- the events and controls;
- S and A, and the ascent check on S.

Raw G is kept in the report, deciding nothing.

**Cost.** One more attention pass per window. That roughly doubles the attention time on the owner's machine; the
alone readings are unchanged.

**What it does not change.** G is still read in a constructed context, with the aim's turn placed directly before
the window. The result is worded as "attention the model gives the goal when it is in view", not as the attention
the entity gave it in the session.

## 3. If accepted

A dated amendment in `product/labelled_logs.md`, as the first one was. And in DECISIONS.md:

- 2026-11-01: **Goal attention is read over a placebo goal** (owner): G is attention on the live aim's turn over
  attention on a length-matched aim from another session in the same slot, the same window read twice. Raw G carried
  a position bias that varied with window and goal length (null 1.2–2.1 in a synthetic check). Amended before any
  data.
