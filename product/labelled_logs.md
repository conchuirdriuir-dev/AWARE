# Labelled logs: goal attention and goal alignment (product step 4)

The implementation record for product step 4: its pre-registration, frozen by the commit before its code, then its
result. The design is the Entity tab's draft "The product's aim: attention on the owner's goal" (2026-11-01), and the
owner's decisions of the same day: in the product the owner's goal takes precedence, and the product is oriented to
the entity's attention on that goal (DECISIONS.md). The labels are the owner's confirmation pass on the 65 departures
(`dara-session-transcripts/study/labels_confirmed.json`); the aims, lines and departures are the Restorations reading
(`study/restorations_reading_v0_1.json`). The corpus is private: references and numbers only, never text.

## Pre-registration (2026-11-01, before its code)

**What it tests.** The batch claim: departures come from losing or misreading the owner's goal, so before a closure
the entity's attention on the goal falls and its attention on its own last thoughts rises, and its work stops matching
the goal. If so, attention on the goal is the right thing for the product to raise.

### Events and controls

- **Departures:** the owner-confirmed real ones, grouped as the text study's run 3 grouped them (closure first, then
  accommodation, then ascent; undecidable-then set aside). **Scored first: the agent's 27 closures.** Reported: the
  agent's 9 accommodations (too few for a bar) and the owner's 23 ascents.
- **Onset:** the departure's time, located in its session's chunks as the text study located it.
- **Controls:** the text study's run 3 controls, unchanged: per session with departures, up to 5 random anchors at the
  agent's text chunks of 200 characters or more, more than W chunks from any departure (seed 0).
- **Windows:** the W = 100 chunks before the anchor, and the 100 after; levels before the anchor, and the change
  across it.
- **The live goal at an anchor:** *local*, the latest aim in that session at or before the anchor; *global*, that
  aim's line. The same rule for departures and controls. Reported beside it for departures: the aim the reading says
  each departed from.

### Readings

**Goal attention and self attention** (the model's attention, read through nnsight). The context is the live aim's
owner turn, then the window's chunks (each capped at 256 tokens), cut from the front to fit T tokens (T = 8,192 on the
8B; 4,096 in the rehearsal here) but always keeping the aim turn first. Attention is taken from every token of the
window's self chunks: averaged over heads and over the last third of the layers, with the sequence's first token left
out (the sink). For each key set (the aim's tokens; the entity's own earlier tokens; other external tokens), observed
attention is divided by what distance alone predicts (the window's own mean attention at each query–key distance,
summed over the set's keys). That gives a **goal attention index G** and a **self attention index S**; 1 means
"as much as recency and size predict".

**Goal alignment** (alone against alone, as validated on the corpus). Each self chunk in the window, read alone,
against the alone readings of all 320 aims' quotes: the live aim's rank among them by distance, scaled so 1 is
nearest and 0 farthest, averaged over the window's self chunks. That is **local alignment A**. The same against the
20 lines' statements gives **global alignment A′**. Read at the last third of the layers, mean of the layer readings.

**The comparison:** alone-against-alone news of the window's arrivals (its median), and the text study's four
statistics from run 3, as the baseline to beat.

### Pass

**Primary, on the agent's closures**, each reading's level in the window before onset against the controls': rank AUC,
one-sided in the predicted direction (S higher, G lower, A lower), with a 90% interval from 2,000 bootstrap draws of
events and controls.

1. **The batch claim is supported** if at least one of S, G and A reaches AUC 0.7 or better with its interval's lower
   bound above 0.5, **and** that AUC beats the best of the text study's four statistics on the same events. Three
   readings are tried, so the interval, not the point, decides.
2. **Ascents are not read as closures:** on the owner's ascents against the same controls, S's AUC (in the closure
   direction) is at most 0.6. An ascent is the origin moving up, which the compass must not break.

Reported, deciding nothing: every reading's change across onset; global alignment A′; accommodation's readings;
alignment with the departed-from aim and, for ascents, with the next aim in the same line; news; per-layer results.

### Amendment, 2026-11-01, before any data

The pre-registered baseline for the attention indices, the window's mean attention at each distance over all keys,
compresses the index it divides: the set being measured feeds its own baseline. In a unit test before any corpus run,
doubling attention on the goal read as G = 1.63, and dragged S to 0.78 though the entity's own tokens were unchanged.
Amended: **expected attention at each distance is the mean over other external tokens** (neither the goal nor the
entity's own), falling back to all keys at a distance where there are none. G then reads attention on the goal against
outside content at the same distance, and S attention on the entity's own tokens against the same. Nothing else
changes.

### Note, 2026-11-01, before any data: how many events are scored

The counts above are the labels': 27 agent closures, 22 owner ascents, 2 agent accommodations (owner-confirmed real;
the 9 named above counted joint and owner accommodations too). The window rule decides which of them can be scored:
an anchor needs its W = 100 chunks before, as run 3's did (its `delta` returns nothing when `i < W`). Seven closures
and eight ascents fall in their session's first 100 chunks, so **20 agent closures and 14 owner ascents are scored**,
against 135 controls. Nothing in the rules changes; this fixes the n each AUC is read on.

### Runs

First a rehearsal here on Qwen3-0.6B (plumbing, draft sets), then the run on the owner's machine with Qwen3-8B, split
across GPU and RAM. Runs resume from checkpoints, one per anchor. The 8B run is the result.
