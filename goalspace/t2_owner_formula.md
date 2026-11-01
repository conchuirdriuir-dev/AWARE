# T2: the owner's formula, inferred from the owner's aims

This file is the implementation record for goal-space test T2 (`proposals/goal-space.md`, accepted by the owner
2026-11-01). It holds the pre-registration, frozen by the commit before any code, and later the result. T2 serves
criterion 3 for the owner's formula: the agent infers the shortest formula behind the owner's statements. A
statement the formula does not explain is flagged.

The corpus (`dara-session-transcripts`) is private. Only references and numbers from it enter this repository or its
records, never text.

## Pre-registration (2026-11-01, before its code)

### Who said what

| Item | By | Used as |
|---|---|---|
| Each aim's quote | **the owner** (verbatim, checked against the transcript by event id) | the data |
| Which owner turns count as aims (320), and their grouping into 20 lines | **a model**: the Restorations reader, a Claude Code session; lines have status "proposed", and the owner's step 2 judgement is not done | the selection of data; the lines are reported only |
| Real, kind and who of the 65 departures | **the owner** (the confirmation pass) | the key for reading 2 |
| Everything computed here | **the code** | |

**Why the 20 lines are not the key.** The proposal planned to score T2 by whether the formula recovers the 20 lines.
The lines turn out to be a model's reading, never confirmed by the owner. Recovering them would measure agreement
between two machine readings, not the owner's formula. T2 is therefore scored against what the owner made:
- the owner's own later aims (does the formula regenerate them?);
- the owner's confirmed ascents (does the formula register where the owner's goal was rewritten?).

Line recovery is reported, labelled as a model's key.

### Material

- **Aims:** the 320 aims of `study/restorations_reading_v0_1.json`, in time order. Each aim's quote is capped at 256
  tokens.
- **Controls:** for each aim, one owner turn from the same session that is not an aim:
  - role human, system turns excluded, at least 40 characters;
  - containing no aim's first 60 characters;
  - nearest in time to the aim, ties broken by seed 0;
  - capped at 256 tokens.

  An aim with no such turn has no control and drops out of reading 1. That count is reported. Controls share their
  aims' sessions and times, so topic and period are matched.
- **Burn-in:** the first 40 aims in time (to 2026-10-07) and their controls. They are used to fit, never scored. That
  leaves 280 aims scored.

### Readings of a text

- **The state:** each text read alone through `substrate.states.Reader`, with the first token left out (the default).
  The layers are the last third of the model's. Each statistic is computed per layer; the primary is its mean over
  those layers.
- **D:** per layer, from the paraphrase sets (`measure_d`; draft until the owner freezes them).

### The formula, and MDL above D

At each aim t, the formula is fitted on the states of all earlier aims (n of them). Controls are handled the same
way, on earlier controls only.

**The model:** probabilistic PCA with k axes:
- a mean μ;
- covariance W Wᵀ + σ² I;
- σ² is the residual variance per dimension, floored at D²/d (d the width).

The floor is MDL above D: variation below D is noise and is not coded.

**Choosing k:** k runs from 0 to min(20, n − 2) and minimises the two-part code on the n earlier states:

  L(k) = −log₂ p(X | PPCA_k) + ½ · (k · (n − 1) + 1) · log₂ n

The parameters are priced within the span of the n states (DECISIONS.md 2026-10-30: the textbook pricing over-admits
axes).

**Code length of a new text:** −log₂ of its density under the formula, in bits. Only differences between code lengths
are read.

**The gain at t:** gₜ = Lₜ(k = 0) − Lₜ(k chosen). That is how many bits the formula's axes save over the mean alone,
in coding the next text before it is seen. This is prequential MDL: each detail is predicted from the code so far.

### Reading 1: the formula regenerates the owner's later aims

For each scored aim t with a control, wₜ = 1 if the aim's gain exceeds its control's gain (each stream fitted on its
own earlier texts), else 0. **W** is the mean of wₜ.

**Pass:** W's 90% interval lies above 0.5, **and** W beats the text baseline's W.
- The interval comes from 2,000 bootstrap draws resampling sessions as blocks.
- The claim is that the owner's aims share a formula that predicts the owner's later aims, more than the owner's
  other talk in the same sessions predicts itself.

### Reading 2: the formula registers the owner's ascents

**Surprisal at aim t:** sₜ = Lₜ(k chosen) − the median of the same quantity over the 20 aims before it. This is how
much harder the aim is to code than recent aims were.

**Events:** the first aim at or after each owner-confirmed ascent, in the same session and after burn-in. Owner-
confirmed means real "yes", kind including ascent, who "owner". There are 16 distinct aims.

**Others:** every other scored aim.

**Pass:** the AUC of s, events over others, is at least 0.7, **and** its 90% interval's lower bound is above 0.5, **and**
it beats the text baseline's AUC.
- The interval comes from 2,000 draws resampling events and others.
- If fewer than 12 events score, the reading is too few to decide.
- The claim is that where the owner rewrote the goal at a higher level, the formula fitted before fails to predict the
  next aim.

### The text baseline

The same procedure, with TF-IDF vectors (word unigrams and bigrams, L2-normalised) in place of states:
- the vocabulary is fitted on the burn-in texts only;
- the text D is the paraphrase sets' spread in that space.

W and the AUC are scale-free, so they compare across the two spaces. Bits do not, and are reported per space only.

### Reported, deciding nothing

- Bits saved per aim and per control.
- k over time.
- Per-layer W and AUC.
- The first aim after each owner-confirmed agent closure (24 events, AUC by the same rule). Closures are the agent's
  departures, so if the owner's formula is what is read, they should move it less than ascents.
- The first aim after each undecidable-then departure (5; too few for a bar).
- **The reader's lines (a model's key):** in the formula fitted on all aims, the AUC of same-line pairs sitting nearer
  than different-line pairs. Also the adjusted Rand index of 20-cluster k-means against the lines.

### Keys judge, never supply

The confirmation labels and the reader's lines are read only by the scoring functions. Those run after every code
length has been computed and checkpointed. The formula never sees a label, a line or a departure.

### Note, 2026-11-01, written with the code, before any data

The rules above leave open whether one owner turn can be the control for two aims. They settle it as follows:
- **A control is not reused.** Aims are taken in time order. If an aim's nearest qualifying turn is already taken, it
  gets the next nearest. A repeated control would be coded almost free the second time, which would inflate the
  controls' gain.
- **The prequential streams begin at the 20th text, not the 40th.** This is so reading 2's running median of 20
  earlier aims exists from the first scored aim. Scoring still starts after burn-in, at aim 41.

Nothing else changes.

### Runs

1. A rehearsal on Qwen3-0.6B with draft sets: plumbing, deciding nothing.
2. The run on the owner's machine with Qwen3-8B, split across GPU and RAM. That run is the result.

Runs resume from per-session checkpoints, as the echo floor's do.

### Limits, stated before the result

- The reader chose which owner turns are aims. A formula over aims may partly recover the reader's notion of an aim.
  The controls are owner turns from the same sessions, and the mean-only baseline absorbs a constant difference in
  register, but a register that varies across aims could still count toward reading 1.
- 16 ascent events is pilot-scale. The interval decides, not the point estimate.
- Aims are the owner's words, read by the model alone. The formula is in the model's goal-space, not in the owner's
  head. A pass shows that the owner's pattern is recoverable from mechanism, which is what the goal statement asks.
