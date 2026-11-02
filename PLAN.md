# Plan

The design is the Big Picture doc, v9 ([Big Picture — v9](https://claude.ai/artifact/ByrrhB2Ue4NS8NzmyZXoCw), a
Claude Doc). Its **Entity** tab is the parent: what the entity is, the substrate decisions, the capacities, and the two
tracks. The **Drift detection (v10)** tab is the product track's full treatment (v9's is kept as the record);
**Tech stack** sets the build order. Gate 0 ends nothing: it is refined until the product passes it, and the doc's
"can end the project" line is ignored (owner, DECISIONS.md).
Where this file and the doc differ, the doc wins and this file is corrected.

**Two tracks, one substrate, kept apart.** The product track (drift detection, then self-correction) has the demo and
fast feedback; the research track (reasoning over the entity's own axes) has no demo yet and is the reason the project
exists. They never share a pass criterion: the product's faster feedback would win every local decision. The
substrate changes only by the owner's decision.

A step's pass criterion is written here before its code, and its thresholds are fixed before it runs. A step is done
when its test passes, not when its code exists.

## Substrate (`substrate/`)

The Entity tab's decisions, as code: the base model frozen (states read through nnterp, `states.py`); the code
retained, not consumed (`axes.py`: every axis considered is kept); admission by two criteria (predictive: lowers
held-out residual by more than D; otherwise structural, tagged untested, until the research track states a test);
provenance logged at the source (`record.py`: self or external on every chunk).

| Step | Test | Status |
|---|---|---|
| 1. Model access and D | states on the real model; D per layer on frozen paraphrase sets | **code built; plumbing passes on real Qwen3-0.6B weights on CPU (2026-10-31); awaits the card and the frozen sets** |
| 2. Logging | per-chunk activation capture, in context and alone; provenance tagged at creation and carried when re-read | **reader and record built; the harness that writes them is product step 4** |

**State** (provisional): the residual stream after a layer, mean-pooled over a chunk's tokens (special tokens and a
sequence's first token left out: an attention sink where there is no begin-of-sequence token), read alone (`states`) or inside the whole sequence (`states_in_context`). **D**: the root-mean-square distance
of a state from its paraphrase set's mean, averaged over sets, per layer; the layer proposed is the one with the
largest between-to-within ratio, and the owner chooses. **Material:** 20 sets of 5 paraphrases
(`material/paraphrases.jsonl`), drafted by the model, frozen by the owner 2026-11-02. **Pass:** states for every text on both models
at every layer asked for, and the same text read twice gives the same state.

**The echo floor** (DECISIONS.md, the event record): the distance below which an arrival is the entity's own content
coming back. It is measured, not assumed; rounds 7a and 7b found that detecting repeated content waits on it.

*Dress rehearsal, written 2026-10-31 before its code* (`python -m substrate.echo_floor`). On Qwen3-0.6B on CPU, with
the draft paraphrase sets, so it is labelled draft throughout: plumbing and a first look, never the result, which is
the 8B on the card with frozen sets.
- **Sessions:** 10, each 8 statements drawn from different sets (seed 0), joined by blank lines and read in one pass;
  each statement's state is read in context.
- **Arrivals**, each read alone and measured against the nearest of its session's in-context states:
  - an **exact echo**, a statement of the session, verbatim;
  - a **paraphrased echo**, another paraphrase of a statement of the session;
  - **news**, a statement from a set the session does not contain.
- **Per layer** (every third of Qwen3-0.6B's 28, and the last): D from the paraphrase sets (`measure_d`, draft);
  the three distances' medians and spreads in units of D; the AUC of news over exact echo and over paraphrased echo;
  and how exact-echo distance varies with the statement's position in its session.
- **How it is read:** repeats are separable from news at a layer if news ranks above exact echo at AUC 0.9 or better.
  The paraphrased echo is reported beside it: whether a repeat in other words is still a repeat is a design question,
  not settled here. The toy's assumption, that an echo sits about √2·D from what it repeats, is checked against the
  measured ratio.

*Dress rehearsal result* (2026-10-31; Qwen3-0.6B on CPU, draft sets; records in `runs/echo-qwen06/` and
`runs/echo-qwen06-skip/`, not committed). **As written, it measured an artefact.** Qwen3 has no begin-of-sequence
token, so the first token of every sequence is an attention sink with massive activations. Every text read alone
starts a sequence and carries it; in a session only the first statement does. So D jumped from 1.3 to about 41 at
layer 3, news landed nearest the first statement (its 5th percentile at 0.26 D), and news ranked above exact echoes at
only 0.59 from layer 3 to 24. Post hoc, with each sequence's first token left out (`--skip-first`):

| Layer | D | Exact echo | Paraphrased echo | News (5th pct) | AUC news over echo / over paraphrase |
|---|---|---|---|---|---|
| 0 | 1.3 | 0.84 | 1.56 | 2.14 (1.83) | 1.00 / 0.98 |
| 3–9 | 2.6–5.2 | 1.00–1.52 | 1.61–1.90 | 2.04–2.54 (1.79–2.28) | 1.00 / 0.96–1.00 |
| 12–18 | 7.7–18 | 1.81–2.18 | 2.06–2.31 | 2.54–2.57 (2.16–2.19) | 0.90–0.99 / 0.74–0.92 |
| 21–27 | 33–69 | 1.32–1.67 | 1.76–2.02 | 2.53–2.73 (2.24–2.47) | 1.00 / 0.97–1.00 |

Medians in units of D. **Repeats are separable from news at every layer,** most cleanly at 21–27, where news's 5th
percentile clears the exact echo's 95th. The toy's √2·D (1.41) is close to the real echo floor at 21–27 and too low in
the middle layers. **Echo distance grows with a statement's position in its session** (correlation 0.6–0.8): the
echo floor is a function of context position, as the biologist review proposed for D (finding 11). Sessions here are
8 statements long, so how far it grows over a real session is not known. A small model on draft sets: plumbing and a
first look.

*Long sessions, written 2026-10-31 before their code* (`python -m substrate.echo_floor --long`). Does the echo floor
hold as a session grows? On Qwen3-4B in bfloat16 on CPU (the 8B does not fit this container's 15 GB; the owner's
RTX 4060 with 32 GB of RAM runs the 8B, split between GPU and RAM), draft sets, first token left out.
- **Sessions:** 4, each holding 14 sets' four paraphrases other than the second (56 statements, shuffled), read in one
  pass. The other 6 sets supply news; each session set's second paraphrase is its paraphrased echo.
- **Measured:** each exact echo's distance to the nearest in-context state, by its original's position (1–8, 9–24,
  25–56); news's distance to the nearest of all 56; both in units of D; the AUC of news over exact echo per bin.
- **How it is read:** the floor holds at length if news ranks above exact echo at AUC 0.9 or better in the last bin
  (25–56) at the layers that separated best in the short sessions (21–27 of 28 there; the 4B's last third here).

*Long-session result* (2026-10-31; Qwen3-4B, bfloat16 on CPU, draft sets, first token left out; 4 sessions of 56
statements; record in `runs/echo-qwen4b-long/`, not committed). Medians, with exact echoes' 95th percentile after the
slash, in units of D:

| Layer | News (5th) | Echo, positions 1–8 | 9–24 | 25–56 | AUC 1–8 / 9–24 / 25–56 |
|---|---|---|---|---|---|
| 8 | 2.18 (1.95) | 1.19 / 1.48 | 1.43 / 1.60 | 1.55 / 1.81 | 1.00 / 1.00 / 1.00 |
| 24 | 2.81 (2.41) | 1.97 / 2.32 | 2.43 / 2.72 | 2.57 / 2.85 | 1.00 / 0.91 / **0.78** |
| 28 | 2.72 (2.39) | 1.56 / 1.81 | 1.99 / 2.25 | 2.25 / 2.56 | 1.00 / 1.00 / 0.95 |
| 32 | 2.64 (2.36) | 1.42 / 1.65 | 1.80 / 2.16 | 2.08 / 2.38 | 1.00 / 1.00 / 0.99 |
| 35 | 2.44 (2.14) | 1.49 / 2.23 | 1.89 / 2.51 | 2.09 / 2.59 | 0.98 / 0.89 / **0.83** |

**Read as written: the floor holds in the last bin at two of the four last-third layers (28, 32), not at 24 or 35.**
At every layer the echo floor grows with position (at 32: 1.42 to 2.08 D), while news stays put, so the margin
narrows as a session grows: at 32 the last bin's 95th percentile (2.38) already meets news's 5th (2.36). These
sessions are about 1,000 tokens; real ones run to tens of thousands. Whether repeats stay separable at real lengths is
not settled by this, and the echo floor has to be read as a function of position, not a constant. Layer 8 holds
cleanly throughout, but an early layer is close to the tokens themselves, so its separation of exact repeats may not
carry to repeats in other words.

*Real sessions, written 2026-10-31 before their code* (`python -m substrate.echo_floor --corpus DIR`). Does the floor
hold at the lengths of real agent work? On the owner's 45 Claude Code sessions (`dara-session-transcripts`, private:
no text from it enters this repository or its records, only session ids, positions and distances).
- **Chunks:** each session's events parsed into chunks with their provenance and arrival cause: the agent's text and
  tool calls are self; tool results are external and elicited; the owner's messages are external and spontaneous.
  Each chunk is capped at 256 tokens (truncation recorded).
- **Windows:** the first W tokens of chunks of each of 10 sessions (W = 4,096 for the rehearsal in this container on
  Qwen3-0.6B; the owner's run sets W up to 32,768 with the 8B), read in one pass.
- **Exact echoes:** the window's self chunks, each read alone, against the nearest in-context state, by the token
  position of the original (0–512, 512–1,024, 1,024–2,048, 2,048–W).
- **News:** self chunks from *other* sessions, read alone, against the nearest in-context state of the window.
- **Reported in units of D** (the paraphrase sets', which are short statements, so also as a fraction of news's
  median), with the AUC of news over exact echo per position bin.
- **How it is read:** the floor holds at real lengths if news ranks above exact echo at AUC 0.9 or better in the last
  position bin, at the layers that held in the long-session test.

*Real-session result* (2026-10-31; Qwen3-0.6B on CPU, draft sets, first token left out; 4,096-token windows of 10
sessions, 160 echoes, 500 news; record in `runs/echo-corpus-qwen06/`, references only, not committed). Exact echo's
median as a fraction of news's median, and the AUC of news over echo, by token position:

| Layer | 0–512 | 512–1,024 | 1,024–2,048 | 2,048–4,096 |
|---|---|---|---|---|
| 0 | 0.39, 0.99 | 0.38, 0.97 | 0.41, 0.98 | 0.41, 0.98 |
| 12 | 0.65, 0.88 | 0.78, 0.76 | 0.90, 0.64 | 0.93, **0.63** |
| 21 | 0.52, 0.97 | 0.68, 0.87 | 0.83, 0.77 | 0.86, **0.75** |
| 27 | 0.51, 0.94 | 0.55, 0.92 | 0.65, 0.79 | 0.65, **0.81** |

**Read as written: the floor does not hold at real lengths.** Within about a thousand tokens of real agent work, an
echo read alone sits nearly as far from its own in-context original as news does, at every layer but the first,
which is close to the tokens themselves. The fault is the comparison, not the floor: an arrival read alone is measured
against states read in context, and context reshapes a state the deeper it sits. **Post hoc, alone against alone**
(the D run's states): a paraphrase of a statement sits nearer its original than the nearest different statement at
AUC 0.92–1.00 at every layer, and position cannot enter. On short, distinct statements only; real chunks of code and
tool output will crowd closer.

*Alone against alone on real sessions, written 2026-10-31 before its code* (`python -m substrate.echo_floor --corpus
DIR --alone`). With news now read alone against alone (DECISIONS.md), does it tell repeats from new content in real
agent work, where chunks of code and tool output crowd close?
- **Chunks:** the first 300 of each of 10 sessions (seed 0), capped at 256 tokens, each read alone.
- **Classes,** by text alone (word 3-gram Jaccard against every earlier chunk in the session; no text recorded):
  *exact repeats* (identical text to an earlier chunk); *near repeats* (overlap 0.8 or more, not identical); *new*
  (overlap below 0.2 with every earlier chunk). Chunks in between are left out.
- **Measured:** each chunk's distance to the nearest earlier chunk's alone reading, in units of D; the AUC of new over
  near repeat per layer; the same AUC in the first and last third of each session's chunks.
- **How it is read:** alone against alone works on real sessions if new ranks above near repeat at AUC 0.9 or better
  at the late layers (the last third), and the AUC in the last third of chunks is within 0.05 of the first third's
  (position does not enter). Exact repeats are reported; they should sit at zero.

*Alone-against-alone result* (2026-10-31; Qwen3-0.6B, draft sets, first token left out; 10 sessions, 2,987 chunks:
48 exact repeats, 15 near repeats, 2,403 new, 521 between; record in `runs/echo-alone-qwen06/`, references only, not
committed). A first run was broken by a reader bug (one-token chunks pooled over nothing, giving NaN) and is not a
result. In units of D:

| Layer | New, median (5th) | Near repeat, median (95th) | Exact repeat, max | AUC: all / first third / last third |
|---|---|---|---|---|
| 0 | 0.80 (0.46) | 0.21 (0.24) | 0.000 | 1.00 / 1.00 / 1.00 |
| 12 | 1.04 (0.60) | 0.20 (0.37) | 0.000 | 1.00 / 1.00 / 1.00 |
| 21 | 1.26 (0.72) | 0.21 (0.37) | 0.000 | 1.00 / 1.00 / 1.00 |
| 27 | 1.59 (0.91) | 0.29 (0.46) | 0.000 | 1.00 / 1.00 / 1.00 |

**Read as written: it works.** New content ranks above near repeats at AUC 1.00 at every layer, in the first third of
sessions and the last alike, and exact repeats sit at zero. Two cautions. Only 15 near repeats were found, and they
are near-repeats by text overlap, the easy kind; a retried call with different arguments, or a repeat in other words,
is not tested here. And **the floor is not D.** New content's nearest earlier chunk sits at 0.5–0.9 D at its 5th
percentile, below D at every layer: real chunks crowd closer than the short paraphrase statements D was measured on.
A floor of D would call much new content a repeat. On this corpus the line falls between near repeats' 95th
percentile (0.24–0.46 D) and new content's 5th (0.46–0.91 D), so the floor has to be measured on real repeats, as here.

*Alone against alone on the 8B* (2026-11-01; Qwen3-8B, bfloat16, split across the owner's RTX 4060 and RAM; draft
sets; the same 10 sessions and 2,987 chunks). In units of D:

| Layer | D | New, median (5th) | Near repeat, median (95th) | AUC: all / first third / last third |
|---|---|---|---|---|
| 0 | 1.88 | 0.63 (0.35) | 0.16 (0.17) | 1.00 / 1.00 / 1.00 |
| 4 | 5.17 | 0.88 (0.50) | 0.25 (0.29) | 1.00 / 1.00 / 1.00 |
| 8–20 | 69–77 | 0.13–0.32 (0.08–0.19) | 0.03–0.05 (0.04–0.06) | 1.00 at every layer |
| 24–32 | 91–153 | 0.52–0.89 (0.30–0.51) | 0.08–0.13 (0.12–0.22) | 1.00 at every layer |
| 35 | 187 | 1.41 (0.77) | 0.29 (0.42) | 1.00 / 1.00 / 1.00 |

**Read as written: it holds on the 8B,** as on the 0.6B: new content above near repeats at AUC 1.00 at every layer,
first and last third alike, exact repeats at zero. The margin is wide: new content's 5th percentile is two to four
times near repeats' 95th. One flag: **D jumps from 5.2 at layer 4 to 69 at layer 8** and stays there, which looks like
massive activations in a few dimensions on tokens other than the first. It does not touch the AUCs (a ratio), but D
is a poor unit at those layers until it is understood. The floor, read off the real distribution: at layers 28–32,
between 0.17–0.22 D and 0.41–0.51 D.

*Retried calls, written 2026-11-01 before their code* (`python -m substrate.echo_floor --corpus DIR --retries`). An
agent stuck in a loop re-issues calls; does news, alone against alone, follow what comes back, not the fact that a call
repeated? On the owner's corpus, references only.
- **Calls and results:** tool calls are linked to their results by call id. A **retried call** has the same tool name
  as an earlier call in the session and arguments at word 3-gram Jaccard 0.5 or more, short of identical; an **exact
  retry** has identical arguments; a **fresh call** has the same tool name but arguments below 0.2 with every earlier
  call of that tool.
- **The results' content,** by text against every earlier chunk: *repeated* (overlap 0.8 or more, or identical),
  *new* (below 0.2), or between (left out).
- **The floor, measured in the same run** from all chunks' classes (as the alone-against-alone check): per layer, the
  midpoint between near and exact repeats' 95th percentile and new chunks' 5th percentile of distance to the nearest
  earlier chunk, alone against alone.
- **Measured:** each result's distance to its nearest earlier chunk; the share of retried and exact-retry calls'
  results below the floor when their content is repeated, and above it when it is new; the same for fresh calls.
- **How it is read:** news follows content on retried calls if, at the late layers (the last third), at least 90% of
  retried calls' results with repeated content read below the floor and at least 90% with new content read above it.
  A class with fewer than 10 results is reported as too few to read. Repeats in other words (the same content
  reworded) cannot be labelled from text, and stay untested.
- First on Qwen3-0.6B here (the first 1,000 chunks of 10 sessions, seed 0, draft sets); then on the owner's machine.
  *Amended before any result, 2026-11-01:* the first 400 chunks held only 30 retried calls and 10 exact retries (call
  counts only; no distance had been computed), too few once split by their results' content, so 1,000.

*Retried calls, result on Qwen3-0.6B* (2026-11-01; float32, draft sets; the first 1,000 chunks of the same 10
sessions). Share of results on the right side of the floor (below it when the content repeats, above it when new),
with counts; the floor in units of D:

| Layer | Floor | Retried call: repeated / new | Exact retry: repeated / new | Fresh call: repeated / new |
|---|---|---|---|---|
| 0 | 0.32 | 1.00 (6) / 0.96 (24) | 1.00 (19) / 1.00 (2) | 0.97 (183) / 0.99 (1,802) |
| 12 | 0.44 | 1.00 (6) / 0.96 (24) | 1.00 (19) / 1.00 (2) | 0.95 (183) / 0.99 (1,802) |
| 21 | 0.51 | 1.00 (6) / 0.96 (24) | 1.00 (19) / 1.00 (2) | 0.95 (183) / 0.98 (1,802) |
| 24 | 0.53 | 1.00 (6) / 0.96 (24) | 1.00 (19) / 1.00 (2) | 0.95 (183) / 0.98 (1,802) |
| 27 | 0.62 | 1.00 (6) / 0.96 (24) | 1.00 (19) / 1.00 (2) | 0.95 (183) / 0.99 (1,802) |

**Read as written: not decided.** Retried calls whose results repeat number 6, under the 10 the rule needs, so that
half is too few to read. The half that can be read passes: 96% of retried calls returning new content read as new
(24 results, at every layer), so news follows what comes back, not the fact that the call repeated. Beside it,
reported only: exact retries returning repeated content read as repeats every time (19), and fresh calls separate at
95% and 98–99% on large counts. Retried calls that bring back the same content are rare in these 10 sessions; deciding
the rule needs more sessions, which is the owner's call (more data, the same rule, written before it is run).

## Product track (`product/`): drift detection, then self-correction

| Step | Doc | Test | Status |
|---|---|---|---|
| 0. Text-only floor | v9 step 0 | the text panel on the owner's 45 sessions | **done by the owner, twice:** text sees the owner leave an aim (AUC 0.65–0.71), not the agent |
| 3. Toy bed | v9, v10 and the v10 draft | below | **eight rounds run (below); detecting repeats now waits on the real model's echo floor** |
| 4. Labelled logs | Tests, in order; Entity tab, the product's aim (draft) | [goal attention and goal alignment](product/labelled_logs.md) on the owner-confirmed departures, the agent's 27 closures first | **8B, 2026-11-02: batch claim not supported (no reading separates closures); ascents pass; placebo stage pending, reported only** |
| 5. Reference scorer, coherence score | Tests, in order | intact versus shuffled work | open: needs the setting and the owner's reference work |
| 6. Compass and harness, arm A; Gate 0 | Tests, in order | the compass predicts drift on held-out stretches above a pre-registered level, and beats self-residual alone | open. Refined until passed; ends nothing |
| 7. Arms S and C; main demo | Main demo | C closes half the A–B gap on both scores and beats S by more than run-to-run noise | open |

### Step 3: the toy bed

Each round's pre-registration and result live in their own file, frozen by the commit before the round's code.

| Round | Design | Result |
|---|---|---|
| [1](product/rounds/round1.md) | v9's detectors: point and trajectory contrast, the observer's band, margin and frame growth | 0 of 3 bars (the margin caught dullness, 0.86, not proliferation); frame growth fails v9's decisive row |
| [2](product/rounds/round2.md) | v10: coupling, dynamic residual, admission, spectral descriptor | 3 of 4; coupling passes its bars; separation fails on the agent's rule for "at zero" |
| [3](product/rounds/round3.md) | sparse externals, at zero against the post-intake reference, the probe, shrinkage, framing | 0 of 6; reviewed: a misspecified observer for held externals, not a lack of power (`reviews/round3/triage.md`) |
| [4](product/rounds/round4.md) | event-locked coupling: innovations, news and gain, a news budget, the probe by rule | 5 of 8; news and gain pass every coupling bar at every echo lag, with honest intervals; misses in the carried-over contrast detectors, the re-read window and the definition of K |
| [5](product/rounds/round5.md) | the event record: separation from events, the probe read by its own responses, the response profile, a blind structure, the freeze rule | 8 of 9, the blind structure among them; separation misses on a news-spike rule (the builder's) and a scalar gain that cannot see a subspace dynamics change; not frozen |
| [6](product/rounds/round6.md) | the news-spike rule fixed; separation narrowed to the product's need; the residual descriptor as discovery's own test; a new blind structure | product 7 of 9; discovery fails; the blind structure (a partner replaying old content beyond the history) read open in every run; not frozen |
| [7a](product/rounds/round7a.md) | arrival cause; short and long news; the partner-side loop on long news over spontaneous arrivals; replay and confirmations as known structures; a new blind structure | 7 of 10; separation passes for the first time; misses: a floor in the wrong unit (replay), closed loops never probed (against the design), every elicited arrival treated as a confirmation (the blind structure) |
| [7b](product/rounds/round7b.md) | 7a's fixes; the probe as a contingency test; entity captured; persistence and recurrence; six reactive or nonlinear conditions; the self-drive term deferred | 7 of 12; entity captured, probing first, short-lag mirrors and the attractor loop work; repeats (replay, the blind retry loop) sit on the toy's floor; not frozen |

## Research track (`research/`): reasoning over the entity's own axes

No demo and no pass bar for the track yet, by design; each probe still has its reading written before its code. The
focus since 2026-11-02 (DECISIONS.md). What it has:

| Item | Test | Status |
|---|---|---|
| The D jump | below | **explained, 2026-11-02: the attention sink landing on the second token in texts beginning "Find"; the reader now finds it by norm (owner); D on the 8B re-measured, J 2.13** |
| Gate 1: negation | mirror, additive shift or contraction toward the origin; the winner beats both others by more than D | `research/gate1.py` built and tested on planted pairs, with a fixed origin; lifted out of the build order; **pre-registered below, centre fitted; pairs elicited and read by the owner; `research/gate1_run.py` runs it** |
| Structural axes | retained, tagged untested (`substrate/axes.py`) | waiting for a structural test from this track |
| Relations, composition, order | angles, nesting, symmetry; predicted position under combined axes; non-commuting moves | no probe yet |

### The D jump

*Written 2026-11-02 before its code* (`python -m research.d_jump`). On Qwen3-8B, D goes from 5.17 at layer 4 to 69 at
layer 8 and stays high, unlike the 0.6B's smooth growth. D is the margin Gate 1 must clear, so what is it measuring?
Qwen3-8B, bfloat16, split across the owner's RTX 4060 and RAM as in `runs/echo-alone-qwen8b/`; the same 20 draft sets
of 5, first token left out. The model is only read; `substrate/` does not change.
- **Layers:** 0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20, 24, 28, 32, 35.
- **Check first:** D recomputed from per-token states must match the saved run within 1% at the layers in common, or
  nothing below is read.
- **The jump:** J = D(8) / D(4), recomputed under each condition as J′. Whichever single step of 4→8 holds the most of
  log J is reported.
- **Three causes, each read on its own; more than one may hold:**
  - *A few tokens.* Each text's largest-norm token (top 1, then top 3, chosen per layer) left out of the mean. Holds if
    J′ ≤ 3 and the control, a random token left out (seed 0), keeps J′ ≥ 0.8 J. Top 1 and top 3 are read separately.
  - *A few dimensions.* The 5 coordinates with most within-set variance, chosen per layer on half the sets and D
    measured with them removed on the other half, both ways, averaged. Holds if J′ ≤ 3 and the control, 5 random
    coordinates (seed 0), keeps J′ ≥ 0.8 J.
  - *Overall scale.* The median token norm's ratio, layer 8 over layer 4. Holds if it is J / 3 or more.
- **Why 3:** the 0.6B's own growth over comparable stretches is about 2 (layers 3 to 9).
- **Reported beside them:** between-to-within ratio per layer and condition; the second and the last token left out;
  which tokens are largest (string, position) and how much of their squared norm sits in their top 5 coordinates; D's
  jump on the same draft sets is the only thing explained, so no corpus text is read.
- **If none holds,** the jump is reported as not explained, and any new cause is written here before it is tried. What
  follows for D is a substrate decision, the owner's.

*Result* (2026-11-02; Qwen3-8B, bfloat16, split across the RTX 4060 and RAM, draft sets; record in
`runs/d-jump-qwen8b/`, not committed). D matches the saved run exactly at every layer in common. J = 13.30, and one
step holds 91% of log J: layer 5→6, ×10.67.

| Cause | J′ | Control J′ (needs ≥ 10.64) | Needs | Holds |
|---|---|---|---|---|
| Top 1 token left out | 2.24 | 13.60 | ≤ 3 | yes |
| Top 3 tokens left out | 2.30 | 13.60 | ≤ 3 | yes |
| 5 coordinates removed | 2.42 | 13.31 | ≤ 3 | yes |
| Overall scale (median token norm, 8 over 4) | 1.85 | none | ≥ 4.43 | no |

| Layer | 4 | 5 | 6 | 8 | 12 | 20 | 24 | 28 | 32 | 35 |
|---|---|---|---|---|---|---|---|---|---|---|
| D | 5.17 | 6.18 | 65.97 | 68.82 | 69.95 | 77.41 | 91.43 | 115.84 | 152.79 | 186.96 |
| D, top token left out | 5.23 | 6.43 | 8.02 | 11.68 | 13.24 | 20.16 | 38.51 | 68.76 | 113.43 | 183.26 |
| Between over D | 1.14 | 1.17 | 0.93 | 0.90 | 0.91 | 0.90 | 0.95 | 1.11 | 1.27 | 1.38 |
| Between over D, top token left out | 1.15 | 1.15 | 1.15 | 1.25 | 1.61 | 1.54 | 1.57 | 1.61 | 1.59 | 1.39 |

**Read as written: a few tokens and a few coordinates both hold; overall scale does not.** Leaving out the second
token does the same as the largest (J′ 2.14); leaving out the last does nothing (13.43).

*Post hoc, no threshold fixed* (`research/d_jump_posthoc.py`, layer 6). The two causes are one: **the attention sink,
landing on the second token in the 4 of 100 texts that begin "Find".** Those are the only "Find" texts, one each in 4
of the 20 sets. In the other 96 the first token has norm 8,297–13,981, about 100% of its squared norm in coordinates
2276, 233 and 4081, and the second token is ordinary (at most 68). In the 4, the first token has norm 160 and the
second 8,755–10,391, 99.9% in those same coordinates. The reader leaves out the first token by position
(`skip_first`), so here it drops an ordinary token and keeps the sink. That one token moves its text's state by about
750, and 4 sets of 20 holding one account for about 60 of D's 66 at layer 6. It still dominates at layer 24 (42 times
the median) and fades to 3 times by layer 35.

**What this leaves.** D on the 8B is inflated at layers 6–32 on these sets, about sixfold at layer 8 and 1.35-fold at
layer 32, and separation is understated. Coordinate removal is not the remedy: 2276 is also a top coordinate of
ordinary tokens at layers 24–35. Not tested: other models, what other first words do this, and whether corpus chunks
carry it. Results in units of D on the 8B (the floors in the 8B table above) use the inflated D; their AUCs are ratios
of distances and do not use D. Corpus chunks are read by the same reader, so a chunk starting this way would carry the
sink; that is unmeasured. What to do about it is a substrate decision, open.

*The sink by norm, written 2026-11-02 before its code* (`python -m research.sink_check`; owner's decision,
DECISIONS.md). **The rule:** in one forward pass, read token norms at the layer a third of the way up (12 of 36, chosen
because the sink is well formed there and the layer is not tuned to a result); a token is the sink if its norm exceeds
5 times the median of the sequence's other tokens' norms, taking the median without the largest token, so that a
two-token text can still flag. Flagged tokens are left out of the mean at every layer read. A text or chunk that would
be left empty keeps its tokens. Special tokens stay out, as before. The rule was chosen after seeing these 100 texts'
norms, so the sets are a check that it does what was meant; the corpus is the first data it has not seen.
- **On the 100 draft texts,** Qwen3-8B, same load and layers as the D jump. It passes if all of:
  1. exactly one token is left out of every text: token 0 in the 96, token 1 in the 4 beginning "Find";
  2. J = D(8) / D(4) is 3 or less;
  3. in the 96, states match the saved position-rule run at the layers in common: the largest distance is at most 1%
     of that state's norm;
  4. the same text read twice gives the same state, as in step 1.
- **On the corpus,** 30 chunks from each of the same 10 sessions (seed 0; capped at 256 tokens; references and norms
  only, no text recorded). It holds for real chunks if at least 99% of chunks of 3 or more tokens have exactly one
  token left out. Reported beside it: shares with none and with two or more; the position of the token left out; how
  many sinks sit somewhere other than token 0 and which is the largest non-flagged ratio. Under 99%, the rule is not
  used on the corpus until the owner has seen why.
- **Reported, not judged:** D and between-to-within per layer under the rule, against the position rule.

*Result* (2026-11-02; record in `runs/sink-check-qwen8b/`, not committed; its first attempt was killed by a time limit
during the corpus part, and that attempt's lines stay in the record).

| Check | Bar | Measured | Passes |
|---|---|---|---|
| One token left out, where expected | 96 at token 0, 4 at token 1 | exactly that | yes |
| J = D(8) / D(4) | ≤ 3 | 2.13 (13.30 under the position rule) | yes |
| Distance from the position rule, the 96 | ≤ 1% | 0 | yes |
| Same text read twice | same state | same | yes |
| **Corpus: exactly one token left out** | **≥ 99%** | **98.98% (290 of 293)** | **no** |

| Layer | 4 | 6 | 8 | 12 | 20 | 24 | 28 | 32 | 35 |
|---|---|---|---|---|---|---|---|---|---|
| D | 5.46 | 8.00 | 11.61 | 13.27 | 19.57 | 37.94 | 68.52 | 113.19 | 184.99 |
| Between over D | 1.09 | 1.17 | 1.24 | 1.55 | 1.62 | 1.61 | 1.64 | 1.63 | 1.39 |

**Read as written: the rule passes on the texts and misses on the corpus by one chunk's worth, so it is not used on the
corpus until the owner has seen why.** Why, post hoc: the other 3 chunks each have tokens 0 and 1 both flagged, at
10,188–10,382 against a median of 65–66, the two equal to the nearest unit within each chunk. So two sinks, both left
out; "exactly one" assumed one sink per sequence. Nothing unflagged came near the factor: the largest other token is 2.90 times the
median, against 5. Every one of the 293 chunks had token 0 flagged, so on this sample no chunk had a sink only on
token 1, and the position rule's fault on real chunks is those 3, which keep one sink. Why two tokens share a sink is
not known. No corpus text was read to look at this.

### Gate 1: negation, standalone

*Written 2026-11-02 before its code; amended only before any run, and any amendment dated here.* The owner's decisions
(DECISIONS.md): standalone on elicited pairs; the centre fitted from the pairs, then compared with candidate origins.
This changes the doc's "about the origin" into two questions, in order: what form does negation take, and is its
centre the origin. What a pass would mean is unchanged: a finding about the code; it cannot end the project.
- **Material:** 20 topics of 10 plain declarative statements, drafted by the agent, labelled as the model's, frozen by
  the owner before the run (`material/negation_statements.jsonl`). No statement begins with an instruction or names
  negation. Each opposite is elicited from Qwen3-8B itself, thinking off, greedy, up to 80 new tokens, by one fixed
  domain-free prompt, "Write the statement that says the opposite of this one. Reply with the statement only.", followed
  by a blank line and the statement; prompt and reply are recorded. The owner reads a random 20 pairs before any
  state is read; a reply that is not an opposite is flagged and left in, and the verdict is reported with and without
  flagged pairs.
- **States:** Qwen3-8B, sink by norm, each statement and opposite read alone. Layers 12, 24, 28, 32. D from the
  paraphrase sets, under the same rule, on sets the owner has frozen.
- **Three forms, centre fitted:** additive, y = x + v; contraction, y = λx + v (fitted centre v / (1 − λ), reported as no
  finite centre if |1 − λ| < 0.1); mirror, u the top direction of x − y and its plane's offset the mean of (x + y)/2 along
  u. **Baseline:** identity, y = x.
- **Two scopes, each form fitted in both, so no form gets a freedom the others lack:** global, one fit across topics,
  held out by topic (5 folds of 4 topics); per topic, one fit per topic, each pair left out in turn within its topic.
- **Verdict, per scope and layer:** the root-mean-square error on held-out pairs. A form wins if it beats both others
  and identity by more than D; otherwise undecided. **A scope's verdict** is the form winning at 3 or more of the 4
  layers; otherwise undecided.
- **Controls, by the same code:** shuffled pairs (each statement with another topic's opposite), and each set's paraphrase
  pairs in place of opposites. If a form wins a scope on shuffled pairs, that form's win there is void.
- **Is the centre the origin? Reported, not judged:** for the winning form, or for each if undecided, the fitted centre's
  distance (the mirror's offset, along u only) from: the paraphrase mean; the mean of the sessions' earliest externals;
  each session's; and the shuffled control's centre. All in units of D.
- **Reported beside it:** the verdict with the centre fixed at the paraphrase mean, as `research/gate1.py` does now. If it
  disagrees with the fitted verdict, both are stated.
- **Amended 2026-11-02, before any state of a pair was read.** *The rule (owner):* contraction with λ = 1 is additive, so
  as first written additive could never win; a test with planted pairs showed it. If contraction's λ (the median over a
  scope's fits) is within 0.1 of 1 it counts as additive: it is no rival and cannot win. Otherwise it stays a rival.
  The verdict as first written is reported beside it. *The flags (owner's reading, `material/negation_review.jsonl`):*
  2 of the 20 are wrong, 2 borderline, 16 fine. "Without flagged" leaves out the wrong; a further line leaves out the
  borderline too. The other 180 are unread, so any wrong ones among them stay in. *Details the text left open (agent):*
  shuffled pairs from one seeded permutation; paraphrase pairs are each set's 10 unordered pairs, so 20 groups of 10, the
  negation pairs' shape, though a text recurs across its set's pairs; a topic under 3 pairs leaves the per-topic scope;
  sessions' origins are their first 5 externals; a mirror's plane passes through its pairs' mean midpoint, and shuffling
  keeps that mean, so its global distance to the shuffled centre is 0 by construction and the angle between the two
  normals is reported instead. Centres for contraction and mirror are reported whichever form wins.
- **Added after seeing the pairs, labelled so, judged by nothing:** the verdict on the 114 pairs whose opposite has a
  negation word and on the other 86, because most opposites are an added "not". Beside these: how far apart statement
  and opposite sit, the statements' own spread within a topic, and token lengths against the paraphrase texts, because
  the statements are short and D was measured on longer texts; the verdict still uses D as written.
- **Not done here:** relations between negation axes (angles, nesting) reuse these states; that probe gets its own
  reading before it runs, so this gate does not become the track.

### Gate 1's origin: how far apart the candidates sit

*Written 2026-11-02 before its code* (`python -m research.origin_candidates`). Input to the owner's choice of `c`, not a
test: nothing here is judged and no candidate is picked. No negation pairs exist, so no Gate 1 hypothesis is touched.
On Qwen3-8B under the sink by norm, draft sets, the same 10 sessions, references and distances only. Runs only if the
sink check holds on the corpus (it missed; the owner released it, DECISIONS.md).
- **Candidates:** the mean of the 100 paraphrase states; each session's earliest externals (the mean of its first 5 and
  first 10 external chunks, read alone, capped at 256 tokens); the mean of the sessions' 5-chunk means; the zero vector.
- **Reported per layer** (8, 12, 24, 28, 32, 35), in units of D under the rule: each candidate's distance from the
  paraphrase mean; session against session; first 5 against first 10 in one session; the paraphrase states' own radius
  about their mean. Because a mirror's fit depends on `c` only along its normal `u`, also 2|Δc·u| for `u` the 190
  directions between set means, as a stand-in for directions that carry meaning (median and 95th percentile).

*Result* (2026-11-02; draft sets, 10 sessions; record in `runs/origin-qwen8b/`, not committed). In units of D:

| Layer | D | Radius of the states | Zero | Mean of sessions | One session (median, range) | Session to session | First 5 to first 10 | Mirror shift (median, 95th) |
|---|---|---|---|---|---|---|---|---|
| 8 | 11.6 | 1.59 | 3.04 | 1.26 | 1.34 (1.26–1.50) | 0.73 | 0.31 | 0.19, 0.44 |
| 12 | 13.3 | 1.85 | 3.85 | 1.66 | 1.75 (1.56–2.00) | 0.85 | 0.36 | 0.28, 0.68 |
| 24 | 37.9 | 1.90 | 5.11 | 2.00 | 2.08 (1.85–2.49) | 1.14 | 0.47 | 0.35, 0.82 |
| 28 | 68.5 | 1.92 | 5.12 | 1.69 | 1.84 (1.70–2.09) | 1.13 | 0.46 | 0.28, 0.75 |
| 32 | 113.2 | 1.92 | 6.41 | 1.66 | 1.80 (1.54–2.22) | 1.09 | 0.46 | 0.30, 0.81 |
| 35 | 185.0 | 1.72 | 7.17 | 2.42 | 2.62 (1.94–3.06) | 1.40 | 0.57 | 0.70, 2.38 |

A session's origin sits about as far from the paraphrase mean as a state does, and sessions sit about 1 D from each
other. Under a fixed origin, a contraction's error moves by (1 − λ) times these: at λ = 0.3 and 1.8 D, about 1.3 D,
more than the margin. A mirror's moves less, under 1 D at the 95th percentile at layers 8–32, on the stand-in
directions only. Nothing here is a Gate 1 result: no negation pair was read.

## Retired

Freeform chat (rung 1), the rung-2 memory harness, the toy world and the port of DARA's kernel, at the tag
`aware-rungs-final` (DECISIONS.md, 2026-10-30).
