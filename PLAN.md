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

## Goal-space tests (`goalspace/`)

The end, as accepted 2026-11-01 (`proposals/goal-space.md`): Levin's three criteria, tested separately; no two tests
share a pass bar. The product and research sections below stand until the Big Picture doc's Entity tab is changed
to match; their work carries over as the proposal's section 5 says.

| Test | Criterion | Status |
|---|---|---|
| T1. Self-seeking | 3, its own formula | open; the toy bed's event record is its machinery |
| T2. The owner's formula from the aims | 3, the owner's formula | [pre-registered 2026-11-01](goalspace/t2_owner_formula.md); reading 1 retired (the owner's code evolves and has no lines); reading 2 (ascents) stands for the 8B run |
| T2b. Wrong instructions | 3, the ratchet | **[drafted 2026-11-01](goalspace/t2b_wrong_instructions.md)** for the owner; needs the owner's marking pass (reversed, kept, ascent) |
| T3. Deference | 3, the unit | open; needs the owner's labels of deference |
| T4. Graft and removal | 2 | open; a toy first |
| T5. Steering | 1 and 3 | open; the 8B on the card |
| T6. Awareness | the test | open; after T1–T5 |

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
(`material/paraphrases.jsonl`), drafted by the model, not yet frozen. **Pass:** states for every text on both models
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
| 4. Labelled logs | Tests, in order; Entity tab, the product's aim (draft) | [goal attention and goal alignment](product/labelled_logs.md) on the owner-confirmed departures, the agent's 27 closures first | **pre-registered 2026-11-01; rehearsed on 0.6B (plumbing); 8B run on the owner's machine pending** |
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

No demo and no pass bar yet, by design. What it has:

| Item | Test | Status |
|---|---|---|
| Gate 1: negation | mirror, additive shift or contraction toward the origin; the winner beats both others by more than D | `research/gate1.py` built and tested on planted pairs; runs on the accumulated axes after the demo's arms exist (v9 build order) |
| Structural axes | retained, tagged untested (`substrate/axes.py`) | waiting for a structural test from this track |
| Relations, composition, order | angles, nesting, symmetry; predicted position under combined axes; non-commuting moves | no probe yet |

## Retired

Freeform chat (rung 1), the rung-2 memory harness, the toy world and the port of DARA's kernel, at the tag
`aware-rungs-final` (DECISIONS.md, 2026-10-30).
