# Plan

The architecture is the DARA Build Map (a Claude Doc, 2026-10-29), read at the meta level (DECISIONS.md, "What AWARE
is"); these rungs fold into its stages once the other session's harness moves in.

Each rung is a test that can fail. Write the test and its pass criteria first, then build the smallest thing that
answers it. A rung is done when its test passes, not when its code exists.

| Rung | Question | Test | Status |
|---|---|---|---|
| 1. Accounting | Can a reviewer that sees only the record keep one aim and notice drift from it, without over-flagging? | a written conversation, turn by turn: offers made, offers held back, the person's answers | drafted on paper |
| 2. Memory of the whole | Can a store and a context builder keep a long story straight for a small local model, better than plain retrieval? | [rung2/SPEC.md](rung2/SPEC.md): six arms, 60 questions, a hand-written novella | **harness built; awaits the story and a real model** |
| 3. The big picture | Can it see what the whole lacks (open threads, the shape, contradictions) and ask about it at the right moment? | sketched below: known inconsistencies planted in a second version of the rung-2 story, on whichever memory wins rung 2 | **sketched; its material exists** |
| 4. Learning the person | Can it learn which questions this person finds worth asking, and which threads they return to? Where DARA's mechanisms compete with the simple version | to write | open |

## The general loop

What every use of AWARE comes down to, whatever the world (owner, 2026-10-30). One model call per turn; everything
else is code, the kernel, and the embedding model:

1. **Record** what happened: what the model was shown, what it did, what came of it. Code.
2. **Check it against what has been learned** (the map of what moves what, the kinds of failure) and find the
   residue. Code and the kernel.
3. **Retrieve what matters now.** The embedding model ranks earlier residue, kinds and record by how much they bear on
   the present situation; rows are embedded once, as they are written. It ranks; it never admits a kind or decides a
   finding, which stays with the kernel's margin. It embeds each item inside its sentence with a task instruction,
   never as a bare word (DECISIONS.md, 2026-10-30, from the negation note).
4. **Build the context:** record excerpts quoted verbatim under a small fixed framing per residue type ("this was done
   and this stayed", "this persisted and nothing that fixes it was tried"), within a fixed token budget. Code; nothing
   generated, so nothing to fact-check.
5. **One call to the model.**

**Two instances of step 4 exist:** the toy world's settled notes (toyworld/SPEC.md, "Stage 2") and rung 2's context
builder (rung2/SPEC.md). Rung 2 already asks the question this loop depends on: whether a store and a builder beat
plain retrieval, which says how much of steps 3 and 4 should be embeddings and how much structure.

**What makes it general.** The channel back to the model is always its context, so no situation needs a channel of
its own. The residue types are defined over the learned map (an action, what it should change, what was then
observed), not over any world. What is still specific to the store and has a route out: the anomaly ids (learned,
the Build Map's anomaly kinds), the action names (from a tool's schema: its type and enum fields), the labels (from the
observation's own field names), and, where there is no schema, free-text actions and observations, which the embedding
model groups into candidate kinds for the kernel to admit or refuse.

**Planned, after stage 2's rerun: which intervention works is learned, not tuned.** The meta level's own moves (show
the record, name the missing action, show the model its own numbers, ask it a question, block an action) are
interventions on the model, and whether the residue clears after them is their effect. Randomising which form a kind's
offer takes, as stage 1 randomised actions, lets the kernel learn under the same margin which form moves this model on
this kind of failure. A different model relearns it.

**Open, not planned:** notes the model writes itself, for situations no residue type covers. Allowed only as
model-proposed records checked by code against the record, every number and claim quoted, as the rung-2 keeper's quotes
are; a note that misstates the record would be the system deceiving its own cognition. It costs a second model call,
which the loop above avoids.

## Next

1. **The toy worlds first** (DECISIONS.md, 2026-10-29): stage 0 ([toyworld/SPEC.md](toyworld/SPEC.md)) has run, and
   Qwen3 14B bare beat Qwen3 8B with stage 0. Stage 1 passed on Qwen3 8B ([toyworld/RESULTS.md](toyworld/RESULTS.md)).
   Stage 2, kinds of failure learned from residue (toyworld/SPEC.md, "Stage 2"), is built, with `latent.py`
   ported; on the owner's stage-1 records it finds two kinds. Next: its run on Qwen3 8B under pressure.
2. **Rung 2 when the story is written:** the owner writes the novella from the outline (its audit fixes applied) and
   freezes the key. The keeper is ready: four dry runs on Qwen3 8B, and a loop pass per chapter.
3. The run, on the owner's story, on Qwen3 8B.

## Porting DARA, stage by stage

DARA's kernel (`src/dara/kernel/`, about 3,950 lines, numpy and the standard library) imports nothing else from DARA, so
it moves in by piece, each when a stage needs it, with a note of where it came from. Its rules (no language model
holding beliefs, depth limits, declared steps) do not come with it. At the meta level its subject is the 8B model's
work: what it was asked, what it did, and what came of it.

| Stage | What it needs | From DARA | Lines |
|---|---|---|---|
| 1. Controllability from interventions | rules of the form "this action, then this anomaly falls", admitted on MDL gain over the intervention log | **ported:** the kernel core: `structure`, `pattern`, `code`, `counts`, `cells`, `naming`, `learn` (the 2-bit margin, admission, retirement, save and load) | about 2,350 |
| 2. Failure kinds from residue | kinds formed from what the held rules leave unexplained, admitted on MDL gain | `latent` (latent classes seeded from residuals, kinds of kinds): **ported** | about 700 |
| 3. The register and negation | the frame the model's reasoning rests on, held as priced alternatives; the model proposes the negation, the law admits it | `frames` (priced frames, admitted and retired by the margin) | about 580 |
| 4. A catalog that compounds | kinds carried across episodes, against a no-lesson baseline | `learn`'s save and load, `latent`'s carried kinds; the baseline arm is new | small |
| Later: offers worth asking | when a question is worth the owner's attention, from the owner's own record | `read/ought.py` (flag or silence, priced) and what it imports | to map |

Not ported: DARA's reader (`read/`: the ERG, binding, talk). The model reads; the ERG returns only if stage 5's
grounded language needs it, or if the negation test (below) shows the keeper losing negation.

**The embedding model** proposes groupings for stage 2's kinds; it never admits one. It embeds each item inside its
sentence with a task instruction, never as a bare word: on bare words every model tested put opposites among the
nearest neighbours 10–17% of the time, in context with the instruction 1–2% (the negation note, Experiment 6).
Qwen3-Embedding-4B at Q4 is the pick; 0.6B if it does not fit beside the 8B.

**Not in DARA's kernel: lookahead.** The kernel admits a rule or class on its immediate gain. The negation note
(Experiments 1–2) found that a real abstraction often costs bits at first and pays only when reused one level up (a
decoy merge saved about 18 bits; the true axis cost about 1), so a one-step test refuses it. The remedy there is a
credit ledger: a candidate is admitted on credit, its description length as debt, and kept only if reuse repays it
within a window. It exists only as the note's scripts (`ledger_test.py`), not in DARA's kernel. Stage 2's design note
expects tier 2's latent classes not to pay in the store world; whether that is the world's size or the one-step test is
open (toyworld/SPEC.md, "Stage 2 design note"). Tier 2 was built with the one-step test before this question was
raised; its silence is read as provisional until it is settled (DECISIONS.md, 2026-10-30).

## From the DARA Agent Architecture Spec and the negation note

Both are the owner's (2026-10-28 and 2026-10-29), written for DARA. What AWARE holds of them:

**In, kept as requirements** (toyworld/SPEC.md, "What every stage keeps"): routing in code (P1), isolated contexts
(P2), MDL admission (P8), a no-lesson baseline (P12, the arms), the ledger's three evidence rules, and goal versus
delusion.

**Superseded on purpose:** "the model proposes, DARA disposes; DARA owns all state" (spec §1–9). In AWARE the model
may write records, labelled as its own; only learned structure must pass MDL (DECISIONS.md, "Two layers of state").

**Planned:** the residue archive and kinds from it (P3, P4: stage 2); negating the shared assumption (P7: stage 3);
the catalog carried across episodes (stage 4); lookahead (above).

**Open, in no stage yet:**
- **The negation test,** set up and waiting for its run (rung2/SPEC.md, "The negation test"): the keeper on *The Told
  Hives*, chapter 1, against the pilot's probes and its ERG-only run. If the keeper loses negation the ERG keeps, the
  ERG returns as the reader of negation scope (spec §12.3).
- **Negation as the operator that proposes abstractions,** ranked by recurrence across codes (the note, Experiment 1).
- **Imagined cases are never evidence:** a case constructed by analogy may direct search and action, never be credited
  (Experiment 3).
- **The self-model tables:** the repayment record per bet shape, and the codes the system caused (spec §12.1).
- **Provenance on every memory:** passive, selected, or self-caused; for story memory, reading in order is the control
  and a retrieved passage cannot confirm the hypothesis that retrieved it.
- **The catalog entry's package** (trigger, test, stop clause, correction: P5), the sizing, stopping, surrender and
  intervention gates (P9), exploring by coverage (P6), and a stance toward other agents' goals (P11).

## Rung 3's test, sketched

**Material.** The owner's consistency audit of the rung-2 outline (2026-10-29, a separate session) found plot holes
and inconsistencies with objects and characters. They are fixed before the rung-2 story is written, since a
contradiction makes its answer key ambiguous. The findings are kept, outside the repository with the rung-2 key,
because they describe the story: they are realistic mistakes a long story picks up, made naturally rather than
invented, and they become rung 3's key.

**The test.** A second version of the rung-2 story with the audit's inconsistencies written back in, and a few more
of each kind, so every kind below has several items. The key lists each one: its kind, the passages that conflict,
and the chapter where it first becomes detectable. The system reads the chapters in order, as rung 2's build does,
and raises an offer whenever it finds something: a question that cites both passages and says what would make it
wrong.

| Kind | Example | Expected difficulty |
|---|---|---|
| An object's state changes with no cause | in her pocket in ch. 5, under the board in ch. 7, nothing moves it | easy: the state timeline |
| A character's fixed traits drift | grey eyes in ch. 1, brown in ch. 12 | easy |
| Someone knows something too early | mentions the forgery in ch. 9, learns of it in ch. 16 | medium: a check per scene |
| A rule is broken | a spell cast with no cost | medium |
| A setup never paid off | the hidden locket never returns | easy, from the loop pass |
| A true plot hole | the ending needs a key never introduced | hard: what the whole lacks |

**Measures.** Items found per kind, and at which chapter; false alarms per chapter, which an offer must not be (the
invariant: offers, not verdicts); and, against the same small model given only what fits its window, how much the
store adds. Pass criteria are written before the run, once the counts per kind are known.

**Open, for when rung 3 is designed:** whether a contradiction the owner introduces on purpose (a revision) can be
told from a mistake, which only the owner settles; and whether the checks run per passage or per chapter.
