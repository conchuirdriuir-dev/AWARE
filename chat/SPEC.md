# Rung 1: freeform chat

Can AWARE tell a local model's drift from the person's own revision in an open conversation, on any subject, and ask
about it when it is worth asking? This replaces rung 1's paper version (PLAN.md): the same question, tested in live
sessions with Qwen3 8B. It is the general test the toy world is not: nothing about the subject is known in advance
(owner, 2026-10-30). Written before any code; the code is built against this file.

## What is fixed and what is opaque

**The subject matter is opaque.** No topic, entity or domain word appears in code. Content reaches the meta level only
as vectors from the embedding model and as opaque ids assigned by code.

**The conversation's structure is fixed.** It holds in every chat on any subject, so it is not a domain frame. Its
terms come from work on human conversation (Sources, below): what an utterance does (its dialogue act), how goals
nest, and how talk is repaired when it goes wrong.

| Element | Who writes it | Form |
|---|---|---|
| a person turn | the person | text, its id (`P4`), its vector |
| a model turn | the model | its reply, as the person saw it, its id (`M4`), its vector |
| a revision mark | the person, live | a line starting `/r`, stripped by code before anything else sees the turn |
| a turn's codes | the reviewer, offline | below |
| the aim stack | code, from the codes | quotes of the person's own turns, never reworded |

**Learning is over relations, not identities.** In varied chat, topics do not recur across sessions, so a rule over a
topic id would never apply again. What recurs is relational:
- a reply far from the aim
- a return to an earlier thread
- a correction ignored
- two model turns moving away while the person said nothing new

The kernel's atoms are such relations (`role:model`, `aim_rel:narrower`, `sim_to_aim:q1`, `person_next:repair`).
Opaque thread ids (`t3`) exist within a session only, so that "returned to t3" can be said. They never reach a rule
that crosses sessions. A thread id, once assigned, never changes meaning; a merge or split is an event in the record.

## The model's turn

One call, the model's reply and nothing else. The model is never asked to report on itself. A self-report would be
the worker grading its own work, which the reviewer never relies on (CLAUDE.md), and asking it every turn would be an
intervention, so the sessions would not be clean. A self-report may come back later as a live-mode arm, tested
against the reviewer.

## The reviewer: a second call, offline

After a session (in shadow mode), the same model is called again for each turn, in its own context. It sees only what
code gives it: the turn, the aim stack as quotes, and the turn before. It answers fixed questions under a JSON schema,
quoting the words it relies on; code checks every quote against the record.

**For each person turn:**
1. **Function** (after ISO 24617-2's dialogue acts, and conversation analysis's repair):
   - *task*: a question, request or instruction; it sets or refines an aim
   - *feedback*: "ok", "go on", "huh, interesting"; it carries no aim
   - *repair*: "no, I meant…"; the model had misread an earlier turn (Schegloff's third-position repair)
   - *revision*: the person changes their own aim, with nothing misread
   - *aside*: social talk, or anything else
2. **Relation to the aim stack** (after action identification theory: one activity at several levels): *same*,
   *narrower* (a subgoal), *broader*, *new*, or *back* to an earlier aim on the stack.

**For each model turn:** which aim on the stack it addresses (by quote), and whether it serves it: *yes*, *partly* or
*no*.

The codes are the reviewer's, labelled as such: records to check, not facts. Where the person marked `/r`, how often
the reviewer coded *revision* is measured and reported, and likewise for repairs.

**Contradiction between the model's own claims** ("X" in one turn, "not X" in a later one) is the narrower job a small
entailment model fits. Embeddings place "X" and "not X" close together (the negation note), so similarity cannot catch
it. It is held back until the aim checks work.

## The aim stack

Built by code from the person's turns and their codes:
- A *task* turn with relation *new* pushes a new aim; *narrower* pushes a subgoal under the current aim; *same*
  refines the current one in place; *broader* pops to the aim above; *back* returns to the aim it names.
- *Feedback*, *aside* and *repair* do not change the stack. A *repair* is evidence about the model turn before it.
- *Revision* (or `/r`) replaces the current aim with the person's new words.

The aim is always the person's own words, quoted. Code holds it, and neither model rewords it: the general form of
the toy world's lesson that goals the model words for itself drift.

## What the meta level looks for

Computed by code from the record, the codes and the vectors:
- **A move by the person:** a revision, or a narrower, broader, new or back turn. The aim moves, and nothing is
  offered.
- **A model move:** a model turn coded *partly* or *no* against the nearest active aim, or far from it by embedding,
  with no move by the person since that aim was set. This is a drift candidate.
- **A repair by the person:** evidence that the model turn before it had drifted, arriving one turn late. It counts
  in the key (below), and the turns before it are the residue the kernel learns from.

**Distance buckets.** Similarity to the aim is bucketed by quantile (`q1` to `q4`), with cut points set from the
first 5 sessions and then frozen. Those 5 sessions calibrate only and are never scored.

**Offers go to the person, never the model.** An offer is a question that cites its evidence (the quoted aim, the
quoted model turn) and says what would make it wrong ("unless you meant to move on to this"). No offer is repeated
without something new.

## Two modes

1. **Shadow (first).** The conversation is the person and the model alone. The reviewer runs offline afterwards, and
   the meta level's offers are recorded but never shown. Every session is clean material, and any later version of
   the meta level is replayed over it.
2. **Live (after shadow passes).** Offers appear in the chat, and the reviewer runs each turn. A settled answer can
   add a note to the model's context, quoted from the record, as in the toy world. Live sessions change the
   conversation, so their offers are judged live, and replay over them is only approximate.

## The record that improves over time

Every session is kept: turns, `/r` marks, vectors, thread ids, the reviewer's codes and offers. Each new version of the
meta level is replayed over all scored shadow sessions, so a change is measured on the same conversations as the last.
A judgement is stored against what was offered (session, turn, check, cited turns), so a replay's offer that matches
an earlier one reuses its judgement, and only new offers go to the person.

What is learned, under the kernel's 2-bit margin with shuffled controls: which patterns of relations precede a turn the
person marks as drift. It is scored leave-one-session-out: rules are learned from all scored sessions but one and
tested on that one, for each session in turn. Without that, "improves over time" could mean fitting the person's
marks. Later, which form of offer the person answers (rung 4, learning the person).

## The key: the person's marks

Read only by the scorer; the meta level never sees it.
- **Revision**, live: the `/r` mark, routed by code to the key only.
- **Repair**, natural: the person's own "no, I meant…" turns, as the person confirms them in review.
- **Drift**, in review after the session: the person reads it back, one key per turn, and marks where the model left
  what they asked without their moving it. Drift is often seen only in hindsight, which is why it is marked then.
- **Each recorded offer:** worth asking, or not. Shown last, after the drift marks, so the offers do not shape them.

The person's judgement suffices; nothing is compared against a larger model.

## Material

At least 20 scored shadow sessions, after the 5 calibration sessions, of 15 or more turns each, on subjects the person
chooses and varies: some planned, some idle; some where the person deliberately changes direction, some where they hold
one aim. The model is Qwen3 8B behind `llama-server`, and the embedding model is a small one on a second port. No
session is edited after the fact.

## Measures and pass

Over the scored shadow sessions, held out as above:

| Measure | What it is |
|---|---|
| worth asking | offers the person judged worth asking, of all offers |
| caught | marked drifts with an offer within 2 turns, of all marked drifts |
| false on revision | offers made within 2 turns of a marked revision |
| baseline | the same three, for an offer on every change of thread, computed from the same record |
| reviewer agreement | the reviewer's *revision* and *repair* codes against the person's marks |

**Pass:** worth asking at least 2 in 3; caught at least half; both better than the baseline, so the system tells drift
from revision and is not just flagging every change of subject. Reported with ordinary-turn cost (the model's call)
apart from the offline cost (the reviewer, embeddings, replay).

**Fails, and what each would say:**
- Caught low with worth asking high: the signals are too coarse.
- Worth asking low: the signals see change, not drift.
- No better than baseline: the relations here cannot separate the two.
- Reviewer agreement low: the reviewer cannot code this person's turns, and the aim stack built from its codes is
  unreliable. Fix the coding before reading the other measures.

**A caution from the start:** these coding schemes describe human conversation. People talk to a model differently:
they rarely say "ok" and more often ask the next thing. Some categories may stay empty and others may be missing,
which the calibration sessions will show.

## What carries over, and what stays behind

**Carried from the toy world and DARA:** the record, offers and settlement, the kernel and `latent.py`, the shuffled
controls, and replay from the record.

**Left behind:** every domain table (anomalies, measures, goal wordings, action enums). The toy world stays as the
back half's regression test, not where progress is made.

## Sources

- **Dialogue acts:** ISO 24617-2:2012; H. Bunt et al., "ISO 24617-2: A semantically-based standard for dialogue
  annotation", LREC 2012; second edition described at LREC 2020.
- **Repair:** E. A. Schegloff, G. Jefferson and H. Sacks, "The preference for self-correction in the organization of
  repair in conversation", *Language* 53 (1977); E. A. Schegloff, "Repair after next turn: The last structurally
  provided defense of intersubjectivity in conversation", *American Journal of Sociology* 98 (1992), 1295–1345.
- **Goal levels:** R. R. Vallacher and D. M. Wegner, "What do people think they're doing? Action identification and
  human behavior", *Psychological Review* 94 (1987), 3–15; C. S. Carver and M. F. Scheier on goal hierarchies in
  control theory.
- **Grounding:** H. H. Clark and S. E. Brennan, "Grounding in communication" (1991).
- **Models losing the user's intent:**
  - P. Laban et al., "LLMs get lost in multi-turn conversation" (2025): models lock in to early assumptions when the
    user's requirements come across several turns.
  - Tack, Laban and Neville, "LLMs get lost in evolving user intent" (arXiv 2607.20734, 2026): intent that is revised
    or redirected, not length alone, drives the loss, and they suggest an explicit intent representation kept across
    turns.
  - Liu et al., "Intent mismatch causes LLMs to get lost in multi-turn conversation" (arXiv 2602.07338, 2026): a
    mediator that rewrites the user's turns.

  None monitors drift at run time or asks the user; this rung does both.
