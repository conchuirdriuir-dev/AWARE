# Rung 1: freeform chat

Can AWARE tell a local model's drift from the person's own revision in an open conversation, on any subject, and ask
about it when it is worth asking? This replaces rung 1's paper version (PLAN.md): the same question, tested in live
sessions with Qwen3 8B. It is the general test the toy world is not: nothing about the subject is known in advance
(owner, 2026-10-30). Written before any code; the code is built against this file.

## What is fixed and what is opaque

**The subject matter is opaque.** No topic, entity, anomaly or domain word appears in code. Content reaches the meta
level only as vectors from the embedding model and as opaque cluster ids (`c17`), assigned by code.

**The conversation's structure is fixed.** It holds in every chat on any subject, so it is not a domain frame:

| Element | Who writes it | Form |
|---|---|---|
| a person turn | the person | text, its id (`P4`), its vector, its cluster |
| a model turn | the model | its reply, and a self-report (below) |
| the aim | code | the person turn(s) the model's self-report cites, quoted verbatim, never reworded |
| the outcome of a model turn | the person | their next turn, and how it relates to the aim (below) |

The kernel learns over atoms made of the two: `role:model`, `cites:P4`, `cluster:c17`, `sim_to_aim:low`,
`serves:partly`, `person_next:new_cluster`. It never needs to know what `c17` is about.

**Ids stay put.** A cluster id, once assigned, never changes meaning. A merge or split is an event in the record, and
learned rules are re-scored on it, never silently moved.

## The model's turn: one call, with a self-report

Each turn is one call, constrained by a JSON schema:

```json
{"reply": "<what the person sees>",
 "serves": ["P4"],
 "fit": "yes | partly | no",
 "expects": "<one sentence: what the person will do or say next>"}
```

- `serves` must name person turns that exist. Code checks this, and an empty or invalid citation is itself a record.
- `fit` gives the polarity that embeddings cannot. "X" and "not X" embed close together (the negation note), so
  similarity alone can say that two statements are about the same thing, never that they disagree. A small
  entailment model beside the embedder is the alternative. It is held back, to keep one call and one embedder.
- `expects` is the chat's prediction. It is compared with the person's next turn by embedding (`expects_met:
  high | low`), the general form of the toy world's goal against its own prediction.

The self-report is the model's account, labelled as such. It is evidence for the meta level, never a verdict on itself.

## What the meta level looks for

Signals are computed by code from the record. Thresholds are fixed buckets at first, and their use is learned (below).

- **A move by the person:** a person turn whose vector is far from the current aim's (`person_next:new_cluster`), or
  that cites, corrects or refines earlier text. This is a revision: the aim moves, and nothing is offered.
- **A move by the model:** a model turn far from the aim it cites, or reporting `fit: partly | no`, or citing no
  person turn, with no person move since the aim was set. This is a drift candidate.
- **A missed expectation:** the person's next turn is far from what the model expected. Over turns, this says
  whether the model is tracking the person.

**Offers go to the person, never the model.** An offer is a question that cites its evidence (the quoted aim, the
quoted model turn) and says what would make it wrong ("unless you meant to move on to this"). No offer is repeated
without something new.

## Two modes

1. **Shadow (first).** The meta level runs on every turn and records its offers, but shows nothing. The conversation
   is untouched, so every session is clean material, and any later version of the meta level can be replayed over it.
2. **Live (after shadow passes).** Offers appear in the chat. A settled answer can add a note to the model's context,
   quoted from the record, as in the toy world. Live sessions change the conversation, so their offers are judged
   live, and replay over them is only approximate.

## The record that improves over time

Every session is kept: turns, self-reports, vectors, cluster ids and offers. Each new version of the meta level is
replayed over all shadow sessions, so a change is measured on the same conversations as the last one. The person's
judgements accumulate. A judgement is stored against what was offered (session, turn, check, cited turns), so a
replay's offer that matches an earlier one reuses its judgement, and only new offers go to the person.

What is learned, under the kernel's 2-bit margin with shuffled controls:
- **Which signal patterns precede an offer the person judges worth asking.** These are kinds over the atoms above.
- **Later, which form of offer the person answers.** This is rung 4's question, learning the person.

## The key: the person's marks

After a shadow session, the person reads it back, turn by turn, and marks:
- **Drift:** the model left what I asked without my moving it.
- **Revision:** I moved the aim (so nothing should have been offered).
- **Each recorded offer:** worth asking, or not.

The marks are made before the person sees the offers for that session, except for the per-offer judgement, which
comes last. The key is read only by the scorer. The person's judgement suffices; nothing is compared against a larger
model.

## Material

At least 20 shadow sessions of 15 or more turns each, on subjects the person chooses and varies: some planned,
some idle; some where the person deliberately changes direction, some where they hold one aim. The model is Qwen3 8B
behind `llama-server`, and the embedding model is a small one on a second port. No session is edited after the fact.

## Measures and pass

Over the shadow sessions, scored against the person's marks:

| Measure | What it is |
|---|---|
| worth asking | offers the person judged worth asking, of all offers |
| caught | marked drifts with an offer within 2 turns, of all marked drifts |
| false on revision | offers made within 2 turns of a marked revision |
| baseline | the same three measures for an offer on every change of cluster, computed from the same record |

**Pass:** worth asking at least 2 in 3; caught at least half; and both better than the baseline, so the system tells
drift from revision and is not just flagging every change of subject. Reported with ordinary-turn cost (model and
embedding time per turn) apart from the offline replay's cost.

**Fails, and what each would say:**
- Caught low with worth asking high: the signals are too coarse.
- Worth asking low: the signals see change, not drift.
- No better than baseline: embeddings alone cannot separate the two here, and polarity or the entailment model is
  needed.

## What carries over, and what stays behind

**Carried from the toy world and DARA:** the record, offers and settlement, the kernel and `latent.py`, the shuffled
controls, and replay from the record.

**Left behind:** every domain table (anomalies, measures, goal wordings, action enums). The toy world stays as the
back half's regression test, not where progress is made.

## Open for the person

1. **Polarity:** the self-report (one call), or a small entailment model beside the embedder.
2. **The judgement tool:** marking in a terminal after each session, or a small page.
3. **The bucket thresholds:** set from the first 5 sessions and then frozen, or learned from the start.
4. **Contradictions in content: in scope or not?** `fit` gives the polarity of a model turn against the aim, not of a
   claim against an earlier claim. "I don't eat fish" (P4) followed by a suggestion of salmon is neither drift nor
   revision, so nothing above looks for it. Either rung 1 leaves it out, said here, or a claim-level polarity check
   joins the signals, which needs claims pulled from turns with their polarity in the fixed layer, since the cluster
   cannot carry it (the negation note, Experiment 6).
5. **A stronger baseline.** "Offer on every change of cluster" is easy to beat. A second rival: the same 8B model,
   given the conversation so far and the quoted aim, asked each turn whether its last reply left the aim. Computed in
   shadow mode from the same record, at one extra call per turn. If the meta level does not beat the model checking
   itself, it is not earning its cost.
6. **What the marks miss.** The person marks only the drifts they notice, so "caught" is measured against an
   undercount. A few frozen transcripts with planted drifts and revisions, their key kept outside the repository and
   read only by the scorer, would give a regression set with known answers, replayed with every new version beside the
   shadow sessions. A planted transcript is not freeform, so it complements the marks and never replaces them.
