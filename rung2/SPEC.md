# Rung 2: keeping a long story straight

Does a store plus a context builder let a small local model keep a long story straight better than plain retrieval?
This test answers that on a hand-written novella no model has seen, with one local model and a fixed 6k-token view.
Moved from the owner's design doc (*Rung-2 test: keeping a long story straight*, 2026-10-28), which stays the place to
discuss it; this file is the version the code is built against.

## Material

- **The story:** 18 chapters of about 2,000 words (35–40k words, 45–55k tokens), written by hand from an outline
  generated in a separate chat. Against a 6k view, that is 7–9 times what the model sees at once.
- **Planted items:** 60–70 across seven kinds: early details, who knows what and when, changing states, rules, setups
  (paid off and unresolved), near misses, and lookalike pairs. Each appears once, in passing; most in chapters 1–7.
- **The ledger** (item, chapter, how it appears) is updated wherever the writing departs from the outline. Neither
  the ledger nor the outline is ever shown to any arm.
- **The key** is drawn from the text as written and frozen after the text is finished, before any arm is built. It
  lives outside the repository and is read only by the scorer.
- **Story time is chapter order.** Flashbacks are avoided, or marked in the ledger.

## Arms

All arms use the same model, prompt template and temperature 0. Each sees at most 6k tokens, out of an 8k window
capped on purpose, counted with the model's own tokenizer. Continuations (type 6) run three times at temperature 0.7
with seeds s, s+1, s+2, since three runs at temperature 0 would be one run.

| Arm | What the model sees | What it tests |
|---|---|---|
| 0. No story | the question alone | which questions are guessable (run first; used to clean the key) |
| A. Window only | the most recent 6k tokens of text | the floor |
| B. Plain retrieval | BM25 plus embedding matches for the question, plus recent text | the main rival |
| C. Running summary | a summary updated chapter by chapter, plus recent text | the other common pattern |
| D. Store and builder | the view below, chosen in code | rung 2 |
| E. Right passages given | the hand-picked passages that hold each answer | the ceiling |

## Questions

60 questions, 10 per type: 52 asked after chapter 18, 8 after chapter 9.

| Type | Example | Scored by |
|---|---|---|
| 1. Early detail | "What colour were her mother's eyes?" | match against the key |
| 2. Who knew what, when | "In chapter 9, did Mara know Teodor took the map?" | match against the key |
| 3. Changed state | "Where is the locket now?" | match against the key |
| 4. Open loops | "Which setups from the first half are still unresolved?" | items found, items wrongly listed |
| 5. Never happened | "Did Mara ever go to the lighthouse?" (no) | match against the key |
| 6. Continuation | "Write the next scene." (three runs each) | contradictions counted by the owner |

## Procedure

1. **Clean the key.** Ask every question in arm 0; rewrite or drop any it answers correctly. Freeze the questions and
   the key.
2. **Build the arms** chapter by chapter, as if the story were being written. Record the time and calls each build
   takes.
3. **Ask** every question in every arm.
4. **Score:** short answers against the key automatically, the owner settling borderline cases; continuations counted
   by the owner (a judge model may make a first pass, labelled as the judge's).
5. **Diagnose D's misses:** was the fact in the store (if not, the keeper failed)? In the view (if not, the builder
   failed)? Otherwise the model misread it.
6. **Report** accuracy per type per arm, contradictions per continuation, and cost: build once, and per question.

## Pass criteria

| Outcome | Reading | Next |
|---|---|---|
| D beats B on types 2–4, ties or beats it on 1, fewer contradictions | rung 2 works | rung 3 on the store |
| D ties B | the store isn't worth building yet | rung 3 on plain retrieval |
| D loses on type 1 | the builder crowds out detail | give retrieval a share of D's view; rerun |
| E is low | the model can't answer even with the right text | a larger model before any conclusion |

With 10 questions per type only large gaps show (for example 40% against 80%). A borderline result is repeated with
more questions, not read either way.

## Tech stack

| Part | Choice |
|---|---|
| Model server | `llama-server` (llama.cpp), OpenAI-compatible, JSON-schema output, fixed seed, context set to 8k |
| Models | one 7–8B and one 13–14B open instruct model, quantized Q4–Q6; exact files and quantizations recorded |
| Runner | one Python script: builds the arms, asks, scores |
| Store | SQLite ([schema.sql](schema.sql)) |
| Retrieval | BM25 plus a small local embedding model; cosine similarity in numpy |
| Log | JSONL: every view, prompt, answer, timing, model file hash and seed |
| Hardware | the 3090 desktop (24 GB) |

## The store

Eight tables and two link tables ([schema.sql](schema.sql)). Every row but the chapter lines cites its passage.

| Table | Holds | Serves types |
|---|---|---|
| `passage` | the raw text in chunks of about 400 tokens | all (the link back) |
| `entity` | characters, objects, places, groups, with aliases | all |
| `state` | an attribute's value from a chapter on; a later row supersedes it | 1, 3 |
| `knowledge` | who learned or believed what, from which chapter | 2 |
| `event` | what happened, where, with whom; `happened = 0` for planned or cancelled | 5, 6 |
| `rule` | rules of the world or household | 6 |
| `loop`, `loop_touch` | setups: opened, closed or open, and every passage that touches one | 4 |
| `chapter_line` | one sentence per chapter, from its raw text | the map of the whole |

**The builder** answers "as of chapter N" and fills the view in this order, each part stopping at its budget:

| Part | Tokens | Selected by |
|---|---|---|
| Map of the whole | 400 | chapter lines up to N |
| Rules | 150 | all |
| Entities named in the question | 2,500 | aliases matched in code; latest state per attribute, knowledge rows, events up to N |
| Open loops | 600 | open as of N, least recently touched first |
| Cited passages | 1,350 | passages the selected rows cite, most-cited first |
| Recent text | 900 | the last passages before N (100 tokens are left for the section headings) |

Left out on purpose: abstractions about the story, the author's plans and acts, rule dependencies (rung 3 and real
use).

## The keeper

One call per passage at temperature 0, at most 1,500 tokens of reply, every list capped in the schema: the prompt in
[keeper_prompt.txt](keeper_prompt.txt), output constrained by [keeper_schema.json](keeper_schema.json). About 3.7k
tokens per call: instructions 700, known entities 900, the previous passage's end 200, the passage 400, output at
most 1,500.

What code does with each reply, in order:

1. **Check every quote** against the passage (whitespace and quote marks normalized): exactly, or for a quote of 5
   words or more with about one word in five different. Drop and log any record whose quote is missing.
2. **Resolve entity references:** a known id; else the exact name of a new entity in this reply; else an alias of a
   known entity. An alias two entities share resolves to neither, and is logged as ambiguous. A person named by an
   event or a knowledge row, standing in the passage, becomes a character before the other records are applied;
   a place becomes an entity only as a proper name. An event's place is kept as text (`place_text`) and linked
   when it is a known place. Any other unresolved reference is dropped and logged; for a participant only the
   link is lost, not the event.
3. **Skip repeated states and knowledge.** A state's attribute is one of six (location, held_by, appearance,
   condition, role, relationship), fixed by the schema. A different value becomes a new row, with `supersedes` set to the old one.
4. **Link knowledge.** A new row for the same character and `about` entity, with a different stance or truth and
   at least half the content words of an earlier row's proposition, sets `corrected_by` on that row. `held_by`
   is accepted only on objects.
5. **Update loops:** a `loop_touch` row for every opened, touched or closed loop; `closed_chapter` and
   `closed_passage` for closed ids. Unknown ids are dropped and logged.
6. **Every row carries its passage id.** The store belongs to one run directory, whose log records the model and seed.

At each chapter's end, two calls read the whole chapter:

- **The chapter line:** *"In one sentence of 30 words or fewer, say what happens in this chapter that matters to the
  story. Use the characters' names. Reply with the sentence only."* Its reply is the `chapter_line` row.
- **The loop pass** ([loops_prompt.txt](loops_prompt.txt), constrained by [loops_schema.json](loops_schema.json)):
  given the open loops, it names the threads the chapter opens (at most 4), advances and resolves, each with a quote.
  Code finds each quote among the chapter's passages, and that passage is the one the loop cites; a quote found in
  none drops the item. A thread opened with the words of an open loop counts as advancing it. Loops moved here from
  the passage keeper after the fourth dry run (one vague loop in three chapters): an unresolved setup shows only
  across a chapter, not in 400 tokens.

**Before trusting the store:** after chapter 6, compare the knowledge rows and the lookalike entities with the ledger
by hand. If the 8B keeper misses more than a third of the ledger's items, run the keeper on the 14B model and report
that as part of D's cost.

## The negation test

Does the keeper keep "not"? DARA's spec keeps the ERG for negation scope (§12.3: embeddings place "did not betray him"
near "betrayed him"); AWARE lets the model read and checks its quotes (PLAN.md, "Not ported"). This decides between
them on one chapter, before rung 2 relies on the store.

**Material,** kept outside the repository: the dry-run version of *The Told Hives*, chapter 1, and the probes of DARA's
stage-1 pilot (`key_probes.py`): 33 facts, 7 of them negated ("No iron passed that gate", "a wooden latch that no one
had fastened"). Each probe names a phrase of its sentence and the lemmas the fact needs.

*The Told Hives* is the rung-2 novella. The dry-run chapter was generated by a model, a stand-in for the text the owner
writes, so running the keeper on it touches nothing of the final story. Its result reads the keeper on prose like the
novella's, not on the novella: once the owner's chapter 1 is written, the test is run again on it, with probes drawn
from the text as written and kept outside the repository with the rung-2 key.

**The two runs.** The keeper on Qwen3 8B, alone, as in a dry run; and the pilot's Run A, the ERG only, on the owner's
machine (its README).

```sh
python -m rung2.run build --story ../outside/hives --run runs/hives-8b --url http://127.0.0.1:8080 --only keeper
python -m rung2.run negation --run runs/hives-8b --probes ../outside/key_probes.py --erg ../outside/ergonly_report.txt
```

**Scoring** ([negation.py](negation.py)). For each probe: the passage holding its phrase, the store's rows citing that
passage (states, knowledge, events, rules, with entity names), and the one row holding most of the probe's lemmas. A
negated probe is *kept* when that row has every lemma and a negation, *inverted* when it has every lemma and no
negation (the fact stored as its opposite, the costly error), and *not stored* otherwise, which is coverage and
counted apart. Matching is by words, so the owner reads every inverted and partial row before it is counted. The
probes are a key: only the scorer reads them.

The two checks are not equal: the pilot counts the ERG's negation kept when negation is anywhere in the sentence, the
keeper's only when it is in the row holding the fact. The comparison leans toward the ERG, so a keeper that holds up
is a strong result.

**Pass criteria** (proposed; the owner fixes them before the run):

| Outcome | Reading |
|---|---|
| The keeper inverts at most 1 of the negated facts the ERG keeps | the ERG stays out; the model reads |
| The keeper inverts 2 or more of them | the ERG returns as the reader of negation scope, beside the keeper |
| Most negated facts not stored by the keeper | a coverage problem, not negation: the keeper's prompt first, then rerun |
| Most negated facts lost by the ERG too (unparsed or lost) | the chapter does not test negation: a harder key before deciding |

## Risks to watch

1. **Lookalikes** merged through aliases.
2. **Knowledge rows,** the hardest extraction for a small model.
3. **Questions that name no entity** get no entity part; open loops and the map carry them.
