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
capped on purpose, counted with the model's own tokenizer.

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
| `loop` | setups: opened, last touched, closed or open | 4 |
| `chapter_line` | one sentence per chapter, from its raw text | the map of the whole |

**The builder** answers "as of chapter N" and fills the view in this order, each part stopping at its budget:

| Part | Tokens | Selected by |
|---|---|---|
| Map of the whole | 400 | chapter lines up to N |
| Rules | 150 | all |
| Entities named in the question | 2,500 | aliases matched in code; latest state per attribute, knowledge rows, events up to N |
| Open loops | 600 | open as of N, least recently touched first |
| Cited passages | 1,350 | passages the selected rows cite, most-cited first |
| Recent text | 1,000 | the last passages before N |

Left out on purpose: abstractions about the story, the author's plans and acts, rule dependencies (rung 3 and real
use).

## The keeper

One call per passage at temperature 0: the prompt in [keeper_prompt.txt](keeper_prompt.txt), output constrained by
[keeper_schema.json](keeper_schema.json). About 3.5k tokens per call: instructions 700, known entities 900, open loops
300, the previous passage's end 200, the passage 400, output at most 1,000.

What code does with each reply, in order:

1. **Check every quote** against the passage (whitespace and quote marks normalized). Drop and log any record whose
   quote is missing.
2. **Resolve entity references:** a known id; else the exact name of a new entity in this reply; else an alias of a
   known entity. Unresolved references are dropped and logged, never created silently.
3. **Skip repeated states.** A different value becomes a new row, with `supersedes` set to the old one.
4. **Link knowledge.** A new row for the same character and `about` entity with a different stance or truth sets
   `corrected_by` on the earlier row.
5. **Update loops:** `last_touched` for touched ids; `closed_chapter` and `closed_passage` for closed ids. Unknown ids
   are dropped and logged.
6. **Stamp every row** with the chapter, passage id, model file and run id.

At each chapter's end a second call reads the whole chapter: *"In one sentence of 30 words or fewer, say what happens
in this chapter that matters to the story. Use the characters' names. Reply with the sentence only."* Its reply is the
`chapter_line` row.

**Before trusting the store:** after chapter 6, compare the knowledge rows and the lookalike entities with the ledger
by hand. If the 8B keeper misses more than a third of the ledger's items, run the keeper on the 14B model and report
that as part of D's cost.

## Risks to watch

1. **Lookalikes** merged through aliases.
2. **Knowledge rows,** the hardest extraction for a small model.
3. **Questions that name no entity** get no entity part; open loops and the map carry them.
