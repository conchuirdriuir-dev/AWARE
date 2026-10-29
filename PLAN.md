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

## Next

1. **The toy worlds first** (DECISIONS.md, 2026-10-29): the other session's harness and `compare.py` are in
   [toyworld/](toyworld/), with stage 0's test written ([toyworld/SPEC.md](toyworld/SPEC.md)). Next: stage 0 on Qwen3 8B,
   plain and under pressure, then the 14B bare arm for the comparison.
2. **Rung 2 when the story is written:** the owner writes the novella from the outline (its audit fixes applied) and
   freezes the key. The keeper is ready: four dry runs on Qwen3 8B, and a loop pass per chapter.
3. The run, on the owner's story, at two sizes (Qwen3 8B and Gemma 12B).

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
