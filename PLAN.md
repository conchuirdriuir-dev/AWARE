# Plan

Each rung is a test that can fail. Write the test and its pass criteria first, then build the smallest thing that
answers it. A rung is done when its test passes, not when its code exists.

| Rung | Question | Test | Status |
|---|---|---|---|
| 1. Accounting | Can a reviewer that sees only the record keep one aim and notice drift from it, without over-flagging? | a written conversation, turn by turn: offers made, offers held back, the person's answers | drafted on paper |
| 2. Memory of the whole | Can a store and a context builder keep a long story straight for a small local model, better than plain retrieval? | [rung2/SPEC.md](rung2/SPEC.md): six arms, 60 questions, a hand-written novella | **harness built; awaits the story and a real model** |
| 3. The big picture | Can it see what the whole lacks (open threads, the shape, contradictions) and ask about it at the right moment? | to write, on whichever memory wins rung 2 | open |
| 4. Learning the person | Can it learn which questions this person finds worth asking, and which threads they return to? Where DARA's mechanisms compete with the simple version | to write | open |

## Next

1. The owner writes the novella from a model-generated outline and freezes the answer key (rung2/SPEC.md, "Material").
2. Meanwhile, on the 3090: a dry run of the keeper on a throwaway public-domain text, whose results do not count, to
   check the prompt and schema on a real 8B model (SPEC.md, "Before trusting the store").
3. The run, on the owner's story.
