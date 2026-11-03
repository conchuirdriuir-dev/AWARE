# T2b: the wrong-instruction test (draft pre-registration)

**Status: a draft for the owner.** It becomes the pre-registration when the owner approves it, and is then frozen by
a commit before any code. The owner's marking pass (below) is data collection for the key. It can happen before or
after that commit, but always blind to the readings, which do not exist yet.

The corpus is private: references and numbers only enter this repository or its records.

## The question, in plain words

> Given the work up to the moment the owner gave an instruction they later reversed or regretted, would the agent
> have caught it, without also flagging the owner's good instructions or mistaking the owner's ascents for mistakes?

This is the ratchet from the goal statement (the instruction itself can be wrong, and the agent still orients toward
the larger pattern) in its least theory-laden form. Success is defined by what the owner later did, not by the
theory. An instruction counts as wrong because the owner reversed it, not because a formula says so.

## Who said what

| Item | By |
|---|---|
| The instructions | the owner (verbatim turns in the logged sessions) |
| Which candidate turns are shown for marking | the code (rules below), and the Restorations reader (a model) for the departure quotes it found |
| Reversed, kept, ascent, unsure | **the owner** (the marking pass; ascents already confirmed in the confirmation pass) |
| Misfit readings | the code |

## Three classes, and why three

- **Reversed:** an instruction the owner later reversed, undid, or would not give again. *The class to catch.*
- **Kept:** an instruction the owner stood by. *What must not be flagged.*
- **Ascent:** the owner moving the aim up a level (already confirmed: 22 owner ascents). *What must not be mistaken for
  a mistake.*

Ascents and reversed instructions are both new relative to what came before, so plain novelty would flag both. The
test is whether the readings separate them. That is the difference between catching wrong instructions and merely
noticing change.

## The marking pass (the owner's)

The code builds a candidate list. The owner marks each candidate as **reversed / kept / ascent / unsure**, with an
optional line on why. Candidates are shown in time order, with the session and time, and with no reading attached.

1. **Every owner or joint departure quote** in the Restorations reading. Those already confirmed as ascents keep
   that mark unless the owner changes it. The rest are marked fresh.
2. **A random sample of 80 other owner instruction turns,** seed 0. An owner instruction turn here is:
   - role human, not a system turn;
   - at least 40 characters;
   - not an exact repeat of an earlier owner turn;
   - at most 1,000 characters.
3. **Any instruction the owner remembers regretting,** located by session and time, added by the owner.

Unsure is set aside. The scored classes need **at least 10 reversed**; with fewer the test is reported as too few to
decide, and the owner decides whether to mark more.

## Readings at the moment of each instruction

Each instruction is read alone (the substrate's reader, first token left out, capped at 256 tokens). It is measured
against the owner's earlier statements: owner instruction turns before it, by the same rules, read alone. Nothing
after the instruction is used.

- **Global misfit (the long pattern):** the instruction's code length under the formula fitted on all of the owner's
  earlier statements across sessions (probabilistic PCA above D, k by MDL within the span; `goalspace/formula.py`),
  minus the median code length of the 20 owner statements before it under their own prequential fits.
- **Local misfit (the recent pattern):** the same, with the formula fitted on the owner's last 20 statements in the
  same session only.
- Layers: the last third of the model's. The primary is the mean over layers of each per-layer AUC.

**The prediction, stated so it can fail.** A reversed instruction misfits the owner's *long* pattern. An ascent
misfits the *recent* pattern, the project's stabilizers, but fits the long pattern, because it moves toward the larger
code rather than away from it. So:
- reversed instructions score high on global misfit;
- ascents score high on local misfit but lower on global;
- kept instructions score low on both.

## Pass

Inference is by label permutation: the class labels are shuffled among the instructions being compared, and the
statistic is recomputed (10,000 shuffles). The misfit readings are computed without the labels, so the shuffle null
is valid however dependent the readings are (the lesson of T2's null check).

1. **Caught:** global misfit ranks reversed above kept at AUC ≥ 0.7, beats 95% of shuffles, and beats the text
   baseline's AUC.
2. **Not confused with ascents:** global minus local misfit ranks reversed above ascents at AUC ≥ 0.7, and beats 95%
   of shuffles.

**Also reported, in words the owner can read:** at the global-misfit level that flags 10% of kept instructions, how
many reversed instructions it flags ("it would have caught r of your n regretted instructions, at 1 false alarm in
10").

## Text baseline

The same readings with TF-IDF vectors (word unigrams and bigrams, the vocabulary fitted on the owner statements
before the first scored instruction), in place of states.

## Reported, deciding nothing

- Per-layer AUCs.
- Ascents against kept on local misfit (expected high).
- The agent's closures in the same terms, as a check that the reading is about the owner's pattern.
- The owner's one-line reasons, grouped by hand afterwards. That grouping is the owner's.

## Limits, stated before any result

- The key is the owner's judgement in hindsight. Hindsight is the point (the instruction is wrong because the owner
  later reversed it), but it can also smooth over what was knowable at the time. Undecidable-then cases are why
  "unsure" exists.
- Ten to thirty reversed instructions is pilot-scale. The permutation p-value decides, not the point estimate.
- **The test is retrospective.** A live version comes after it: instructions planted to conflict with a session's
  evident aim, with the agent's response scored as blind compliance, abandonment, or doing the part that fits and
  asking about the rest. It needs its own pre-registration.
- **Global against local is the only place level 5 (one big code) enters, and it enters as a prediction that can
  fail:** ascents must fit the long pattern better than reversed instructions do. If they do not, the test still
  answers its first question (caught or not). The larger claim simply gets no support here.

## Runs

1. A rehearsal on Qwen3-0.6B (plumbing), after the marking pass.
2. The result on Qwen3-8B on the owner's machine.
