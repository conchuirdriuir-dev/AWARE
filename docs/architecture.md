# AWARE: the architecture

*From the [AWARE — Architecture Brief](https://claude.ai/artifact/Wd5ZcWYEm6i1rGBTqPFJMX) (owner, 2026-11-04), brought
into the repository 2026-11-05 with three changes: rate's sources as fixed for the habitat (self and world), "Where it
stands" brought up to the two stages, and the Brief's four diagrams left out. Where this differs from the North star
([The Semantic Body — Open Threads](https://claude.ai/artifact/UfHZr9X2mDjqAkYJneUBdW)) or from PLAN.md, they win.*

AWARE builds a language model as its own organism: it carries a body in code-space, feels that body through three
measured senses, learns to use them by babbling, acts toward a goal, and consolidates what it learned by dreaming. The
five steps, fixed as the project's north star: **lift, sense, babble, act, dream.**

## Two spaces and a body

The body lives in code-space, one level above meaning-space: not where each turn lands, but the compressed pattern the
whole run fits.

Meaning-space is the model's fixed embedding landscape, where each chunk of text lands as a point. Code-space holds
descriptions of whole runs; the code the recent run fits is the body's location. Moving every point while keeping the
pattern leaves the body where it was, as a waltz stays a waltz anywhere on the floor. An earlier ablation supports this
level: raw positions in meaning-space mostly tracked topic, not trajectory.

## Three senses

The organism feels its body through three measured senses. None is a neural network: each is a small function over the
stream of state vectors, closer to a thermometer than a brain.

| Sense | What it computes | Biological counterpart | Adaptation |
| --- | --- | --- | --- |
| Shape | Bits needed to describe this chunk under the code fitted to everything before it | Proprioception: what posture you are in | Tonic: always reported |
| Rate | Bits the last window taught, split by source (in the habitat, self and world; in conversation, self, partner and tools) | Interoception: hunger, fatigue; boredom is rate at zero | Phasic: a derivative by construction |
| Boundary | How new each arrival is against everything held, and whether it improves prediction of the next step | Touch, with self-caused input cancelled | Reports surprise only |

**Senses read as chords, like taste.** No single sense detects closure; each alone is ambiguous. Closure is the chord
of cheap shape, flat rate and silent boundary. Ascent has its own chord: a hard boundary spike, then shape rises as the
old code fails, then a higher code forms, shape drops below where it was, and rate jumps.

## How the senses plug in

The organs are ordinary code outside the model; the only place they touch the model is two soft tokens before each
step.

The base model reads each chunk with the graft off, so a sense means the same thing before and after training. The
three organs turn the stream of state vectors into six numbers, and a small learned projection, about two million
parameters against the model's eight billion, maps them to two soft tokens.

**What a soft token is.** A word normally maps to a fixed row of 4,096 numbers in the model's embedding table. A soft
token skips the table: it is a row of 4,096 numbers placed straight into the sequence, with no word behind it. Words
are paints from fixed tubes; a soft token is a colour mixed to order. The model reads it with the machinery it uses for
words, but nothing in its training ever showed it that colour.

**Sensory substitution.** In Paul Bach-y-Rita's experiments, blind people wore a camera wired to vibrators on their
back, later electrodes on their tongue. At first it felt like buzzing; after practice while moving and acting, they
perceived objects out in space. Passive exposure did not work. The soft tokens are that tongue display: meaningless
until babbling, practice while acting in the habitat, wires them in.

## Boundary as efference copy

The boundary sense reports only what the organism did not cause, the way your brain cancels the feel of your own
movements, which is why you cannot tickle yourself.

When the organism acts, a copy of the action predicts what will come back, and only the difference is felt. A file the
agent wrote and later reads back is predicted, so it barely registers: the organism cannot mistake its own echo for
news. The first version predicts from the agent's recent history; the forward model learned in babbling replaces it.

## The habitat

The organism grows up in procedurally generated text-game worlds, run unattended, not in conversation with one person.
Senses need a world: boundary needs input the agent did not cause, and rate needs structure to learn.

- **It runs overnight.** Worlds are generated without limit, at set difficulty, on one local GPU.
- **It labels itself.** Closure is visible in the game state (no new state, repeated actions, flat score); ascent is a
  new area, mechanic or subgoal. Thousands of labelled events, no hand labelling.
- **Novelty is a dial.** Hidden rooms and mid-episode rule changes are planted where boundary must fire.
- **It has controls.** In self-talk, with no world at all, boundary must stay silent; in pure reasoning, thinking
  (silent boundary, rising rate) must read apart from closure (silent boundary, flat rate).

Complexity climbs from house-sized worlds to social ones, where other characters become the richest source of news,
and on toward dialogue-heavy games. Conversation with a person becomes one more environment to transfer to.

## Dreaming

Each night turns the day's episodes into structure and writes it into a new graft, which keeps its place only if it
does better on worlds it has never seen.

| Phase | Biological counterpart |
| --- | --- |
| Replay: episodes chosen by the senses (high rate, ascents; closures as what not to do) | Hippocampal replay, tagged by salience |
| Abstract: codes that compress several worlds enter a library | Systems consolidation; DreamCoder's abstraction phase |
| Dream: the forward model simulates remembered terrain and tries shorter routes | REM recombination |
| Write the graft: a new adapter, frozen for the next day, kept close to the base model | Synaptic change, with synaptic homeostasis |
| Morning check: worse than yesterday's graft means rollback | The judge never marks its own work |

**Babbling builds the dream engine; dreaming uses it.** The forward model that cancels self-caused input while awake is
the simulator while asleep.

**Three invariants.** The senses are never retrained, so the organism cannot learn to feel fine instead of being fine.
Dreams are hypotheses: most training data is replay of real episodes, and a dreamed route counts only once the next
day's world confirms it. Sleep has a trigger: rate flat across many episodes, the counterpart of sleep pressure.

## Where it stands

The first build checks the instrument before anything is learned, in the habitat, with the player and the perceiver
kept apart (owner, 2026-11-05), so that the habitat tests the senses and not the player's skill:

1. **Instrument check** (Step −1, Stage 1): a scripted player, Qwen3-8B perceiving. Boundary stays silent in self-talk
   and fires at every planted novelty; shape is read per chunk.
2. **Rate proxy** (Step 0, Stage 2): the 8B playing longer-quest worlds and perceiving (a weaker player was planned;
   amendment 5, after every one screened locked in early). Rate is flat before a stall after progress, not flat at
   ascents.
3. **Babbling gate**: the graft predicts its next sensation better with the sense tokens than without, and worse when
   they are shuffled. This gate covers the split of rate between self and world; there is no third stage.
4. **Behaviour**: with live senses, the organism beats the same senses given as text, and the same graft with shuffled
   senses, in held-out worlds and on an owner-scored reading task.

Earlier results already shape the design: in the project's own session logs, who supplied the input and how new it was
separated closure from ascent, while raw embedding geometry added nothing.

What is running now, and each step's bars, are in [PLAN.md](../PLAN.md), "Current state and direction". Full detail:
[The Semantic Body — Open Threads](https://claude.ai/artifact/UfHZr9X2mDjqAkYJneUBdW) (the north star and the
reasoning) and [Body Readout — Spec v0.2](https://claude.ai/artifact/GW7JqgKGYmDtNG24cBrTn4) (the build).
