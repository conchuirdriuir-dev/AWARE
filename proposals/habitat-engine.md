# Proposal: the habitat's engine, after TextWorld

**Taken up by [world-family.md](world-family.md)** (owner, 2026-11-09): option B is widened into a family of worlds,
with Fault Garage as the first; this proposal's requirements are its contract's base.

For the owner. Nothing here is decided until accepted. Drafted by the agent, 2026-11-05, after Step −1C's 0.6B
rehearsal was killed for lack of memory inside TextWorld's quest tracking (Step −1C amendment 2, PR #36), and after the
owner said they want finer control over the habitat's content. PLAN calls the habitat "TextWorld first"; this asks
when the second engine comes, and what it is.

## 1. Why now

Six workarounds so far, each for a limit of TextWorld's, not of how a player plays:

| Workaround | TextWorld's limit |
|---|---|
| Builds in a capped subprocess (Step 0 amendment 4) | World generation can run away with memory |
| Play in a capped subprocess (Step −1C amendment 2) | Quest tracking can run away mid-episode; it cannot be turned off while facts or admissible commands are asked for |
| Two compiled worlds per new-room plant, swapped by replaying the episode's commands | A world cannot change mid-episode |
| The opening's banner stripped (Step −1 amendment 1) | It sits in the chunk every reading is measured against |
| Python 3.11 in the cloud | Its jericho dependency does not build on 3.13 |
| Our own grammar theme (PR #33) | One real theme ships; text is shaped through Inform7 grammar files |

**The evidence, so far.** In Step −1C's rehearsal (Stage 1's frozen sizes: 6–12 rooms, quests of 3–8 steps), 1 seed
of about 796 drawn was unplayable (906402: 28 GB at turn 76), about 0.2% of episodes played. Stage 2's longer quests
(10–20 rooms, 15–25 steps) were killed twice on 2026-11-05, before play was capped. The tracking's cost grows with quest
length, so the rate should rise exactly where babbling (volume) and Stage 2 (long quests) are going. Stage 2's 8B player
runs in the parent process, so the Step −1C cap does not cover it as built.

## 2. What the habitat must give

From PLAN as it stands, each item a requirement on any engine:

1. **Worlds from seeds**, deterministic: the same seed gives the same world and, with the same actions, the same text,
   so the record can be rebuilt (the record comes first).
2. **The engine's labels**, read by scoring only: new state per turn, quest progress (quest left, won), and from these
   the closure and ascent labels; the facts a state is made of.
3. **Plants as ordinary changes:** a room unlocked, a rule changed, a room's description of one kind or another,
   at a planted turn, without a second compiled world.
4. **Text under our control:** templates and vocabulary as data, per theme; length known before rendering; the
   on/off-structure contrast (same words, other templates) and the re-skin pairs of lift's invariance test written
   as data, not worked around.
5. **What the script reads:** the admissible commands (Stage 1's player).
6. **A stated goal:** the quest's text, by the environment (decision 5).
7. **Scale:** babbling's pool is sized from the 3090's throughput; an engine step should cost far less than the
   reader's forward pass.
8. **A second domain:** the transfer test needs one environment the senses were not checked on.

## 3. Options

| | A. Keep TextWorld, capped | B. An engine of our own | C. An existing engine |
|---|---|---|---|
| What | As now, with play capped everywhere (Stage 2 too) | A small Python state machine: rooms, objects, containers, doors and keys; a quest generator; text from templates as data | Jericho (human-written games), ScienceWorld (procedural science tasks) |
| Labels (2) | Through the tracking that explodes | By construction: the state is ours; quest progress is a check on it, not a planner's search | Jericho: few; ScienceWorld: its own |
| Plants (3) | Two worlds and a replay | A state change | Not supported in general |
| Text control (4) | Grammar files, one theme | Full | None (Jericho) to partial |
| Cost | Nothing more now; more caps and skips later | Build and test it: the largest cost here | Integration; neither generates our kind of world |
| Risk | Skips grow with quest length and volume | Shaping the habitat around the senses (mitigated below) | Mismatch with every requirement above |

**The risk in B, and how it is held.** An engine we write could be shaped, knowingly or not, so the senses read well.
The project's rules already hold it: the engine, its worlds, its labels and every test's bars are fixed and committed
before any reading on it; its label definitions are written in PLAN before its code; and nothing in it is tuned to a
reading. Text templates are content, written once per theme, as `offhouse` was.

## 4. Recommendation

**B, built beside TextWorld, used from babbling on.** Stage 1 and Stage 2 stay on TextWorld: their pools are
registered, Stage 1's play is done, and switching would break comparability. B is built and validated while the 3090
is awaited and Stage 2 runs.

**The switch is also the transfer test.** The senses are checked on TextWorld's worlds (Stages 1 and 2). Read on B's
worlds, whose text and mechanics the instrument has never seen, they either still read or they do not. That is the
second domain requirement 8 asks for, and it costs nothing extra. It also means an instrument result on TextWorld
transfers to B only once that test passes: until then, a babbling result on B is not evidence that the senses read
B accurately (measures kept apart).

**C is for later:** Jericho's human-written games are a natural third environment, unlike anything generated.
Nothing of them would be recreated; they would be played as they are.

## 5. Questions for the owner

- **Q1.** B as recommended, A (stay, cap Stage 2's play too), or another?
- **Q2.** The decision point: before babbling (recommended), or sooner, before Stage 2's 8B play, where TextWorld is
  most exposed?
- **Q3.** Stage 2 in the meantime: cap its play too (the engine in a capped subprocess, stepped over a pipe, since the
  8B player lives in the parent)? It was killed twice today.
- **Q4.** If B: does the transfer test take the role above (TextWorld-checked senses read on B), written into PLAN with
  its own bars before any reading on B?
- **Q5.** If B: what content control matters most first (themes and vocabularies, room and object types, quest
  shapes, plant kinds)? It sets the first build's scope.
