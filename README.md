# AWARE

AWARE studies goal-directedness in a language-model agent by treating the model as an organism. It gives the model
a body in a space one level above meaning, lets it feel that body through a few measured senses, and asks, by
experiment, what those senses let it do. The method is borrowed from developmental biology, in the spirit of Michael
Levin's TAME framework and the FieldSHIFT pivot: where FieldSHIFT carries neuroscience into anatomical morphospace,
AWARE carries morphospace one step further, into a *code-space* over the model's own embedding space, with game turns
in place of minutes.

Everything runs locally on one machine (Qwen3-8B, Hugging Face transformers, numpy). The base model's weights are
never changed; anything learned is a removable adapter (a "graft").

The picture is the North star in [The Semantic Body — Open Threads](https://claude.ai/artifact/UfHZr9X2mDjqAkYJneUBdW);
the current build is [Body Readout — Spec v0.2](https://claude.ai/artifact/GW7JqgKGYmDtNG24cBrTn4). Where this
repository differs from the North star, the North star wins. A longer account of the design is in
[docs/architecture.md](docs/architecture.md).

## The idea in five steps

**Lift, sense, babble, act, dream.**

1. **Lift.** Each chunk of text lands as a point in the model's fixed embedding space (meaning-space). The body is not
   any one point but the compressed code that the agent's recent trajectory fits, in code-space. Move every point
   while keeping the pattern and the body stays where it was, as a waltz stays a waltz anywhere on the floor.
2. **Sense.** Three small measured functions, not neural networks, read the body each turn:

   | Sense | What it computes | Biological counterpart |
   |---|---|---|
   | Shape | Bits needed to describe this turn under the code fitted so far | Proprioception |
   | Rate | Bits the last window taught, split by source (self, world) | Interoception; boredom is rate at zero |
   | Boundary | How new an arrival is, with self-caused input cancelled | Touch, with efference copy |

   No single sense detects a stall; each alone is ambiguous. A stall is a chord: cheap shape, flat rate, silent
   boundary.
3. **Babble.** The senses reach the model as two learned soft tokens, meaningless at first, like Bach-y-Rita's
   tongue display. Practice while acting wires them in, and trains a forward model that predicts the agent's own
   effects.
4. **Act.** The agent pursues goals stated by its environment (a game's quest), with its senses live.
5. **Dream.** Offline, the forward model replays and recombines the day's episodes; the result is written to a new
   graft, which is kept only if it does better on worlds it has never seen.

The agent grows up in procedurally generated text-game worlds ([TextWorld](https://github.com/microsoft/TextWorld)),
run unattended, so no measurement depends on one person's conversations.

## How it is tested

Each step's pass criterion is written in [PLAN.md](PLAN.md) before its code, and its thresholds are fixed before it
runs. Decisions and their reasons are logged in [DECISIONS.md](DECISIONS.md). A failed bar is recorded, not moved.

The senses are calibrated before the subject, in two stages, so that the habitat tests the instrument and not the
player's skill:

| Stage | Who plays | Who reads | Asks |
|---|---|---|---|
| 1 (Step −1) | A scripted player, alternating explore and loop phases | Qwen3-8B | Does each sense read what it claims, under known conditions? |
| 2 (Step 0) | Qwen3-8B, on longer-quest worlds | Qwen3-8B | Does the reading anticipate a stall after progress, and not read an ascent as one? |

Each chunk is read alone, from its own text, so who wrote the actions does not enter the reading. Stage 2 was to be
played by a weaker model; every one screened locked in from its first turns, and only the 8B gave progress and then a
stall, so the 8B plays it (Step 0 amendment 5). Its reading is then of its own play: the measure compares coding with
and without the world's replies, so self-familiarity should largely cancel. There is no third stage; the split between
self and world is covered by the babbling gate. Stage 1 gates what follows; Stage 2 informs.

Then babbling (does the model use its senses, and does acting beat passive exposure?), behaviour (does it do better
with live senses than with the same readings given as text, or shuffled?), and dreaming.

The longer-term test, following the planarian rewrite experiments: edit the agent's target state directly, with its
inputs unchanged, and compare that intervention with prompting and with retraining. That places the agent on TAME's
axis of persuadability by measurement.

## Status

*Updated 2026-11-05.*

- Stage 1 (scripted player): the registered play is done (160 episodes, every plant delivered). The 8B's reading,
  and its self-talk, wait for a 24 GB GPU.
- Stage 2: the 8B plays (amendment 5), after Qwen3-1.7B and 4B, screened with and without TextWorld's command forms,
  fell short of the selection criteria. Its screen, k, play and reading wait for the 24 GB GPU.
- Earlier runs on Qwen3-0.6B and a 4-bit 8B rehearsed the harness end to end and found the design faults that led to
  the staged schedule; they decide nothing. Records are in PLAN.md, "Body readout", "The habitat", "Runs".
- Results: none yet decide a bar. *(This line is replaced by the first result.)*

The current state is always in [PLAN.md](PLAN.md), "Current state and direction".

## The repository

| Path | What it holds |
|---|---|
| `body/` | The habitat, the senses and the step harnesses (`habitat.py`, `stepm1.py`, `step0_habitat.py`; `step0.py` is Step 0's form on the owner's sessions, paused) |
| `goalspace/` | The code-length formula behind lift and rate |
| `substrate/` | Model access, the echo floor, the record with provenance, the geometry core |
| `material/` | Registered world seeds and measured floors |
| `probes/` | Small one-off measurements (memory, concept probes) |
| `tests/` | Tests; `pytest -m model` for those that load a model |
| `docs/` | The architecture, at more length than here |
| `product/`, `research/` | Earlier tracks, resting and parked; their records stand |
| `proposals/`, `reviews/` | Design proposals and outside reviews |

## Running

Python 3.11+. TextWorld needs Linux, macOS or WSL.

```sh
pip install -r requirements.txt
python -m pytest                       # numpy-only tests
pip install -r requirements-model.txt  # torch, transformers, nnsight, nnterp, TextWorld; bitsandbytes optional
python -m pytest -m model              # loads Qwen3-0.6B
```

Each habitat step plays, reads and scores. The record comes first: `play` appends every turn as it happens, `read`
derives states from the recorded text alone, and only `score` reads the keys. Stage 1 (Step −1): the script plays
the calibration and Step −1 pools with no model or GPU; the 8B writes its own self-talk, then reads everything.

```sh
python -m body.stepm1 play  --script --run runs/stepm1 --pool calibration
python -m body.stepm1 play  --script --run runs/stepm1 --pool step_minus_1
python -m body.stepm1 play  --model Qwen/Qwen3-8B --run runs/stepm1 --pool self_talk
python -m body.stepm1 read  --model Qwen/Qwen3-8B --run runs/stepm1
python -m body.stepm1 score --run runs/stepm1
```

Stage 2 (Step 0's habitat form) plays its own worlds (`material/habitat_worlds_stage2.json`), the 8B told the
command forms: first a screen on mirror seeds (labels only), then the calibration pool, k, and Step 0's pool.

```sh
python -m body.step0_habitat play  --model Qwen/Qwen3-8B --forms --run runs/s2-screen --pool calibration --limit 30 --rehearsal
python -m body.step0_habitat play  --model Qwen/Qwen3-8B --forms --run runs/s2-screen --pool step0_habitat --limit 40 --rehearsal
python -m body.step0_habitat screen --run runs/s2-screen
python -m body.step0_habitat play  --model Qwen/Qwen3-8B --forms --run runs/step0h --pool calibration
python -m body.step0_habitat k     --run runs/step0h
python -m body.step0_habitat play  --model Qwen/Qwen3-8B --forms --run runs/step0h --pool step0_habitat
python -m body.step0_habitat read  --model Qwen/Qwen3-8B --run runs/step0h
python -m body.step0_habitat score --run runs/step0h --k K
```

Useful flags: `--limit n` plays only the first n seeds of a pool; `--rehearsal` plays mirror seeds instead, spending
no registered seed, and is required by the rehearsal-only flags (`--nf4` for a 4-bit player, `--context`,
`--generator`). `read` takes `--floor` (the echo floor; `material/echo_floor_8b.json` by default, and another model
needs its own, made by `body.floor_file`) and `--gpu-memory`. PLAN.md gives each step's pools and flags.

Qwen3-8B in bf16 needs a 24 GB GPU, to play and to read; the script's play, and rehearsals with smaller or 4-bit
models, run on an 8 GB card.

The earlier tracks' commands (the product rounds, `substrate.measure_d`, `substrate.echo_floor`) are in this file's
history, before 2026-11-05, and in PLAN.md under their sections.

## History

AWARE grew out of an earlier architecture, DARA, and went through several framings: a drift detector for one
person's sessions, then reasoning over the agent's own axes, then the organism. The earlier work is kept at the tag
`aware-rungs-final` and in PLAN.md, "The record". Its main lesson shaped the current design: tests built on one
person's conversation logs measured that person as much as the agent, so the agent now grows up in its own worlds.

MIT License.
