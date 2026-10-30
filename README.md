# AWARE

The goal, in the owner's words: "The agent should function as if it is aware, even without metacognition machinery."

AWARE is an agent on a local 8B model that stays self-coherent, revises itself as its situation changes, and stays on
its goal over long work in a domain it was not prepared for. Its orientation comes from the geometry of the model's own
states, not from a structure built by hand. The design is the Big Picture doc
([Big Picture — v8](https://claude.ai/artifact/RmZatX3o5ZrAkJToUvSsQW), a Claude Doc, with its Tech stack tab); this
repository builds it, step by step, each step's test written before the step ([PLAN.md](PLAN.md)).

It runs on one machine, without cloud services: Hugging Face transformers with nnsight for the model's activations,
numpy for the geometry.

## Status

The framing changed on 2026-10-30 (DECISIONS.md). The earlier work (freeform chat, the rung-2 memory harness, the toy
world, the port of DARA's kernel) is retired, kept at the tag `aware-rungs-final`. Step 2 of the plan, the CPU work
before the GPU arrives, is under way.

## Running

Python 3.11+. The geometry, toy bed and compass need only numpy:

```sh
pip install -r requirements.txt
python -m pytest
```

Reading the model's states needs torch, transformers, nnsight and nnterp:

```sh
pip install -r requirements-model.txt
```
