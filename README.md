# AWARE

The goal, in the owner's words: "The agent should function as if it is aware, even without metacognition machinery."

The entity is a trajectory through a frozen local model's activation space, with an observer riding alongside it: a
retained low-dimensional code fit to the trajectory, and a model of the trajectory's dynamics in that code. Everything
the project does is a use of that observer. The design is the Big Picture doc
([Big Picture — v9](https://claude.ai/artifact/ByrrhB2Ue4NS8NzmyZXoCw), a Claude Doc; its Entity tab is the parent).

Two tracks share one substrate and are kept apart ([PLAN.md](PLAN.md)):
- **Product: drift detection, then self-correction** (`product/`). The demo: it catches the loop from the shape, not
  the words.
- **Research: reasoning over the entity's own axes** (`research/`). Relations, operations, composition and order. No
  demo yet; the reason the project exists.
- **Substrate** (`substrate/`): model access and D, the record with provenance, the geometry core, and the retained
  axes with their two admission criteria.

It runs on one machine, without cloud services: Hugging Face transformers with nnsight for activations, numpy for the
geometry.

## Status

The framing changed on 2026-10-30 and again on 2026-10-31 (DECISIONS.md). The earlier work (freeform chat, the rung-2
memory harness, the toy world, the port of DARA's kernel) is retired at the tag `aware-rungs-final`. The product
track's toy bed is built and its first pre-registered run fails three of four bars (PLAN.md); model access awaits the
GPU and a reachable Hugging Face.

## Running

Python 3.11+. The geometry, axes, toy bed and Gate 1 need only numpy:

```sh
pip install -r requirements.txt
python -m pytest
python -m product.toybed                     # the toy bed against its pre-registered bars
```

Reading the model's states needs torch, transformers, nnsight and nnterp:

```sh
pip install -r requirements-model.txt
python -m substrate.measure_d --model meta-llama/Llama-3.1-8B-Instruct --run runs/d-llama   # D per layer, frozen sets
python -m pytest -m model                                                                    # Qwen3-0.6B, plumbing only
```
