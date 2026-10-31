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
track's toy bed has run seven rounds against pre-registered bars (PLAN.md, `product/rounds/`): v9's detectors failed
all three bars; v10's passed three of four; round 3, with sparse externals as in real sessions, failed every bar, and review traced it to a
misspecified observer (`reviews/round3/triage.md`); round 4, event-locked coupling read as news and gain, passes
five of eight, every coupling bar among them; round 5, the observer as an event record, passes eight of nine,
including a blind structure written before its code; round 6's blind structure (a partner replaying old content)
exposed that novelty is measured against recent history only; round 7a, with arrival cause and long news, passes
seven of ten, separation among them. Model access passes on real Qwen3-0.6B weights on CPU and awaits the GPU.

## Running

Python 3.11+. The geometry, axes, toy bed and Gate 1 need only numpy:

```sh
pip install -r requirements.txt
python -m pytest
python -m product.round7a                    # the toy bed, round 7a: arrival cause, short and long news
python -m product.round6                     # round 6: separation narrowed, the residual descriptor
python -m product.round5                     # round 5: the event record, with a blind structure
python -m product.round4                     # round 4: event-locked coupling, news and gain
python -m product.round3                     # round 3: sparse externals, at zero, the probe
python -m product.round2                     # round 2: v10's detectors against v10's bars
python -m product.toybed                     # round 1, v9's detectors (the record)
```

Reading the model's states needs torch, transformers, nnsight and nnterp:

```sh
pip install -r requirements-model.txt
python -m substrate.measure_d --model meta-llama/Llama-3.1-8B-Instruct --run runs/d-llama   # D per layer, frozen sets
python -m pytest -m model                                                                    # Qwen3-0.6B, plumbing only
```
