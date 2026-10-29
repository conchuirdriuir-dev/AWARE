# AWARE

A local language model with a structure that supports metacognition, built on one machine and run without cloud
services. The model does the talking and the work. Around it, code keeps a record of everything, a store of the whole
(a story, a project, a long conversation) and a reviewer that sees only the record. The reviewer offers questions to
the person, never to the model, and the person settles them.

AWARE starts from a finding of the owner's layered-metacognition experiments: with a structure that routes what each
part sees, much of metacognition is reachable with an ordinary model. Each capability is added as a rung whose test is
written before anything is built ([PLAN.md](PLAN.md)). DARA and DARA-PALS are references for it, not dependencies.

## Status

Rung 2's test is specified ([rung2/SPEC.md](rung2/SPEC.md)); nothing is built yet.

## Running

Nothing to run yet. The planned stack: Python 3.11+ with the standard library, `llama.cpp`'s `llama-server` with
local open-weights models, SQLite, and numpy for retrieval.
