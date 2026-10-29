# AWARE

A local language model with a structure that supports metacognition, built on one machine and run without cloud
services. The model does the talking and the work. Around it, code keeps a record of everything, a store of the whole
(a story, a project, a long conversation) and a reviewer that sees only the record. The reviewer offers questions to
the person, never to the model, and the person settles them.

AWARE starts from a finding of the owner's layered-metacognition experiments: with a structure that routes what each
part sees, much of metacognition is reachable with an ordinary model. Each capability is added as a rung whose test is
written before anything is built ([PLAN.md](PLAN.md)). DARA and DARA-PALS are references for it, not dependencies.

## Status

Rung 2's harness is built ([rung2/SPEC.md](rung2/SPEC.md)) and tested end to end with a fake model. It has not yet run
on a real model or on the owner's story.

## Running rung 2

Python 3.11+ with numpy (`pip install -r requirements.txt`), and `llama.cpp`'s `llama-server`.

```sh
# the test model, its window capped at 8k; and a small embedding model for arm B
llama-server -m MODEL.gguf -c 8192 --port 8080
llama-server -m EMBEDDING.gguf --embeddings --port 8081

# chapters as story/ch01.md ... ch18.md; one run directory per model
python -m rung2.run build --story story --run runs/8b --url http://127.0.0.1:8080 --embed-url http://127.0.0.1:8081
python -m rung2.run ask   --run runs/8b --questions questions.jsonl --evidence evidence.jsonl
python -m rung2.run score --run runs/8b --key ../outside/key.jsonl --settled settled.jsonl   # writes runs/8b/report.md
```

`--fake` in place of `--url` runs the whole pipeline with canned replies: it tests the plumbing, never the result.
The file formats are in `rung2/run.py` and `rung2/score.py`. Run the tests with `python -m pytest`.
