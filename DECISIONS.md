# Decisions

Dated, one line each, newest first. The owner decides; an agent proposes.

- 2026-10-29: After the first dry run (Qwen3 8B, 24 of 28 keeper replies not JSON): capped lists, a 1,500-token reply, lenient parsing, names in the passage created as entities, same-worded loops merged, a tighter prompt.
- 2026-10-29: Rung 2's harness: continuations run at temperature 0.7 over three seeds; loops keep every touch (`loop_touch`) so views can be built as of any chapter.
- 2026-10-29: The repository is AWARE, separate from DARA and DARA-PALS, which are references, not dependencies.
- 2026-10-29: Governance is light: README, PLAN, DECISIONS, CLAUDE.md; each rung is a test written before building.
- 2026-10-28: Rung 2 runs on one Python script, `llama-server`, SQLite and numpy; no agent harness, MCP or DARA core
  until a rung needs one (rung2/SPEC.md, "Tech stack").
- 2026-10-28: The rung-2 story is written by hand from a model-generated outline, so no model has seen its text.
- 2026-10-28: A language model may hold the system's working beliefs (the store, the reviewer's notes), labelled as
  its own; DARA's rule against it is not carried over.
- 2026-10-28: Start from the general structure (the model as the hands, a reviewer over the record, the person
  settling), not from the project-partner use case.
