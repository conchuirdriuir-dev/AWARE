"""The body readout: a measured, simplified model of where the entity is and what is moving it (the attention schema).

Content-agnostic: it reads a record (chunks with provenance) through `substrate/` and the goal-space formula's code
lengths (`goalspace/`), and reports a body state per chunk plus a fixed-template readout for the actor. Nothing here
names a session, a codebase or a poem. Imports `substrate/` and `goalspace/`; never `product/` or `research/`.

Spec: [Body Readout — Spec v0.1](https://claude.ai/artifact/BBcL4c3mN6uUthsJfrqBMo), a Claude Doc. No code yet: its
tests are written in PLAN.md before any build (owner, 2026-11-03, DECISIONS.md).
"""
