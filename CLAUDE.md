# Working on AWARE

Read [README.md](README.md) and [PLAN.md](PLAN.md). The design is the Big Picture doc linked there (v9; its Entity tab is
the parent); it takes precedence over this repository. Work serves the current step's test; build the simplest thing
that can pass it. Propose a change to the doc or the plan rather than widening a build.

- **Two tracks, kept apart.** Product (`product/`) and research (`research/`) never share a pass criterion, and neither
  imports the other; both import `substrate/`. Product work may not delete, refit away or repurpose retained or
  structural axes. A change to `substrate/` is a substrate decision: the owner's, with a line in DECISIONS.md.
- **Watch for the known failure mode.** The capacity with the cleanest test re-derives everything around itself (DARA,
  then drift detection in v1–v9). If a change makes the research track serve the product's bar, stop and say so.
- **The record comes first.** Append everything; anything else is derived from it and can be rebuilt. Activations are
  derived: a cache, rebuilt by replaying the forward pass from the recorded text, model and revision.
- **Label who said what:** the person, the model, the code. Stated and inferred are kept apart.
- **Tests before builds.** A step's pass criterion and thresholds are written before its code and fixed before it runs.
- **Keys judge, never supply.** The reference scorer and any answer key are read only by scoring, never by an arm.
- **Route in code.** What each arm sees is set by code, never by asking a model to ignore something; arm S never sees
  arm C's geometry.
- **No hand-building in discovery.** Self-negation is domain-free and written once; nothing is rebuilt for the
  transfer domain. A domain-specific fix in discovery or the compass hides hand-building.
- Add a document only when a step needs one; record decisions in DECISIONS.md, one line each.
