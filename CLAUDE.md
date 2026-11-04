# Working on AWARE

Read [README.md](README.md) and [PLAN.md](PLAN.md), starting with PLAN's "Current state and direction": it says what is
running and what comes next. Update that section whenever either changes. The picture is the North star in
[The Semantic Body — Open Threads](https://claude.ai/artifact/UfHZr9X2mDjqAkYJneUBdW) (fixed 2026-11-04): the model as its
own organism, five steps (lift, sense, babble, act, dream). It takes precedence over this repository. The current build is
[Body Readout — Spec v0.2](https://claude.ai/artifact/GW7JqgKGYmDtNG24cBrTn4). The Big Picture doc (v9) is kept as the
implementation reference only (coupling, the echo floor, D, the event record, consolidation safeguards, the graft rule,
the tech stack); where it differs from the North star, the North star wins. Work serves the current step's test; build
the simplest thing that can pass it. Propose a change to the doc or the plan rather than widening a build.

- **One project, one loop, measures kept apart** (owner, 2026-11-02, framed before the North star; its rules still
  bind, and whose goal the loop serves in the habitat is open, PLAN.md). The user's goal, drift against it, a judge, an
  intervention, consolidation. No two tests share a pass criterion, and a functional measure (does the agent hold the
  user's goal better?) never stands in for an instrument measure (is this reading accurate?), or the reverse. The base
  weights never change; what is learned is a removable graft. `product/` and `research/` stay
  separate directories and neither imports the other; both import `substrate/`. Work may not delete, refit away or repurpose
  retained or structural axes. A change to `substrate/` is the owner's decision, with a line in DECISIONS.md.
- **Watch for the known failure mode.** The capacity with the cleanest test re-derives everything around itself (DARA,
  then drift detection in v1–v9, then the dyad oscillation). If a change lets a functional result stand in for an
  instrument's accuracy, or the reverse, stop and say so.
- **The judge never marks its own work.** What gets consolidated is checked against the user's confirmations and held-out
  tasks the judge never sees, with rollback.
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
