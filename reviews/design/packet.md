# Design packet: the entity and its observer

For a reviewer outside machine learning. This is the design only, as accepted on 2026-10-31; no code, results or
the builders' reasoning are included.

## The goal

The author's words: "The agent should function as if it is aware, even without metacognition machinery." A fresher
statement: "The agent should be dynamic and self-correct to pursue goals over perturbations."

## The entity

The entity is a trajectory through a frozen language model's activation space: one state per chunk of text the agent
produces (a message, an action). The model's weights never change, so the space is a stable frame. External chunks
arrive between the agent's own: tool results, files it reads, a person's messages. Every chunk is tagged with its
provenance: self (the entity produced it) or external. Provenance follows origin, not channel: the entity's own words
read back to it are still self. The origin is the earliest externals (the task statement, the opening turns), frozen,
and every quantity is taken about their centre. D is the unit of distance: the spread between two readings of the same
content (paraphrases), measured on the real model.

An observer rides alongside the trajectory. It keeps a low-dimensional code fitted to the trajectory, retained rather
than consumed, and reads events. Its uses, called capacities: noticing drift; self-correcting; noticing that its
partner is captured (a person inside the loop who echoes the entity while believing they steer); discovering its own
axes; transferring to a new setting with nothing rebuilt; seeking where the world has structure its code lacks;
reasoning over its own axes (a separate research track).

## The observer as an event record

Events are of three kinds: arrivals (externals), probes (the entity asks a question, and the partner's responses
arrive), and corrections (pruning the entity's own context, or re-inserting the origin).

Each event has an **innovation**: what it carried that the entity's recent history (its last 21 states) and the
external before it did not. The innovation's size is the event's **news**. Each event also has a **response**: how
the entity moved over the following steps. **Gain** is how much of an innovation shows up in the entity's response,
pooled across many arrivals. **Unexplained surprise** is movement that no innovation accounts for.

The compass reads three things:

| Reading | Means | What the entity does |
|---|---|---|
| News present, gain open | On track (progress if surprise is above its floor) | Continue |
| News present, gain at zero | Closed loop on the entity's side: it no longer listens | Prune its own context |
| News at zero | Closed loop on the partner's side: arrivals carry nothing the entity did not already hold | Probe; flag the capture |
| News low, gain intact | Not a loop: the entity listens, little is arriving | Probe |
| Surprise with no news | The entity is generating: a discovery candidate, or lost if gain is also gone | Admit a new axis, or re-localize |

News below D is no news (it is the spread of re-reading the same thing). Gain is judged against the session's own gain
just after an opening intake; "at zero" means confidently below half of it. A reading that is neither is undecided,
and the entity probes. How long an input stays causally active (the response profile) is measured on the model, like
D. The **frame** is the span of admitted innovation directions: directions the world supplied. Seeking and probing are
one act: buying news where it is missing.

## What is tested so far

Only in a simulated toy: linear dynamics with noise, where planted situations (the entity stops listening; the
partner echoes the entity; healthy work with less novel input; a burst of noise; a change of topic; a change in the
entity's own dynamics) are known, and the readings are checked against them.
