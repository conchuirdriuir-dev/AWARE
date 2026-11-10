# Proposal: the cause test, for boundary's forward model

**Not decided.** Drafted by the agent, 2026-11-09, at the owner's request, from the discussion of what the forward
model would have to do for "efference copy" to be the right name. No code is written and nothing has been run for it.
Every number marked (agent) is the agent's proposal. It reads AWARE at `0028780` and Fault Garage at `a4af8c4`.

## 1. Why

Boundary v0 is each reply's distance to the nearest earlier chunk of the episode (`body/stepm1.news`). It reads
whether an arrival has been **seen**. The North star's boundary reads whether an arrival was **caused**: "what
surprised it from outside, via efference copy". These are two different questions, and they cross:

| | The agent caused it | The agent did not |
|---|---|---|
| **Not seen this episode** | v0: news. Efference copy: quiet | both: news |
| **Seen this episode** | both: quiet | v0: quiet. Efference copy: news |

On the diagonal the two readings agree, so no test there can tell them apart. The claim that the organism knows what
it caused lives in the other two cells.

**What the tickle test covers** (PLAN, "After the gate"): it is Step −1 again, read by boundary v1.

| Tickle test part | Cell | Tells v1 from v0? |
|---|---|---|
| Exact self-talk | caused, seen | No: both quiet |
| Planted novelty | not caused, not seen | No: both news |
| Restated self-talk (at most 5%; v0 78.8%) | caused, not seen | Yes, but only for an **echo**: the reply is the action in other words |
| Control 4, self-caused against spontaneous novelty | — | Reported, deciding nothing |

So one off-diagonal cell is gated in its weakest form and the other is not asked. A forward model that learned "the
reply resembles my last action" passes the restated part without predicting any consequence of acting. And nothing
asks for news from a repeat. The cause test asks both, with the reply held the same and only its cause changed.

## 2. The reading under test

Boundary v1 as the tickle test defines it: reply t+1's distance from the forward model's prediction, made at the last
token of action t+1, in the forward model's 64-dimensional space, in units of v1's own floor. The forward model is the
frozen graft of decision 1. Nothing here changes v1, the graft or `substrate/`.

Read beside it on the same replies, deciding nothing except in bar 3: v0; and the residual of Step 2's linear history
predictor (the action's state and the mean of the last 10 chunks).

**It runs only if bar 3 of the babbling gate and the tickle test pass.** On a fail of either, boundary stays v0 and
there is no v1 to read.

## 3. The test: one reply, two causes (a dissociation, as Step −1C and area's)

**Why Fault Garage.** Its world has the control the family table names (world-family.md, section 3: "self-caused
events against external ones"): a scan lists "lost communication with M" when the technician has unplugged module M,
flagged self-caused by the oracle, and lists the same code when module M is itself the fault
(`circuit.module_dead`). One content, two causes, labelled by the world. Wording is data there since Fault Garage
PR #14, so the reply can be reworded on every visit.

A scripted player, level 2, wording `fresh`. Each episode: the greedy player for 10 turns (the burn, as every sense),
then a short fixed script ending in one **scored scan** at the same turn in both kinds of a contrast, so every
comparison is matched on position by design (the owner's position rule, 2026-11-08). The episode ends after the
scored scan. The kind is in the seed's key, read only by scoring.

**Contrast A, the consequence** (not seen; caused against not caused). The scored scan lists lost communication with
M, and no earlier chunk of the episode states that code.

- **Caused:** module M is healthy. The script unplugs M, then scans.
- **Not caused:** module M is the fault. The script makes a command that changes nothing in M's place (a `lookup`),
  then scans.

Efference copy says: quiet when caused, news when not. v0 should not part them: the scored reply has the same
content, in a phrasing new to the episode, in both kinds.

**Contrast B, the return** (seen; caused against not caused). Early in the script, in both kinds: unplug M, scan
(the code appears, caused), plug M, clear. Then:

- **Caused again:** the script unplugs M once more, then scans. The code is back, and the action explains it.
- **Not caused:** the script leaves M plugged and turns the harness wiggle on; M has a fault that shows only over
  bumps, so the code is back with no unplugging. Then it scans. *Not buildable in Fault Garage as it stands* (checked
  at `a4af8c4`, below).

Efference copy says: quiet when caused again, news when not. v0 should not part them: in both kinds the scored reply
repeats a content the episode has already shown.

*The agent's note:* the two kinds of a contrast must differ in one command, because the cause is the command. What is
held the same is the scored reply's content, the scan that elicits it, the turn, and the wording stream.

*What "not caused" has to mean.* v1 is the forward model's residual, so it is quiet for anything the context predicts,
whoever caused it. A code that returns after `clear` because a steady fault is still there is not caused by the agent,
and is also no surprise. So each not-caused kind must be unpredictable from the episode so far. In contrast A the
seeds are kept only where the oracle's surprise at the scored scan is at least 1 bit (agent). In contrast B it is why
the return has to come from an intermittent fault.

*Contrast B and the catalogue (checked 2026-11-09 on 300 layouts).* Only connections can be intermittent
(`circuit.CONNECTIONS`); a dead module cannot. No bumps-only supply or ground wire takes its module off the bus (0 of
2,372 cases give the codes that unplugging gives: 3 ohms sets a low-voltage code at most), and a bumps-only open bus
link adds the isolated module's own lost-communication codes, so the scan's content differs. Contrast B therefore
needs one addition to Fault Garage, a module fault that shows only over bumps (for example the module's own
connector, open over bumps). That is a change to the fault catalogue: the owner's decision there, with its
DECISIONS line and tests. Until then contrast A stands alone, and bar 2 is a slot.

**Pass, all three (agent):**

1. **The consequence:** AUC of not-caused over caused, v1 at the scored scan, at least 0.80, its 90% lower bound
   above 0.70.
2. **The return:** the same bar on contrast B.
3. **Apart from v0:** on each contrast, the 90% interval of (v1's AUC − v0's AUC) lies entirely above zero.

80 episodes a contrast, 40 of each kind, 2,000 bootstrap draws resampling episodes. Seeds are a new range registered
in Fault Garage first; records cross stamped with the world's commit, the reader's commit and the model revision. A
rehearsal on the 0.6B decides nothing; the deciding reading is the 8B in bf16.

Reported, deciding nothing: the linear history predictor's residual on the same scans (section 5, the second risk);
v1 by layer; the oracle's surprise in bits at each scored scan (zero when caused, by construction); contrast A with
"unplug M, plug M, clear" in the not-caused kind, so the word *unplug* is in context in both kinds.

**What a pass means.** In a world built for it, v1 reads cause apart from novelty: the same reply is quiet after the
action that brings it and news without it. By the transfer rule, the general claim needs a world not built for it.
What a pass does not show is in section 6.

**What a fail means.** Bars 1 and 2 fail, tickle test passed: v1 is a better novelty reading than v0 and cancels
echoes, but does not predict consequences. It stays in the tokens as the tickle test decides; the North star's wording
"via efference copy" is not yet earned, and Dream's use of the forward model as a simulator has no instrument support.
Bar 1 passes and bar 2 fails: v1 predicts consequences but still treats a repeat as old news.

## 4. Before babbling: v0 reads the pool (stage A, can run now)

The plant is only sound if nothing but the cause differs between the kinds. v0 can check that today, with no graft:

- Play both contrasts' pools with the script (CPU), read v0 on the 8B.
- **Bar (agent):** v0's AUC between the kinds of each contrast has a 90% interval that includes 0.5 and an upper
  bound at most 0.65.
- Above it, the kinds differ in something v0 can see (a wording, a length, an earlier chunk that states the code),
  and the script is repaired before any v1 reading. The window is not changed to suit a v1 result.

This also gives bar 3's v0 side in advance, on rehearsal seeds, never on the deciding pool.

## 5. Risks

- **A world shaped for the sense.** Held by the transfer rule: the script, seeds, wording setting and bars are fixed
  before any read; the self-caused flag is the world's, written before this proposal; a general claim needs
  TextWorld, with its own bars.
- **The episode may already hold the answer.** The linear history predictor sees the mean of the last 10 chunks,
  and in the caused kinds that mean includes "unplug M". If its residual parts the kinds as well as v1 does, then an
  episode-local predictor conditioned on the action is enough, and babbling's lifetime model is not what earns the
  result. That would be a finding about where efference copy lives, told to the owner, not a fail of the test.
- **The forward model never met the action.** If babbling's world is TextWorld (it is chosen by the 8B's measured
  reach), the graft has never seen an unplugging, and a fail would mean inexperience. The cause test is read in the
  world the graft babbled in. In Fault Garage, the babbling record must hold enough unplug-then-scan turns: the
  scripted players never unplug (R4 had none), and the 8B's first play there stayed key-off. Fewer than 50 such turns
  in the babbling record (agent): too few to decide.
- **The actions are not its own.** The script acts and the graft reads, as the gate scores the forward model on the
  base model's play. The test asks whether the prediction depends on the action in context, not whether the organism
  chose it. The own-action form (the grafted player's live play, self-caused scans against others) is reported with
  Step 2's live play.
- **Keys supply.** v1 is fixed before the pool is read; no readout is fitted to the self-caused flag.
- **The catalogue.** Contrast B needs a module fault that shows only over bumps, and the catalogue has none
  (section 3). Adding one changes the oracle's hypothesis space and every prior, so earlier pools must still rebuild
  byte for byte: it belongs behind a setting that is off by default, as wording was.

## 6. What it does not test

Measures stay apart. This is an instrument test of v1 in one world. It does not show that the organism behaves
better for having v1 (Step 3), that acting matters to learning the forward model (the yoked arm, still a slot), or
that v1 holds in a world not built for it (the transfer rule).

## 7. Questions for the owner

- **Q1.** The two cells as the content of "efference copy", with the cause test as their instrument test, after the
  tickle test?
- **Q2.** What it gates. The agent's proposal: v1 enters the tokens on the tickle test, as planned; the staged
  schedule's row "Boundary, efference copy" is marked passed only on the cause test; and Dream's simulator waits on
  bar 1.
- **Q3.** The contrasts and bars as drafted (0.80 with a lower bound above 0.70; apart from v0; 40 a kind)?
- **Q4.** Stage A now, on rehearsal seeds, before babbling runs on the 8B?
- **Q5.** If babbling's world is TextWorld: a TextWorld form of the cause test, drafted separately, or the test read
  in Fault Garage after a second babbling there?
- **Q6.** Control 4 (self-caused against spontaneous novelty, reported in the tickle test): kept as it is, or
  replaced by this?
- **Q7.** Contrast B: add an intermittent module fault to Fault Garage's catalogue (off by default), or run
  contrast A alone and leave the return as a slot?
