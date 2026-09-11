# FUTURE.md — The Parking Lot

Ideas live here so they stop living in the codebase. Nothing in this file is deleted from
the vision; it is sequenced behind Phase 1. Nothing in this file may be implemented until
the Phase 1 result is published.

Rule: when the escalation urge hits mid-build, append it here with one line of context
and close the file.

## Parked from the original Venjix vision

- **Memory dynamics** — strengthening, decay, merging, consolidation (complementary
  learning systems style). Earns entry only if retrieval-mode results show stale-memory
  failures worth modeling.
- **Self-model / autobiographical identity** — capabilities, limitations, reputation,
  continuity across sessions.
- **Reflection layer as a distinct component** — currently subsumed into prediction-error
  calibration; revisit if calibration alone proves too coarse.
- **Multi-agent society** — shared memories, trust scoring, negotiation, coalitions.
- **Persistent NPCs** — market-tested and failed for others (Inworld pivot, Altera
  rebrand); revisit only with a specific paying-buyer thesis.
- **Robotics / physical embodiment** — same cognition, different action layer. Far future.

## Parked expansions from planning conversations

- **Simulated embodiment testbed** — Habitat or AI2-THOR navigation under nonstationarity
  (move furniture / lock doors after learning the layout). The most legitimate Phase 2+
  candidate: same hypothesis, embodied instantiation, still laptop-scale.
- **Robostral Navigate integration** — Mistral's single-RGB-camera navigation model as an
  action layer. Blocked anyway: no public API/weights as of July 2026, enterprise contact
  only.
- **Learned neural arbiter** — anything beyond the contextual bandit (RL-trained arbiter,
  learned value-of-computation). Only if the bandit clearly beats the heuristic.
- **Transition-dynamics shifts** — wall/door toggles as a second shift type (config flag
  exists; implementation is Phase 2). Phase 1 shifts are reward relocation only.
- **Standard benchmark ports** — ALFWorld / WebArena variants with injected
  nonstationarity, for external validity. Phase 2 if Phase 1 lands.
  *2026-09-10 refinement (post exp-v1):* adopt-and-modify rather than build. Order:
  **MiniGrid/BabyAI** first (same shape as the Phase 1 gridworld, Gym API, language
  missions; shifts = relocate goal/keys/doors between episodes; keeps modes, cost
  accounting, pairing and analysis unchanged), then **ALFWorld** (move objects /
  receptacles between episodes) only after the *policy gate* shows the chosen model
  functions there; WebArena out of budget. Carry-overs from exp-v1: run the policy
  gate before registering anything; register success-per-dollar as a pooled ratio or
  paired log-ratio, not a mean of per-condition ratios; pick a substrate whose
  non-thinking policy moves, or budget for thinking.
- **Additional agent modes** — asking a human, delegating to a sub-agent, tool synthesis.
- **Richer world models** — fine-tuned or distilled world models instead of prompted LLM
  rollouts.
- **Harness crossover** — applying the arbiter to a repo-tuned coding agent (simulate
  blast radius before editing; Onboarding Mirror groundwork). Separate project; do not
  merge codebases.

