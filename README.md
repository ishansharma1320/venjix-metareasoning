# Venjix — Phase 1: Adaptive Metareasoning Under Nonstationarity

A study of when LLM agents should **act directly, retrieve past experience, simulate
outcomes, or gather new evidence** — and whether online prediction-error calibration of
that choice beats fixed strategies and simple threshold heuristics.

## Pre-registered hypothesis (July 2026, before any experiments were run)

In environments that undergo distribution shift, a prediction-error-calibrated arbiter
over the four agent modes will outperform:

1. fixed single-mode agents (reactive-only, retrieve-only, simulate-only),
2. fixed mode mixtures,
3. a prediction-error **threshold heuristic**,

on **success-per-dollar** (task success normalized by token + tool cost).

Predicted mechanism: at a shift, prediction error spikes; the arbiter shifts spend away
from simulation/retrieval toward evidence-gathering, then reverts as it recalibrates.

**Committed in advance:** the arbiter "wins" only if it beats the threshold heuristic by
≥10% relative on success-per-dollar with non-overlapping bootstrap 95% CIs across 20
seeds. Anything less counts as "matches," and the project publishes as a study of how
little machinery arbitration actually requires. This criterion will not be revised after
the first experiment runs. Negative results publish identically to positive ones.

## Phase 1 deliverable

- One toy nonstationary environment with configurable, silent shift schedules
- Reactive baseline agent
- JSONL episode logs with per-step cost accounting
- (Subsequent weeks: remaining baselines → heuristic → contextual-bandit arbiter →
  writeup with the mode-choice-over-time plot)

## Definition of done

Public repo + writeup posted by **August 14, 2026**, regardless of outcome.

## Status

- [x] Environment + shift scheduler (tested)
- [ ] Reactive baseline
- [ ] Episode logging + cost accounting
- [ ] Retrieve-only, simulate-only, fixed-mixture baselines
- [ ] Threshold heuristic
- [ ] Contextual-bandit arbiter
- [ ] Shift experiments + plots
- [ ] Writeup

## Non-goals (see FUTURE.md)

Embodiment, memory dynamics, multi-agent systems, NPCs, neural arbiters, products.
This repo tests one claim.