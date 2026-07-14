# Venjix — Phase 1

## What this project is

An empirical study of adaptive metareasoning for LLM agents under nonstationarity.

**Hypothesis:** An arbiter that uses online prediction-error calibration to choose between
agent modes — act directly, retrieve past experience, simulate outcomes, gather new
evidence — outperforms (a) fixed single-mode agents, (b) fixed mixtures, and (c) a simple
prediction-error threshold heuristic, on **success-per-dollar** in environments that undergo
distribution shift.

**Predicted signature behavior:** at a distribution shift, the arbiter's prediction error
spikes; it should reduce reliance on simulation and retrieval, temporarily increase
evidence-gathering, then return to cheaper modes as it recalibrates. The
mode-choice-over-time plot at the shift point is the centerpiece result.

**Acceptable outcomes (both are wins):**
- Learned/contextual arbiter beats the threshold heuristic → method contribution.
- Threshold heuristic matches the arbiter → study contribution ("arbitration helps but
  needs almost no machinery"). Do not torture results into a method paper.

## Phase 1 finish line (FIXED — do not expand)

1. One toy nonstationary environment (rules/rewards silently shift at configurable steps).
2. One reactive baseline agent running in it.
3. Episode logs with per-step cost accounting (tokens, tool calls, wall time).

That is the entire Phase 1 deliverable. Later weeks add: remaining baselines
(retrieve-only, simulate-only, fixed mixture), the threshold heuristic, then the
contextual-bandit arbiter — in that order, each only after the previous is running.

## Scope rule (non-negotiable)

**Scope is fixed. Any idea beyond the current step — memory decay, embodiment,
Habitat/AI2-THOR, Robostral, learned neural arbiters, multi-agent systems, self-models,
NPCs, product features — must be appended to FUTURE.md, never implemented.**
If the user asks for an out-of-scope feature mid-session, add it to FUTURE.md and remind
them of this rule.

## Baseline hierarchy (build in this order)

1. Fixed single-mode agents: reactive-only, retrieve-only, simulate-only.
2. Fixed mixtures (static mode proportions, no adaptivity).
3. Prediction-error threshold heuristic (~2 lines of decision logic).
4. Contextual bandit arbiter over rolling prediction-error features. No neural nets in v1.

Each rung only matters if it beats the rung below. The heuristic (3) is the killer
baseline — take it seriously.

## Metrics (all required, logged per episode)

- Task success rate
- Total cost (LLM tokens in/out, tool calls, simulated-step count)
- **Success-per-dollar** (primary)
- Recovery time after each shift point (see Design decision 7; regret-vs-oracle is out of scope)
- Mode-choice distribution over time (for the signature plot)

## Tech choices

- Python 3.11+, single package, `pyproject.toml`, uv or pip.
- Environment: pure-Python gridworld (decided — see Design decisions). No game engines,
  no simulators, no external benchmark suites in Phase 1.
- LLM calls behind a single thin client interface so models are swappable and every call
  is cost-logged. Support a mock/deterministic model for fast tests.
- Logging: JSONL episode logs, one file per run, plus a run manifest (config hash, seed,
  shift schedule). Plots with matplotlib. No dashboards.
- Determinism: seeds everywhere; every experiment reproducible from config + seed.
- Tests: pytest for environment dynamics and shift scheduling (these must be provably
  correct or all results are garbage).

## Working style for Claude Code

- Propose plans before writing code for any new module (Plan mode expected).
- The logging schema and environment/agent interfaces are foundational — flag any change
  to them loudly instead of silently refactoring.
- Prefer boring, readable code over clever abstractions. This is a research harness,
  not a framework.
- When a result looks good, first look for the bug that would produce it.

## Design decisions (ratified — resolve ambiguity here, not in code)

These answer the open questions surfaced in the first comprehension review. If
implementation reveals one of these is unworkable, stop and flag it; do not silently
substitute a different design.

1. **Costing model.** Synthetic price table defined as config constants: real API
   per-token prices (input/output, per 1M tokens) for the chosen model. Simulated
   rollout steps cost whatever LLM calls they consume under the same table. Probe
   actions cost environment steps (bounded by the step budget), not dollars. Wall time
   is logged but never priced. The mock model's token counts are priced with the same
   table so mock-mode comparisons remain meaningful. The writeup includes one alternate
   price table as a sensitivity check.
2. **Prediction-error signal.** Before each action, the agent's world model predicts the
   next observation and reward; after acting, the misprediction is scored (binary for
   discrete observations). The arbitration signal is an EWMA of the misprediction rate
   (window/decay in config). The threshold heuristic and the bandit arbiter consume the
   IDENTICAL EWMA — never two different signals.
3. **Mode semantics (gridworld).**
   - `act`: one cheap policy call on the current observation.
   - `retrieve`: nearest-match lookup over an append-only episodic log of
     (state, action, outcome) tuples. The log has no decay, merging, or consolidation —
     memory *dynamics* are parked in FUTURE.md; the dumb log is in scope.
   - `simulate`: k-step world-model rollouts over candidate actions before committing;
     each simulated step is cost-accounted.
   - `gather_evidence`: a probe/look action revealing local map information; consumes a
     step, makes no task progress.
4. **Environment.** Gridworld. Phase 1 shift type: silent reward relocation, magnitude
   parameterized by relocation distance. Transition-dynamics shifts (wall/door toggles)
   are a config-flagged Phase 2 feature — parked.
5. **Arbitration granularity.** Per-step. A fixed mixture is an i.i.d. per-step draw
   from a static mode distribution.
6. **Success.** Binary: goal reached within the episode step budget. No partial credit.
7. **Recovery (replaces "regret").** Steps after a shift until rolling success-per-dollar
   returns to 90% of its pre-shift EWMA. Regret-vs-oracle is out of scope.
8. **Statistical plan and decision criterion (pre-committed).** 20 seeds per condition;
   shift schedules drawn from a fixed, versioned set; bootstrap 95% CIs on all reported
   metrics. The arbiter "wins" only if it beats the threshold heuristic by ≥10% relative
   on success-per-dollar with non-overlapping CIs. Anything less is "matches" and
   triggers the study framing. This criterion may not be revised after the first
   experiment runs.
9. **Naming.** The mode/baseline is called `retrieve` / "retrieve-only" everywhere.
   "Memory-only" is deprecated.

## Deadline

Public repo + writeup posted by **August 14, 2026**, regardless of which baseline wins.
Finishing is the deliverable.