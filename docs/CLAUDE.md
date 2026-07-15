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

## Amendments (pre-experiment, on record — 2026-07-14, before any bandit code exists)

These were made during baseline implementation, before the first experiment run and
before rung 4. Git history timestamps them. The writeup must report all three.

1. **`pe_threshold` default 0.25 (signal params: `ewma_alpha=0.3`, `pe_threshold=0.25`).**
   Correctness repair, not tuning: from a calm signal (EWMA = 0), a single binary
   misprediction yields EWMA = alpha = exactly 0.3, and the strict `>` comparison
   against a 0.3 threshold could therefore never fire on first evidence of a shift —
   the parameter was dead, not suboptimal. The repair direction biases AGAINST the
   pre-registered hypothesis: it makes the threshold heuristic (the killer baseline)
   stronger, so the arbiter's ≥10% win criterion gets harder, not easier.
2. **Disproof-aware retrieval query.** `believed_goal()` skips goal evidence that a
   LATER entry in the same log contradicts (agent stood on that cell, reward 0). The
   log itself stays append-only — no decay/merging/consolidation (still parked). Without
   this, every memory-bearing mode deadlocks magnetically on stale evidence after a
   shift, and all arbitration experiments (heuristic AND bandit, which share the mode
   implementations) would measure the deadlock artifact rather than arbitration.
3. **Experiment set frozen pre-bandit: `experiments/exp-v1.json`.** Regimes, shift
   schedules, seeds, episode counts, and agent parameters are fixed before rung 4
   exists, spanning difficulty (grid size, relocation distance, probe bluntness,
   shift frequency) so a "matches" verdict is a finding about arbitration, not about
   an easy toy. May not be revised after the first bandit run.
4. **Rung-4 guard: the bandit's context-feature list is part of the experiment.** It
   must be declared and locked in the implementation plan BEFORE any bandit code is
   written. The comparison is "learned policy vs. fixed rule on the same signal";
   every feature beyond the shared EWMA is machinery that the study framing counts
   against the method framing.
5. **Bandit registration completed pre-first-run (2026-07-14, same day, zero bandit
   runs executed).** Context features LOCKED at `(1, ewma, has_belief)` — exactly the
   heuristic's inputs, nothing more. Algorithm: LinUCB (ridge lambda = 1),
   deterministic tie-breaking. Reward `r = env_reward − cost_weight × step_cost_usd`
   ties the bandit to success-per-dollar. Frozen hyperparameters `ucb_alpha = 1.0`,
   `cost_weight = 100.0` added to `experiments/exp-v1.json` `agent_params` in the same
   commit that implements the bandit — completing a registration the freeze opened, not
   revising results-adjacent choices. Adding the two `RunConfig` fields changes every
   condition's `config_hash`; no experiment results existed, and hashes are final from
   this commit on.

6. **Amendment 6 (2026-07-14) — final registration items; the registration is CLOSED
   after this commit.** Zero experiment-set runs have been executed.
   - **(a) Model swap: `claude-haiku-4-5` → `Qwen/Qwen3-4B` served on vLLM**
     *(model name corrected — see the Correction note below; originally misrecorded as
     Qwen3-8B).* The hypothesis is about arbitration over modes, not about any
     particular model's quality, so it is model-agnostic; a small open model
     self-served on vLLM gives the throughput and cost profile 600 conditions need (no
     provider rate limits, no per-call billing risk). The substrate is validated by the
     pre-declared calibration criterion: the calm-state world-model probe
     (`runs/calibration/20260715T001705-Qwen-Qwen3-4B`, 500 stratified cases) measured
     a misprediction rate of **0.024, 95% CI [0.012, 0.038] — GREEN** (bands
     pre-declared: <0.15 GREEN / <0.25 YELLOW / ≥0.25 RED), leaving `pe_threshold =
     0.25` an order of magnitude of headroom above the noise floor. Dollar accounting
     uses a market-rate proxy price table of $0.10 input / $0.30 output per MTok; the
     previous `claude-haiku-4-5` table ($1/$5) is retained as Design decision 1's
     alternate table for the writeup's sensitivity check.
   - **(b) Zero-cost metric rule.** Retrieve-only spends $0, making success-per-dollar
     degenerate (division by zero). The full-roster comparison is therefore presented
     as a cost-vs-success Pareto plot; the success-per-dollar ratio is reserved for
     dollar-spending agents. The pre-registered bandit-vs-heuristic criterion
     (Design decision 8) is untouched — both agents spend on every step's world-model
     call, so their ratio is well-defined.
   - **(c) Pairing: relocations are drawn from a dedicated RNG independent of agent
     state.** Previously relocation draws consumed the env RNG and excluded the
     agent's current cell, so the goal sequence depended on agent behavior — unpaired
     comparisons and an agent→environment leak. Now a dedicated stream (seeded from
     the run seed) draws relocations, excluding only the old goal and the start cell.
     Goal sequences are identical across agents for the same (regime, seed): paired
     comparisons, lower variance. The analysis path must assert shift-count equality
     and goal-sequence prefix consistency across agents per (regime, seed), loudly
     flagging conditions where an efficient agent finished before late shifts fired.
   - RunConfig default-model and price changes move every condition's `config_hash`
     one final time; hashes are frozen from this commit forward. *(Superseded — see the
     Correction note below.)*

   **Correction (2026-07-14).** Amendment 6a as originally committed **misnamed the
   model as `Qwen/Qwen3-8B`**. The deployed, calibration-probed, and intended substrate
   is **`Qwen/Qwen3-4B`** — a transcription error between the deployment and the
   registration text, not a change of substrate. Evidence: the serving endpoint has
   only ever hosted Qwen3-4B, and the calibration report
   (`runs/calibration/20260715T001705-Qwen-Qwen3-4B`: rate 0.024, CI [0.012, 0.038],
   GREEN) — run **before any experiment-set run and before this correction** —
   records the actual substrate. Zero experiment-set runs have been executed. This
   note exists so the error stays visible; the amendment is not silently rewritten.
   - The $0.10/$0.30 market-proxy price table is **retained unchanged**: it is a
     synthetic accounting table per Design decision 1, and if anything it is
     conservative (over-priced) for a 4B model.
   - **cost_weight interaction, disclosed:** `cost_weight = 100.0` was frozen
     (Amendment 5) while the registered price table was still $1/$5; under the
     $0.10/$0.30 table the bandit's effective cost-aversion is therefore 10× weaker
     than under the table it was sized against. This is internally consistent — the
     bandit optimizes exactly the metric the analysis measures, both computed under
     the registered table — but the price-table-relative meaning of `cost_weight`
     belongs in the writeup's sensitivity section alongside the alternate table.
   - The model string moves every condition's `config_hash`; **hashes are final from
     this correction commit**, superseding the same claim in Amendment 6, which was
     voided by the transcription error.

## Deadline

Public repo + writeup posted by **August 14, 2026**, regardless of which baseline wins.
Finishing is the deliverable.