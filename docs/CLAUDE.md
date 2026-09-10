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
     pre-declared calibration criterion: the calm-state world-model probe against the
     FINAL serving config (temperature 0, thinking disabled, guided_regex constrained
     decoding on world-model calls) — `runs/calibration/20260715T005234-Qwen-Qwen3-4B`,
     500 stratified cases — measured a misprediction rate of **0.024, 95% CI
     [0.012, 0.038] — GREEN** (bands pre-declared: <0.15 GREEN / <0.25 YELLOW / ≥0.25
     RED), leaving `pe_threshold = 0.25` an order of magnitude of headroom above the
     noise floor. All 15 residual parse failures classify as `out_of_range` (the model
     names the unclipped boundary-adjacent cell; the parser's bounds check + stay-in-
     place fallback neutralize it) — zero truncation, zero format drift under
     constrained decoding. Dollar accounting
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

7. **Analysis plan registered pre-data (2026-07-15): see `docs/ANALYSIS.md`.** Governs
   shift-exposure handling (full-data headline, pre-declared equal-exposure robustness
   check, shift-indexed mechanism analyses, exposure reported not adjusted), the
   assertion set the analysis must pass before producing a verdict, and the
   WIN/MATCHES output. Registered before any real-model experiment-set run existed;
   may not be revised after the first real run.

8. **Amendment 8 (2026-07-15) — substrate: `Qwen/Qwen3-8B` served via OpenRouter.**
   Zero experiment-set runs executed. The registered 4B substrate was lost
   operationally: the self-hosted Vast endpoint died mid-preparation (dead worker,
   silent recreation, unreliable provisioning), and Qwen3-4B has been delisted from
   managed serving (verified: absent from the OpenRouter catalog). Selection was a
   cost-benefit over the catalog — free tiers excluded by rate-limit arithmetic
   (~1,000 requests/day vs the set's ~1.9M calls), then paid candidates compared on
   price, provider count, quantization stability, and registration distance.
   `qwen/qwen3-8b` chosen as the family-minimal step: same family, dense, nearest
   available size (and the id Amendment 6a originally recorded).
   - **Serving config**: OpenRouter, provider pinned to Alibaba (single provider =
     one quantization for the whole run), temperature 0, seed 0, reasoning disabled
     via OpenRouter's normalized field. `guided_regex` does not exist on this path;
     format robustness rests on the parser's bounds check + stay-in-place fallback
     (measured benign in the v1 probe).
   - **Prices become the real per-token prices of the chosen serving path** per
     Design decision 1: $0.117 input / $0.455 output per MTok (projected set cost
     ≈ $32). The haiku table stays the sensitivity alternate. The `cost_weight`
     disclosure in the Amendment 6 Correction applies with the new denominator.
   - **Calibration probe (the pre-declared gate) re-run against this exact config:**
     `runs/calibration/20260716T185958-Qwen-Qwen3-8B` (500 stratified cases) —
     misprediction rate **0.030, 95% CI [0.016, 0.046] — GREEN**, ~8× headroom under
     `pe_threshold = 0.25`. All 28 parse failures classify as `out_of_range`
     (unclipped boundary cell; format-perfect replies; every one neutralized by the
     bounds check + stay-in-place fallback and scored correct) — zero truncation,
     zero format drift even without constrained decoding.
   - Hashes move with model + prices and are final from this commit, superseding
     prior finality claims.

9. **Candidate evaluation on record (2026-07-16/17): `google/gemma-3-4b-it` REJECTED
   by the calibration gate. No substrate change — Amendment 8 stands.**
   - **Why it was evaluated**: qwen3-8b's sole provider (Alibaba) chronically
     rate-limits upstream — measured net throughput 4–5 calls/s through sustained 429
     storms, projecting 5–6 days of run exposure. Candidate burst tests (identical
     3×32-concurrent probes, same account, same moment) found gemma-3-4b-it via
     DeepInfra at 30.5 calls/s burst and **102.5 calls/s sustained over 4 minutes
     (24,668 calls, zero 429s)** — ~20× the throttled rate, ~40% cheaper.
   - **Why it was rejected**: the pre-declared calibration gate returned **RED**:
     misprediction rate **0.354, 95% CI [0.312, 0.398]**
     (`runs/calibration/20260717T035200-google-gemma-3-4b-it`) — a noise floor ABOVE
     `pe_threshold = 0.25`, meaning the arbitration signal would fire on calm-state
     noise and the experiment would measure model incompetence, not nonstationarity.
     Failures: 72 out_of_range + 25 format_drift (19% parse-error rate), 62%
     misprediction on probe actions. Every operational metric was superior; the
     substrate is simply not competent at the task. Rejection is exactly the gate's
     purpose (bands: ≥0.25 RED = substrate fails, escalate deliberately).
   - Registered substrate, prices, and hashes remain exactly as Amendment 8 set them.

10. **Amendment 9 (2026-09-09) — serving path: `Qwen/Qwen3-8B` self-served on vLLM
    (RunPod Secure Cloud, one A100 80GB). Same registered model; zero conditions
    carried forward.**
    - **Why**: the Amendment 8 run (OpenRouter → Alibaba) was executing on a Raspberry
      Pi when the Pi failed from thermal damage and its filesystem was corrupted; the
      in-progress results are lost. Independently, Alibaba's upstream throttle held net
      throughput at 4–6 calls/s (item 9), i.e. 8–9 days of exposure per attempt for a
      set of ~4.3M calls (measured: ~7,200 calls per condition, ~136 input / ~13 output
      tokens per call). A dedicated GPU removes the rate limit entirely.
    - **Serving config** (the exact config the calibration gate is re-run against):
      canonical Hugging Face `Qwen/Qwen3-8B` weights in bf16 (no quantization — the
      Alibaba path's quantization was never disclosed), vLLM OpenAI-compatible server,
      temperature 0, seed 0, thinking disabled via `chat_template_kwargs`
      (`enable_thinking: false`), `max_tokens 256`, **`guided_regex` constrained
      decoding restored on world-model calls exactly as Amendment 6a registered it**.
      Policy (action-choice) calls stay unconstrained as always registered — considered
      and declined on 2026-09-09: measured reply length is identical with and without
      the constraint (8B unconstrained probe 13.3 output tokens/call vs 4B constrained
      probe 14.3), so there is no waste to repair, and constraining them would alter the
      logged parse-error fallback and the policy-heavy agents' dollar cost for no
      correctness reason. Served by **vLLM 0.29.0 (cu129 build) on torch 2.13.0+cu129**
      (host driver CUDA 12.8; the PyPI default cu130 build cannot initialize CUDA on
      it), `--max-model-len 4096 --max-num-seqs 256 --gpu-memory-utilization 0.90`.
    - **Client repair (correctness, not tuning), same day, before any experiment
      run:** vLLM renamed the constrained-decoding request field. The client's
      Amendment 6a-era `guided_regex` is *accepted but silently ignored* by vLLM 0.29
      — verified live by sending a regex the model would never emit (`ZEBRA-[0-9]{3}`):
      `guided_regex` → unconstrained reply; `structured_outputs: {"regex": …}` →
      `ZEBRA-100`. `llm.py` now sends `structured_outputs`. Without this repair the
      registered "constrained decoding on world-model calls" would have been a
      no-op. Bias direction: neutral (shared substrate; both arbiters' world-model
      calls are affected identically). **Supplementary data point**: the gate run
      accidentally executed on this serving path *before* the repair (i.e.
      unconstrained) measured 0.032, 95% CI [0.018, 0.048], GREEN, 27 parse errors —
      `runs/calibration/20260909T205745-Qwen-Qwen3-8B` — statistically
      indistinguishable from the Amendment 8 OpenRouter probe (0.030, 28 errors),
      which is itself evidence the two serving paths host the same substrate.
    - **Substrate uniformity**: Amendment 8's rule (one serving path for the whole run)
      is kept. Every condition produced under the OpenRouter/Alibaba path — the three
      complete and the partial runs on the laptop, and whatever the Pi held — is
      **discarded**; the laptop copies are moved to `runs/exp-v1-openrouter-abandoned`
      (kept for the record, never analyzed). All 600 conditions are run fresh on the
      new path, resume-by-config-hash starting from an empty `runs/exp-v1`.
    - **Prices**: the Amendment 8 table ($0.117 input / $0.455 output per MTok) is
      **retained unchanged** as the registered synthetic accounting table (Design
      decision 1 wording: "synthetic price table defined as config constants").
      Self-hosting has no per-token price; the table is now a market-rate reference for
      this model class, not the serving bill (for the record: Alibaba Model Studio's
      list price for qwen3-8b on 2026-09-09 is $0.18 / $0.70; the actual marginal cost of
      the set is GPU time, ≈ $1.59/h × 6–8 h). Keeping the table keeps every
      `config_hash` unchanged. **Bias direction: neutral** — both agents in the
      pre-registered criterion are priced under the same table with `cost_weight`
      unchanged, and the shared substrate cannot favor either arbiter.
    - **Calibration gate (of record)**: the pre-declared calm-state probe, re-run
      against this exact serving config with constrained decoding enforced, **before
      any experiment-set run**: `runs/calibration/20260909T205928-Qwen-Qwen3-8B`
      (500 stratified cases, 64 workers) — misprediction rate **0.032, 95% CI
      [0.018, 0.048] — GREEN** (bands <0.15 GREEN / <0.25 YELLOW / ≥0.25 RED), ~8×
      headroom under `pe_threshold = 0.25`. Parse failures: 27, all `out_of_range`
      (the unclipped boundary cell, neutralized by the bounds check + stay-in-place
      fallback), **zero format_drift, zero truncation**. Identical point estimate and
      CI to the Amendment 8 probe on the OpenRouter/Alibaba path.
    - **Concurrency**: the batch executor runs at 96–128 workers against a localhost
      endpoint on the same pod. Execution order and concurrency cannot affect results
      (conditions are independent; see `batch.py`).
    - Registered model id, agents, regimes, seeds, prices, and hashes remain exactly as
      Amendment 8 set them. Registered on record before the first call to the new
      serving path.
    - **Run record (2026-09-10)**: 600/600 conditions complete, 0 failures, ~2.5M
      calls in 3.3 h at ~211 calls/s; pairing verification: 0 prefix divergences, 35
      (regime, seed) groups with unequal shift counts (the pre-declared
      finished-before-late-shift case; handled per docs/ANALYSIS.md, not adjusted).

11. **Amendment 10 (2026-09-10) — analysis computation details, declared BEFORE any
    analysis code is run on the data.** docs/ANALYSIS.md registers *what* is computed;
    two mechanical choices it leaves open are fixed here so neither can be tuned after
    a first look. No result has been computed at the time of this commit.
    - **Per-condition metric**: success-per-dollar (SPD) = successes / total
      `cost_usd` over the condition's 40 episodes (registered price table). Conditions
      with zero spend (retrieve-only) are excluded from any ratio (Amendment 6b) and
      appear only on the Pareto plot.
    - **Headline aggregation (Design decision 8)**: bandit vs heuristic are **paired by
      (regime, seed)** — 100 pairs. The statistic is the relative difference of mean
      SPD, (mean_bandit − mean_heuristic) / mean_heuristic, over all 100 pairs pooled
      across the five regimes. Bootstrap: **stratified by regime, resampling the 20
      seeds within each regime with replacement, pairs kept intact**, 10,000 resamples,
      RNG seed 0, percentile 95% CIs on each agent's mean SPD and on the relative
      difference. **WIN iff** relative difference ≥ 0.10 **and** the bandit's mean-SPD
      CI lower bound exceeds the heuristic's CI upper bound (the registered
      "non-overlapping" reading); else MATCHES. Per-regime numbers are reported as a
      table but do not enter the verdict. The §2 equal-exposure recomputation uses the
      identical procedure on the subset of pairs with equal shift counts.
    - **Recovery time (Design decision 7)**: per (regime, seed, shift k), for each agent
      that experienced shift k: episode-level SPD EWMA with **alpha = 0.3 (the signal's
      registered alpha; no new constant)**, baseline = EWMA value at the last episode
      that ended before the shift's global step; recovered at the first later episode
      whose EWMA ≥ 0.9 × baseline; recovery time = global steps from the shift to the
      end of that episode. Runs that never recover before the condition ends are
      **censored at the condition's final step and counted separately** (reported, not
      dropped). Comparisons at shift k are among experiencers only (ANALYSIS.md §3).
    - **Signature plot**: per-mode choice frequency by step offset from shift k, window
      −60…+180 global steps, averaged over (group, k) among experiencers, for every
      agent that chooses modes (heuristic, bandit, mixture; single-mode agents are flat
      by construction).
    - **Parse-error rate (assertion 3)**: fraction of step records with
      `parse_error = true`, per agent. **Goal-landed-on-cell (assertion 4)**: shift
      records whose `new_goal` equals the agent's position at that global step.
    - **Bias direction**: these choices are symmetric between bandit and heuristic
      (same pairing, same estimator, same alpha inherited from the shared signal). The
      stratified paired bootstrap is the *tighter* of the reasonable choices, which
      makes non-overlap easier to reach — noted so the writeup can say so; the
      alternative (pooled unpaired resampling) is reported as a sensitivity line.
    - The analysis script implements exactly this and docs/ANALYSIS.md, runs the
      assertion set first, and refuses to print a verdict if any hard stop fails.

12. **Analysis record (2026-09-10) — registered verdict: `MATCHES`.** Outputs in
    `results/exp-v1-analysis/` (report.json, summary.txt, signature.png, pareto.png).
    Assertion set: 600/600 complete, 0 prefix divergences, 35 unequal-exposure groups,
    parse-error rate 0.000 for every agent, goal-landed-on-cell 11/1906 (0.58%).
    - **Headline (registered estimator)**: bandit mean SPD 1061 [875, 1283] vs
      heuristic 1792 [986, 2813]; relative difference −0.41 [−0.57, −0.05]. Margin not
      met, CIs overlap → MATCHES. Equal-exposure robustness (90 pairs): MATCHES,
      −0.50 [−0.62, −0.23]. Per-regime: bandit ahead in 3/5 regimes (hard-11-blunt-probe
      +∞ since heuristic = 0, hard-15-far +0.60, mid-11 +0.62), behind in easy-7 (−0.64)
      and turbulent-11 (−0.13).
    - **What drives the estimator**: the registered SPD is the mean of per-condition
      ratios. Heuristic conditions in easy-7 that succeed at near-zero spend produce
      ratios up to 42,966 successes/$ and dominate the mean. **Post-hoc sensitivity,
      labeled as such and NOT the verdict**: pooled ratio-of-means (total successes /
      total dollars) is bandit 493/$ vs heuristic 136/$ (3.6×). The Pareto data
      (registered §4 / Amendment 6b) shows the bandit at 0.684 success
      [0.620, 0.745] vs heuristic 0.204 [0.133, 0.279] at the same cost per episode
      ($0.0014 vs $0.0015). Both views go in the writeup; the verdict line is the
      registered one.
    - **Two anomalies investigated before reporting ("look for the bug")**:
      reactive-only success 0.000 and heuristic `gather_evidence` share ≈ 0.000. Root
      cause is the substrate, verified verbatim (`results/exp-v1-analysis/policy-reply-
      diagnostic.md`): the `act` policy call returns the single word `probe` on every
      step, even when the probe result already reveals the goal. Not a parser or
      harness defect; the frozen agent prompt was not changed. The heuristic therefore
      almost never holds a belief its world model can be wrong about (2 mispredictions
      in 59,814 sampled steps) and its signal never fires; it succeeds only when the
      start cell is within probe radius of the goal.
    - **Mechanism**: the bandit's signature at a shift is mild (gather_evidence share
      0.21 → 0.25 in the 30 steps after a shift; retrieve dips 0.43 → 0.34), not the
      predicted sharp spike; the heuristic is flat (98% act). Recovery: bandit median
      20 steps with 63/297 censored; heuristic median 8 steps but 303/310 of its
      baselines are zero (it had no pre-shift successes to recover to).
    - **The study result**: retrieve-only (random walk + episodic memory, $0) reaches
      0.627 success, within the bandit's CI at zero LLM spend. Arbitration helps over
      every dollar-spending fixed strategy, but the cheapest strategy needs no
      arbitration at all. This is the "how little machinery" framing the registration
      committed to publishing.
    - **Follow-up diagnostic (same day)**: the same model with thinking ON moves
      sensibly on the same prompts (right when the probe finds nothing, down toward a
      revealed (2, 1) offset) at 1,000–4,500 output tokens per call. The registered
      thinking-off config, not the model, is what killed the act policy. exp-v1 stands
      as recorded; this motivates exp-v2 below.

13. **Amendment 11 (2026-09-10) — exp-v2 registered: same substrate, bounded thinking
    on policy calls. Zero exp-v2 runs executed at this commit.** Spec:
    `experiments/exp-v2.json` = exp-v1 byte-for-byte (regimes, seeds, episodes, agents,
    agent params, prices, prompts) plus **one field, `policy_thinking_budget: 512`**.
    exp-v2 is a separate pre-registered experiment published alongside exp-v1, never a
    re-run or replacement of it.
    - **The change**: Qwen3 thinking enabled on policy (act-mode) calls only, capped by
      a token budget using Qwen's two-phase recipe (generate with `max_tokens =
      budget + 16`; if the think block did not close, resend with the partial
      reasoning + a fixed closing line and `continue_final_message` so the model
      answers from what it has — a "forced answer"). World-model calls stay exactly as
      exp-v1: thinking off, regex-constrained (a grammar would mask the think block,
      and that path is separately calibrated). `RunConfig.policy_thinking_budget`
      (None = exp-v1) is omitted from the hash when None: **every exp-v1 hash is
      unchanged (test-pinned)**; exp-v2 hashes differ and cannot collide.
    - **Budget schedule, declared now**: **512 first**. The **sole registered fallback
      is 1024**, taken only if the policy gate is RED at 512. No other budget will be
      tried; unbounded thinking is not a condition (the point is whether bounded
      reasoning buys competence at a cost the arbiter can trade off).
    - **Policy gate (the check exp-v1 lacked), before the 600-run**: reactive-only on
      easy-7, 20 seeds × 10 episodes, exact exp-v2 serving config
      (`scripts/policy_gate.py`). **GREEN iff move fraction ≥ 0.50 AND success rate
      ≥ 0.15**; else RED. The gate detects a *dead* policy (exp-v1: 0.0 moves, 0.0
      success), not superiority: the mock random policy scores 0.21 on easy-7
      (`runs/exp-v1-mock`), with binomial sd ≈ 0.029 over the gate's 200 episodes, so
      0.15 is its lower 95% bound — GREEN means "moves, and not significantly worse
      than random". (Declared after the mock dress rehearsal of the script scored
      0.205, which showed a bar at the random point estimate fails a random policy on
      noise; declared before any exp-v2 call to the model.) The world-model
      calibration gate is re-run too if the serving stack changes at all.
    - **Prices unchanged**: thinking tokens are output tokens under the registered
      table, so an act step costs ~$0.0002–0.0005 instead of ~$0.00002 — act becomes an
      expensive mode. That is the experiment, not a confound; `cost_weight = 100` now
      bites on act and the writeup says so.
    - **Schema addition, flagged**: step records gain `forced_answer` (int, nullable;
      additive). Analysis reports the forced-answer rate among act steps per agent
      (reported, not a hard stop — the gate governs competence).
    - **Bias direction**: neutral in mechanism (every agent's act mode gets the same
      policy). In practice it strengthens the *heuristic* most, since the heuristic
      is act-dominant — i.e. it makes the killer baseline stronger, the honest kind of
      change.
    - Analysis plan for exp-v2 is docs/ANALYSIS.md + Amendment 10 unchanged; the
      script takes `--runs runs/exp-v2 --spec experiments/exp-v2.json`.
    - **Gate record (2026-09-10, same pod/stack as exp-v1's Amendment 9 config;
      world-model gate re-run first: 0.032 [0.018, 0.048] GREEN, identical to
      exp-v1).** Reports in `results/exp-v2-policy-gate/`. **Zero exp-v2 experiment-set
      conditions were run.**

      | budget | episodes | move fraction (bar ≥ 0.50) | success (bar ≥ 0.15) | forced-answer rate | output tokens / policy call | verdict |
      |---|---|---|---|---|---|---|
      | 512 | 200 | 0.012 | 0.075 | 1.000 | 532 | **RED** |
      | 1024 (registered fallback) | 200 | 0.082 | 0.365 | 0.963 | 1047 | **RED** |

      At 512 the policy is dead exactly as in exp-v1: every call hits the budget and
      the forced answer is "probe". At 1024 it is *not dead*: it follows a revealed
      probe offset to the goal in a few steps (the first ~20 episodes per condition
      were 20/20 successes at ~5 steps each), but once a shift moves the goal out of
      probe range it mostly probes again — success 0.365 (well above the 0.21 random
      floor) with only 8% of steps moving. **The registered schedule is exhausted:
      exp-v2 halts here.** Cost of the two gates ≈ $3.
    - **Gate-design note, on record**: the move-fraction criterion was written to catch
      "never moves" (exp-v1's 0.000); it also rejects a probe-heavy policy that
      succeeds 36.5% of the time. Whether that policy is "alive enough" to run exp-v2
      on is a judgment the gate as written does not make, and it is NOT re-made here.
      Any continuation (a redesigned gate criterion, or a larger budget) is a new
      amendment with its own justification and bias statement, decided by Ishan.

## Deadline

Public repo + writeup posted by **August 14, 2026**, regardless of which baseline wins.
Finishing is the deliverable.