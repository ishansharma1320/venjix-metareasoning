# exp-v3 plan — MiniGrid port (M0 approved 2026-09-11)

Status: **plan approved and committed (M0); decisions (a)–(d) resolved below.** This
is an engineering + design plan, not the registration: the registration proper (spec,
price table, metric constants) is milestone M3 and happens *after* the policy gate has
chosen a substrate (M2). Code for new modules starts only after this commit
(CLAUDE.md working style).

## 1. Goal

Same hypothesis, same six agents, same cost accounting, same paired-shift design,
same analysis machinery — on a standard environment instead of the Phase 1 toy. Fixes
the three exp-v1 lessons (docs/RESULTS.md carry-overs): policy gate before
registration, a substrate whose policy moves, and a pooled success-per-dollar metric.

## 2. Environment

- **Package**: `minigrid` 3.1.0 on `gymnasium` 1.3 (deps: numpy, pygame-ce; no display
  needed). Verified 2026-09-11: actions `left=0, right=1, forward=2` (pickup/drop/
  toggle/done unused); `obs = {image (7,7,3), direction, mission}`;
  `env.unwrapped.agent_pos / agent_dir` exposed; `grid.set(x, y, Goal())` relocates
  the goal. The package's zero-dependency rule is amended for Phase 2 (dependency
  group `minigrid`).
- **Base task**: `MiniGrid-Empty-{N}x{N}-v0` — "get to the green goal square". The goal
  is a real object that is only *seen* when inside the agent's 7×7 forward view cone.
  Partial observability replaces Phase 1's probe as the evidence mechanism. (BabyAI
  `GoToLocal` with distractor objects is the natural second regime family; **not in
  v3** unless decision (a) says otherwise.)
- **Shift** (unchanged concept): at registered global steps, relocate the Goal object
  by Manhattan distance *d* via `grid.set`, drawn from the dedicated paired RNG
  (Amendment 6c), excluding the old goal cell and the agent's cell. Silent: the
  mission string never changes.
- **Episode**: MiniGrid's `max_steps` overridden per regime (Phase 1 used 4×N); success
  = goal reached (binary; MiniGrid's shaped reward is ignored, Design decision 6).
  Reaching the goal ends the episode, exactly as Phase 1.
- **Observation given to the LLM** (text): absolute `(x, y)` and facing direction, the
  mission, and the view cone rendered as a small ASCII block with the goal marked if
  visible plus its offset — decision (b). Absolute coordinates keep the world model
  and retrieval memory comparable with Phase 1.

## 3. Modes on an egocentric action set

| mode | Phase 1 | exp-v3 |
|---|---|---|
| act | LLM picks up/down/left/right/probe | LLM picks **left / right / forward** |
| retrieve | greedy step toward remembered goal | deterministic egocentric planner (turn toward, then forward) toward the remembered absolute goal; seeded random turn/forward when no belief |
| simulate | world-model rollouts over 4 moves, depth 3 | rollouts over the 3 actions, depth 3; world model predicts `(x, y, dir, reward)` under a regex |
| gather_evidence | probe (reveals goal offset within radius) | **one `left` turn** — rotates the view cone; evidence = goal visible or not — decision (c) |

Memory: append `(pos, dir, action, next_pos, reward, goal_seen_at)`; `believed_goal()`
= last cell where the goal was seen or reached, with the same disproof rule
(Amendment 2). Heuristic and bandit logic unchanged; context features unchanged
`(1, ewma, has_belief)`.

## 4. Substrate — policy gate BEFORE registration (M2)

Candidates, cheapest first, all served on one RunPod A100 with the existing pod
scripts: `Qwen/Qwen3-8B` no-thinking (the Phase 1 baseline; expected dead),
`Qwen/Qwen3-30B-A3B-Instruct-2507` (non-thinking MoE, ~3B active — near 8B speed),
`Qwen/Qwen3-32B` no-thinking. Gate = reactive-only on the easiest regime
(Empty-8x8), 20 seeds × 10 episodes. **Criterion (fixed after the mock rehearsal,
before any model call): success rate ≥ the lower 95% bound of the mock random policy
on the same gate, p − 1.96·√(p(1−p)/200), where p is measured by running
`scripts/policy_gate.py --backend mock` — no move-fraction criterion (exp-v2
lesson).** The bar is written into this section (§4a) before the first model call.
The smallest GREEN substrate is chosen; its per-token serving price becomes the
registered price table (decision (d)). Budget ≈ $3. Serving (decided 2026-09-11,
cheapest pod = A100 80 GB at $1.59/h): `Qwen/Qwen3-8B` and
`Qwen/Qwen3-30B-A3B-Instruct-2507` in bf16 as published; the 32B as Qwen's own
`Qwen/Qwen3-32B-FP8` checkpoint (33 GB; weight-only FP8 on Ampere). **Rule: the
checkpoint that is gated is the checkpoint that runs in M4** — precision is part of
substrate identity (Amendment 8's one-provider-one-quantization principle), so gating
and running never differ. KV-cache quantization is irrelevant at ~300-token requests.

### 4a. Declared gate bar (2026-09-11, from the mock floor run; before any model call)

`scripts/policy_gate.py --set experiments/exp-v3.json --backend mock` on empty-8,
20 seeds × 10 episodes: mock random policy success **p = 0.320** (move fraction
0.692, 5,239 steps). Lower 95% bound = 0.320 − 1.96·√(0.320·0.680/200) = **0.255**.
**The M2 gate passes iff a candidate's reactive-only success rate on the same gate
is ≥ 0.255.** No other criterion.

## 5. Metric (fixes exp-v1 finding 1)

Success-per-dollar per (agent, regime) = **pooled**: total successes / total dollars
over the 20 seeds. Headline: bandit vs heuristic pooled SPD over all regimes; paired
stratified-by-regime bootstrap on (regime, seed) groups, 10k resamples, seed 0;
WIN iff relative difference ≥ 0.10 and the 95% CI of the difference excludes 0
(declared this way because the pooled statistic has no per-agent CI overlap to
compare). Equal-exposure robustness, recovery, signature, exposure table, Pareto:
as docs/ANALYSIS.md + Amendment 10. Zero-spend agents excluded from ratios.

## 6. Regimes (proposal; frozen at M3)

| regime | grid | max steps | shifts (global step, distance) |
|---|---|---|---|
| empty-8 | 8×8 | 32 | (120, 3), (280, 4) |
| empty-12 | 12×12 | 48 | (180, 5), (360, 8), (540, 6) |
| empty-16-far | 16×16 | 64 | (180, 7), (360, 12), (540, 9) |
| empty-12-turbulent | 12×12 | 48 | six shifts, distance 4–6 |

20 seeds × 40 episodes × 6 agents × 4 regimes = **480 conditions**.

### 6a. Mock dress rehearsal (2026-09-11, structural only — no model-behaviour claims)

480/480 mock conditions ran, 0 failures, pairing prefix-consistent. Mean shifts
experienced per agent (of 2 / 3 / 3 / 6 scheduled): empty-8 1.7–1.85, empty-12
2.0–2.95, empty-16-far 2.45–3.0, empty-12-turbulent 4.9–6.0; only 7/480 conditions
saw zero shifts. Mock steps per episode range from ~8 (retrieve, empty-8) to ~53
(reactive, empty-16-far). **Schedules in §6 are kept as drafted.** The rehearsal
also caught a real defect (zero placeholder prices → undefined SPD → summary crash),
fixed before any real data exists.

## 7. Cost & time (estimate, to be replaced by the gate's measurements)

Prompt ≈ 250–350 tokens (view block + mission) vs 136 in Phase 1; call counts similar
to exp-v1 if the policy moves (fewer if episodes end early). 30B-A3B decodes at roughly
8B speed. **≈ $15–30, 6–10 h on one A100.**

## 8. Engineering

New/changed modules (each with tests before use; foundational schema changes flagged):
- `venjix/minigrid_env.py` — adapter: `reset(seed)`, `step(action)`, observation
  dataclass (pos, dir, mission, view text, goal_visible, goal_offset), goal
  relocation hook for the shift scheduler, deterministic under seed.
- `venjix/shifts.py` — relocation via the adapter instead of the Phase 1 gridworld.
- `venjix/agents.py` — action vocabulary and planner parameterized by environment
  (Phase 1 gridworld agents keep working; tests stay green).
- `venjix/world.py` — egocentric prediction prompt + regex `(x, y, dir, reward)`.
- `venjix/logs.py` — step record gains `dir` and `goal_visible` (additive).
- `venjix/calibration.py` — case generator for the egocentric world model.
- `scripts/policy_gate.py` — success-only criterion; `experiments/exp-v3.json`.
- Mock model: 3-action policy + perfect egocentric world model for the dress rehearsal.

## 9. Milestones (each ends in a commit; no milestone starts before the previous lands)

- **M0** this plan approved and committed (decisions a–d resolved).
- **M1** adapter, agents, world model, tests; full mock dress rehearsal of the set
  (`runs/exp-v3-mock`), analysis script runs end-to-end on it. **DONE 2026-09-11**
  (commits 6c725db…66e44df): 164 tests; rehearsal 480/480, 0 prefix divergences,
  36 unequal-exposure groups, analysis + plots end-to-end. Note for M3: the analysis
  headline still implements Amendment 10's mean-of-ratios; the pooled estimator (§5)
  is added at M3 as Amendment 13, before any real exp-v3 data exists.
- **M2** policy gate on the candidate substrates (pod, ≈ $5–8); substrate chosen.
- **M3** registration freeze: `exp-v3.json`, price table, metric constants, analysis
  plan addendum; world-model calibration gate on the chosen substrate.
- **M4** the 480-condition run (pod, watchers, half-hourly sync).
- **M5** analysis, findings appended to docs/RESULTS.md.

## 10. Decisions (Ishan, 2026-09-11, before any exp-v3 code)

- **(a) Regimes: Empty-only.** Four `MiniGrid-Empty` regimes; BabyAI GoToLocal stays
  in FUTURE.md.
- **(b) Observation: absolute `(x, y, dir)` + rendered view cone.** World model and
  retrieval memory stay comparable with Phase 1; partial observability still bites.
- **(c) gather_evidence = one `left` turn.** One decision = one env step, always.
- **(d) Prices: the chosen substrate's real per-token serving price** (Design decision
  1 as written). Phase 1 dollar comparability is given up knowingly.
