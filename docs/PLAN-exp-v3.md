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

### 4b. M2 gate record and a flagged gate defect (2026-09-11 19:43 UTC, written
BEFORE the 30B-A3B and 32B gate results existed)

`Qwen/Qwen3-8B` (no thinking): success **0.270**, lower-95 bound 0.208, 4,823 steps —
**GREEN by the §4a rule** (≥ 0.255). But **4,823/4,823 actions are `forward`**: the
policy never turns; it succeeds only when the goal lies on its start row ahead of it
(episodes of exactly 4 steps) and never after a shift moves it. That is a degenerate
policy the gate failed to catch, for the mirror-image reason exp-v2's gate failed:
"not significantly worse than random" cannot separate a lucky constant policy from a
live one when the constant policy's success is near random's.

**Proposed correction (Ishan decides; not applied to any result in this section):**
add a degeneracy check in the spirit of the gate's stated purpose ("detect a dead
policy"): **RED if any single action accounts for > 90% of gate steps**, regardless of
success. Bias direction: it can only *exclude* substrates, never admit one, and it is
declared before the remaining candidates' results are known. If adopted, the 8B
verdict becomes RED and the smallest GREEN candidate among the rest is chosen; if
not adopted, the §4 rule stands and the 8B is the substrate as written.

### 4c. M2 round 1 complete; prompt defect found and fixed (2026-09-11 ~20:10 UTC)

All three candidates on the round-1 prompt: **identical** 4,823 steps, success 0.270,
`forward` on 100% / 100% / 97% of steps (8B / 30B-A3B / 32B-FP8). Three models of very
different size failing identically is a prompt defect. Diagnostic on the live 30B
(`scripts/pod/prompt_diagnostic_v3.py`, logs in `results/exp-v3-gate/`): with a wall
directly ahead and the goal 3 cells to its right-hand side, the model reads the
world-frame offset (dx, dy) = (0, 3) as "directly in front, 3 steps away", calls the
cell ahead "floor" (it cannot read the ASCII view), and answers `forward` — with or
without thinking, and even when told in words that the cell ahead is wall while the
offset stays world-frame. The same facts stated **egocentrically** ("directly ahead
of you: a wall … the goal is 3 cells to your right") get `right` every time, thinking
off, on every phrasing tried.

**Decision (b) is revised before registration:** the policy LLM sees an egocentric
summary — what is directly ahead (floor / wall / the goal, and free cells until a
wall), what is on the left- and right-hand cells, and the goal as "N cells ahead and M
to your left/right" — no coordinates, no ASCII view. Absolute (x, y, facing) stays in
the code for memory, the planner and the world model, unchanged. This is not tuning
to a result: it is the same information the round-1 prompt carried, re-encoded so the
model can act on it; it applies identically to every agent's act mode; and every
candidate is **re-gated with the final prompt** (round 2) before any choice is made.

### 4d. M2 round 2 — final prompt, all candidates re-gated (2026-09-11 20:08–20:14 UTC)

| candidate | success (bar ≥ 0.255) | steps | actions | max action share | verdict |
|---|---|---|---|---|---|
| `Qwen/Qwen3-8B` (bf16, no thinking) | **1.000** | 1,495 | forward 85%, right 15% | 0.855 | GREEN |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` (bf16) | 0.560 | 3,624 | left 41%, forward 31%, right 28% | 0.406 | GREEN |
| `Qwen/Qwen3-32B-FP8` | 0.820 | 3,273 | forward 82%, left 18% | 0.815 | GREEN |

Reports: `results/exp-v3-gate/round2/`. **Substrate by the §4 rule (smallest GREEN):
`Qwen/Qwen3-8B`, bf16, thinking off, on the A100 pod** — the exp-v1 model and
serving stack, now with a live act policy. The §4b degeneracy proposal is moot for
the choice (the 8B's max action share is 0.855 < 0.90) but is carried into the
registration as a standing gate criterion. Prompt tokens per policy call ≈ 170
(vs 240 for the ASCII-view prompt). M2 cost ≈ $2.

### 4e. World-model gate on the chosen substrate: RED on the world-frame prompt;
egocentric world model (2026-09-11 ~20:20 UTC, before registration)

`runs/calibration/20260911T201627-Qwen-Qwen3-8B` (500 MiniGrid cases, world-frame
prompt "POSITION / FACING / ACTION → NEXT / FACING / REWARD"): misprediction rate
**0.644 [0.602, 0.686] — RED**. Failure classes: forward moved along the wrong
axis (183/500; e.g. facing down → y−1), "turn left" answered as "now facing left"
(54), right turns wrong in facing and/or position (74). The same world-frame
rotation reasoning that broke the policy prompt (§4c) breaks the world model.

**Design change, declared before registration:** turns are deterministic and known
to the agent, so predicting them adds noise to the signal, never information. The
MiniGrid world model now takes **egocentric facts computed by the code from the
absolute state** — what is directly ahead (floor / a wall, from the room geometry)
and where the believed goal is in the agent's frame — and the LLM predicts only
`MOVED: yes|no` and `REWARD: 0|1`; the code applies the turn and the move. The
signal's meaning is preserved exactly: a misprediction is a wrong move outcome or a
wrong reward expectation — the latter being the stale-belief surprise the study
measures. Calibration cases are unchanged (same generator, same strata); the mock
rehearsal is re-run; **the world-model gate is re-run on the 8B with this prompt
and must be GREEN before M3.** If it is not, the substrate escalates per §4 (same
prompt, next candidate), on record.

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
