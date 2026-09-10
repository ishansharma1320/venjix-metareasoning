# Policy-reply diagnostic (2026-09-10, post-analysis, evidence only)

**Why this exists.** The registered analysis showed the reactive agent at 0/4000
episode successes and the heuristic choosing `gather_evidence` on ~0 of 148k steps.
Both trace to the same fact: on this substrate the `act` mode's policy call returns
`probe` on essentially every step (reactive: 54,080/54,080 actions; bandit act-mode:
4,388/4,392; mixture act-mode: 8,040/8,045). The parser takes the *first* action word
in the reply and the prompt itself contains the phrase "probe result", so a parser
artifact had to be ruled out.

**Method.** Four representative observations were sent through the exact
`ReactiveAgent` prompt to the same model id (`Qwen/Qwen3-8B`, via OpenRouter, thinking
disabled, temperature 0) and the raw reply captured before parsing. Four calls, not an
experiment run; no result was computed from them.

| observation | raw reply | parsed |
|---|---|---|
| start (0,0), steps 0, no probe yet | `probe` | probe |
| (0,0), steps 1, last probe result None (goal out of range) | `probe` | probe |
| (0,0), steps 1, last probe result (2, 1) — goal offset revealed | `probe` | probe |
| (3,3), steps 10, no info | `probe` | probe |

**Conclusion.** The reply is literally the single word `probe`; the parser reproduces
it faithfully. This is the model's genuine (degenerate) policy, not a harness bug. It
is a substrate limitation of the frozen agent design on an 8B model and is disclosed
as such; the prompt is part of the registered agent and was not changed post hoc.

**Follow-up (same day): is it the model or the registered config?** The same four
prompts re-sent with Qwen3's thinking mode ON (the registered config has it OFF,
Amendments 6a/8/9, for cost and output format):

| observation | thinking OFF (registered) | thinking ON | output tokens (ON) |
|---|---|---|---|
| start, no probe yet | `probe` | `probe` | 1,993 |
| after probe, goal out of range | `probe` | `right` | 4,497 |
| after probe, goal offset (2, 1) | `probe` | `down` (correct direction) | 2,552 |
| mid-grid, no info | `probe` | `probe` | 1,093 |

So the model *can* act on the observation; the no-thinking config is what reduces its
policy to "probe". The trade-off that made thinking-off the registered choice is real:
~1,000–4,500 output tokens per policy call versus 1, i.e. ~200× the decode cost —
on exp-v1's call volume, infeasible on this budget. This is a substrate-config
limitation of exp-v1, disclosed; it was not changed post hoc. The natural follow-up
(pre-registered as its own experiment, never a re-run of exp-v1) is a substrate whose
non-thinking policy moves — a larger model, or a policy gate declared before the run.

**Consequences for the roster (all measured, none adjusted):**
- `act` is a dead mode: it never moves. Reactive-only success = 0.000.
- The heuristic is act-dominant by construction (`gather if ewma > 0.25 else retrieve
  if belief else act`). It acquires beliefs only from probes, so its world-model
  predictions are almost never surprised (2 mispredictions in 59,814 sampled steps)
  and its signal never crosses the threshold. It succeeds only when the start cell is
  within probe radius of the goal (success 0.204; 0.000 in hard-11-blunt-probe).
- The bandit learns away from `act` (23% of steps, mostly early) toward retrieve /
  gather / simulate: success 0.684 at the same cost per episode as the heuristic.
- Retrieve-only (seeded random walk until the goal is found, then memory) reaches
  0.627 success at $0 — nearly the bandit's success rate with zero LLM spend.
