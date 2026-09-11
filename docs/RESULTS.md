# Results log

Running record of what the experiments have established. Each entry cites the
registered plan it answers and the committed artifacts. Publication happens when this
log holds something worth publishing (Amendment 12); until then this file *is* the
deliverable. Nothing here is revised after the fact — corrections are appended.

## exp-v1 — arbitration over four modes on Qwen3-8B (no thinking), 2026-09-10

Registered: docs/CLAUDE.md Amendments 3, 5, 6, 8, 9, 10; docs/ANALYSIS.md.
Data: 600/600 conditions, 0 failures, pairing intact (0 prefix divergences, 35
unequal-exposure groups). Artifacts: `results/exp-v1-analysis/`.

**Registered verdict: MATCHES.** Bandit mean success-per-dollar 1061 [875, 1283] vs
threshold heuristic 1792 [986, 2813]; relative difference −0.41 [−0.57, −0.05].
Equal-exposure robustness: MATCHES.

### Findings

1. **The pre-registered estimator inverted what the raw data shows.** Mean of
   per-condition success/cost ratios is dominated by a few heuristic runs that succeed
   at near-zero spend (ratios up to 42,966 per dollar). On the registered Pareto view
   the bandit reaches 0.684 success [0.620, 0.745] vs the heuristic's 0.204
   [0.133, 0.279] at the same cost per episode. Post-hoc pooled ratio-of-means
   (labeled, never the verdict): bandit 493/$ vs heuristic 136/$. **Lesson:** register
   a pooled ratio or paired log-ratio next time, never a mean of ratios.
2. **The act mode was dead on this substrate.** Qwen3-8B with thinking disabled
   answers the policy prompt with the single word `probe` on every step, even when the
   probe result already reveals the goal's offset (verbatim evidence:
   `results/exp-v1-analysis/policy-reply-diagnostic.md`). Reactive-only success: 0.000
   over 4,000 episodes. Not a parser or harness defect; the frozen prompt was not
   changed. The same model with thinking ON moves sensibly at 1,000–4,500 output
   tokens per call.
3. **The killer baseline could not engage.** The heuristic is act-dominant by
   construction and acquires beliefs only from probes, so its world model was almost
   never surprised (2 mispredictions in 59,814 sampled steps) and its signal never
   fired (gather_evidence share ≈ 0). It succeeded only when the start cell was within
   probe radius of the goal. The MATCHES verdict is therefore "bandit did not beat a
   heuristic that could not function", not evidence about arbitration vs heuristics
   in general.
4. **Arbitration beat every fixed strategy that spends money.** Bandit 0.684 success
   vs mixture 0.600 (higher cost), simulate 0.427 (5× cost), reactive 0.000.
   The bandit learned away from the dead act mode (23% of steps, mostly early).
5. **The cheapest strategy needed no arbitration.** Retrieve-only (seeded random walk
   until the goal is found, then episodic memory; zero LLM calls) reached 0.627
   success — inside the bandit's CI at $0. Partly a fact about this environment
   (small grids, modest relocations); the "how little machinery" framing the
   registration committed to is the correct one.
6. **The shift signature was mild.** Bandit gather_evidence share 0.21 → 0.25 in the
   30 steps after a shift, retrieve 0.43 → 0.34; heuristic flat. Not the predicted
   sharp spike (`results/exp-v1-analysis/signature.png`).
7. **Cost.** 2.5M calls in 3.3 h on one A100 (~$5.50 GPU); ~$9 including setup.

## exp-v2 — same substrate, bounded thinking on policy calls, 2026-09-10

Registered: Amendment 11. Budget 512, sole fallback 1024. Policy gate: reactive-only,
easy-7, 20 seeds × 10 episodes; GREEN iff move fraction ≥ 0.50 and success ≥ 0.15.
Artifacts: `results/exp-v2-policy-gate/`. **Zero experiment conditions run.**

| budget | move fraction | success | forced-answer rate | tokens / policy call | verdict |
|---|---|---|---|---|---|
| 512 | 0.012 | 0.075 | 1.000 | 532 | RED |
| 1024 | 0.082 | 0.365 | 0.963 | 1047 | RED |

### Findings

8. **512 tokens of thinking is not enough to un-kill the policy.** Every call hits the
   budget; a model cut off mid-thought answers `probe`.
9. **1024 tokens produces a policy that reads evidence but barely explores.** When the
   probe reveals the goal's offset it walks there in ~5 steps (first ~20 episodes per
   condition: 20/20 successes). Once a shift moves the goal out of probe range it
   mostly probes again. Success 0.365 (well above the 0.21 random floor) with 8% of
   steps moving.
10. **The gate criterion measured the wrong thing.** "Move fraction ≥ 0.50" was written
    to catch a never-moving policy; it also rejects a probe-heavy policy that succeeds
    a third of the time. Recorded, not revised (Amendment 11 gate-design note). **Lesson:**
    gate on success against a random floor, and gate the policy *before* registering
    anything around a substrate.
11. **Cost.** Two gates ≈ $3.

## Carry-overs for the next experiment (exp-v3, MiniGrid/BabyAI — see FUTURE.md)

- Policy gate first, on candidate substrates, before registration.
- Register success-per-dollar as a pooled ratio or paired log-ratio.
- Pick a substrate whose non-thinking policy moves, or budget for bounded thinking
  at a level the gate has already shown to work.
- Keep: paired shift RNG, resume-by-config-hash, world-model calibration gate,
  assertion set before verdict, amendments with git timestamps.
