# Registered Analysis Plan (pre-data — 2026-07-15)

Registered BEFORE any real-model experiment-set run exists. The mock rehearsal
(structural dress rehearsal, no model behavior claims) is the only data consulted, and
only for the exposure-delta expectation in §2. This plan may not be revised after the
first real run; deviations would be reported as such in the writeup.

## The shift-exposure question

Shifts fire on global env steps, so the number of shifts an agent experiences is
endogenous to its own efficiency. The mock rehearsal showed unequal shift counts in
42/100 (regime, seed) groups — the dominant regime of the data, not an edge case. The
four rules below govern how analysis handles this.

## 1. Headline criterion: full data, no exclusions

The registered ≥10% bandit-vs-heuristic success-per-dollar comparison (Design
decision 8: 20 seeds, bootstrap 95% CIs, non-overlapping required) is computed over
complete runs exactly as frozen. Unequal shift exposure is part of the phenomenon —
a faster agent genuinely lives through fewer upheavals; that is the world being
simulated, not a measurement artifact. No agent can game exposure in reverse: burning
steps to dodge a late shift costs episodes and budget directly. Excluding or
truncating data to equalize exposure would discard real outcomes to manufacture a
symmetry the environment does not have.

## 2. Pre-declared robustness check (the insurance)

**Expectation, written before real data exists** (from the mock rehearsal, structural
only): the bandit-minus-heuristic shift-count delta across the 100 groups was
**mean +0.26, sd 0.73, exactly 0 in 79/100 groups** (bandit slightly over-exposed in
19 groups — UCB exploration makes it marginally slower). The endogeneity argument in
§1 rests on this near-parity.

**Registered secondary check:** the headline comparison is recomputed restricted to
groups where bandit and heuristic experienced **equal shift counts**. Declared now,
before real data, so it is a robustness check and not a post-hoc escape hatch; it
protects the result in both directions, whichever way the verdict prints. Both the
full-data and equal-exposure numbers are reported; the full-data number is the
registered verdict.

## 3. Mechanism analyses condition on shift index

Goal sequences are paired (Amendment 6c), so shift *k* in a (regime, seed) group is
the identical event for every agent that reaches it. Recovery time (Design decision
7), the mode-choice-over-time signature plot, and any per-shift comparison are
computed per (group, shift k), including only agents that experienced shift k, with
comparisons at k made only among experiencers. Mechanism evidence stays paired where
pairing actually holds instead of averaging over mismatched exposure sets.

## 4. Exposure is reported, not adjusted

The writeup includes a shifts-experienced table (agent × regime) with one
interpretation sentence: exposure differences across the full roster (retrieve-only's
efficiency vs. reactive's wandering as the extremes) are a finding about the modes —
efficient strategies buy themselves calmer lives — not noise to be corrected.

## Assertion set (analysis refuses to produce a verdict if any fails)

1. **Pairing prefix consistency** per (regime, seed): every agent's
   (at_step, old_goal, new_goal) shift sequence is a prefix of the group's longest
   (verify_pairing; divergence = harness bug, hard stop).
2. **Shift-count table** per (regime, seed, agent) computed and attached; unequal
   groups counted and enumerated (expected, per §1 — reported, not fatal).
3. **Parse-error rate per agent** reported; hard stop if any agent exceeds 5%
   (calibration floor was 3.2% on world-model calls, all neutralized out_of_range;
   materially higher rates mean the substrate drifted from its calibrated state).
4. **Goal-landed-on-agent-cell events** counted (mock rehearsal: 16/1847 = 0.87%);
   reported alongside the exposure table.
5. **Completeness**: only complete runs (full episode count) enter analysis;
   incomplete dirs are listed. All 600 registered conditions must be present.

## Verdict output

The analysis terminates by printing exactly one of:

- `VERDICT: WIN` — bandit beats heuristic by ≥10% relative on success-per-dollar
  with non-overlapping bootstrap 95% CIs (full data, §1). Method framing.
- `VERDICT: MATCHES` — anything less. Study framing ("arbitration helps but needs
  almost no machinery"). Published identically.

followed by the same computation on the §2 equal-exposure subset, labeled
`robustness (equal exposure): ...`, and the Pareto plot data (Amendment 6b) for the
full roster. Negative results publish identically to positive ones.
