"""Registered analysis of the frozen experiment set.

Implements docs/ANALYSIS.md (registered 2026-07-15, pre-data) exactly as fixed by
Amendment 10 (2026-09-10, pre-analysis). Nothing here is tunable from the CLI
beyond output paths and the bootstrap resample count/seed, which are declared.

Order of operations (ANALYSIS.md):
  1. assertion set — the analysis refuses to produce a verdict on any hard stop
  2. headline: bandit vs heuristic success-per-dollar, full data (§1)
  3. robustness: same computation on the equal-exposure subset (§2)
  4. mechanism: recovery time + mode-choice signature, per shift index (§3)
  5. exposure table + Pareto data for the full roster (§4, Amendment 6b)

Stdlib only, like the rest of the package. Plots live in scripts/analyze.py.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from venjix.batch import _run_is_complete, verify_pairing
from venjix.experiments import conditions, load_spec

# Registered constants (Amendment 10). Not CLI-tunable on purpose.
WIN_RELATIVE_MARGIN = 0.10          # Design decision 8
RECOVERY_ALPHA = 0.3                # the signal's registered alpha, no new constant
RECOVERY_FRACTION = 0.9             # Design decision 7
SIGNATURE_WINDOW = (-60, 180)       # global-step offsets around a shift
SIGNATURE_BIN = 10
PARSE_ERROR_HARD_STOP = 0.05        # assertion 3
RATIO_AGENTS = ("bandit", "heuristic")
MODE_CHOOSING_AGENTS = ("heuristic", "bandit", "mixture")
MODES = ("act", "retrieve", "simulate", "gather_evidence")

_STEP_KEYS = ("episode", "global_step", "mode", "parse_error", "pos", "cost_usd",
              "forced_answer")  # forced_answer: absent pre-exp-v2 -> None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

@dataclass
class Run:
    regime: str                 # schedule version, e.g. "exp-v1/hard-15-far"
    agent: str
    seed: int
    config_hash: str
    run_dir: Path
    n_episodes: int
    episodes: list = field(default_factory=list)   # episode records, in order
    shifts: list = field(default_factory=list)     # shift records, in order
    steps: list = field(default_factory=list)      # trimmed step records, in order

    @property
    def regime_name(self) -> str:
        return self.regime.split("/")[-1]

    @property
    def complete(self) -> bool:
        return len(self.episodes) >= self.n_episodes

    @property
    def successes(self) -> int:
        return sum(1 for e in self.episodes if e["success"])

    @property
    def cost_usd(self) -> float:
        return sum(e["cost_usd"] for e in self.episodes)

    @property
    def spd(self) -> float | None:
        """Success-per-dollar; None for zero-spend conditions (Amendment 6b)."""
        return None if self.cost_usd <= 0 else self.successes / self.cost_usd

    @property
    def final_step(self) -> int:
        return self.steps[-1]["global_step"] if self.steps else 0

    def episode_end_steps(self) -> dict[int, int]:
        """episode index -> global step of its last env step."""
        ends: dict[int, int] = {}
        for s in self.steps:
            ends[s["episode"]] = s["global_step"]
        return ends


def load_runs(out_root: str | Path) -> list[Run]:
    """Every run dir under out_root (complete or not; completeness is decided
    by the assertion set, which also lists the incomplete ones)."""
    runs = []
    for manifest_path in sorted(Path(out_root).glob("*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        config = manifest["config"]
        run = Run(
            regime=config["schedule"]["version"],
            agent=config["agent"],
            seed=config["seed"],
            config_hash=manifest["config_hash"],
            run_dir=manifest_path.parent,
            n_episodes=config["n_episodes"],
        )
        jsonl = manifest_path.parent / "episodes.jsonl"
        if jsonl.exists():
            with jsonl.open() as fh:
                for line in fh:
                    record = json.loads(line)
                    kind = record.get("type")
                    if kind == "step":
                        run.steps.append({k: record.get(k) for k in _STEP_KEYS})
                    elif kind == "shift":
                        run.shifts.append(record)
                    elif kind == "episode":
                        run.episodes.append(record)
        runs.append(run)
    return runs


# --------------------------------------------------------------------------- #
# Assertion set
# --------------------------------------------------------------------------- #

def run_assertions(runs: list[Run], out_root: str | Path, spec: dict) -> dict:
    """ANALYSIS.md assertion set. Returns a report with `hard_stop` and reasons.
    Complete runs only enter the analysis (assertion 5)."""
    reasons = []
    complete = [r for r in runs if r.complete]
    incomplete = [str(r.run_dir) for r in runs if not r.complete]

    # 5. completeness against the registered set
    registered = {c.config.config_hash(): c for c in conditions(spec)}
    have = defaultdict(list)
    for r in complete:
        have[r.config_hash].append(r)
    missing = sorted(
        f"{c.regime}/{c.agent}/seed{c.seed}"
        for h, c in registered.items() if h not in have
    )
    unregistered = sorted(str(r.run_dir) for r in complete if r.config_hash not in registered)
    duplicates = sorted(h for h, rs in have.items() if len(rs) > 1)
    if missing:
        reasons.append(f"{len(missing)} registered conditions missing/incomplete")
    if duplicates:
        reasons.append(f"{len(duplicates)} config hashes have multiple complete runs")

    # 1. pairing prefix consistency (harness truth from batch.verify_pairing)
    count_mismatches, prefix_divergences, ignored = verify_pairing(out_root)
    if prefix_divergences:
        reasons.append(f"{len(prefix_divergences)} pairing prefix divergences (harness bug)")

    # 2. shift-count table
    shift_counts: dict[str, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for r in complete:
        shift_counts[r.regime_name][r.seed][r.agent] = len(r.shifts)
    unequal_groups = []
    for regime, by_seed in shift_counts.items():
        for seed, by_agent in by_seed.items():
            if len(set(by_agent.values())) > 1:
                unequal_groups.append({"regime": regime, "seed": seed, "counts": by_agent})

    # 3. parse-error rate per agent
    parse = defaultdict(lambda: [0, 0])
    for r in complete:
        for s in r.steps:
            parse[r.agent][0] += bool(s["parse_error"])
            parse[r.agent][1] += 1
    parse_rate = {a: (n / t if t else 0.0) for a, (n, t) in sorted(parse.items())}
    for agent, rate in parse_rate.items():
        if rate > PARSE_ERROR_HARD_STOP:
            reasons.append(f"parse-error rate {rate:.3f} for {agent} exceeds {PARSE_ERROR_HARD_STOP}")

    # Amendment 11 (exp-v2): forced-answer rate among act-mode steps, per agent.
    # Reported, not a hard stop (the policy gate governs competence).
    forced = defaultdict(lambda: [0, 0])
    for r in complete:
        for s in r.steps:
            if s["mode"] == "act" and s.get("forced_answer") is not None:
                forced[r.agent][0] += s["forced_answer"]
                forced[r.agent][1] += 1
    forced_rate = {a: (n / t if t else 0.0) for a, (n, t) in sorted(forced.items())}

    # 4. goal-landed-on-agent-cell events
    landed = 0
    total_shifts = 0
    for r in complete:
        pos_at = {s["global_step"]: tuple(s["pos"]) for s in r.steps}
        for sh in r.shifts:
            total_shifts += 1
            if pos_at.get(sh["at_step"]) == tuple(sh["new_goal"]):
                landed += 1

    return {
        "hard_stop": bool(reasons),
        "reasons": reasons,
        "registered_conditions": len(registered),
        "complete_runs": len(complete),
        "incomplete_dirs": incomplete,
        "missing_conditions": missing,
        "unregistered_runs": unregistered,
        "duplicate_hashes": duplicates,
        "pairing": {
            "prefix_divergences": prefix_divergences,
            "count_mismatch_groups": len(count_mismatches),
            "ignored_incomplete": ignored,
        },
        "shift_counts": {
            regime: {str(seed): by_agent for seed, by_agent in sorted(by_seed.items())}
            for regime, by_seed in sorted(shift_counts.items())
        },
        "unequal_exposure_groups": unequal_groups,
        "parse_error_rate": parse_rate,
        "forced_answer_rate_act_steps": forced_rate,
        "goal_landed_on_agent_cell": {
            "events": landed, "shifts": total_shifts,
            "rate": (landed / total_shifts if total_shifts else 0.0),
        },
    }


# --------------------------------------------------------------------------- #
# Headline + robustness (Design decision 8, ANALYSIS.md §1–2, Amendment 10)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Pair:
    regime: str
    seed: int
    bandit_spd: float
    heuristic_spd: float
    bandit_shifts: int
    heuristic_shifts: int
    # raw counts for the pooled estimator (exp-v3 plan §5 / Amendment 13)
    bandit_successes: int = 0
    bandit_cost: float = 0.0
    heuristic_successes: int = 0
    heuristic_cost: float = 0.0

    @property
    def equal_exposure(self) -> bool:
        return self.bandit_shifts == self.heuristic_shifts


def build_pairs(runs: list[Run]) -> list[Pair]:
    by_key: dict[tuple[str, int], dict[str, Run]] = defaultdict(dict)
    for r in runs:
        if r.complete and r.agent in RATIO_AGENTS:
            by_key[(r.regime_name, r.seed)][r.agent] = r
    pairs = []
    for (regime, seed), d in sorted(by_key.items()):
        if set(d) != set(RATIO_AGENTS):
            continue
        b, h = d["bandit"], d["heuristic"]
        if b.spd is None or h.spd is None:
            continue  # zero-spend — cannot happen for these agents, guarded anyway
        pairs.append(Pair(regime, seed, b.spd, h.spd, len(b.shifts), len(h.shifts),
                          b.successes, b.cost_usd, h.successes, h.cost_usd))
    return pairs


def headline_pooled(pairs: list[Pair], resamples: int = 10_000, seed: int = 0) -> dict:
    """exp-v3 estimator (PLAN §5, registered as Amendment 13 at M3): POOLED
    success-per-dollar = total successes / total dollars over all pairs, per agent;
    relative difference of the two pooled ratios; stratified-by-regime paired
    bootstrap over (regime, seed) groups; WIN iff rel >= 0.10 AND the 95% CI of the
    relative difference excludes 0. Immune to the low-spend-ratio blow-up that
    inverted exp-v1 (docs/RESULTS.md finding 1)."""
    if not pairs:
        return {"n_pairs": 0, "verdict": None, "estimator": "pooled_ratio"}
    rng = random.Random(seed)
    strata = defaultdict(list)
    for p in pairs:
        strata[p.regime].append(p)
    strata_lists = list(strata.values())

    def pooled(sample):
        sb = sum(p.bandit_successes for p in sample) / max(sum(p.bandit_cost for p in sample), 1e-12)
        sh = sum(p.heuristic_successes for p in sample) / max(sum(p.heuristic_cost for p in sample), 1e-12)
        return sb, sh

    spd_b, spd_h = pooled(pairs)
    rel = _rel(spd_b, spd_h)
    boots_b, boots_h, boots_rel = [], [], []
    for _ in range(resamples):
        sample = []
        for group in strata_lists:
            n = len(group)
            sample.extend(group[rng.randrange(n)] for _ in range(n))
        mb, mh = pooled(sample)
        boots_b.append(mb)
        boots_h.append(mh)
        boots_rel.append(_rel(mb, mh))
    ci_rel = _ci(boots_rel)
    win = rel >= WIN_RELATIVE_MARGIN and ci_rel[0] > 0
    per_regime = {}
    for regime, group in sorted(strata.items()):
        gb, gh = pooled(group)
        per_regime[regime] = {"n": len(group), "bandit_pooled_spd": gb, "heuristic_pooled_spd": gh,
                              "relative_difference": _rel(gb, gh),
                              "bandit_wins_pairs": sum(p.bandit_spd > p.heuristic_spd for p in group)}
    return {
        "estimator": "pooled_ratio",
        "n_pairs": len(pairs), "resamples": resamples, "bootstrap_seed": seed,
        "bandit_pooled_spd": spd_b, "bandit_ci": _ci(boots_b),
        "heuristic_pooled_spd": spd_h, "heuristic_ci": _ci(boots_h),
        "relative_difference": rel, "relative_difference_ci": ci_rel,
        "ci_excludes_zero": ci_rel[0] > 0, "margin_met": rel >= WIN_RELATIVE_MARGIN,
        "verdict": "WIN" if win else "MATCHES",
        "per_regime": per_regime,
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = q * (len(sorted_values) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)


def _ci(values: list[float]) -> tuple[float, float]:
    v = sorted(values)
    return _percentile(v, 0.025), _percentile(v, 0.975)


def _rel(mean_b: float, mean_h: float) -> float:
    return (mean_b - mean_h) / mean_h if mean_h else float("inf")


def headline(pairs: list[Pair], resamples: int = 10_000, seed: int = 0) -> dict:
    """Registered estimator: relative difference of mean SPD over paired
    (regime, seed) groups; stratified-by-regime paired bootstrap; WIN iff
    rel >= 0.10 AND CI_low(bandit) > CI_high(heuristic)."""
    if not pairs:
        return {"n_pairs": 0, "verdict": None}
    rng = random.Random(seed)
    strata = defaultdict(list)
    for p in pairs:
        strata[p.regime].append(p)
    strata_lists = list(strata.values())

    mean_b = sum(p.bandit_spd for p in pairs) / len(pairs)
    mean_h = sum(p.heuristic_spd for p in pairs) / len(pairs)
    boots_b, boots_h, boots_rel = [], [], []
    for _ in range(resamples):
        sb = sh = 0.0
        for group in strata_lists:
            n = len(group)
            for _ in range(n):
                p = group[rng.randrange(n)]
                sb += p.bandit_spd
                sh += p.heuristic_spd
        mb, mh = sb / len(pairs), sh / len(pairs)
        boots_b.append(mb)
        boots_h.append(mh)
        boots_rel.append(_rel(mb, mh))
    ci_b, ci_h, ci_rel = _ci(boots_b), _ci(boots_h), _ci(boots_rel)
    rel = _rel(mean_b, mean_h)
    win = rel >= WIN_RELATIVE_MARGIN and ci_b[0] > ci_h[1]

    # Amendment 10 sensitivity line: pooled, unpaired, unstratified resampling.
    rng2 = random.Random(seed + 1)
    bs, hs = [p.bandit_spd for p in pairs], [p.heuristic_spd for p in pairs]
    n = len(pairs)
    sens_rel = []
    for _ in range(resamples):
        mb = sum(bs[rng2.randrange(n)] for _ in range(n)) / n
        mh = sum(hs[rng2.randrange(n)] for _ in range(n)) / n
        sens_rel.append(_rel(mb, mh))

    # per-regime table (reported, does not enter the verdict)
    per_regime = {}
    for regime, group in sorted(strata.items()):
        gb = sum(p.bandit_spd for p in group) / len(group)
        gh = sum(p.heuristic_spd for p in group) / len(group)
        per_regime[regime] = {
            "n": len(group), "bandit_mean_spd": gb, "heuristic_mean_spd": gh,
            "relative_difference": _rel(gb, gh),
            "bandit_wins_pairs": sum(p.bandit_spd > p.heuristic_spd for p in group),
        }

    return {
        "n_pairs": len(pairs),
        "resamples": resamples,
        "bootstrap_seed": seed,
        "bandit_mean_spd": mean_b,
        "bandit_ci": ci_b,
        "heuristic_mean_spd": mean_h,
        "heuristic_ci": ci_h,
        "relative_difference": rel,
        "relative_difference_ci": ci_rel,
        "cis_non_overlapping": ci_b[0] > ci_h[1],
        "margin_met": rel >= WIN_RELATIVE_MARGIN,
        "verdict": "WIN" if win else "MATCHES",
        "sensitivity_unpaired_relative_difference_ci": _ci(sens_rel),
        "per_regime": per_regime,
    }


# --------------------------------------------------------------------------- #
# Mechanism (ANALYSIS.md §3, Design decision 7, Amendment 10)
# --------------------------------------------------------------------------- #

def recovery_times(run: Run) -> list[dict]:
    """Per shift k of this run: steps from the shift until the episode-level SPD
    EWMA (alpha 0.3) returns to 90% of its pre-shift value; censored at the
    run's final step if it never does."""
    ends = run.episode_end_steps()
    ewma = None
    ewma_by_episode: list[tuple[int, int, float]] = []  # (episode, end_step, ewma)
    for e in run.episodes:
        spd = (1.0 if e["success"] else 0.0) / e["cost_usd"] if e["cost_usd"] > 0 else 0.0
        ewma = spd if ewma is None else RECOVERY_ALPHA * spd + (1 - RECOVERY_ALPHA) * ewma
        ewma_by_episode.append((e["episode"], ends.get(e["episode"], 0), ewma))

    out = []
    for k, sh in enumerate(run.shifts):
        at = sh["at_step"]
        before = [x for x in ewma_by_episode if x[1] < at]
        after = [x for x in ewma_by_episode if x[1] >= at]
        baseline = before[-1][2] if before else None
        entry = {"shift_index": k, "at_step": at, "baseline_ewma": baseline,
                 "recovered": False, "recovery_steps": None, "censored_at": None,
                 "zero_baseline": baseline == 0.0}
        if baseline is None:
            entry["no_pre_shift_episode"] = True
            out.append(entry)
            continue
        for _, end_step, value in after:
            if value >= RECOVERY_FRACTION * baseline:
                entry["recovered"] = True
                entry["recovery_steps"] = end_step - at
                break
        if not entry["recovered"]:
            entry["censored_at"] = run.final_step - at
        out.append(entry)
    return out


def recovery_summary(runs: list[Run]) -> dict:
    """Per agent: recovery stats among experiencers; plus the paired
    bandit-vs-heuristic comparison at each (group, k) both experienced."""
    per_agent = defaultdict(lambda: {"shifts": 0, "recovered": 0, "censored": 0,
                                     "no_pre_shift_episode": 0, "zero_baseline": 0,
                                     "recovery_steps": []})
    per_run: dict[tuple[str, int, str], list[dict]] = {}
    for r in runs:
        if not r.complete:
            continue
        rec = recovery_times(r)
        per_run[(r.regime_name, r.seed, r.agent)] = rec
        a = per_agent[r.agent]
        for x in rec:
            a["shifts"] += 1
            if x.get("no_pre_shift_episode"):
                a["no_pre_shift_episode"] += 1
                continue
            a["zero_baseline"] += bool(x["zero_baseline"])
            if x["recovered"]:
                a["recovered"] += 1
                a["recovery_steps"].append(x["recovery_steps"])
            else:
                a["censored"] += 1
    summary = {}
    for agent, a in sorted(per_agent.items()):
        steps = sorted(a["recovery_steps"])
        summary[agent] = {
            "shifts_experienced": a["shifts"],
            "recovered": a["recovered"],
            "censored": a["censored"],
            "no_pre_shift_episode": a["no_pre_shift_episode"],
            "zero_baseline_cases": a["zero_baseline"],
            "median_recovery_steps": _percentile(steps, 0.5) if steps else None,
            "mean_recovery_steps": (sum(steps) / len(steps)) if steps else None,
        }
    # paired at shift k, experiencers only (§3)
    paired = defaultdict(lambda: {"n": 0, "bandit_faster": 0, "heuristic_faster": 0,
                                  "both_recovered": 0, "mean_diff_steps": []})
    for (regime, seed, agent), rec in per_run.items():
        if agent != "bandit":
            continue
        h = per_run.get((regime, seed, "heuristic"))
        if not h:
            continue
        for k in range(min(len(rec), len(h))):
            b, hh = rec[k], h[k]
            key = f"k={k}"
            paired[key]["n"] += 1
            bt = b["recovery_steps"] if b["recovered"] else b["censored_at"]
            ht = hh["recovery_steps"] if hh["recovered"] else hh["censored_at"]
            if bt is None or ht is None:
                continue
            if b["recovered"] and hh["recovered"]:
                paired[key]["both_recovered"] += 1
                paired[key]["mean_diff_steps"].append(bt - ht)
            if bt < ht:
                paired[key]["bandit_faster"] += 1
            elif ht < bt:
                paired[key]["heuristic_faster"] += 1
    for key, d in paired.items():
        diffs = d.pop("mean_diff_steps")
        d["mean_diff_steps_bandit_minus_heuristic"] = (sum(diffs) / len(diffs)) if diffs else None
    return {"per_agent": summary, "paired_by_shift_index": dict(sorted(paired.items()))}


def mode_signature(runs: list[Run]) -> dict:
    """Mode-choice frequency by binned step offset from each shift, averaged
    over (group, k) among experiencers, for mode-choosing agents."""
    lo, hi = SIGNATURE_WINDOW
    counts = {a: defaultdict(lambda: defaultdict(int)) for a in MODE_CHOOSING_AGENTS}
    for r in runs:
        if not r.complete or r.agent not in counts:
            continue
        for sh in r.shifts:
            at = sh["at_step"]
            for s in r.steps:
                off = s["global_step"] - at
                if off < lo or off > hi:
                    continue
                b = (off // SIGNATURE_BIN) * SIGNATURE_BIN
                counts[r.agent][b][s["mode"]] += 1
    out = {}
    for agent, by_bin in counts.items():
        rows = []
        for b in sorted(by_bin):
            total = sum(by_bin[b].values())
            rows.append({"offset_bin": b, "n_steps": total,
                         **{m: by_bin[b].get(m, 0) / total for m in MODES}})
        out[agent] = rows
    return out


# --------------------------------------------------------------------------- #
# Exposure table + Pareto (ANALYSIS.md §4, Amendment 6b)
# --------------------------------------------------------------------------- #

def exposure_table(runs: list[Run]) -> dict:
    acc = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r.complete:
            acc[r.agent][r.regime_name].append(len(r.shifts))
    return {a: {reg: sum(v) / len(v) for reg, v in sorted(by.items())}
            for a, by in sorted(acc.items())}


def pareto(runs: list[Run], resamples: int = 10_000, seed: int = 0) -> dict:
    """Per agent: mean success rate and mean cost per episode over its 100
    conditions, with percentile bootstrap CIs (simple, unstratified)."""
    rng = random.Random(seed)
    by_agent = defaultdict(list)
    for r in runs:
        if r.complete:
            by_agent[r.agent].append((r.successes / len(r.episodes),
                                      r.cost_usd / len(r.episodes), r.spd))
    out = {}
    for agent, rows in sorted(by_agent.items()):
        n = len(rows)
        succ = [x[0] for x in rows]
        cost = [x[1] for x in rows]
        bs, bc = [], []
        for _ in range(resamples):
            idx = [rng.randrange(n) for _ in range(n)]
            bs.append(sum(succ[i] for i in idx) / n)
            bc.append(sum(cost[i] for i in idx) / n)
        spds = [x[2] for x in rows if x[2] is not None]
        out[agent] = {
            "n_conditions": n,
            "success_rate": sum(succ) / n, "success_rate_ci": _ci(bs),
            "cost_per_episode_usd": sum(cost) / n, "cost_per_episode_ci": _ci(bc),
            "mean_spd": (sum(spds) / len(spds)) if spds else None,  # None = zero-spend
        }
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def analyze(out_root: str | Path, spec_path: str | Path, resamples: int = 10_000,
            seed: int = 0) -> dict:
    spec = load_spec(spec_path)
    runs = load_runs(out_root)
    report = {"runs_root": str(out_root), "spec": str(spec_path)}
    report["assertions"] = run_assertions(runs, out_root, spec)
    complete = [r for r in runs if r.complete]
    report["exposure_table"] = exposure_table(complete)
    report["pareto"] = pareto(complete, resamples, seed)
    report["recovery"] = recovery_summary(complete)
    report["signature"] = mode_signature(complete)
    if report["assertions"]["hard_stop"]:
        report["verdict"] = None
        return report
    pairs = build_pairs(complete)
    # Estimator is part of the registration: exp-v1/v2 = Amendment 10 mean of
    # per-condition ratios; exp-v3 = pooled ratio (PLAN §5 / Amendment 13).
    metric = spec.get("metric", "mean_of_ratios")
    estimator = headline_pooled if metric == "pooled_ratio" else headline
    report["metric"] = metric
    report["headline"] = estimator(pairs, resamples, seed)
    equal = [p for p in pairs if p.equal_exposure]
    report["robustness_equal_exposure"] = estimator(equal, resamples, seed)
    report["verdict"] = report["headline"]["verdict"]
    return report


def format_summary(report: dict) -> str:
    """Human-readable summary; ends with the registered verdict lines."""
    a = report["assertions"]
    lines = ["== Assertion set =="]
    lines.append(f"complete runs: {a['complete_runs']} / registered {a['registered_conditions']}"
                 f"  (incomplete dirs: {len(a['incomplete_dirs'])}, missing: {len(a['missing_conditions'])})")
    lines.append(f"pairing: prefix divergences {len(a['pairing']['prefix_divergences'])}, "
                 f"unequal-exposure groups {len(a['unequal_exposure_groups'])}")
    lines.append("parse-error rate: " + ", ".join(f"{k} {v:.4f}" for k, v in a["parse_error_rate"].items()))
    if a.get("forced_answer_rate_act_steps"):
        lines.append("forced-answer rate (act steps): " + ", ".join(
            f"{k} {v:.3f}" for k, v in a["forced_answer_rate_act_steps"].items()))
    g = a["goal_landed_on_agent_cell"]
    lines.append(f"goal landed on agent cell: {g['events']}/{g['shifts']} shifts ({g['rate']:.2%})")
    if a["hard_stop"]:
        lines.append("HARD STOP: " + "; ".join(a["reasons"]))
        lines.append("no verdict produced")
        return "\n".join(lines)

    lines.append("\n== Exposure (mean shifts experienced, agent x regime) ==")
    for agent, by in report["exposure_table"].items():
        lines.append(f"{agent:<10} " + "  ".join(f"{reg}={v:.2f}" for reg, v in by.items()))

    lines.append("\n== Pareto (full roster) ==")
    for agent, p in report["pareto"].items():
        spd = "zero-spend" if p["mean_spd"] is None else f"{p['mean_spd']:.1f}"
        lines.append(f"{agent:<10} success {p['success_rate']:.3f} "
                     f"[{p['success_rate_ci'][0]:.3f}, {p['success_rate_ci'][1]:.3f}]  "
                     f"cost/ep ${p['cost_per_episode_usd']:.4f}  mean SPD {spd}")

    lines.append("\n== Recovery (Design decision 7) ==")
    for agent, r in report["recovery"]["per_agent"].items():
        med = "n/a" if r["median_recovery_steps"] is None else f"{r['median_recovery_steps']:.0f}"
        lines.append(f"{agent:<10} shifts {r['shifts_experienced']}, recovered {r['recovered']}, "
                     f"censored {r['censored']}, median recovery {med} steps, "
                     f"zero-baseline {r['zero_baseline_cases']}")
    for k, d in report["recovery"]["paired_by_shift_index"].items():
        md = d["mean_diff_steps_bandit_minus_heuristic"]
        lines.append(f"  paired {k}: n={d['n']} bandit faster {d['bandit_faster']}, "
                     f"heuristic faster {d['heuristic_faster']}, "
                     f"mean diff (b-h) {'n/a' if md is None else f'{md:+.1f}'} steps")

    h = report["headline"]
    if not h.get("n_pairs"):
        lines.append("\n== Headline ==")
        lines.append("no bandit/heuristic pairs with a defined success-per-dollar "
                     "(zero-spend conditions?) — no verdict")
        lines.append("")
        lines.append("VERDICT: NONE")
        return "\n".join(lines)
    if h.get("estimator") == "pooled_ratio":
        lines.append("\n== Headline (full data, pooled success-per-dollar — exp-v3 §5) ==")
        lines.append(f"pairs: {h['n_pairs']}  resamples: {h['resamples']}  seed: {h['bootstrap_seed']}")
        lines.append(f"bandit    pooled SPD {h['bandit_pooled_spd']:.2f}  95% CI [{h['bandit_ci'][0]:.2f}, {h['bandit_ci'][1]:.2f}]")
        lines.append(f"heuristic pooled SPD {h['heuristic_pooled_spd']:.2f}  95% CI [{h['heuristic_ci'][0]:.2f}, {h['heuristic_ci'][1]:.2f}]")
        lines.append(f"relative difference {h['relative_difference']:+.3f}  95% CI "
                     f"[{h['relative_difference_ci'][0]:+.3f}, {h['relative_difference_ci'][1]:+.3f}]  "
                     f"(margin >= {WIN_RELATIVE_MARGIN:.2f}: {h['margin_met']}; CI excludes 0: {h['ci_excludes_zero']})")
        for regime, pr in h["per_regime"].items():
            lines.append(f"  {regime:<20} bandit {pr['bandit_pooled_spd']:.2f} vs heuristic "
                         f"{pr['heuristic_pooled_spd']:.2f}  rel {pr['relative_difference']:+.3f}  "
                         f"bandit wins {pr['bandit_wins_pairs']}/{pr['n']} pairs")
        lines.append("")
        lines.append(f"VERDICT: {h['verdict']}")
        r = report["robustness_equal_exposure"]
        if r.get("verdict") is None:
            lines.append("robustness (equal exposure): no equal-exposure pairs")
        else:
            lines.append(f"robustness (equal exposure): {r['verdict']} on {r['n_pairs']} pairs — "
                         f"rel {r['relative_difference']:+.3f} CI [{r['relative_difference_ci'][0]:+.3f}, "
                         f"{r['relative_difference_ci'][1]:+.3f}]")
        return "\n".join(lines)
    lines.append("\n== Headline (full data, Design decision 8 / Amendment 10) ==")
    lines.append(f"pairs: {h['n_pairs']}  resamples: {h['resamples']}  seed: {h['bootstrap_seed']}")
    lines.append(f"bandit    mean SPD {h['bandit_mean_spd']:.2f}  95% CI [{h['bandit_ci'][0]:.2f}, {h['bandit_ci'][1]:.2f}]")
    lines.append(f"heuristic mean SPD {h['heuristic_mean_spd']:.2f}  95% CI [{h['heuristic_ci'][0]:.2f}, {h['heuristic_ci'][1]:.2f}]")
    lines.append(f"relative difference {h['relative_difference']:+.3f}  95% CI "
                 f"[{h['relative_difference_ci'][0]:+.3f}, {h['relative_difference_ci'][1]:+.3f}]  "
                 f"(margin >= {WIN_RELATIVE_MARGIN:.2f}: {h['margin_met']}; CIs non-overlapping: {h['cis_non_overlapping']})")
    lines.append(f"sensitivity (unpaired pooled bootstrap) rel-diff CI "
                 f"[{h['sensitivity_unpaired_relative_difference_ci'][0]:+.3f}, "
                 f"{h['sensitivity_unpaired_relative_difference_ci'][1]:+.3f}]")
    for regime, pr in h["per_regime"].items():
        lines.append(f"  {regime:<16} bandit {pr['bandit_mean_spd']:.2f} vs heuristic "
                     f"{pr['heuristic_mean_spd']:.2f}  rel {pr['relative_difference']:+.3f}  "
                     f"bandit wins {pr['bandit_wins_pairs']}/{pr['n']} pairs")

    lines.append("")
    lines.append(f"VERDICT: {h['verdict']}")
    r = report["robustness_equal_exposure"]
    if r.get("verdict") is None:
        lines.append("robustness (equal exposure): no equal-exposure pairs")
    else:
        lines.append(f"robustness (equal exposure): {r['verdict']} on {r['n_pairs']} pairs — "
                     f"rel {r['relative_difference']:+.3f} CI [{r['relative_difference_ci'][0]:+.3f}, "
                     f"{r['relative_difference_ci'][1]:+.3f}]; bandit CI [{r['bandit_ci'][0]:.2f}, "
                     f"{r['bandit_ci'][1]:.2f}] vs heuristic CI [{r['heuristic_ci'][0]:.2f}, {r['heuristic_ci'][1]:.2f}]")
    return "\n".join(lines)
