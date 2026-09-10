"""Analysis tests on synthetic run dirs that follow the real log schema.

The registered estimator and verdict rule (Amendment 10) are exercised with
constructed data whose answer is known; the real data is never touched here.
"""

import json
from pathlib import Path

from venjix.analysis import (
    Pair,
    Run,
    build_pairs,
    headline,
    load_runs,
    mode_signature,
    recovery_times,
    run_assertions,
)

SPEC = {
    "version": "t",
    "n_seeds": 2,
    "n_episodes": 3,
    "model": "m",
    "prices": {"input_per_mtok_usd": 1.0, "output_per_mtok_usd": 1.0},
    "agents": ["heuristic", "bandit"],
    "agent_params": {"sim_depth": 3, "ewma_alpha": 0.3, "pe_threshold": 0.25,
                     "mixture_weights": [0.25, 0.25, 0.25, 0.25],
                     "ucb_alpha": 1.0, "cost_weight": 100.0},
    "regimes": [{"name": "r", "env": {"size": 7, "probe_radius": 3}, "shifts": [[4, 3]]}],
}


def write_run(root: Path, regime: str, agent: str, seed: int, *, successes, cost_each,
              n_episodes=3, steps_per_episode=3, shift_at=4, parse_errors=0,
              config_hash=None, modes=("act",), land_goal_on_agent=False):
    """A run dir with the real record shapes: steps, one shift, episode records."""
    run_dir = root / f"{agent}-{seed}"
    run_dir.mkdir(parents=True)
    manifest = {
        "run_id": run_dir.name, "created_at": "2026-09-10T00:00:00+00:00",
        "config_hash": config_hash or f"{regime}-{agent}-{seed}",
        "package_version": "0.1.0",
        "config": {"agent": agent, "seed": seed, "n_episodes": n_episodes,
                   "schedule": {"version": f"t/{regime}", "entries": [{"at_step": shift_at, "distance": 3}]}},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    lines = []
    g = 0
    pe_left = parse_errors
    for ep in range(n_episodes):
        for s in range(steps_per_episode):
            g += 1
            lines.append({"type": "step", "episode": ep, "step_in_episode": s + 1, "global_step": g,
                          "mode": modes[g % len(modes)], "action": "up",
                          "parse_error": pe_left > 0, "pos": [1, 1], "reward": 0, "done": False,
                          "success": False, "probe_result": None, "llm_calls": 1,
                          "input_tokens": 10, "output_tokens": 1, "cost_usd": cost_each / steps_per_episode,
                          "wall_time_ms": 1.0, "prediction_error": None, "signal_ewma": None})
            pe_left -= 1
            if g == shift_at:
                lines.append({"type": "shift", "global_step": g, "at_step": g,
                              "old_goal": [5, 5], "new_goal": [1, 1] if land_goal_on_agent else [6, 6],
                              "requested_distance": 3, "actual_distance": 3})
        lines.append({"type": "episode", "episode": ep, "success": ep < successes,
                      "steps_used": steps_per_episode, "llm_calls": steps_per_episode,
                      "input_tokens": 30, "output_tokens": 3, "cost_usd": cost_each, "wall_time_ms": 3.0})
    (run_dir / "episodes.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    return run_dir


def test_load_runs_reads_schema_and_spd(tmp_path):
    write_run(tmp_path, "r", "bandit", 0, successes=2, cost_each=0.5)
    runs = load_runs(tmp_path)
    assert len(runs) == 1
    r = runs[0]
    assert r.regime_name == "r" and r.agent == "bandit" and r.seed == 0
    assert r.complete and r.successes == 2 and abs(r.cost_usd - 1.5) < 1e-12
    assert abs(r.spd - 2 / 1.5) < 1e-12
    assert len(r.shifts) == 1 and len(r.steps) == 9


def test_headline_win_and_matches_rules():
    # bandit clearly better in every pair -> WIN
    pairs = [Pair("r", s, 10.0 + (s % 3), 5.0 + (s % 3), 1, 1) for s in range(20)]
    h = headline(pairs, resamples=500, seed=0)
    assert h["verdict"] == "WIN" and h["margin_met"] and h["cis_non_overlapping"]
    # identical -> MATCHES (rel diff 0)
    pairs = [Pair("r", s, 5.0, 5.0, 1, 1) for s in range(20)]
    assert headline(pairs, resamples=200, seed=0)["verdict"] == "MATCHES"
    # +5% only -> MATCHES even if CIs separate
    pairs = [Pair("r", s, 5.25, 5.0, 1, 1) for s in range(20)]
    h = headline(pairs, resamples=200, seed=0)
    assert h["verdict"] == "MATCHES" and not h["margin_met"]


def test_headline_is_deterministic_for_a_seed():
    pairs = [Pair("r", s, 6.0 + s * 0.1, 5.0 + (s % 2), 1, 1) for s in range(20)]
    a = headline(pairs, resamples=300, seed=0)
    b = headline(pairs, resamples=300, seed=0)
    assert a["bandit_ci"] == b["bandit_ci"] and a["relative_difference_ci"] == b["relative_difference_ci"]


def test_equal_exposure_subset_selection(tmp_path):
    write_run(tmp_path, "r", "bandit", 0, successes=1, cost_each=1.0)
    write_run(tmp_path, "r", "heuristic", 0, successes=1, cost_each=1.0)
    runs = load_runs(tmp_path)
    pairs = build_pairs(runs)
    assert len(pairs) == 1 and pairs[0].equal_exposure
    pairs[0] = Pair("r", 0, 1.0, 1.0, 2, 1)
    assert not pairs[0].equal_exposure


def test_assertions_hard_stop_on_parse_errors_and_missing(tmp_path):
    spec = dict(SPEC)
    # only one of the four registered conditions present, and it has 40% parse errors
    write_run(tmp_path, "r", "bandit", 0, successes=1, cost_each=1.0, parse_errors=4)
    runs = load_runs(tmp_path)
    a = run_assertions(runs, tmp_path, spec)
    assert a["hard_stop"]
    assert any("missing" in r for r in a["reasons"])
    assert any("parse-error" in r for r in a["reasons"])
    assert a["parse_error_rate"]["bandit"] > 0.05


def test_goal_landed_on_agent_cell_is_counted(tmp_path):
    write_run(tmp_path, "r", "bandit", 0, successes=1, cost_each=1.0, land_goal_on_agent=True)
    write_run(tmp_path, "r", "heuristic", 0, successes=1, cost_each=1.0)
    a = run_assertions(load_runs(tmp_path), tmp_path, SPEC)
    assert a["goal_landed_on_agent_cell"] == {"events": 1, "shifts": 2, "rate": 0.5}


def test_recovery_time_basic_and_censored(tmp_path):
    # episodes end at global steps 3, 6, 9; shift at 4 -> baseline is episode 0's EWMA.
    # successes=3: every episode succeeds with equal cost -> EWMA flat -> recovers at
    # the first post-shift episode end (step 6): recovery = 2 steps.
    r = load_runs(write_run(tmp_path / "a", "r", "bandit", 0, successes=3, cost_each=1.0).parent)[0]
    rec = recovery_times(r)
    assert rec[0]["recovered"] and rec[0]["recovery_steps"] == 2
    # successes=1: only episode 0 succeeds -> post-shift EWMA decays below 90% of
    # baseline forever -> censored at the run's final step (9 - 4 = 5).
    r = load_runs(write_run(tmp_path / "b", "r", "bandit", 0, successes=1, cost_each=1.0).parent)[0]
    rec = recovery_times(r)
    assert not rec[0]["recovered"] and rec[0]["censored_at"] == 5


def test_mode_signature_bins_and_shares(tmp_path):
    write_run(tmp_path, "r", "bandit", 0, successes=1, cost_each=1.0,
              modes=("act", "gather_evidence"))
    rows = mode_signature(load_runs(tmp_path))["bandit"]
    assert rows, "bandit must have signature rows"
    for row in rows:
        assert abs(sum(row[m] for m in ("act", "retrieve", "simulate", "gather_evidence")) - 1.0) < 1e-9
    assert any(r["offset_bin"] < 0 for r in rows) and any(r["offset_bin"] >= 0 for r in rows)
