"""Policy gate — run BEFORE any experiment-set condition on a new substrate.

exp-v1 gated only the world model; its act policy turned out to be dead (always
"probe"). This gate probes the ACT POLICY under the exact serving config:
reactive-only on the spec's easiest regime, 20 seeds x 10 episodes.

Criteria (fixed per environment, not flags):

  gridworld (exp-v2, Amendment 11):  GREEN iff move fraction >= 0.50 AND success >= 0.15
  minigrid  (exp-v3, PLAN §4):       GREEN iff success >= --success-bar, where the bar is
      the lower 95% bound of the MOCK random policy on this very gate
      (p - 1.96*sqrt(p(1-p)/n)), computed by running this script with --backend mock
      and declared in the plan BEFORE any model call. No move-fraction criterion
      (exp-v2 lesson: it rejected a live, probe-heavy policy).

    uv run python scripts/policy_gate.py --set experiments/exp-v3.json --backend mock
    uv run python scripts/policy_gate.py --set experiments/exp-v3.json --backend vllm --success-bar 0.xx
"""

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from venjix.experiments import conditions, load_spec
from venjix.llm import make_client
from venjix.runner import run

GATE_SEEDS = 20
GATE_EPISODES = 10
# gridworld bands (Amendment 11)
MOVE_FRACTION_GREEN = 0.50
SUCCESS_RATE_GREEN_GRIDWORLD = 0.15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="experiments/exp-v2.json")
    parser.add_argument("--backend", choices=("vllm", "mock"), default="vllm")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None, help="override the spec's model id (candidate substrates at M2)")
    parser.add_argument("--regime", default=None, help="gate regime (default: the spec's first regime)")
    parser.add_argument("--out", default="runs/policy-gate")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--budget", type=int, default=None,
                        help="gate-only override of the spec's policy_thinking_budget; recorded in the report")
    parser.add_argument("--success-bar", type=float, default=None,
                        help="minigrid: declared success bar (lower 95%% bound of the mock random policy)")
    args = parser.parse_args()

    spec = load_spec(args.set)
    if args.budget is not None:
        spec = dict(spec, policy_thinking_budget=args.budget)
    if args.model is not None:
        spec = dict(spec, model=args.model)
    env_kind = spec.get("env_kind", "gridworld")
    regime = args.regime or (spec["regimes"][0]["name"] if env_kind == "minigrid" else "easy-7")
    budget = spec.get("policy_thinking_budget")
    if env_kind == "minigrid" and args.backend != "mock" and args.success_bar is None:
        raise SystemExit("minigrid gate needs --success-bar (declared from the mock run) before a model call")

    conds = [c for c in conditions(spec, agents=["reactive"]) if c.regime == regime and c.seed < GATE_SEEDS]
    assert len(conds) == GATE_SEEDS, f"expected {GATE_SEEDS} gate conditions, got {len(conds)}"
    tag = spec["model"].replace("/", "-") if args.backend != "mock" else "mock"
    out_dir = Path(args.out) / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{spec['version']}-{tag}-budget{budget}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"policy gate: {spec['version']} {regime} reactive x {GATE_SEEDS} seeds x {GATE_EPISODES} episodes, "
          f"model {spec['model']}, thinking budget {budget}, backend {args.backend}")

    def one(c):
        cfg = replace(c.config, n_episodes=GATE_EPISODES)
        client = make_client(args.backend, cfg.model, seed=c.seed, base_url=args.base_url,
                             thinking_budget=cfg.policy_thinking_budget)
        return c.seed, run(cfg, client, out_dir)

    steps = moves = successes = episodes = forced = policy_calls = out_tokens = in_tokens = 0
    evidence_actions = {"gridworld": "probe", "minigrid": "left"}[env_kind]
    action_counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for seed, summary in pool.map(one, conds):
            for line in (Path(summary.run_dir) / "episodes.jsonl").open():
                r = json.loads(line)
                if r["type"] == "step":
                    steps += 1
                    moves += r["action"] != evidence_actions
                    action_counts[r["action"]] = action_counts.get(r["action"], 0) + 1
                    policy_calls += r["llm_calls"]
                    out_tokens += r["output_tokens"]
                    in_tokens += r["input_tokens"]
                    forced += r.get("forced_answer") or 0
                elif r["type"] == "episode":
                    episodes += 1
                    successes += r["success"]
            print(f"  seed {seed}: {summary.successes}/{summary.episodes} successes, ${summary.cost_usd:.4f}")

    move_fraction = moves / steps if steps else 0.0
    success_rate = successes / episodes if episodes else 0.0
    implied_bar = max(0.0, success_rate - 1.96 * math.sqrt(success_rate * (1 - success_rate) / max(episodes, 1)))
    if env_kind == "minigrid":
        bar = args.success_bar
        verdict = ("GREEN" if bar is not None and success_rate >= bar else
                   "RED" if bar is not None else "N/A (mock floor run)")
        criterion = f"success >= {bar}" if bar is not None else "mock floor run: no verdict"
    else:
        verdict = ("GREEN" if move_fraction >= MOVE_FRACTION_GREEN and success_rate >= SUCCESS_RATE_GREEN_GRIDWORLD
                   else "RED")
        criterion = f"move >= {MOVE_FRACTION_GREEN} and success >= {SUCCESS_RATE_GREEN_GRIDWORLD}"
    report = {
        "spec": args.set, "version": spec["version"], "env_kind": env_kind, "model": spec["model"],
        "backend": args.backend, "policy_thinking_budget": budget, "regime": regime,
        "seeds": GATE_SEEDS, "episodes_per_seed": GATE_EPISODES, "steps": steps,
        "move_fraction": move_fraction, "success_rate": success_rate,
        "success_rate_lower95": implied_bar,
        "forced_answer_rate": (forced / steps if steps else 0.0),
        "mean_output_tokens_per_policy_call": (out_tokens / policy_calls if policy_calls else 0.0),
        "mean_input_tokens_per_policy_call": (in_tokens / policy_calls if policy_calls else 0.0),
        "criterion": criterion, "success_bar": args.success_bar, "verdict": verdict,
        # informational (PLAN §4b proposal): share of the most frequent action
        "action_counts": action_counts,
        "max_action_share": (max(action_counts.values()) / steps if steps else 0.0),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nsteps {steps}  move fraction {move_fraction:.3f}  success rate {success_rate:.3f} "
          f"(lower 95% bound {implied_bar:.3f})  forced-answer rate {report['forced_answer_rate']:.3f}  "
          f"tokens/policy call in {report['mean_input_tokens_per_policy_call']:.0f} / out "
          f"{report['mean_output_tokens_per_policy_call']:.0f}  max action share {report['max_action_share']:.3f} "
          f"{dict(sorted(action_counts.items(), key=lambda kv: -kv[1]))}")
    print(f"POLICY GATE: {verdict}  ({criterion})")
    print(f"report: {out_dir}/report.json")


if __name__ == "__main__":
    main()
