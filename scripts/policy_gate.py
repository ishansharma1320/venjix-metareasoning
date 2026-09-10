"""Policy gate (Amendment 11) — run BEFORE any exp-v2 experiment-set condition.

exp-v1 gated only the world model; its act policy turned out to be dead (always
"probe"). This gate probes the ACT POLICY under the exact exp-v2 serving config:
reactive-only on the spec's easy-7 regime, 20 seeds x 10 episodes, and prints a
pre-declared verdict:

    GREEN  iff  move fraction >= 0.50  AND  success rate >= 0.15
    RED    otherwise

The gate detects a DEAD policy (exp-v1: 0.0 moves, 0.0 success), not superiority.
The mock random policy scores 0.21 on easy-7 (runs/exp-v1-mock); over the gate's
200 episodes its binomial sd is ~0.029, so 0.15 ≈ its lower 95% bound: GREEN means
"moves, and not significantly worse than random". (The mock dress rehearsal of this
script scored 0.205 — a bar at 0.21 would have failed a random policy on noise.)
Bands are fixed here, not flags.

    uv run python scripts/policy_gate.py --set experiments/exp-v2.json --backend vllm
"""

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from venjix.experiments import conditions, load_spec
from venjix.llm import make_client
from venjix.runner import run

GATE_REGIME = "easy-7"
GATE_SEEDS = 20
GATE_EPISODES = 10
MOVE_FRACTION_GREEN = 0.50
SUCCESS_RATE_GREEN = 0.15  # ~lower 95% bound of the mock random policy (0.21) over 200 episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", default="experiments/exp-v2.json")
    parser.add_argument("--backend", choices=("vllm", "mock"), default="vllm")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--out", default="runs/policy-gate")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--budget", type=int, default=None,
                        help="gate-only override of the spec's policy_thinking_budget "
                             "(the registered fallback schedule); recorded in the report")
    args = parser.parse_args()

    spec = load_spec(args.set)
    if args.budget is not None:
        spec = dict(spec, policy_thinking_budget=args.budget)
    budget = spec.get("policy_thinking_budget")
    conds = [c for c in conditions(spec, agents=["reactive"])
             if c.regime == GATE_REGIME and c.seed < GATE_SEEDS]
    assert len(conds) == GATE_SEEDS, f"expected {GATE_SEEDS} gate conditions, got {len(conds)}"
    out_dir = Path(args.out) / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-budget{budget}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"policy gate: {spec['version']} {GATE_REGIME} reactive x {GATE_SEEDS} seeds x "
          f"{GATE_EPISODES} episodes, thinking budget {budget}, backend {args.backend}")

    from concurrent.futures import ThreadPoolExecutor

    def one(c):
        cfg = replace(c.config, n_episodes=GATE_EPISODES)
        client = make_client(args.backend, cfg.model, seed=c.seed, base_url=args.base_url,
                             thinking_budget=cfg.policy_thinking_budget)
        return c.seed, run(cfg, client, out_dir), client

    steps = moves = successes = episodes = forced = policy_calls = out_tokens = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for seed, summary, client in pool.map(one, conds):
            run_dir = Path(summary.run_dir)
            for line in (run_dir / "episodes.jsonl").open():
                r = json.loads(line)
                if r["type"] == "step":
                    steps += 1
                    moves += r["action"] != "probe"
                    policy_calls += r["llm_calls"]
                    out_tokens += r["output_tokens"]
                    forced += r.get("forced_answer") or 0
                elif r["type"] == "episode":
                    episodes += 1
                    successes += r["success"]
            print(f"  seed {seed}: {summary.successes}/{summary.episodes} successes, ${summary.cost_usd:.4f}")

    move_fraction = moves / steps if steps else 0.0
    success_rate = successes / episodes if episodes else 0.0
    verdict = ("GREEN" if move_fraction >= MOVE_FRACTION_GREEN and success_rate >= SUCCESS_RATE_GREEN
               else "RED")
    report = {
        "spec": args.set, "version": spec["version"], "policy_thinking_budget": budget,
        "regime": GATE_REGIME, "seeds": GATE_SEEDS, "episodes_per_seed": GATE_EPISODES,
        "steps": steps, "move_fraction": move_fraction, "success_rate": success_rate,
        "forced_answer_rate": (forced / steps if steps else 0.0),
        "mean_output_tokens_per_policy_call": (out_tokens / policy_calls if policy_calls else 0.0),
        "bands": {"move_fraction_green": MOVE_FRACTION_GREEN, "success_rate_green": SUCCESS_RATE_GREEN},
        "verdict": verdict,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nsteps {steps}  move fraction {move_fraction:.3f}  success rate {success_rate:.3f}  "
          f"forced-answer rate {report['forced_answer_rate']:.3f}  "
          f"mean output tokens/policy call {report['mean_output_tokens_per_policy_call']:.0f}")
    print(f"POLICY GATE: {verdict}  (GREEN iff move >= {MOVE_FRACTION_GREEN} and success >= {SUCCESS_RATE_GREEN})")
    print(f"report: {out_dir}/report.json")


if __name__ == "__main__":
    main()
