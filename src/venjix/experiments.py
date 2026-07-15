"""Loader for the frozen experiment set (experiments/exp-v1.json).

The JSON file is the pre-registered artifact (docs/CLAUDE.md Amendment 3); this
module makes the freeze executable: every (regime, agent, seed) condition
materializes into a validated RunConfig, so the set can be enumerated, hashed,
and run without any hand-assembled configs sneaking in.

Agents listed in the set but not yet implemented (the bandit, until rung 4) are
part of the frozen registration and are skipped by the loader until they exist.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from venjix.config import (
    AGENT_TYPES,
    GridworldConfig,
    PriceTable,
    RunConfig,
    ShiftEntry,
    ShiftSchedule,
)

DEFAULT_SET = Path(__file__).resolve().parents[2] / "experiments" / "exp-v1.json"


@dataclass(frozen=True)
class Condition:
    regime: str
    agent: str
    seed: int
    config: RunConfig


def load_spec(path: str | Path = DEFAULT_SET) -> dict:
    return json.loads(Path(path).read_text())


def conditions(spec: dict, agents: list[str] | None = None) -> list[Condition]:
    if agents is None:
        agents = [a for a in spec["agents"] if a in AGENT_TYPES]
    params = spec["agent_params"]
    prices = PriceTable(**spec["prices"]) if "prices" in spec else PriceTable()
    result = []
    for regime in spec["regimes"]:
        schedule = ShiftSchedule(
            version=f"{spec['version']}/{regime['name']}",
            entries=tuple(ShiftEntry(step, dist) for step, dist in regime["shifts"]),
        )
        env = GridworldConfig(**regime["env"])
        for agent in agents:
            for seed in range(spec["n_seeds"]):
                config = RunConfig(
                    env=env,
                    schedule=schedule,
                    seed=seed,
                    n_episodes=spec["n_episodes"],
                    model=spec["model"],
                    prices=prices,
                    agent=agent,
                    mixture_weights=(
                        tuple(params["mixture_weights"]) if agent == "mixture" else None
                    ),
                    sim_depth=params["sim_depth"],
                    ewma_alpha=params["ewma_alpha"],
                    pe_threshold=params["pe_threshold"],
                    ucb_alpha=params["ucb_alpha"],
                    cost_weight=params["cost_weight"],
                )
                result.append(
                    Condition(regime=regime["name"], agent=agent, seed=seed, config=config)
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen experiment set tools.")
    parser.add_argument("--set", default=str(DEFAULT_SET))
    parser.add_argument("--list", action="store_true", help="enumerate conditions")
    parser.add_argument("--regime")
    parser.add_argument("--agent")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mock", action="store_true", help="shortcut for --backend mock")
    parser.add_argument(
        "--backend", choices=("mock", "vllm", "vast", "anthropic"), default=None
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--out", default="runs")
    args = parser.parse_args()

    spec = load_spec(args.set)
    all_conditions = conditions(spec)

    if args.list or args.regime is None:
        registered = spec["agents"]
        implemented = sorted({c.agent for c in all_conditions})
        print(f"{spec['version']} (frozen {spec['frozen_at']})")
        print(f"regimes: {[r['name'] for r in spec['regimes']]}")
        print(f"agents registered: {registered} (implemented: {implemented})")
        print(f"seeds per condition: {spec['n_seeds']}")
        print(f"runnable conditions now: {len(all_conditions)}")
        return

    matches = [
        c
        for c in all_conditions
        if c.regime == args.regime and c.agent == args.agent and c.seed == args.seed
    ]
    if not matches:
        raise SystemExit(f"no condition ({args.regime}, {args.agent}, {args.seed})")
    condition = matches[0]

    from venjix.llm import make_client
    from venjix.runner import run

    backend = args.backend or ("mock" if args.mock else "vllm")
    client = make_client(
        backend, condition.config.model, seed=condition.seed, base_url=args.base_url
    )
    summary = run(condition.config, client, args.out)
    print(
        f"{condition.regime}/{condition.agent}/seed{condition.seed}: "
        f"{summary.successes}/{summary.episodes} successes, "
        f"{summary.shifts} shifts, ${summary.cost_usd:.4f} -> {summary.run_dir}"
    )


if __name__ == "__main__":
    main()
