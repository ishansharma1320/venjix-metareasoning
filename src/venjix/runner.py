"""Run loop: env + scheduler + agent + logger. Owns the global step counter
(persists across episode resets — that is what makes mid-episode shifts work).
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from venjix.agents import (
    Agent,
    FixedMixtureAgent,
    ReactiveAgent,
    RetrieveOnlyAgent,
    SimulateOnlyAgent,
    ThresholdHeuristicAgent,
)
from venjix.config import (
    GridworldConfig,
    PriceTable,
    RunConfig,
    ShiftEntry,
    ShiftSchedule,
)
from venjix.gridworld import Gridworld
from venjix.llm import AnthropicModel, LLMClient, MockModel
from venjix.logs import EpisodeLogger, now_ms
from venjix.shifts import ShiftScheduler


@dataclass(frozen=True)
class RunSummary:
    run_dir: str
    episodes: int
    successes: int
    shifts: int
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


def build_agent(config: RunConfig, client: LLMClient) -> Agent:
    if config.agent == "reactive":
        return ReactiveAgent(client, config.env)
    if config.agent == "retrieve":
        return RetrieveOnlyAgent(config.seed)
    if config.agent == "simulate":
        return SimulateOnlyAgent(client, config.env, config.seed, config.sim_depth)
    if config.agent == "heuristic":
        return ThresholdHeuristicAgent(
            client, config.env, config.seed, config.ewma_alpha, config.pe_threshold
        )
    return FixedMixtureAgent(
        client, config.env, config.seed, config.mixture_weights, config.sim_depth
    )


def run(config: RunConfig, client: LLMClient, out_root: str | Path) -> RunSummary:
    env = Gridworld(config.env, config.seed)
    scheduler = ShiftScheduler(config.schedule)
    agent = build_agent(config, client)
    logger = EpisodeLogger(out_root, config)

    global_step = 0
    successes = 0
    shifts = 0
    total_cost = 0.0
    with logger:
        for episode in range(config.n_episodes):
            obs = env.reset()
            ep_calls = ep_in = ep_out = 0
            ep_cost = ep_wall = 0.0
            while not obs.done:
                t0 = now_ms()
                calls0 = client.total_calls
                in0, out0 = client.total_input_tokens, client.total_output_tokens
                decision = agent.choose(obs)
                prev_obs = obs
                obs = env.step(decision.action)
                agent.observe(prev_obs, decision.action, obs)
                wall_ms = now_ms() - t0

                step_in = client.total_input_tokens - in0
                step_out = client.total_output_tokens - out0
                step_cost = config.prices.cost_usd(step_in, step_out)
                global_step += 1
                logger.log_step(
                    episode=episode,
                    step_in_episode=obs.steps_used,
                    global_step=global_step,
                    mode=decision.mode,
                    action=decision.action,
                    parse_error=decision.parse_error,
                    pos=obs.pos,
                    reward=obs.reward,
                    done=obs.done,
                    success=obs.success,
                    probe_result=obs.probe_result,
                    llm_calls=client.total_calls - calls0,
                    input_tokens=step_in,
                    output_tokens=step_out,
                    cost_usd=step_cost,
                    wall_time_ms=wall_ms,
                    prediction_error=getattr(agent, "last_prediction_error", None),
                    signal_ewma=getattr(agent, "signal_value", None),
                )
                ep_calls += client.total_calls - calls0
                ep_in += step_in
                ep_out += step_out
                ep_cost += step_cost
                ep_wall += wall_ms

                for record in scheduler.maybe_shift(global_step, env):
                    logger.log_shift(record, global_step)
                    shifts += 1

            logger.log_episode(
                episode=episode,
                success=obs.success,
                steps_used=obs.steps_used,
                llm_calls=ep_calls,
                input_tokens=ep_in,
                output_tokens=ep_out,
                cost_usd=ep_cost,
                wall_time_ms=ep_wall,
            )
            successes += int(obs.success)
            total_cost += ep_cost

    return RunSummary(
        run_dir=str(logger.run_dir),
        episodes=config.n_episodes,
        successes=successes,
        shifts=shifts,
        llm_calls=client.total_calls,
        input_tokens=client.total_input_tokens,
        output_tokens=client.total_output_tokens,
        cost_usd=total_cost,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reactive baseline.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=7)
    parser.add_argument(
        "--agent",
        choices=("reactive", "retrieve", "simulate", "mixture", "heuristic"),
        default="reactive",
    )
    parser.add_argument("--ewma-alpha", type=float, default=0.3)
    parser.add_argument("--pe-threshold", type=float, default=0.25)
    parser.add_argument(
        "--weights", default=None,
        help="mixture only: 4 comma-separated weights over act,retrieve,simulate,gather",
    )
    parser.add_argument("--sim-depth", type=int, default=3)
    parser.add_argument("--shift-at", type=int, default=25)
    parser.add_argument("--shift-distance", type=int, default=4)
    parser.add_argument("--mock", action="store_true", help="use the offline mock model")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--out", default="runs")
    args = parser.parse_args()

    weights = (
        tuple(float(w) for w in args.weights.split(",")) if args.weights else None
    )
    config = RunConfig(
        env=GridworldConfig(size=args.size),
        schedule=ShiftSchedule(
            version="cli-demo-v1",
            entries=(ShiftEntry(args.shift_at, args.shift_distance),),
        ),
        seed=args.seed,
        n_episodes=args.episodes,
        model="mock" if args.mock else args.model,
        prices=PriceTable(),
        agent=args.agent,
        mixture_weights=weights,
        sim_depth=args.sim_depth,
        ewma_alpha=args.ewma_alpha,
        pe_threshold=args.pe_threshold,
    )
    client: LLMClient = (
        MockModel(seed=args.seed) if args.mock else AnthropicModel(args.model)
    )
    summary = run(config, client, args.out)
    print(
        f"run: {summary.run_dir}\n"
        f"episodes: {summary.episodes}  successes: {summary.successes}  "
        f"shifts: {summary.shifts}\n"
        f"llm_calls: {summary.llm_calls}  tokens in/out: "
        f"{summary.input_tokens}/{summary.output_tokens}  "
        f"cost: ${summary.cost_usd:.6f}"
    )


if __name__ == "__main__":
    main()
