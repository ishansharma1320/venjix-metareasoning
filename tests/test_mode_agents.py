from collections import Counter

from venjix.agents import (
    FixedMixtureAgent,
    ReactiveAgent,
    RetrieveOnlyAgent,
    SimulateOnlyAgent,
)
from venjix.config import GridworldConfig
from venjix.gridworld import Gridworld
from venjix.llm import MockModel
from venjix.memory import Experience


def run_episode(env, agent):
    obs = env.reset()
    visited = [obs.pos]
    while not obs.done:
        decision = agent.choose(obs)
        prev_obs = obs
        obs = env.step(decision.action)
        agent.observe(prev_obs, decision.action, obs)
        visited.append(obs.pos)
    return obs, visited


CONFIG = GridworldConfig(size=5, start=(0, 0), goal=(3, 2))  # Manhattan distance 5


# --- retrieve-only -----------------------------------------------------------


def test_retrieve_cold_start_walk_is_seed_deterministic():
    def actions(seed):
        agent = RetrieveOnlyAgent(seed=seed)
        env = Gridworld(CONFIG, seed=0)
        obs = env.reset()
        return [agent.choose(obs).action for _ in range(10)]

    assert actions(1) == actions(1)
    assert actions(1) != actions(2)


def test_retrieve_learns_goal_then_fails_after_silent_shift():
    env = Gridworld(CONFIG, seed=0)
    agent = RetrieveOnlyAgent(seed=1)

    # Random-walk episodes until the goal is stumbled upon once.
    for _ in range(50):
        obs, _ = run_episode(env, agent)
        if obs.success:
            break
    assert obs.success, "random walk never found the goal within 50 episodes"
    assert agent.memory.believed_goal() == (3, 2)

    # With the goal remembered, the next episode is Manhattan-optimal and free.
    obs, _ = run_episode(env, agent)
    assert obs.success and obs.steps_used == 5

    # Silent shift: the agent walks confidently to the STALE goal and fails.
    # This is the study's core mechanism appearing in a test.
    env.relocate_goal(4)
    obs, visited = run_episode(env, agent)
    assert (3, 2) in visited  # went straight to the remembered (now empty) cell
    assert not obs.success  # oscillates near the stale goal until budget death
    assert agent.memory.believed_goal() == (3, 2)  # belief stays stale


def test_retrieve_makes_zero_llm_calls():
    env = Gridworld(CONFIG, seed=0)
    agent = RetrieveOnlyAgent(seed=1)
    run_episode(env, agent)
    # No client anywhere in the agent: nothing to bill. Its Decision cost is 0
    # by construction; this test documents the contract.
    assert not hasattr(agent, "client") and not hasattr(agent, "world")


# --- simulate-only -----------------------------------------------------------


def test_simulate_cold_start_pays_full_rollout_budget():
    client = MockModel(seed=0)
    agent = SimulateOnlyAgent(client, CONFIG, seed=1, sim_depth=3)
    env = Gridworld(CONFIG, seed=0)
    agent.choose(env.reset())
    # No believed goal: no early termination, 4 candidates x 3 steps each.
    assert client.total_calls == 4 * 3


def test_simulate_navigates_optimally_with_believed_goal():
    client = MockModel(seed=0)
    agent = SimulateOnlyAgent(client, CONFIG, seed=1, sim_depth=8)
    agent.memory.append(Experience((3, 1), "right", (3, 2), 1, None))
    env = Gridworld(CONFIG, seed=0)
    obs, _ = run_episode(env, agent)
    assert obs.success and obs.steps_used == 5  # Manhattan-optimal
    assert client.total_calls >= 4 * 5  # every step paid for its rollouts


def test_simulate_costs_more_per_step_than_reactive():
    sim_client, react_client = MockModel(seed=0), MockModel(seed=0)
    sim_agent = SimulateOnlyAgent(sim_client, CONFIG, seed=1, sim_depth=3)
    react_agent = ReactiveAgent(react_client, CONFIG)
    env = Gridworld(CONFIG, seed=0)
    obs = env.reset()
    sim_agent.choose(obs)
    react_agent.choose(obs)
    assert react_client.total_calls == 1
    assert sim_client.total_calls > react_client.total_calls
    assert sim_client.total_input_tokens > react_client.total_input_tokens


# --- fixed mixture -----------------------------------------------------------


def make_mixture(seed, weights, sim_depth=3):
    return FixedMixtureAgent(MockModel(seed=0), CONFIG, seed, weights, sim_depth)


def test_mixture_draws_are_seed_reproducible():
    def modes(seed):
        agent = make_mixture(seed, (0.25, 0.25, 0.25, 0.25))
        env = Gridworld(CONFIG, seed=0)
        obs = env.reset()
        return [agent.choose(obs).mode for _ in range(50)]

    assert modes(5) == modes(5)
    assert modes(5) != modes(6)


def test_mixture_frequencies_match_weights():
    weights = (0.4, 0.3, 0.2, 0.1)
    agent = make_mixture(seed=7, weights=weights)
    env = Gridworld(CONFIG, seed=0)
    obs = env.reset()
    counts = Counter(agent.choose(obs).mode for _ in range(2000))
    for mode, weight in zip(("act", "retrieve", "simulate", "gather_evidence"), weights):
        assert abs(counts[mode] / 2000 - weight) < 0.04


def test_degenerate_act_mixture_reproduces_reactive():
    agent = make_mixture(seed=3, weights=(1.0, 0.0, 0.0, 0.0))
    reference = ReactiveAgent(MockModel(seed=0), CONFIG)
    env = Gridworld(CONFIG, seed=0)
    obs = env.reset()
    for _ in range(10):
        mixed, pure = agent.choose(obs), reference.choose(obs)
        assert mixed.mode == "act"
        assert mixed.action == pure.action
        obs = env.step(mixed.action)
        if obs.done:
            obs = env.reset()


def test_gather_evidence_mode_emits_probe():
    agent = make_mixture(seed=3, weights=(0.0, 0.0, 0.0, 1.0))
    env = Gridworld(CONFIG, seed=0)
    decision = agent.choose(env.reset())
    assert (decision.mode, decision.action) == ("gather_evidence", "probe")


def test_mixture_modes_share_one_memory():
    agent = make_mixture(seed=3, weights=(0.0, 0.0, 0.0, 1.0))
    env = Gridworld(CONFIG, seed=0)
    obs = env.reset()
    # Walk adjacent to the goal, then probe via gather_evidence.
    for action in ("down", "down", "down", "right"):
        prev = obs
        obs = env.step(action)
        agent.observe(prev, action, obs)
    decision = agent.choose(obs)  # probe at (3, 1), goal (3, 2) within radius
    prev = obs
    obs = env.step(decision.action)
    agent.observe(prev, decision.action, obs)
    assert agent.memory.believed_goal() == (3, 2)
    # The retrieve sub-policy sees the shared evidence immediately.
    assert agent._retrieve.choose(obs).action == "right"
