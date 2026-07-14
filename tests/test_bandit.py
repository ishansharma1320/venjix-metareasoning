import math

import pytest

from venjix.agents import MIXTURE_MODES, BanditArbiterAgent
from venjix.bandit import LinUCB, solve_linear
from venjix.config import GridworldConfig, PriceTable
from venjix.gridworld import ACTIONS, Gridworld
from venjix.llm import MockModel
from venjix.memory import Experience
from venjix.signal import EwmaPredictionError

CONFIG = GridworldConfig(size=5, start=(0, 0), goal=(3, 2))


def make_agent(client=None, seed=1, **kwargs):
    return BanditArbiterAgent(
        client or MockModel(seed=0), CONFIG, seed, prices=PriceTable(), **kwargs
    )


# --- LinUCB math (plumbing only — learned outcomes are the experiment's job) --


def test_solve_linear_against_hand_computed_values():
    assert solve_linear([[2, 0, 0], [0, 3, 0], [0, 0, 4]], [2, 6, 8]) == [1, 2, 2]
    x = solve_linear([[2, 1], [1, 3]], [3, 5])
    assert math.isclose(x[0], 0.8) and math.isclose(x[1], 1.4)
    with pytest.raises(ValueError):
        solve_linear([[1, 1], [1, 1]], [1, 2])  # singular


def test_ucb_tries_every_arm_before_exploiting():
    bandit = LinUCB(n_arms=4, dim=3, ucb_alpha=1.0)
    x = (1.0, 0.5, 1.0)
    chosen = []
    for _ in range(4):
        arm = bandit.select(x)
        chosen.append(arm)
        bandit.update(arm, x, 0.0)  # zero reward: only widths differentiate
    assert sorted(chosen) == [0, 1, 2, 3]


def test_learns_context_dependent_arm_preference():
    bandit = LinUCB(n_arms=2, dim=3, ucb_alpha=0.1)
    hi, lo = (1.0, 0.9, 1.0), (1.0, 0.0, 1.0)
    for _ in range(100):
        bandit.update(0, hi, 1.0)
        bandit.update(0, lo, -1.0)
        bandit.update(1, hi, -1.0)
        bandit.update(1, lo, 1.0)
    assert bandit.select(hi) == 0
    assert bandit.select(lo) == 1


def test_selection_is_deterministic():
    def run():
        bandit = LinUCB(n_arms=4, dim=3, ucb_alpha=1.0)
        choices = []
        for i in range(30):
            x = (1.0, (i % 10) / 10, float(i % 2))
            arm = bandit.select(x)
            choices.append(arm)
            bandit.update(arm, x, 0.1 * (arm + 1))
        return choices

    assert run() == run()


def test_linucb_validation():
    for bad in ((1, 3, 1.0), (4, 0, 1.0), (4, 3, -0.5)):
        with pytest.raises(ValueError):
            LinUCB(*bad)


# --- BanditArbiterAgent plumbing ----------------------------------------------


def force_arm(agent, mode):
    """Pre-train so the given arm dominates under ucb_alpha=0."""
    context = (1.0, 0.0, 1.0)
    target = MIXTURE_MODES.index(mode)
    for arm in range(len(MIXTURE_MODES)):
        agent.bandit.update(arm, context, 1.0 if arm == target else -1.0)


def test_uses_the_shared_signal_class_and_locked_context():
    agent = make_agent()
    assert isinstance(agent.signal, EwmaPredictionError)  # identical EWMA protocol
    assert agent.bandit.dim == 3  # (1, ewma, has_belief) — locked, nothing more


def test_signal_cost_parity_with_heuristic():
    client = MockModel(seed=0)
    agent = make_agent(client=client, ucb_alpha=0.0)
    agent.memory.append(Experience((3, 1), "right", (3, 2), 1, None))
    force_arm(agent, "retrieve")
    env = Gridworld(CONFIG, seed=0)
    decision = agent.choose(env.reset())
    assert decision.mode == "retrieve"
    assert client.total_calls == 1  # retrieve is free; the 1 call is the predict


def test_bandit_reward_arithmetic_is_exact():
    client = MockModel(seed=0)
    agent = make_agent(client=client, ucb_alpha=0.0, cost_weight=100.0)
    agent.memory.append(Experience((3, 1), "right", (3, 2), 1, None))
    force_arm(agent, "retrieve")
    env = Gridworld(CONFIG, seed=0)
    obs = env.reset()
    decision = agent.choose(obs)
    prev_obs = obs
    obs = env.step(decision.action)
    agent.observe(prev_obs, decision.action, obs)
    step_cost = PriceTable().cost_usd(
        client.total_input_tokens, client.total_output_tokens
    )
    assert agent.last_bandit_reward == obs.reward - 100.0 * step_cost


def test_every_arm_yields_legal_actions_and_exploration_covers_all_modes():
    client = MockModel(seed=0)
    agent = make_agent(client=client)  # ucb_alpha 1.0: UCB explores
    env = Gridworld(CONFIG, seed=0)
    obs = env.reset()
    modes = set()
    for _ in range(60):
        decision = agent.choose(obs)
        assert decision.action in ACTIONS
        modes.add(decision.mode)
        prev_obs = obs
        obs = env.step(decision.action)
        agent.observe(prev_obs, decision.action, obs)
        if obs.done:
            obs = env.reset()
    assert modes == set(MIXTURE_MODES)
