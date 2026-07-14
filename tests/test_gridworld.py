import random

import pytest

from venjix.config import GridworldConfig
from venjix.gridworld import ACTIONS, OUT_OF_RANGE, Gridworld


def make_env(**overrides):
    seed = overrides.pop("seed", 0)
    config = GridworldConfig(**{"size": 5, "goal": (4, 4), **overrides})
    env = Gridworld(config, seed=seed)
    env.reset()
    return env


def test_movement_and_boundary_clipping():
    env = make_env(start=(0, 0))
    assert env.step("up").pos == (0, 0)  # clipped at top edge
    assert env.step("left").pos == (0, 0)  # clipped at left edge
    assert env.step("down").pos == (1, 0)
    assert env.step("right").pos == (1, 1)
    assert env.step("up").pos == (0, 1)


def test_reward_and_success_only_on_goal():
    env = make_env(goal=(0, 2))
    obs = env.step("right")
    assert (obs.reward, obs.done, obs.success) == (0, False, False)
    obs = env.step("right")
    assert obs.pos == (0, 2)
    assert (obs.reward, obs.done, obs.success) == (1, True, True)


def test_step_budget_exhaustion_fails_episode():
    env = make_env()  # budget defaults to 4 * size = 20
    for _ in range(19):
        obs = env.step("up")  # bounces on the top edge, never reaches (4, 4)
        assert not obs.done
    obs = env.step("up")
    assert obs.done and not obs.success and obs.reward == 0
    with pytest.raises(RuntimeError):
        env.step("up")


def test_probe_reveals_offset_within_radius_only():
    env = make_env(goal=(0, 3), probe_radius=3)
    obs = env.step("probe")
    assert obs.probe_result == (0, 3)
    assert obs.pos == (0, 0) and obs.steps_used == 1

    env = make_env(goal=(0, 3), probe_radius=2)
    assert env.step("probe").probe_result == OUT_OF_RANGE


def test_moves_do_not_carry_probe_result():
    env = make_env()
    assert env.step("down").probe_result is None


def test_goal_persists_across_resets():
    config = GridworldConfig(size=7)
    env = Gridworld(config, seed=123)
    env.reset()
    goal = env.goal
    env.step("down")
    env.reset()
    assert env.goal == goal
    env.reset()
    assert env.goal == goal


def test_relocation_is_silent_in_observations():
    shifted, control = make_env(seed=7), make_env(seed=7)
    assert shifted.step("down") == control.step("down")
    shifted.relocate_goal(2)
    assert shifted.goal != control.goal
    # identical agent state, moved goal: the observation stream must not differ
    assert shifted.step("down") == control.step("down")
    assert shifted.step("right") == control.step("right")


def test_determinism_from_config_and_seed():
    config = GridworldConfig(size=7)
    env_a, env_b = Gridworld(config, seed=42), Gridworld(config, seed=42)
    assert env_a.goal == env_b.goal

    script = random.Random(0)
    env_a.reset(), env_b.reset()
    for step in range(200):
        action = script.choice(ACTIONS)
        obs_a, obs_b = env_a.step(action), env_b.step(action)
        assert obs_a == obs_b
        if step in (60, 140):
            assert env_a.relocate_goal(3) == env_b.relocate_goal(3)
        if obs_a.done:
            assert env_a.reset() == env_b.reset()
    assert env_a.goal == env_b.goal


def test_config_validation():
    with pytest.raises(ValueError):
        GridworldConfig(size=5, start=(5, 0))
    with pytest.raises(ValueError):
        GridworldConfig(size=5, goal=(0, 5))
    with pytest.raises(ValueError):
        GridworldConfig(size=5, start=(1, 1), goal=(1, 1))
    with pytest.raises(RuntimeError):
        Gridworld(GridworldConfig(), seed=0).step("up")  # step before reset
    env = make_env()
    with pytest.raises(ValueError):
        env.step("teleport")
