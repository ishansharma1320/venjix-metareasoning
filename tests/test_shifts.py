import random

import pytest

from venjix.config import GridworldConfig, ShiftEntry, ShiftSchedule
from venjix.gridworld import ACTIONS, Gridworld, _manhattan
from venjix.shifts import ShiftScheduler


def test_fires_exactly_at_configured_steps():
    env = Gridworld(GridworldConfig(size=7, goal=(3, 3)), seed=0)
    env.reset()
    scheduler = ShiftScheduler(
        ShiftSchedule(version="test-v1", entries=(ShiftEntry(3, 2), ShiftEntry(7, 1)))
    )
    fired = {}
    for global_step in range(1, 13):
        env.step("up")  # bounces on the top edge; episode never ends early
        for record in scheduler.maybe_shift(global_step, env):
            fired.setdefault(global_step, []).append(record)
    assert sorted(fired) == [3, 7]
    assert [r.at_step for step in (3, 7) for r in fired[step]] == [3, 7]
    assert all(len(records) == 1 for records in fired.values())


def test_relocation_at_exact_requested_distance():
    env = Gridworld(GridworldConfig(size=11, goal=(5, 5)), seed=1)
    env.reset()
    scheduler = ShiftScheduler(
        ShiftSchedule(version="test-v1", entries=(ShiftEntry(1, 3),))
    )
    env.step("down")
    (record,) = scheduler.maybe_shift(1, env)
    assert record.requested_distance == record.actual_distance == 3
    assert _manhattan(record.old_goal, record.new_goal) == 3
    assert record.old_goal == (5, 5) and env.goal == record.new_goal


def test_oversized_distance_falls_back_to_max_achievable():
    config = GridworldConfig(size=4, start=(0, 0), goal=(1, 1))
    env = Gridworld(config, seed=2)
    env.reset()
    old_goal, new_goal, actual = env.relocate_goal(100)

    excluded = {old_goal, (0, 0)}  # goal itself, agent-at-start, start
    max_achievable = max(
        _manhattan(old_goal, (r, c))
        for r in range(4)
        for c in range(4)
        if (r, c) not in excluded
    )
    assert actual == max_achievable == _manhattan(old_goal, new_goal)


def test_relocation_never_lands_on_start_or_old_goal():
    # Goal (2, 2), distance 1 on a 3x3 grid: in-grid ring is {(1, 2), (2, 1)}.
    # Amendment 6c: only the old goal and start are excluded — never anything
    # derived from agent state (the agent's cell IS a legal landing spot).
    for seed in range(50):
        env = Gridworld(GridworldConfig(size=3, start=(0, 0), goal=(2, 2)), seed=seed)
        env.reset()
        old_goal, new_goal, actual = env.relocate_goal(1)
        assert new_goal in {(1, 2), (2, 1)}
        assert new_goal not in {(0, 0), old_goal}
        assert actual == 1


def test_relocation_is_independent_of_agent_position():
    # Amendment 6c: identical (config, seed) must yield identical goal
    # sequences no matter what the agent did — the pairing property.
    config = GridworldConfig(size=7, goal=(3, 3))
    env_a, env_b = Gridworld(config, seed=9), Gridworld(config, seed=9)
    env_a.reset(), env_b.reset()
    for action in ("down", "down", "right", "down"):  # only env_a's agent moves
        env_a.step(action)
    for distance in (3, 5, 2):
        assert env_a.relocate_goal(distance) == env_b.relocate_goal(distance)


def test_full_run_reproducibility():
    def run():
        env = Gridworld(GridworldConfig(size=7), seed=42)
        scheduler = ShiftScheduler(
            ShiftSchedule(
                version="test-v1", entries=(ShiftEntry(40, 3), ShiftEntry(120, 5))
            )
        )
        script = random.Random(0)
        env.reset()
        observations, records = [], []
        for global_step in range(1, 201):
            obs = env.step(script.choice(ACTIONS))
            observations.append(obs)
            records.extend(scheduler.maybe_shift(global_step, env))
            if obs.done:
                env.reset()
        return observations, records

    first, second = run(), run()
    assert first == second
    assert len(first[1]) == 2


def test_schedule_validation():
    with pytest.raises(ValueError):
        ShiftSchedule(version="v", entries=(ShiftEntry(7, 1), ShiftEntry(3, 1)))
    with pytest.raises(ValueError):
        ShiftSchedule(version="v", entries=(ShiftEntry(3, 1), ShiftEntry(3, 2)))
    with pytest.raises(ValueError):
        ShiftSchedule(version="v", entries=(ShiftEntry(3, 0),))
    env = Gridworld(GridworldConfig(), seed=0)
    with pytest.raises(ValueError):
        env.relocate_goal(0)
