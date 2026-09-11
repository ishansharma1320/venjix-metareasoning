"""exp-v3 MiniGrid adapter, spec, planner, world model, and mock end-to-end run.
Skipped when the optional `minigrid` group is not installed."""

import random

import pytest

from venjix.config import GridworldConfig, PriceTable, RunConfig, ShiftEntry, ShiftSchedule
from venjix.envspec import DIRS, MINIGRID, minigrid_next, spec_for
from venjix.gridworld import Observation
from venjix.llm import MockModel
from venjix.world import MINIGRID_PREDICT_RESPONSE_REGEX, MiniGridWorldModel

minigrid = pytest.importorskip("minigrid")

from venjix.minigrid_env import START, MiniGridEmpty, render_view  # noqa: E402


def cfg(size=8, budget=32):
    return GridworldConfig(size=size, step_budget=budget)


def test_reset_step_and_determinism():
    a = MiniGridEmpty(cfg(), seed=3)
    b = MiniGridEmpty(cfg(), seed=3)
    oa, ob = a.reset(), b.reset()
    assert oa.pos == START and oa.dir == "right" and oa.steps_used == 0 and not oa.done
    assert a.goal == b.goal and a.goal != START and 1 <= a.goal[0] <= 6 and 1 <= a.goal[1] <= 6
    for act in ("forward", "left", "forward", "right", "forward"):
        oa, ob = a.step(act), b.step(act)
        assert (oa.pos, oa.dir, oa.reward, oa.done) == (ob.pos, ob.dir, ob.reward, ob.done)
    assert oa.steps_used == 5
    # goal persists across resets (only relocate_goal moves it)
    g = a.goal
    a.reset()
    assert a.goal == g


def test_dynamics_match_spec_truth_and_walls_block():
    env = MiniGridEmpty(cfg(), seed=1)
    obs = env.reset()
    state = (obs.pos, obs.dir)
    rng = random.Random(0)
    for _ in range(60):
        act = rng.choice(("left", "right", "forward"))
        expect = minigrid_next(8, state[0], state[1], act)
        if env._done:
            obs = env.reset(); state = (obs.pos, obs.dir); continue
        obs = env.step(act)
        if obs.success:
            # reaching the goal ends the episode; position is the goal cell
            assert obs.pos == env.goal and obs.done
            obs = env.reset(); state = (obs.pos, obs.dir); continue
        assert (obs.pos, obs.dir) == expect
        state = (obs.pos, obs.dir)
    # facing the left wall from (1,1): forward stays
    env2 = MiniGridEmpty(cfg(), seed=1); env2.reset()
    env2.step("left"); env2.step("left")  # facing left
    o = env2.step("forward")
    assert o.pos == START and o.dir == "left"


def test_goal_visibility_offset_and_success():
    env = MiniGridEmpty(cfg(size=8, budget=32), seed=7)
    env.reset()
    # force a known goal straight ahead: (4, 1) with agent at (1,1) facing right
    env._goal = (4, 1); env._place_goal()
    obs = env._observation(env._env.gen_obs()["image"], 0, False, False)
    assert obs.probe_result == (3, 0)          # in view, offset (dx, dy)
    assert "G" in obs.view
    o = env.step("forward"); o = env.step("forward"); o = env.step("forward")
    assert o.success and o.reward == 1 and o.done and o.pos == (4, 1)
    # goal behind the agent is not in view
    env3 = MiniGridEmpty(cfg(), seed=7); env3.reset()
    env3._goal = (1, 6); env3._place_goal()
    obs3 = env3._observation(env3._env.gen_obs()["image"], 0, False, False)
    assert obs3.probe_result is None


def test_relocate_goal_is_paired_and_at_distance():
    a = MiniGridEmpty(cfg(size=12, budget=48), seed=5)
    b = MiniGridEmpty(cfg(size=12, budget=48), seed=5)
    a.reset(); b.reset()
    for d in (3, 5, 8):
        ra, rb = a.relocate_goal(d), b.relocate_goal(d)
        assert ra == rb
        old, new, actual = ra
        assert new != old and new != START and actual == d
        assert 1 <= new[0] <= 10 and 1 <= new[1] <= 10
    # the Goal object really moved on the MiniGrid grid
    from minigrid.core.world_object import Goal
    goals = [(x, y) for x in range(12) for y in range(12) if isinstance(a._env.grid.get(x, y), Goal)]
    assert goals == [a.goal]


def test_render_view_shape_and_agent_marker():
    import numpy as np
    img = np.ones((7, 7, 3), dtype=int)  # all "empty"
    img[3, 0, 0] = 8   # goal at top-center
    img[0, 0, 0] = 2   # wall corner
    text = render_view(img)
    rows = text.split("\n")
    assert len(rows) == 7 and all(len(r) == 7 for r in rows)
    assert rows[0][3] == "G" and rows[0][0] == "#" and rows[6][3] == "A"


def test_planner_turns_then_moves():
    plan = MINIGRID.plan
    assert plan(((1, 1), "right"), (4, 1)) == "forward"
    assert plan(((1, 1), "down"), (4, 1)) == "left"     # down -> right is a left turn
    assert plan(((1, 1), "up"), (4, 1)) == "right"      # up -> right is a right turn
    assert plan(((1, 1), "left"), (4, 1)) == "left"     # 180: either; we choose left
    assert plan(((3, 3), "right"), (3, 6)) == "right"   # x aligned, need down
    assert plan(((3, 3), "down"), (3, 6)) == "forward"
    assert plan(((3, 3), "right"), (3, 3)) is None
    # a planned walk from start reaches any goal
    for goal in [(6, 6), (1, 6), (6, 1), (3, 5)]:
        state = ((1, 1), "right"); steps = 0
        while state[0] != goal and steps < 40:
            act = plan(state, goal); state = minigrid_next(8, state[0], state[1], act); steps += 1
        assert state[0] == goal


def test_gather_never_spins_forever():
    rng = random.Random(0)
    assert MINIGRID.gather(None, rng) == "left"
    assert MINIGRID.gather("left", rng) in ("right", "forward")


def test_mock_world_model_predicts_minigrid_truth():
    client = MockModel(seed=0)
    wm = MiniGridWorldModel(client, cfg())
    p = wm.predict(((2, 2), "up"), "forward", (2, 1))
    assert (p.next_pos, p.next_dir, p.reward, p.parse_error) == ((2, 1), "up", 1, False)
    p = wm.predict(((1, 1), "left"), "forward", None)
    assert (p.next_pos, p.next_dir, p.reward) == ((1, 1), "left", 0)   # wall blocks
    p = wm.predict(((3, 3), "right"), "right", None)
    assert (p.next_pos, p.next_dir) == ((3, 3), "down")
    assert p.next_state == ((3, 3), "down")
    import re
    assert re.fullmatch(MINIGRID_PREDICT_RESPONSE_REGEX, p.raw_text)


def test_egocentric_summary_and_goal_offsets():
    from venjix.envspec import egocentric_goal_offset, egocentric_summary
    # world (dx, dy) -> (ahead, right) per facing
    assert egocentric_goal_offset("right", (3, 0)) == (3, 0)
    assert egocentric_goal_offset("right", (0, 3)) == (0, 3)      # +y is to the right when facing right
    assert egocentric_goal_offset("down", (0, 3)) == (3, 0)       # ahead
    assert egocentric_goal_offset("down", (3, 0)) == (0, -3)      # to the left
    assert egocentric_goal_offset("left", (-2, 0)) == (2, 0)
    assert egocentric_goal_offset("up", (0, -2)) == (2, 0)
    assert egocentric_goal_offset("up", (2, 0)) == (0, 2)
    # real observations: wall ahead + goal to the right-hand side
    env = MiniGridEmpty(cfg(), seed=3); o = env.reset()
    for _ in range(5): o = env.step("forward")
    env._goal = (6, 4); env._place_goal()
    o = env._observation(env._env.gen_obs()["image"], 0, False, False)
    s = egocentric_summary(o)
    assert s["ahead"].startswith("a wall") and s["goal"] == "in view, 3 cells to your right"
    assert "wall" in s["left"] and "floor" in s["right"]
    # goal straight ahead from the start
    env2 = MiniGridEmpty(cfg(), seed=3); o2 = env2.reset(); env2._goal = (4, 1); env2._place_goal()
    o2 = env2._observation(env2._env.gen_obs()["image"], 0, False, False)
    s2 = egocentric_summary(o2)
    assert s2["goal"] == "in view, 3 cells ahead" and s2["ahead"].startswith("open floor")
    prompt = MINIGRID.policy_prompt(cfg(), o2)
    assert "3 cells ahead" in prompt and "(x, y)" not in prompt and "View:" not in prompt


def test_mock_policy_uses_prompt_vocabulary():
    client = MockModel(seed=0)
    view = "\n".join(["......." for _ in range(6)] + ["...A..."])
    obs = Observation(pos=(2, 2), reward=0, done=False, success=False, probe_result=None,
                      steps_used=0, dir="right", view=view)
    text = client.complete(MINIGRID.policy_prompt(cfg(), obs)).text
    assert text.split(": ")[1] in ("left", "right", "forward")
    # a malformed (ragged / absent) view must not crash the prompt builder
    for bad_view in ("...\n.......\n..", "", None):
        bad = Observation(pos=(2, 2), reward=0, done=False, success=False, probe_result=None,
                          steps_used=0, dir="right", view=bad_view)
        p = MINIGRID.policy_prompt(cfg(), bad)
        assert "Reply with exactly one action word" in p and "not visible" in p


@pytest.mark.parametrize("agent", ["reactive", "retrieve", "simulate", "mixture", "heuristic", "bandit"])
def test_mock_end_to_end_minigrid_run(tmp_path, agent):
    from venjix.runner import run
    import json
    config = RunConfig(
        env=cfg(size=8, budget=32),
        schedule=ShiftSchedule(version="t/mg", entries=(ShiftEntry(20, 3),)),
        seed=2, n_episodes=3, model="mock", prices=PriceTable(), agent=agent,
        env_kind="minigrid",
    )
    summary = run(config, MockModel(seed=2), tmp_path)
    assert summary.episodes == 3
    lines = [json.loads(l) for l in open(tmp_path / summary.run_dir.split("/")[-1] / "episodes.jsonl")]
    steps = [r for r in lines if r["type"] == "step"]
    assert steps and all(r["action"] in ("left", "right", "forward") for r in steps)
    assert all(r["dir"] in DIRS for r in steps)
    assert sum(1 for r in lines if r["type"] == "shift") == 1
    if agent in ("heuristic", "bandit"):
        assert all(r["prediction_error"] in (0, 1) for r in steps)


def test_minigrid_calibration_cases_and_mock_probe():
    from collections import Counter
    from venjix.calibration import build_report, generate_cases_minigrid, run_probe
    cases = generate_cases_minigrid([8, 12], 100, seed=0)
    assert len(cases) == 100
    strata = Counter(c.stratum for c in cases)
    assert strata == {"forward": 40, "blocked_forward": 30, "turn": 30}
    for c in cases:
        assert c.facing in DIRS and c.truth_dir in DIRS
        assert (c.truth_next, c.truth_dir) == minigrid_next(c.size, c.pos, c.facing, c.action)
        if c.stratum == "blocked_forward":
            assert c.truth_next == c.pos
        if c.stratum == "forward":
            assert c.truth_next != c.pos
        if c.belief == "unknown":
            assert c.believed_goal is None and c.truth_reward == 0
    results, usage = run_probe(cases, lambda: MockModel(seed=0), workers=4, env_kind="minigrid")
    report = build_report(results, usage, model="mock", backend="mock", seed=0, resamples=200)
    assert report["overall"]["misprediction_rate"] == 0.0 and report["verdict"] == "GREEN"
    assert usage["llm_calls"] == 100


def test_env_kind_hash_compat_and_spec_lookup():
    env = cfg()
    sched = ShiftSchedule(version="t/r", entries=(ShiftEntry(4, 3),))
    base = RunConfig(env=env, schedule=sched, seed=0, n_episodes=1)
    assert "env_kind" not in base.to_dict()
    mg = RunConfig(env=env, schedule=sched, seed=0, n_episodes=1, env_kind="minigrid")
    assert mg.to_dict()["env_kind"] == "minigrid" and mg.config_hash() != base.config_hash()
    assert spec_for("gridworld").kind == "gridworld" and spec_for("minigrid") is MINIGRID
    with pytest.raises(ValueError):
        RunConfig(env=env, schedule=sched, seed=0, n_episodes=1, env_kind="atari")
