"""Environment specs: everything an agent needs to know about an environment
that is NOT arbitration logic (exp-v3 plan §3, §8).

The heuristic and the bandit must be byte-identical across environments — that
is the scientific requirement. So the environment-specific pieces (action
vocabulary, random-walk moves, the gather_evidence primitive, the planner toward
a believed goal, the policy prompt, and the world model) live here, and the
agents in agents.py consult a spec instead of module constants. GRIDWORLD is
the Phase 1 spec and the default everywhere, so Phase 1 behaviour and tests are
unchanged.

State: the world model predicts and the signal scores a *state*. For the Phase 1
gridworld that is the position; for MiniGrid it is (position, facing).
"""

import random
from dataclasses import dataclass

from venjix.config import GridworldConfig
from venjix.gridworld import ACTIONS as GRID_ACTIONS
from venjix.gridworld import Observation
from venjix.llm import LLMClient
from venjix.world import MiniGridWorldModel, WorldModel

GRID_MOVES = ("up", "down", "left", "right")

# MiniGrid conventions (minigrid.core): agent_dir 0=right(+x) 1=down(+y) 2=left 3=up;
# `left` action decrements agent_dir mod 4, `right` increments, `forward` moves one
# cell in the facing direction unless blocked. Walkable cells: 1..size-2 (outer wall).
MINIGRID_ACTIONS = ("left", "right", "forward")
DIRS = ("right", "down", "left", "up")
DIR_DELTA = {"right": (1, 0), "down": (0, 1), "left": (-1, 0), "up": (0, -1)}


@dataclass(frozen=True)
class EnvSpec:
    kind: str
    actions: tuple[str, ...]        # full vocabulary the parser accepts
    move_actions: tuple[str, ...]   # seeded random-walk vocabulary (cold starts)
    fallback_action: str            # parser failure -> never makes task progress

    # --- state -----------------------------------------------------------
    def state(self, obs: Observation):
        raise NotImplementedError

    def pos_of(self, state) -> tuple[int, int]:
        raise NotImplementedError

    # --- primitives ------------------------------------------------------
    def gather(self, last_action: str | None, rng: random.Random) -> str:
        raise NotImplementedError

    def plan(self, state, goal: tuple[int, int]) -> str | None:
        """Deterministic step toward `goal`; None when already there."""
        raise NotImplementedError

    def policy_prompt(self, env_config: GridworldConfig, obs: Observation) -> str:
        raise NotImplementedError

    def make_world_model(self, client: LLMClient, env_config: GridworldConfig):
        raise NotImplementedError


class GridworldSpec(EnvSpec):
    """Phase 1 toy gridworld — verbatim the pre-exp-v3 agent behaviour."""

    def __init__(self) -> None:
        super().__init__("gridworld", GRID_ACTIONS, GRID_MOVES, "probe")

    def state(self, obs: Observation):
        return obs.pos

    def pos_of(self, state) -> tuple[int, int]:
        return state

    def gather(self, last_action: str | None, rng: random.Random) -> str:
        # probe unless the previous action was a probe (probing never moves — a
        # pure-probe mode would pin the agent to one cell), else a seeded move.
        if last_action == "probe":
            return rng.choice(self.move_actions)
        return "probe"

    def plan(self, state, goal: tuple[int, int]) -> str | None:
        pos = state
        if pos[0] != goal[0]:
            return "down" if pos[0] < goal[0] else "up"
        if pos[1] != goal[1]:
            return "right" if pos[1] < goal[1] else "left"
        return None

    def policy_prompt(self, env_config: GridworldConfig, obs: Observation) -> str:
        rules = (
            f"You control an agent on a {env_config.size}x{env_config.size} grid. "
            f"A hidden goal cell gives reward 1; reaching it ends the episode. "
            f"Positions are (row, col) with (0, 0) at the top-left. "
            f"Actions: up, down, left, right (move one cell), or probe "
            f"(reveals the goal's offset if it is within Manhattan distance "
            f"{env_config.probe_radius}, costs one step, no movement). "
            f"You have {env_config.step_budget} steps per episode."
        )
        return (
            f"{rules}\n"
            f"Current position: {obs.pos}. Steps used: {obs.steps_used}.\n"
            f"Last probe result: {obs.probe_result}.\n"
            f"Reply with exactly one action word: up, down, left, right, or probe."
        )

    def make_world_model(self, client: LLMClient, env_config: GridworldConfig):
        return WorldModel(client, env_config)


def minigrid_turn(direction: str, action: str) -> str:
    i = DIRS.index(direction)
    if action == "left":
        return DIRS[(i - 1) % 4]
    if action == "right":
        return DIRS[(i + 1) % 4]
    return direction


def minigrid_next(size: int, pos: tuple[int, int], direction: str, action: str):
    """Ground-truth MiniGrid-Empty dynamics: (next_pos, next_dir). Walkable
    cells are 1..size-2 in both axes; a blocked forward stays in place."""
    if action in ("left", "right"):
        return pos, minigrid_turn(direction, action)
    dx, dy = DIR_DELTA[direction]
    nx, ny = pos[0] + dx, pos[1] + dy
    if 1 <= nx <= size - 2 and 1 <= ny <= size - 2:
        return (nx, ny), direction
    return pos, direction


def egocentric_goal_offset(direction: str, offset: tuple[int, int]) -> tuple[int, int]:
    """World-frame (dx, dy) -> (ahead, right) in the agent's frame."""
    dx, dy = offset
    if direction == "right":
        return dx, dy
    if direction == "down":
        return dy, -dx
    if direction == "left":
        return -dx, -dy
    return -dy, dx  # up


def egocentric_summary(obs: Observation) -> dict[str, str]:
    """Agent-relative description of the observation, derived from the rendered
    view (rows top = farthest, agent at the bottom center) and the goal offset."""
    rows = [r for r in obs.view.split("\n") if r] if obs.view else []
    if not rows or any(len(r) != len(rows[0]) for r in rows) or len(rows[0]) < 3:
        rows = []  # malformed / absent view: everything reads as "not visible"
    h = len(rows)
    c = (len(rows[0]) // 2) if rows else 0
    # cells of floor straight ahead before a wall / the edge of sight
    free = 0
    for j in range(h - 2, -1, -1):
        ch = rows[j][c]
        if ch in ("#", "?"):
            break
        free += 1
    ahead_cell = rows[h - 2][c] if h >= 2 else "?"
    if h < 2:
        free = 0
    if ahead_cell == "G":
        ahead = "the goal, 1 cell away"
    elif ahead_cell in ("#", "?"):
        ahead = "a wall (moving forward does nothing)"
    else:
        ahead = f"open floor, {free} cell{'s' if free != 1 else ''} until a wall"
    left_cell = rows[h - 1][c - 1] if h and c >= 1 else "?"
    right_cell = rows[h - 1][c + 1] if h and c + 1 < len(rows[h - 1]) else "?"
    name = {"#": "a wall", ".": "open floor", "G": "the goal", "?": "not visible"}
    left = name.get(left_cell, "not visible")
    right = name.get(right_cell, "not visible")
    if obs.probe_result is None or obs.dir is None:
        goal = "not in view"
    else:
        a, r = egocentric_goal_offset(obs.dir, obs.probe_result)
        parts = []
        if a > 0:
            parts.append(f"{a} cell{'s' if a != 1 else ''} ahead")
        if r > 0:
            parts.append(f"{r} cell{'s' if r != 1 else ''} to your right")
        elif r < 0:
            parts.append(f"{-r} cell{'s' if -r != 1 else ''} to your left")
        goal = "in view, " + " and ".join(parts) if parts else "right here"
    return {"ahead": ahead, "left": left, "right": right, "goal": goal}


class MiniGridSpec(EnvSpec):
    """MiniGrid-Empty, egocentric actions (exp-v3 plan §3, decisions b, c)."""

    def __init__(self) -> None:
        super().__init__("minigrid", MINIGRID_ACTIONS, MINIGRID_ACTIONS, "left")

    def state(self, obs: Observation):
        return (obs.pos, obs.dir)

    def pos_of(self, state) -> tuple[int, int]:
        return state[0]

    def gather(self, last_action: str | None, rng: random.Random) -> str:
        # Decision (c): one `left` turn rotates the view cone; mirror Phase 1's
        # "don't repeat the evidence primitive twice" rule so a gather burst
        # cannot pin the agent spinning in place.
        if last_action == "left":
            return rng.choice(("right", "forward"))
        return "left"

    def plan(self, state, goal: tuple[int, int]) -> str | None:
        pos, direction = state
        if pos == goal:
            return None
        if goal[0] != pos[0]:
            target = "right" if goal[0] > pos[0] else "left"
        else:
            target = "down" if goal[1] > pos[1] else "up"
        if direction == target:
            return "forward"
        delta = (DIRS.index(target) - DIRS.index(direction)) % 4
        return "right" if delta == 1 else "left"  # 2 or 3 -> left (2 = either)

    def policy_prompt(self, env_config: GridworldConfig, obs: Observation) -> str:
        """EGOCENTRIC encoding (PLAN-exp-v3 §4c, 2026-09-11): the M2 diagnostic showed
        every candidate model reads world-frame (x, y, dx, dy) + an ASCII view as
        "goal ahead" and walks into walls, while the same facts in agent-relative
        words are acted on correctly. Absolute coordinates stay in the code (memory,
        planner, world model); the policy LLM sees only what is relative to it."""
        ego = egocentric_summary(obs)
        rules = (
            f"You control an agent in a {env_config.size}x{env_config.size} grid room with walls "
            f"around it. A goal square gives reward 1; stepping onto it ends the episode. "
            f"You can only see the area in front of you. Actions: left (turn 90 degrees left, "
            f"no movement), right (turn 90 degrees right, no movement), forward (move one cell "
            f"ahead; a wall blocks it and you stay put). You have {env_config.step_budget} "
            f"steps per episode."
        )
        return (
            f"{rules}\n"
            f"Steps used: {obs.steps_used}.\n"
            f"Directly ahead of you: {ego['ahead']}.\n"
            f"On your left-hand side: {ego['left']}. On your right-hand side: {ego['right']}.\n"
            f"Goal: {ego['goal']}.\n"
            f"Reply with exactly one action word: left, right, or forward."
        )

    def make_world_model(self, client: LLMClient, env_config: GridworldConfig):
        return MiniGridWorldModel(client, env_config)


GRIDWORLD = GridworldSpec()
MINIGRID = MiniGridSpec()
SPECS = {"gridworld": GRIDWORLD, "minigrid": MINIGRID}


def spec_for(kind: str) -> EnvSpec:
    try:
        return SPECS[kind]
    except KeyError:
        raise ValueError(f"unknown env_kind {kind!r}; known: {sorted(SPECS)}") from None
