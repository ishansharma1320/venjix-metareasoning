"""MiniGrid-Empty adapter (exp-v3 plan §2).

Wraps `minigrid.envs.EmptyEnv` behind the same reset/step/relocate_goal/goal
surface as the Phase 1 Gridworld so the runner, scheduler, and logger do not
change. Differences, all deliberate:

- The goal is OURS, not MiniGrid's: EmptyEnv regenerates its grid on every
  reset with the goal in the bottom-right corner; we clear that and place the
  goal at our persisted position (uniform over walkable cells at construction,
  excluding the start; moved ONLY by relocate_goal — the shift is the sole
  nonstationarity, exactly as Phase 1).
- Partial observability replaces the probe: `probe_result` carries the goal's
  (dx, dy) offset whenever the goal is inside the agent's view cone, else None.
  The EpisodicLog reads it unchanged (offset relative to the agent's position).
- Coordinates are MiniGrid's (x, y): x = column, y = row, (1, 1) the top-left
  walkable cell; the outer ring is wall. Facing is one of right/down/left/up.
- Success is binary "stepped onto the goal"; MiniGrid's shaped reward is ignored.

`minigrid` / `gymnasium` are imported lazily so the core package still imports
without them (dependency group `minigrid`).
"""

import random

from venjix.config import GridworldConfig
from venjix.gridworld import Observation, _manhattan

START = (1, 1)   # EmptyEnv's default agent start; facing right (dir 0)
VIEW = 7         # MiniGrid default agent_view_size; the plan fixes it

_GLYPH = {0: "?", 1: ".", 2: "#", 3: ".", 8: "G", 10: "A"}  # OBJECT_TO_IDX subset


def render_view(image) -> str:
    """ASCII rows of the (VIEW, VIEW, 3) egocentric image: top row = farthest,
    the agent at the bottom-center cell."""
    rows = []
    for j in range(image.shape[1]):
        row = []
        for i in range(image.shape[0]):
            row.append("A" if (i == VIEW // 2 and j == VIEW - 1) else _GLYPH.get(int(image[i, j, 0]), "?"))
        rows.append("".join(row))
    return "\n".join(rows)


class MiniGridEmpty:
    def __init__(self, config: GridworldConfig, seed: int):
        from minigrid.core.actions import Actions
        from minigrid.core.world_object import Goal
        from minigrid.envs import EmptyEnv

        self._Actions = Actions
        self._Goal = Goal
        self.config = config
        self.size = config.size
        self._seed = seed
        self._rng = random.Random(seed)
        # Dedicated relocation stream (Amendment 6c): identical (config, seed,
        # schedule) -> identical goal sequences for every agent.
        self._shift_rng = random.Random(f"{seed}-shifts")
        self._env = EmptyEnv(size=config.size, max_steps=config.step_budget,
                             agent_view_size=VIEW, render_mode=None)
        self._env.reset(seed=seed)
        self._goal = self._rng.choice(self._walkable(exclude={START}))
        self._done = True
        self._episode = 0

    # --- helpers ---------------------------------------------------------
    def _walkable(self, exclude: set) -> list[tuple[int, int]]:
        n = self.size
        return [(x, y) for x in range(1, n - 1) for y in range(1, n - 1) if (x, y) not in exclude]

    def _place_goal(self) -> None:
        grid = self._env.grid
        for x in range(self.size):
            for y in range(self.size):
                if isinstance(grid.get(x, y), self._Goal):
                    grid.set(x, y, None)
        grid.set(self._goal[0], self._goal[1], self._Goal())

    def _pos(self) -> tuple[int, int]:
        p = self._env.agent_pos
        return (int(p[0]), int(p[1]))

    def _dir(self) -> str:
        return ("right", "down", "left", "up")[int(self._env.agent_dir)]

    def _probe_result(self):
        gx, gy = self._goal
        if self._env.in_view(gx, gy):
            x, y = self._pos()
            return (gx - x, gy - y)
        return None

    def _observation(self, image, reward: int, done: bool, success: bool) -> Observation:
        return Observation(
            pos=self._pos(), reward=reward, done=done, success=success,
            probe_result=self._probe_result(), steps_used=int(self._env.step_count),
            dir=self._dir(), view=render_view(image),
        )

    # --- Gridworld-compatible surface -----------------------------------
    @property
    def goal(self) -> tuple[int, int]:
        """For the scheduler, logger, and tests only — agents must not read this."""
        return self._goal

    def reset(self) -> Observation:
        # Reseed per episode so MiniGrid's own RNG never introduces cross-agent
        # divergence (EmptyEnv is deterministic anyway; belt and braces).
        obs, _ = self._env.reset(seed=self._seed * 100_003 + self._episode)
        self._episode += 1
        self._place_goal()
        self._done = False
        # in_view depends on the freshly generated grid; recompute the image
        # after placing our goal so the first observation shows it if visible.
        obs = self._env.gen_obs()
        return self._observation(obs["image"], 0, False, False)

    def step(self, action: str) -> Observation:
        if self._done:
            raise RuntimeError("episode is done; call reset() first")
        if action not in ("left", "right", "forward"):
            raise ValueError(f"unknown action {action!r}")
        obs, reward, terminated, truncated, _ = self._env.step(getattr(self._Actions, action))
        success = bool(terminated) and self._pos() == self._goal
        self._done = bool(terminated) or bool(truncated)
        return self._observation(obs["image"], 1 if success else 0, self._done, success)

    def relocate_goal(self, distance: int) -> tuple[tuple[int, int], tuple[int, int], int]:
        """Identical algorithm to Gridworld.relocate_goal over walkable cells:
        uniform over cells at exactly `distance` (closest achievable ring
        otherwise), excluding only the old goal and the start cell."""
        if distance < 1:
            raise ValueError(f"relocation distance must be >= 1, got {distance}")
        candidates = self._walkable(exclude={self._goal, START})
        best_gap = min(abs(_manhattan(self._goal, c) - distance) for c in candidates)
        ring = [c for c in candidates if abs(_manhattan(self._goal, c) - distance) == best_gap]
        old = self._goal
        self._goal = self._shift_rng.choice(ring)
        self._place_goal()
        return old, self._goal, _manhattan(old, self._goal)
