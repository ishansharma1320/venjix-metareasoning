"""Toy nonstationary gridworld.

Coordinates are (row, col); (0, 0) is the top-left. Actions "up"/"down" change the
row, "left"/"right" the column; moves off the edge clip in place. "probe" is the
gather_evidence primitive: it consumes a step, never moves the agent, and reveals
the goal's offset only within `probe_radius` (Manhattan).

The goal is hidden and persists across episode resets — it moves ONLY via
`relocate_goal`, so scheduled shifts are the sole source of nonstationarity.
Observations never mention the goal or a shift; shifts are silent by construction.

Observation is a foundational interface (see docs/CLAUDE.md): agents, the world
model, and the episode logger all build on it. Changes here must be flagged loudly.
"""

import random
from dataclasses import dataclass

from venjix.config import GridworldConfig

ACTIONS = ("up", "down", "left", "right", "probe")
_MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

OUT_OF_RANGE = "out_of_range"


@dataclass(frozen=True)
class Observation:
    pos: tuple[int, int]
    reward: int
    done: bool
    success: bool
    probe_result: tuple[int, int] | str | None  # offset | OUT_OF_RANGE | None
    steps_used: int


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class Gridworld:
    def __init__(self, config: GridworldConfig, seed: int):
        self.config = config
        self._rng = random.Random(seed)
        # Dedicated relocation stream (Amendment 6c): shift draws must be
        # independent of agent state so goal sequences are PAIRED across agents
        # for the same (config, seed).
        self._shift_rng = random.Random(f"{seed}-shifts")
        self._pos = config.start
        self._steps_used = 0
        self._done = True  # must reset() before step()
        self._goal = self._draw_initial_goal()

    def _draw_initial_goal(self) -> tuple[int, int]:
        if self.config.goal is not None:
            return self.config.goal
        size = self.config.size
        cells = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if (r, c) != self.config.start
        ]
        return self._rng.choice(cells)

    @property
    def goal(self) -> tuple[int, int]:
        """For the scheduler, logger, and tests only — agents must not read this."""
        return self._goal

    def reset(self) -> Observation:
        self._pos = self.config.start
        self._steps_used = 0
        self._done = False
        return Observation(
            pos=self._pos,
            reward=0,
            done=False,
            success=False,
            probe_result=None,
            steps_used=0,
        )

    def step(self, action: str) -> Observation:
        if self._done:
            raise RuntimeError("episode is done; call reset() first")
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")

        self._steps_used += 1
        probe_result: tuple[int, int] | str | None = None
        if action == "probe":
            offset = (self._goal[0] - self._pos[0], self._goal[1] - self._pos[1])
            in_range = _manhattan(self._pos, self._goal) <= self.config.probe_radius
            probe_result = offset if in_range else OUT_OF_RANGE
        else:
            dr, dc = _MOVES[action]
            last = self.config.size - 1
            self._pos = (
                min(max(self._pos[0] + dr, 0), last),
                min(max(self._pos[1] + dc, 0), last),
            )

        success = self._pos == self._goal
        self._done = success or self._steps_used >= self.config.step_budget
        return Observation(
            pos=self._pos,
            reward=1 if success else 0,
            done=self._done,
            success=success,
            probe_result=probe_result,
            steps_used=self._steps_used,
        )

    def relocate_goal(self, distance: int) -> tuple[tuple[int, int], tuple[int, int], int]:
        """Move the goal ~`distance` Manhattan steps; returns (old, new, actual_distance).

        Samples from the dedicated shift RNG (Amendment 6c), uniformly over
        in-grid cells at exactly `distance` from the current goal, excluding
        only the old goal and the start cell — never anything derived from
        agent state, so identical (config, seed, schedule) yields identical
        goal sequences for every agent. (The goal may therefore land on the
        agent's current cell; rare, and symmetric across paired runs.) If the
        exact ring is empty (corner + oversized distance), falls back to the
        cells whose distance is closest to the request — for an oversized
        request that is the maximum achievable distance. The actual distance is
        returned so the caller can log it; nothing about the shift enters
        observations.
        """
        if distance < 1:
            raise ValueError(f"relocation distance must be >= 1, got {distance}")
        size = self.config.size
        excluded = {self._goal, self.config.start}
        candidates = [
            (r, c)
            for r in range(size)
            for c in range(size)
            if (r, c) not in excluded
        ]
        best_gap = min(abs(_manhattan(self._goal, cell) - distance) for cell in candidates)
        ring = [
            cell
            for cell in candidates
            if abs(_manhattan(self._goal, cell) - distance) == best_gap
        ]
        old_goal = self._goal
        self._goal = self._shift_rng.choice(ring)
        return old_goal, self._goal, _manhattan(old_goal, self._goal)
