"""Append-only episodic log (Design decision 3).

The log persists across episode resets for the whole run — the goal is stable
until a silent shift, so remembered evidence goes stale exactly when the study
needs it to. NO decay, merging, or consolidation: memory *dynamics* are parked
in FUTURE.md; only this dumb log is in scope.

`believed_goal()` is the concrete instantiation of "nearest-match lookup" in
this env: sparse reward makes generic nearest-state matching degenerate, so the
lookup specializes to the most recent entry carrying direct goal evidence — a
reward hit, or a probe offset.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Experience:
    pos: tuple[int, int]
    action: str
    next_pos: tuple[int, int]
    reward: int
    probe_result: tuple[int, int] | str | None


class EpisodicLog:
    def __init__(self) -> None:
        self._entries: list[Experience] = []

    def append(self, experience: Experience) -> None:
        self._entries.append(experience)

    @property
    def entries(self) -> tuple[Experience, ...]:
        return tuple(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def believed_goal(self) -> tuple[int, int] | None:
        """Most recent direct goal evidence, or None on cold start."""
        for exp in reversed(self._entries):
            if exp.reward == 1:
                return exp.next_pos
            if isinstance(exp.probe_result, tuple):
                # probe offset is relative to where the agent stood when probing
                return (exp.pos[0] + exp.probe_result[0], exp.pos[1] + exp.probe_result[1])
        return None
