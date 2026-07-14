"""Shift scheduler: fires silent goal relocations at configured global step counts.

The run loop owns the scheduler and the global step counter, which persists across
episode resets — that is what makes mid-episode shifts possible and episode-boundary
schedules just a special case of where the shift steps land.
"""

from dataclasses import dataclass

from venjix.config import ShiftSchedule
from venjix.gridworld import Gridworld


@dataclass(frozen=True)
class ShiftRecord:
    at_step: int
    requested_distance: int
    actual_distance: int
    old_goal: tuple[int, int]
    new_goal: tuple[int, int]


class ShiftScheduler:
    def __init__(self, schedule: ShiftSchedule):
        self.schedule = schedule
        self._next_index = 0

    def maybe_shift(self, global_step: int, env: Gridworld) -> list[ShiftRecord]:
        """Fire every not-yet-fired entry with at_step <= global_step.

        Call once after each env step with the cumulative step count. With unit
        increments at most one entry fires per call; firing all due entries keeps
        the scheduler correct even if a caller skips steps.
        """
        records = []
        while (
            self._next_index < len(self.schedule.entries)
            and self.schedule.entries[self._next_index].at_step <= global_step
        ):
            entry = self.schedule.entries[self._next_index]
            old_goal, new_goal, actual = env.relocate_goal(entry.distance)
            records.append(
                ShiftRecord(
                    at_step=entry.at_step,
                    requested_distance=entry.distance,
                    actual_distance=actual,
                    old_goal=old_goal,
                    new_goal=new_goal,
                )
            )
            self._next_index += 1
        return records
