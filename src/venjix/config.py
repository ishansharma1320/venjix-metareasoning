"""Configuration dataclasses. Frozen so a config can never drift mid-run."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GridworldConfig:
    size: int = 11
    start: tuple[int, int] = (0, 0)
    probe_radius: int = 3
    # 0 means "use the default of 4 * size"; frozen dataclasses can't compute
    # defaults from other fields.
    step_budget: int = 0
    # None = draw the initial goal from the seeded RNG. Explicit placement is for
    # tests and controlled experiments.
    goal: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.step_budget == 0:
            object.__setattr__(self, "step_budget", 4 * self.size)
        if self.size < 2:
            raise ValueError("grid must have at least 2 cells so goal != start")
        for name, cell in (("start", self.start), ("goal", self.goal)):
            if cell is not None and not (
                0 <= cell[0] < self.size and 0 <= cell[1] < self.size
            ):
                raise ValueError(f"{name} {cell} outside {self.size}x{self.size} grid")
        if self.goal is not None and self.goal == self.start:
            raise ValueError("goal must differ from start")


@dataclass(frozen=True)
class ShiftEntry:
    at_step: int  # global env step count at which the shift fires
    distance: int  # requested Manhattan relocation distance


@dataclass(frozen=True)
class ShiftSchedule:
    version: str
    entries: tuple[ShiftEntry, ...] = field(default=())

    def __post_init__(self) -> None:
        steps = [e.at_step for e in self.entries]
        if steps != sorted(steps) or len(steps) != len(set(steps)):
            raise ValueError("shift entries must be sorted by at_step and unique")
        for e in self.entries:
            if e.distance < 1:
                raise ValueError(f"shift distance must be >= 1, got {e.distance}")
