"""Configuration dataclasses. Frozen so a config can never drift mid-run."""

import hashlib
import json
from dataclasses import asdict, dataclass, field


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


@dataclass(frozen=True)
class PriceTable:
    """Synthetic price table (Design decision 1). Defaults are claude-haiku-4-5's
    real API prices as of July 2026. The same table prices the mock model so
    mock-mode comparisons stay meaningful. Wall time is logged, never priced."""

    input_per_mtok_usd: float = 1.00
    output_per_mtok_usd: float = 5.00

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok_usd
            + output_tokens * self.output_per_mtok_usd
        ) / 1_000_000


AGENT_TYPES = ("reactive", "retrieve", "simulate", "mixture")


@dataclass(frozen=True)
class RunConfig:
    env: GridworldConfig
    schedule: ShiftSchedule
    seed: int
    n_episodes: int
    model: str = "claude-haiku-4-5"
    prices: PriceTable = PriceTable()
    agent: str = "reactive"
    # weights over (act, retrieve, simulate, gather_evidence); mixture only.
    # None with agent="mixture" resolves to uniform.
    mixture_weights: tuple[float, float, float, float] | None = None
    sim_depth: int = 3

    def __post_init__(self) -> None:
        if self.agent not in AGENT_TYPES:
            raise ValueError(f"agent must be one of {AGENT_TYPES}, got {self.agent!r}")
        if self.sim_depth < 1:
            raise ValueError(f"sim_depth must be >= 1, got {self.sim_depth}")
        if self.agent == "mixture":
            if self.mixture_weights is None:
                object.__setattr__(self, "mixture_weights", (0.25, 0.25, 0.25, 0.25))
            weights = self.mixture_weights
            if len(weights) != 4 or any(w < 0 for w in weights):
                raise ValueError("mixture_weights must be 4 non-negative floats")
            if abs(sum(weights) - 1.0) > 1e-9:
                raise ValueError(f"mixture_weights must sum to 1, got {sum(weights)}")
        elif self.mixture_weights is not None:
            raise ValueError("mixture_weights only apply to agent='mixture'")

    def to_dict(self) -> dict:
        return asdict(self)

    def config_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
