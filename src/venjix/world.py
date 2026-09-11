"""Prompted-LLM world model (Design decision 3: simulate = k-step rollouts).

One LLM call per simulated step — rollout cost flows through the client's
cumulative usage counters and lands in the step's cost_usd automatically
(Design decision 1). Richer world models (fine-tuned/distilled) are parked in
FUTURE.md.

The prompt uses labeled fields so both real models and MockModel's PREDICT
branch can answer it robustly. On an unparseable reply the fallback is
stay-in-place with reward 0, so a garbage model can never fabricate progress.
"""

import re
from dataclasses import dataclass

from venjix.config import GridworldConfig
from venjix.llm import PREDICT_MARKER, LLMClient

_NEXT_RE = re.compile(r"NEXT:\s*\((\d+)\s*,\s*(\d+)\)")
_REWARD_RE = re.compile(r"REWARD:\s*([01])")

# Constrained-decoding pattern for vLLM guided_regex: the reply can ONLY be the
# exact NEXT/REWARD shape the parser accepts. Kept in lockstep with the parser
# regexes above.
PREDICT_RESPONSE_REGEX = r"NEXT: \([0-9]{1,2}, [0-9]{1,2}\) REWARD: [01]"


@dataclass(frozen=True)
class Prediction:
    next_pos: tuple[int, int]
    reward: int
    parse_error: bool
    # None when parsed cleanly; "extraction" = reply didn't match the format;
    # "out_of_range" = matched but named a cell outside the grid.
    error_kind: str | None = None
    raw_text: str = ""  # for diagnostics (calibration probe); agents ignore it
    next_dir: str | None = None  # MiniGrid facing after the action; None for Phase 1

    @property
    def next_state(self):
        """What the signal compares against EnvSpec.state(obs)."""
        return self.next_pos if self.next_dir is None else (self.next_pos, self.next_dir)


def build_predict_prompt(
    size: int,
    pos: tuple[int, int],
    action: str,
    believed_goal: tuple[int, int] | None,
) -> str:
    goal_text = str(believed_goal) if believed_goal is not None else "unknown"
    return (
        f"{PREDICT_MARKER} the next state on a {size}x{size} grid.\n"
        f"Rules: positions are (row, col) with (0, 0) top-left; up/down change the "
        f"row, left/right the column; moves off the edge stay in place; probe does "
        f"not move; reward is 1 only when the next position is the goal, else 0.\n"
        f"GRID: {size}\n"
        f"POSITION: {pos}\n"
        f"ACTION: {action}\n"
        f"BELIEVED_GOAL: {goal_text}\n"
        f"Reply exactly in the form: NEXT: (row, col) REWARD: 0 or 1"
    )


class WorldModel:
    def __init__(self, client: LLMClient, env_config: GridworldConfig):
        self.client = client
        self.size = env_config.size

    def predict(
        self,
        pos: tuple[int, int],
        action: str,
        believed_goal: tuple[int, int] | None,
    ) -> Prediction:
        prompt = build_predict_prompt(self.size, pos, action, believed_goal)
        response = self.client.complete(prompt, response_regex=PREDICT_RESPONSE_REGEX)

        next_match = _NEXT_RE.search(response.text)
        reward_match = _REWARD_RE.search(response.text)
        if next_match is None or reward_match is None:
            return Prediction(
                next_pos=pos, reward=0, parse_error=True,
                error_kind="extraction", raw_text=response.text,
            )
        next_pos = (int(next_match.group(1)), int(next_match.group(2)))
        if not (0 <= next_pos[0] < self.size and 0 <= next_pos[1] < self.size):
            return Prediction(
                next_pos=pos, reward=0, parse_error=True,
                error_kind="out_of_range", raw_text=response.text,
            )
        return Prediction(
            next_pos=next_pos,
            reward=int(reward_match.group(1)),
            parse_error=False,
            raw_text=response.text,
        )


# --------------------------------------------------------------------------- #
# MiniGrid (exp-v3): egocentric state = (position, facing)
# --------------------------------------------------------------------------- #

_FACING_RE = re.compile(r"FACING:\s*(right|down|left|up)", re.IGNORECASE)
MINIGRID_DIRS = ("right", "down", "left", "up")

# Constrained-decoding pattern (structured_outputs) for the MiniGrid world model.
MINIGRID_PREDICT_RESPONSE_REGEX = (
    r"NEXT: \([0-9]{1,2}, [0-9]{1,2}\) FACING: (right|down|left|up) REWARD: [01]"
)


def build_minigrid_predict_prompt(
    size: int,
    pos: tuple[int, int],
    direction: str,
    action: str,
    believed_goal: tuple[int, int] | None,
) -> str:
    goal_text = str(believed_goal) if believed_goal is not None else "unknown"
    return (
        f"{PREDICT_MARKER} the next state in a {size}x{size} MiniGrid room.\n"
        f"Rules: coordinates are (x, y); walkable cells have x and y from 1 to {size - 2}, "
        f"the outer ring is wall; facing is one of right, down, left, up; the action "
        f"left turns 90 degrees left and right turns 90 degrees right without moving "
        f"(right -> down -> left -> up -> right when turning right); forward moves one "
        f"cell in the facing direction unless a wall blocks it, in which case the "
        f"position stays; reward is 1 only when the next position is the goal, else 0.\n"
        f"GRID: {size}\n"
        f"POSITION: {pos}\n"
        f"FACING: {direction}\n"
        f"ACTION: {action}\n"
        f"BELIEVED_GOAL: {goal_text}\n"
        f"Reply exactly in the form: NEXT: (x, y) FACING: direction REWARD: 0 or 1"
    )


class MiniGridWorldModel:
    """Same contract as WorldModel, over (pos, facing) states. Fallback on an
    unparseable reply: stay in place, keep facing, reward 0."""

    def __init__(self, client: LLMClient, env_config: GridworldConfig):
        self.client = client
        self.size = env_config.size

    def predict(self, state, action: str, believed_goal: tuple[int, int] | None) -> Prediction:
        pos, direction = state
        prompt = build_minigrid_predict_prompt(self.size, pos, direction, action, believed_goal)
        response = self.client.complete(prompt, response_regex=MINIGRID_PREDICT_RESPONSE_REGEX)
        next_match = _NEXT_RE.search(response.text)
        facing_match = _FACING_RE.search(response.text)
        reward_match = _REWARD_RE.search(response.text)
        if next_match is None or facing_match is None or reward_match is None:
            return Prediction(next_pos=pos, next_dir=direction, reward=0, parse_error=True,
                              error_kind="extraction", raw_text=response.text)
        next_pos = (int(next_match.group(1)), int(next_match.group(2)))
        if not (1 <= next_pos[0] <= self.size - 2 and 1 <= next_pos[1] <= self.size - 2):
            return Prediction(next_pos=pos, next_dir=direction, reward=0, parse_error=True,
                              error_kind="out_of_range", raw_text=response.text)
        return Prediction(next_pos=next_pos, next_dir=facing_match.group(1).lower(),
                          reward=int(reward_match.group(1)), parse_error=False,
                          raw_text=response.text)
