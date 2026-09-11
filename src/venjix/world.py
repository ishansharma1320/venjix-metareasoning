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

MINIGRID_DIRS = ("right", "down", "left", "up")
_MG_DELTA = {"right": (1, 0), "down": (0, 1), "left": (-1, 0), "up": (0, -1)}
_MOVED_RE = re.compile(r"MOVED:\s*(yes|no)", re.IGNORECASE)

# Constrained-decoding pattern (structured_outputs) for the MiniGrid world model.
MINIGRID_PREDICT_RESPONSE_REGEX = r"MOVED: (yes|no) REWARD: [01]"


def minigrid_turn(direction: str, action: str) -> str:
    i = MINIGRID_DIRS.index(direction)
    if action == "left":
        return MINIGRID_DIRS[(i - 1) % 4]
    if action == "right":
        return MINIGRID_DIRS[(i + 1) % 4]
    return direction


def egocentric_offset(direction: str, dx: int, dy: int) -> tuple[int, int]:
    """World (dx, dy) -> (ahead, right) in the agent's frame."""
    if direction == "right":
        return dx, dy
    if direction == "down":
        return dy, -dx
    if direction == "left":
        return -dx, -dy
    return -dy, dx


def describe_goal_offset(ahead: int, right: int) -> str:
    if ahead == 0 and right == 0:
        return "right here"
    parts = []
    if ahead > 0:
        parts.append(f"{ahead} cell{'s' if ahead != 1 else ''} ahead")
    elif ahead < 0:
        parts.append(f"{-ahead} cell{'s' if -ahead != 1 else ''} behind")
    if right > 0:
        parts.append(f"{right} cell{'s' if right != 1 else ''} to your right")
    elif right < 0:
        parts.append(f"{-right} cell{'s' if -right != 1 else ''} to your left")
    return " and ".join(parts)


def build_minigrid_predict_prompt(
    size: int,
    pos: tuple[int, int],
    direction: str,
    action: str,
    believed_goal: tuple[int, int] | None,
) -> str:
    """EGOCENTRIC world-model prompt (exp-v3 PLAN §4e). The code converts the
    absolute state into agent-frame facts (what is directly ahead, where the
    believed goal is relative to the agent); the model predicts only what carries
    information for the signal — whether `forward` moves, and the reward."""
    dx, dy = _MG_DELTA[direction]
    nx, ny = pos[0] + dx, pos[1] + dy
    ahead = "floor" if (1 <= nx <= size - 2 and 1 <= ny <= size - 2) else "a wall"
    if believed_goal is None:
        goal_text = "unknown"
    else:
        a, r = egocentric_offset(direction, believed_goal[0] - pos[0], believed_goal[1] - pos[1])
        goal_text = describe_goal_offset(a, r)
    return (
        f"{PREDICT_MARKER} the outcome of one action in a grid room with walls around it.\n"
        f"Rules: left and right turn you 90 degrees in place (no movement, reward 0); "
        f"forward moves you one cell ahead unless a wall is directly ahead, in which "
        f"case you do not move; reward is 1 only if you end the action standing on the "
        f"goal, else 0.\n"
        f"ACTION: {action}\n"
        f"DIRECTLY_AHEAD: {ahead}\n"
        f"BELIEVED_GOAL: {goal_text}\n"
        f"Reply exactly in the form: MOVED: yes or no REWARD: 0 or 1"
    )


class MiniGridWorldModel:
    """Same contract as WorldModel, over (pos, facing) states. The LLM answers
    MOVED / REWARD from egocentric facts; the code applies the deterministic
    turn and the move. Fallback on an unparseable reply: stay in place, keep
    facing, reward 0."""

    def __init__(self, client: LLMClient, env_config: GridworldConfig):
        self.client = client
        self.size = env_config.size

    def predict(self, state, action: str, believed_goal: tuple[int, int] | None) -> Prediction:
        pos, direction = state
        prompt = build_minigrid_predict_prompt(self.size, pos, direction, action, believed_goal)
        response = self.client.complete(prompt, response_regex=MINIGRID_PREDICT_RESPONSE_REGEX)
        moved_match = _MOVED_RE.search(response.text)
        reward_match = _REWARD_RE.search(response.text)
        next_dir = minigrid_turn(direction, action)
        if moved_match is None or reward_match is None:
            return Prediction(next_pos=pos, next_dir=next_dir, reward=0, parse_error=True,
                              error_kind="extraction", raw_text=response.text)
        moved = moved_match.group(1).lower() == "yes" and action == "forward"
        dx, dy = _MG_DELTA[direction]
        next_pos = (pos[0] + dx, pos[1] + dy) if moved else pos
        if not (1 <= next_pos[0] <= self.size - 2 and 1 <= next_pos[1] <= self.size - 2):
            # the model said "moved" into the wall ring: score it as a wrong
            # position prediction, never as a crash
            next_pos = (min(max(next_pos[0], 1), self.size - 2), min(max(next_pos[1], 1), self.size - 2))
        return Prediction(next_pos=next_pos, next_dir=next_dir,
                          reward=int(reward_match.group(1)), parse_error=False,
                          raw_text=response.text)
