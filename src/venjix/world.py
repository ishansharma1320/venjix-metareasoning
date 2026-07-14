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


@dataclass(frozen=True)
class Prediction:
    next_pos: tuple[int, int]
    reward: int
    parse_error: bool


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
        response = self.client.complete(prompt)

        next_match = _NEXT_RE.search(response.text)
        reward_match = _REWARD_RE.search(response.text)
        if next_match is None or reward_match is None:
            return Prediction(next_pos=pos, reward=0, parse_error=True)
        next_pos = (int(next_match.group(1)), int(next_match.group(2)))
        if not (0 <= next_pos[0] < self.size and 0 <= next_pos[1] < self.size):
            return Prediction(next_pos=pos, reward=0, parse_error=True)
        return Prediction(
            next_pos=next_pos, reward=int(reward_match.group(1)), parse_error=False
        )
