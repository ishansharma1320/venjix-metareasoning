"""Agents. The Decision shape is foundational (see docs/CLAUDE.md): every
baseline and the arbiter log through it, so `mode` exists from day one.

`mode` names the decision procedure (Design decision 5: chosen per step);
`action` is the env primitive. A reactive agent that answers "probe" is still
mode="act" — gather_evidence as a *mode* arrives with the later baselines.
"""

import re
from dataclasses import dataclass

from venjix.config import GridworldConfig
from venjix.gridworld import ACTIONS, Observation
from venjix.llm import LLMClient

_ACTION_RE = re.compile(r"\b(" + "|".join(ACTIONS) + r")\b", re.IGNORECASE)

# Deterministic fallback for unparseable responses: probe never makes task
# progress, so parser failures can't fake success, and they stay visible in the
# action stats alongside the parse_error flag.
FALLBACK_ACTION = "probe"


@dataclass(frozen=True)
class Decision:
    mode: str
    action: str
    parse_error: bool


def parse_action(text: str) -> tuple[str, bool]:
    match = _ACTION_RE.search(text)
    if match is None:
        return FALLBACK_ACTION, True
    return match.group(1).lower(), False


class ReactiveAgent:
    """One cheap policy call on the current observation (Design decision 3).
    No history, no memory — that is the point of this baseline."""

    def __init__(self, client: LLMClient, env_config: GridworldConfig):
        self.client = client
        self._rules = (
            f"You control an agent on a {env_config.size}x{env_config.size} grid. "
            f"A hidden goal cell gives reward 1; reaching it ends the episode. "
            f"Positions are (row, col) with (0, 0) at the top-left. "
            f"Actions: up, down, left, right (move one cell), or probe "
            f"(reveals the goal's offset if it is within Manhattan distance "
            f"{env_config.probe_radius}, costs one step, no movement). "
            f"You have {env_config.step_budget} steps per episode."
        )

    def choose(self, obs: Observation) -> Decision:
        prompt = (
            f"{self._rules}\n"
            f"Current position: {obs.pos}. Steps used: {obs.steps_used}.\n"
            f"Last probe result: {obs.probe_result}.\n"
            f"Reply with exactly one action word: up, down, left, right, or probe."
        )
        response = self.client.complete(prompt)
        action, parse_error = parse_action(response.text)
        return Decision(mode="act", action=action, parse_error=parse_error)
