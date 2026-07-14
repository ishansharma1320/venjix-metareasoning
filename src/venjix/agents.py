"""Agents. The Decision shape is foundational (see docs/CLAUDE.md): every
baseline and the arbiter log through it, so `mode` exists from day one.

`mode` names the decision procedure (Design decision 5: chosen per step);
`action` is the env primitive. A reactive agent that answers "probe" is still
mode="act" — gather_evidence as a *mode* arrives with the later baselines.
"""

import random
import re
from dataclasses import dataclass

from venjix.config import GridworldConfig
from venjix.gridworld import ACTIONS, Observation
from venjix.llm import LLMClient
from venjix.memory import EpisodicLog, Experience
from venjix.world import WorldModel

MOVE_ACTIONS = ("up", "down", "left", "right")
MIXTURE_MODES = ("act", "retrieve", "simulate", "gather_evidence")

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


class Agent:
    """FOUNDATIONAL INTERFACE (see docs/CLAUDE.md). The runner calls
    choose -> env.step -> observe on every step. `observe` is a no-op by
    default; memory-bearing agents override it to record outcomes."""

    def choose(self, obs: Observation) -> Decision:
        raise NotImplementedError

    def observe(self, prev_obs: Observation, action: str, obs: Observation) -> None:
        return None


def greedy_step(pos: tuple[int, int], goal: tuple[int, int]) -> str | None:
    """Deterministic step toward goal: close the row gap first, then the
    column. None when already on the goal cell."""
    if pos[0] != goal[0]:
        return "down" if pos[0] < goal[0] else "up"
    if pos[1] != goal[1]:
        return "right" if pos[1] < goal[1] else "left"
    return None


def parse_action(text: str) -> tuple[str, bool]:
    match = _ACTION_RE.search(text)
    if match is None:
        return FALLBACK_ACTION, True
    return match.group(1).lower(), False


class ReactiveAgent(Agent):
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


class RetrieveOnlyAgent(Agent):
    """Nearest-match lookup over the episodic log (Design decision 3). Zero LLM
    calls, so its per-step dollar cost is 0 — cheap but brittle under shift,
    which is exactly its role in the study. Cold start (or standing on a stale
    believed goal that paid nothing): seeded random walk."""

    def __init__(self, seed: int, memory: EpisodicLog | None = None):
        self.memory = memory if memory is not None else EpisodicLog()
        self._rng = random.Random(seed)

    def choose(self, obs: Observation) -> Decision:
        goal = self.memory.believed_goal()
        action = greedy_step(obs.pos, goal) if goal is not None else None
        if action is None:
            action = self._rng.choice(MOVE_ACTIONS)
        return Decision(mode="retrieve", action=action, parse_error=False)

    def observe(self, prev_obs: Observation, action: str, obs: Observation) -> None:
        self.memory.append(
            Experience(prev_obs.pos, action, obs.pos, obs.reward, obs.probe_result)
        )


class SimulateOnlyAgent(Agent):
    """k-step world-model rollouts over the 4 move candidates before committing
    (Design decision 3). Every simulated step is one LLM call, metered into the
    env step's cost by the runner's counter snapshot. With no believed goal the
    rollouts predict no reward anywhere and the agent falls back to a seeded
    random walk — while still paying for its rollout calls (honest accounting
    of a mode that simulates without evidence)."""

    def __init__(
        self,
        client: LLMClient,
        env_config: GridworldConfig,
        seed: int,
        sim_depth: int = 3,
        memory: EpisodicLog | None = None,
    ):
        self.world = WorldModel(client, env_config)
        self.memory = memory if memory is not None else EpisodicLog()
        self.sim_depth = sim_depth
        self._rng = random.Random(seed)

    def choose(self, obs: Observation) -> Decision:
        goal = self.memory.believed_goal()
        best_action: str | None = None
        best_steps: int | None = None
        for candidate in MOVE_ACTIONS:
            steps = self._rollout(obs.pos, candidate, goal)
            if steps is not None and (best_steps is None or steps < best_steps):
                best_action, best_steps = candidate, steps
        if best_action is None:
            best_action = self._rng.choice(MOVE_ACTIONS)
        return Decision(mode="simulate", action=best_action, parse_error=False)

    def _rollout(
        self,
        pos: tuple[int, int],
        first_action: str,
        goal: tuple[int, int] | None,
    ) -> int | None:
        """Predicted number of steps to reward (<= sim_depth), or None.
        Rollout policy after the first action: greedy toward the believed goal;
        with no goal, keep heading the same way."""
        action = first_action
        for depth in range(1, self.sim_depth + 1):
            prediction = self.world.predict(pos, action, goal)
            if prediction.reward == 1:
                return depth
            pos = prediction.next_pos
            if goal is not None:
                action = greedy_step(pos, goal) or action
        return None

    def observe(self, prev_obs: Observation, action: str, obs: Observation) -> None:
        self.memory.append(
            Experience(prev_obs.pos, action, obs.pos, obs.reward, obs.probe_result)
        )


class FixedMixtureAgent(Agent):
    """Static mode proportions, no adaptivity: an i.i.d. per-step draw over
    (act, retrieve, simulate, gather_evidence) — Design decision 5. All modes
    share one episodic log; gather_evidence emits the probe action."""

    def __init__(
        self,
        client: LLMClient,
        env_config: GridworldConfig,
        seed: int,
        weights: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
        sim_depth: int = 3,
    ):
        self.weights = weights
        self.memory = EpisodicLog()
        self._act = ReactiveAgent(client, env_config)
        self._retrieve = RetrieveOnlyAgent(seed, memory=self.memory)
        self._simulate = SimulateOnlyAgent(
            client, env_config, seed, sim_depth, memory=self.memory
        )
        self._rng = random.Random(seed)

    def choose(self, obs: Observation) -> Decision:
        mode = self._rng.choices(MIXTURE_MODES, weights=self.weights)[0]
        if mode == "act":
            return self._act.choose(obs)
        if mode == "retrieve":
            return self._retrieve.choose(obs)
        if mode == "simulate":
            return self._simulate.choose(obs)
        return Decision(mode="gather_evidence", action="probe", parse_error=False)

    def observe(self, prev_obs: Observation, action: str, obs: Observation) -> None:
        self.memory.append(
            Experience(prev_obs.pos, action, obs.pos, obs.reward, obs.probe_result)
        )
