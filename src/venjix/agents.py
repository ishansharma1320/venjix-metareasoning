"""Agents. The Decision shape is foundational (see docs/CLAUDE.md): every
baseline and the arbiter log through it, so `mode` exists from day one.

`mode` names the decision procedure (Design decision 5: chosen per step);
`action` is the env primitive. A reactive agent that answers "probe" is still
mode="act" — gather_evidence as a *mode* arrives with the later baselines.
"""

import random
import re
from dataclasses import dataclass

from venjix.bandit import LinUCB
from venjix.config import GridworldConfig, PriceTable
from venjix.gridworld import ACTIONS, Observation
from venjix.llm import LLMClient
from venjix.memory import EpisodicLog, Experience
from venjix.signal import EwmaPredictionError
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


def gather_action(last_action: str | None, rng: random.Random) -> str:
    """The gather_evidence policy shared by signal-bearing agents: probe unless
    the previous action was a probe (probing never moves — a pure-probe mode
    would pin the agent to one cell), else a seeded random move."""
    if last_action == "probe":
        return rng.choice(MOVE_ACTIONS)
    return "probe"


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


class ThresholdHeuristicAgent(Agent):
    """Rung 3 — the killer baseline. The ENTIRE arbitration logic:

        mode = "gather_evidence" if ewma > threshold
               else ("retrieve" if memory.believed_goal() else "act")

    Signal per Design decision 2: before acting, one world-model call predicts
    (next_pos, reward) for the chosen action; after acting, the misprediction
    is scored binarily and folded into the shared EWMA. That prediction call is
    the signal's honest per-step cost. probe_result is not part of the scored
    observation — the world model predicts position and reward only.

    gather_evidence policy: probe unless the previous action was a probe, else
    a seeded random move — a probe/move sweep, since probing never moves and a
    pure-probe mode would pin the agent to one cell.
    """

    def __init__(
        self,
        client: LLMClient,
        env_config: GridworldConfig,
        seed: int,
        ewma_alpha: float = 0.3,
        pe_threshold: float = 0.25,
    ):
        self.memory = EpisodicLog()
        self.world = WorldModel(client, env_config)
        self.signal = EwmaPredictionError(ewma_alpha)
        self.pe_threshold = pe_threshold
        self._act = ReactiveAgent(client, env_config)
        self._rng = random.Random(seed)
        self._last_action: str | None = None
        self._pending_prediction = None
        # Read by the runner after observe(); null until the first step lands.
        self.last_prediction_error: int | None = None
        self.signal_value: float | None = None

    def choose(self, obs: Observation) -> Decision:
        goal = self.memory.believed_goal()
        if self.signal.value > self.pe_threshold:
            mode = "gather_evidence"
        else:
            mode = "retrieve" if goal is not None else "act"

        parse_error = False
        if mode == "gather_evidence":
            action = gather_action(self._last_action, self._rng)
        elif mode == "retrieve":
            action = greedy_step(obs.pos, goal) or self._rng.choice(MOVE_ACTIONS)
        else:
            act_decision = self._act.choose(obs)
            action, parse_error = act_decision.action, act_decision.parse_error

        self._pending_prediction = self.world.predict(obs.pos, action, goal)
        return Decision(mode=mode, action=action, parse_error=parse_error)

    def observe(self, prev_obs: Observation, action: str, obs: Observation) -> None:
        self.memory.append(
            Experience(prev_obs.pos, action, obs.pos, obs.reward, obs.probe_result)
        )
        self._last_action = action
        prediction = self._pending_prediction
        self._pending_prediction = None
        if prediction is None:
            return
        mispredicted = (
            prediction.next_pos != obs.pos or prediction.reward != obs.reward
        )
        self.last_prediction_error = int(mispredicted)
        self.signal_value = self.signal.update(mispredicted)


class BanditArbiterAgent(Agent):
    """Rung 4 — LinUCB arbiter over the four modes.

    LOCKED context features (docs/CLAUDE.md Amendment 4, rung-4 plan):
    x = (1, ewma, has_belief) — exactly the inputs of the heuristic's fixed
    rule, nothing more, so any win is attributable to learning the mapping.

    Signal protocol and per-step signal cost are identical to the heuristic
    (same EwmaPredictionError class, one world-model predict per step). Bandit
    reward is r = env_reward - cost_weight * step_cost_usd, computed from the
    same client counters and price table the runner logs — the bandit optimizes
    the pre-registered metric, not raw success. Credit assignment is myopic by
    design (that is what makes it a bandit; RL arbiters are parked).
    """

    def __init__(
        self,
        client: LLMClient,
        env_config: GridworldConfig,
        seed: int,
        prices: PriceTable,
        sim_depth: int = 3,
        ewma_alpha: float = 0.3,
        ucb_alpha: float = 1.0,
        cost_weight: float = 100.0,
    ):
        self.client = client
        self.prices = prices
        self.cost_weight = cost_weight
        self.memory = EpisodicLog()
        self.world = WorldModel(client, env_config)
        self.signal = EwmaPredictionError(ewma_alpha)
        self.bandit = LinUCB(n_arms=len(MIXTURE_MODES), dim=3, ucb_alpha=ucb_alpha)
        self._act = ReactiveAgent(client, env_config)
        self._simulate = SimulateOnlyAgent(
            client, env_config, seed, sim_depth, memory=self.memory
        )
        self._rng = random.Random(seed)
        self._last_action: str | None = None
        self._pending = None  # (context, arm, prediction, input0, output0)
        self.last_prediction_error: int | None = None
        self.signal_value: float | None = None
        self.last_bandit_reward: float | None = None

    def choose(self, obs: Observation) -> Decision:
        input0 = self.client.total_input_tokens
        output0 = self.client.total_output_tokens
        goal = self.memory.believed_goal()
        context = (1.0, self.signal.value, 1.0 if goal is not None else 0.0)
        arm = self.bandit.select(context)
        mode = MIXTURE_MODES[arm]

        parse_error = False
        if mode == "act":
            act_decision = self._act.choose(obs)
            action, parse_error = act_decision.action, act_decision.parse_error
        elif mode == "retrieve":
            action = (greedy_step(obs.pos, goal) if goal is not None else None) or (
                self._rng.choice(MOVE_ACTIONS)
            )
        elif mode == "simulate":
            action = self._simulate.choose(obs).action
        else:
            action = gather_action(self._last_action, self._rng)

        prediction = self.world.predict(obs.pos, action, goal)
        self._pending = (context, arm, prediction, input0, output0)
        return Decision(mode=mode, action=action, parse_error=parse_error)

    def observe(self, prev_obs: Observation, action: str, obs: Observation) -> None:
        self.memory.append(
            Experience(prev_obs.pos, action, obs.pos, obs.reward, obs.probe_result)
        )
        self._last_action = action
        if self._pending is None:
            return
        context, arm, prediction, input0, output0 = self._pending
        self._pending = None
        mispredicted = (
            prediction.next_pos != obs.pos or prediction.reward != obs.reward
        )
        self.last_prediction_error = int(mispredicted)
        self.signal_value = self.signal.update(mispredicted)
        step_cost = self.prices.cost_usd(
            self.client.total_input_tokens - input0,
            self.client.total_output_tokens - output0,
        )
        self.last_bandit_reward = obs.reward - self.cost_weight * step_cost
        self.bandit.update(arm, context, self.last_bandit_reward)


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
