"""Thin LLM client interface (foundational — see docs/CLAUDE.md).

Every call flows through LLMClient.complete(), which maintains cumulative usage
counters. The runner meters per-step cost by snapshotting those counters around
each agent call, so agents that make many calls per step (simulate mode, later)
are accounted for without any schema change.

AnthropicModel uses stdlib urllib on purpose: it keeps the package
zero-dependency, and this machine's network blocks PyPI's file host anyway.
"""

import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

# World-model prediction prompts start with this marker (see world.py, which
# imports it — llm.py must not import world.py).
PREDICT_MARKER = "PREDICT"

_FIELD_RES = {
    "grid": re.compile(r"GRID:\s*(\d+)"),
    "pos": re.compile(r"POSITION:\s*\((\d+)\s*,\s*(\d+)\)"),
    "action": re.compile(r"ACTION:\s*(\w+)"),
    "goal": re.compile(r"BELIEVED_GOAL:\s*(?:\((\d+)\s*,\s*(\d+)\)|(unknown))"),
}
_MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


class LLMClient:
    def __init__(self) -> None:
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def complete(self, prompt: str) -> LLMResponse:
        response = self._complete(prompt)
        self.total_calls += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        return response

    def _complete(self, prompt: str) -> LLMResponse:
        raise NotImplementedError


class MockModel(LLMClient):
    """Deterministic offline model for tests and harness debugging.

    Without a script, it answers a policy prompt with a legal action chosen by a
    stable hash of (seed, prompt) — same prompt, same answer, across processes.
    With `scripted`, it replays the given responses in order (for parser tests).
    Token counts are length-derived so cost math is exactly testable.
    """

    ACTIONS = ("up", "down", "left", "right", "probe")

    def __init__(self, seed: int = 0, scripted: list[str] | None = None):
        super().__init__()
        self._seed = seed
        self._scripted = list(scripted) if scripted is not None else None

    def _complete(self, prompt: str) -> LLMResponse:
        if self._scripted is not None:
            if not self._scripted:
                raise RuntimeError("MockModel script exhausted")
            text = self._scripted.pop(0)
        elif prompt.startswith(PREDICT_MARKER):
            text = self._predict_reply(prompt)
        else:
            digest = hashlib.sha256(f"{self._seed}:{prompt}".encode()).digest()
            action = self.ACTIONS[int.from_bytes(digest[:4], "big") % len(self.ACTIONS)]
            text = f"action: {action}"
        return LLMResponse(
            text=text,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
        )

    @staticmethod
    def _predict_reply(prompt: str) -> str:
        """Answer a world-model prompt as a *competent* model: true clipped
        dynamics, reward 1 iff the next cell is the stated believed goal.
        Deterministic and exact so harness tests can assert optimal behavior."""
        size = int(_FIELD_RES["grid"].search(prompt).group(1))
        pos_match = _FIELD_RES["pos"].search(prompt)
        pos = (int(pos_match.group(1)), int(pos_match.group(2)))
        action = _FIELD_RES["action"].search(prompt).group(1).lower()
        goal_match = _FIELD_RES["goal"].search(prompt)
        goal = (
            None
            if goal_match.group(3) is not None
            else (int(goal_match.group(1)), int(goal_match.group(2)))
        )

        dr, dc = _MOVES.get(action, (0, 0))  # probe/unknown: stay in place
        last = size - 1
        next_pos = (min(max(pos[0] + dr, 0), last), min(max(pos[1] + dc, 0), last))
        reward = 1 if goal is not None and next_pos == goal else 0
        return f"NEXT: {next_pos} REWARD: {reward}"


class OpenAICompatibleClient(LLMClient):
    """Chat-completions client for vLLM-served open models (Amendment 6a) —
    or any OpenAI-compatible endpoint. Same counter interface as the rest;
    vLLM returns usage on every response. temperature=0 for greedy decoding
    (per-call determinism is still not guaranteed under server batching —
    reproducibility tests run on the mock only).

    Reasoning-tuned models (e.g. Qwen3) may emit <think>...</think> traces;
    they are stripped before parsing so an action word mentioned inside the
    reasoning cannot hijack the policy parser. Prefer serving with thinking
    disabled anyway (vLLM: --chat-template-kwargs '{"enable_thinking": false}').
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 256,
        timeout: float = 120.0,
    ):
        super().__init__()
        base = (
            base_url
            or os.environ.get("OPENAI_BASE_URL")
            or "http://localhost:8000"
        ).rstrip("/")
        self.url = f"{base}/v1/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

    def _complete(self, prompt: str) -> LLMResponse:
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        request = urllib.request.Request(
            self.url,
            data=body,
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as raw:
            payload = json.load(raw)
        text = payload["choices"][0]["message"]["content"] or ""
        text = _THINK_RE.sub("", text).strip()
        usage = payload["usage"]
        return LLMResponse(
            text=text,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
        )


def make_client(
    backend: str, model: str, seed: int = 0, base_url: str | None = None
) -> LLMClient:
    if backend == "mock":
        return MockModel(seed=seed)
    if backend == "anthropic":
        return AnthropicModel(model)
    if backend == "vllm":
        return OpenAICompatibleClient(model, base_url=base_url)
    raise ValueError(f"unknown backend {backend!r}")


class AnthropicModel(LLMClient):
    """Minimal Messages API client. Not exercised by tests (no network in CI);
    smoke-tested manually with a real key."""

    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 64):
        super().__init__()
        self.model = model
        self.max_tokens = max_tokens
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("no API key: pass api_key or set ANTHROPIC_API_KEY")

    def _complete(self, prompt: str) -> LLMResponse:
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        request = urllib.request.Request(
            ANTHROPIC_API_URL,
            data=body,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=60) as raw:
            payload = json.load(raw)
        text = "".join(
            block["text"] for block in payload["content"] if block["type"] == "text"
        )
        usage = payload["usage"]
        return LLMResponse(
            text=text,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
        )
