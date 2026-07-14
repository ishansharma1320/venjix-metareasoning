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
import urllib.request
from dataclasses import dataclass

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


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
        else:
            digest = hashlib.sha256(f"{self._seed}:{prompt}".encode()).digest()
            action = self.ACTIONS[int.from_bytes(digest[:4], "big") % len(self.ACTIONS)]
            text = f"action: {action}"
        return LLMResponse(
            text=text,
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
        )


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
