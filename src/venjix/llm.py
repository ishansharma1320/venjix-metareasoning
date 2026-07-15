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
import ssl
import time
import urllib.error
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


def _strip_thinking(text: str) -> str:
    """Remove <think> traces — including an UNCLOSED trace from a response
    truncated mid-reasoning (everything from '<think>' onward is dropped), so
    reasoning text can never reach the action parser."""
    text = _THINK_RE.sub("", text)
    return text.split("<think>")[0].strip()


def _extract_text(message: dict) -> str:
    """Assistant text from a chat-completions message.

    vLLM servers running a reasoning parser (e.g. --reasoning-parser qwen3)
    can return content=null with the ENTIRE answer in reasoning_content — even
    when the request disables thinking (observed live on Qwen3-4B). Prefer
    content; fall back to reasoning_content only when content is empty. We
    always request thinking off, so the fallback carries the short final
    answer, and the think-strip guards the truncated-trace case regardless."""
    text = _strip_thinking(message.get("content") or "")
    if text:
        return text
    return _strip_thinking(message.get("reasoning_content") or "")


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

    def complete(self, prompt: str, response_regex: str | None = None) -> LLMResponse:
        """response_regex: optional constrained-decoding pattern. Honored by the
        vLLM-backed clients (guided_regex); a no-op for mock/anthropic. Used by
        world-model prediction calls to pin the NEXT/REWARD reply format."""
        response = self._complete(prompt, response_regex)
        self.total_calls += 1
        self.total_input_tokens += response.input_tokens
        self.total_output_tokens += response.output_tokens
        return response

    def _complete(self, prompt: str, response_regex: str | None) -> LLMResponse:
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

    def _complete(self, prompt: str, response_regex: str | None = None) -> LLMResponse:
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

    def _complete(self, prompt: str, response_regex: str | None = None) -> LLMResponse:
        request_body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            # vLLM-standard: disables Qwen3-style thinking, which otherwise
            # burns the whole token budget on a truncated <think> trace and
            # returns empty content. Ignored by templates without the kwarg.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if response_regex is not None:
            request_body["guided_regex"] = response_regex  # vLLM constrained decoding
        body = json.dumps(request_body).encode()
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
        text = _extract_text(payload["choices"][0]["message"])
        usage = payload["usage"]
        return LLMResponse(
            text=text,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
        )


class VastServerlessClient(LLMClient):
    """Vast.ai serverless wrapper around a vLLM worker (two-hop protocol):

    1. POST https://run.vast.ai/route/ with {endpoint, api_key, cost} — returns
       the worker URL plus a one-time signature, so a fresh route is fetched
       for every call.
    2. POST {worker}/v1/chat/completions with the signed route passed through
       as `auth_data` and the standard OpenAI/vLLM payload under `payload`.
       Workers use a self-signed cert, so TLS verification is disabled for the
       worker hop only (the route hop verifies normally).

    `cost` is Vast's per-request cost claim, sized to max_tokens as in their
    examples. Transient failures (route or worker) retry with backoff — one
    dead step would otherwise kill a whole 40-episode condition. Same counter
    interface and <think>-stripping as the other clients.
    """

    ROUTE_URL = "https://run.vast.ai/route/"

    def __init__(
        self,
        model: str,
        endpoint: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 256,
        timeout: float = 120.0,
        retries: int = 3,
    ):
        super().__init__()
        self.model = model
        self.endpoint = endpoint or os.environ.get("VAST_ENDPOINT", "qwen-llm")
        self._api_key = api_key or os.environ.get("VAST_API_KEY")
        if not self._api_key:
            raise ValueError("no Vast API key: pass api_key or set VAST_API_KEY")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self._worker_ctx = ssl.create_default_context()
        self._worker_ctx.check_hostname = False
        self._worker_ctx.verify_mode = ssl.CERT_NONE

    def _post_json(self, url: str, body: dict, context: ssl.SSLContext | None = None) -> dict:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout, context=context) as raw:
            return json.load(raw)

    def _complete(self, prompt: str, response_regex: str | None = None) -> LLMResponse:
        inner_payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
            # see OpenAICompatibleClient: keep Qwen3 thinking off
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if response_regex is not None:
            inner_payload["guided_regex"] = response_regex  # vLLM constrained decoding
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                route = self._post_json(
                    self.ROUTE_URL,
                    {
                        "endpoint": self.endpoint,
                        "api_key": self._api_key,
                        "cost": self.max_tokens,
                    },
                )
                payload = self._post_json(
                    f"{route['url'].rstrip('/')}/v1/chat/completions",
                    {
                        "auth_data": route,  # signed route, passed through as-is
                        "payload": inner_payload,
                    },
                    context=self._worker_ctx,
                )
                text = _extract_text(payload["choices"][0]["message"])
                usage = payload["usage"]
                return LLMResponse(
                    text=text,
                    input_tokens=usage["prompt_tokens"],
                    output_tokens=usage["completion_tokens"],
                )
            except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < self.retries - 1:
                    time.sleep(2.0**attempt)
        raise RuntimeError(
            f"vast call failed after {self.retries} attempts: {last_error!r}"
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
    if backend == "vast":
        return VastServerlessClient(model)
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

    def _complete(self, prompt: str, response_regex: str | None = None) -> LLMResponse:
        # response_regex ignored: the Anthropic API has no guided decoding;
        # the parser's fallback handles format drift there.
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
